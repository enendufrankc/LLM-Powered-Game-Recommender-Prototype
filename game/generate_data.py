import json
import os
import random
import time
from typing import List, Dict, Union
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
OUT_FILE = os.path.join(os.getcwd(), "game", "database", "games.json")

class Game(BaseModel):
    id: str
    name: str
    theme: str
    volatility: Literal["Low", "Medium", "High"]
    rtp: float
    reels: int
    paylines: Union[int, Literal["cluster"]]
    max_win_x: int
    special_features: List[str]
    art_style: str
    provider_style: str
    target_audience: str
    description: str
    tags: List[str]

class GameCollection(BaseModel):
    games: List[Game]
    
def llm_generate_games(openai_client, n=12, model="gpt-4o-2024-08-06") -> List[Dict]:
    system_prompt = f"""
    You are a slot game data generator. Create diverse, realistic slot machine games
    with varied themes, volatility levels, RTPs (92-97%), art styles, and features.
    Make each game unique with creative names and engaging descriptions.
    You must generate exactly {n} games in your response.
    Use the below JSON format for your response:
    note: id's should an alphanumeric string
    """

    user_prompt = f'Generate {n} unique slot game entries. Each game should be completely different with varied themes, features, and characteristics. Your response should be in JSON format with exactly {n} games in the games array.'

    for attempt in range(3):
        try:
            response = openai_client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=GameCollection,
                max_tokens=4000,
                temperature=0.8,
            )
            
            game_collection = response.choices[0].message.parsed
            
            if not game_collection or len(game_collection.games) < n:
                print(f"Generated only {len(game_collection.games)} games, trying again...")
                continue
            
            # Convert to dict format
            return [game.model_dump() for game in game_collection.games]

        except Exception as e:
            print(f"LLM generation attempt {attempt+1} failed: {e}")
            time.sleep(1 + attempt * 2)
    
    raise RuntimeError("LLM generation failed after retries")


def generate_games_in_batches(client, total_games=100, batch_size=10):
    """Generate games in batches and return all games"""
    all_games = []
    num_batches = total_games // batch_size
    
    print(f"Generating {total_games} games in {num_batches} batches of {batch_size}...")
    
    for batch_num in range(num_batches):
        print(f"\nGenerating batch {batch_num + 1}/{num_batches}...")
        try:
            batch_games = llm_generate_games(client, n=batch_size)
            all_games.extend(batch_games)
            print(f"Successfully generated {len(batch_games)} games. Total: {len(all_games)}")
            
            # Add a small delay between batches to be respectful to the API
            time.sleep(2)
            
        except Exception as e:
            print(f"Failed to generate batch {batch_num + 1}: {e}")
            continue
    
    return all_games

def save_games_to_json(games, filename):
    """Save games to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"games": games}, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(games)} games to {filename}")
    except Exception as e:
        print(f"Error saving games to file: {e}")
        raise
    
def main():
    """Main pipeline to generate and save games"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Set it in your environment and re-run.")
        print("Temporary (PowerShell): $env:OPENAI_API_KEY = 'sk-...'")
        print("Persistent (PowerShell): setx OPENAI_API_KEY \"sk-...\"  (restart shell after setx)")
        return

    client = OpenAI(api_key=api_key)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    # Generate new games
    new_games = generate_games_in_batches(client, total_games=100, batch_size=10)
    
    # Combine with existing games
    all_games = new_games
    
    # Save to file
    save_games_to_json(all_games, OUT_FILE)
    
    print(f"\nPipeline completed!")
    print(f"Total games in file: {len(all_games)}")
    print(f"File saved to: {OUT_FILE}")
    
main()