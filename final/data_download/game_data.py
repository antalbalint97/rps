import json
from rpsa_client import RPSAClient

# --- Configuration ---
API_KEY = "5229c6e5626944f39dda2d28662a69ee"
BASE_URL = "https://rockpapercode.onespire.hu/api/v1/public"
client = RPSAClient(api_key=API_KEY, base_url=BASE_URL)

# --- Download Latest Arena Game Results ---
def download_latest_arena_data(output_path="latest_arena_games.json"):
    arenas = client.list_regular_arenas(page=1, per_page=1).data
    if not arenas:
        raise ValueError("No arenas found.")
    latest_arena = arenas[0]

    games = client.list_arena_games(arena_id=latest_arena.id, page=1, per_page=200).data
    data_to_save = []

    for game in games:
        try:
            results = client.get_game_results(game.id)
            data_to_save.append({
                "game_id": game.id,
                "strategy_a_id": game.strategy_a_id,
                "strategy_b_id": game.strategy_b_id,
                "results": [r.model_dump() for r in results]
            })
        except Exception as e:
            print(f"Failed to fetch game {game.id}: {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved {len(data_to_save)} games to {output_path}")

# Run it
if __name__ == "__main__":
    download_latest_arena_data()
