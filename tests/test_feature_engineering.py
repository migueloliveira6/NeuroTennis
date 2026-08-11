import pandas as pd

from clustering.style_clustering.feature_engineering import build_player_season_table


def test_service_hold_rate_uses_break_points_saved_instead_of_converted() -> None:
    player_match = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Test Player",
                "player_name_clean": "test player",
                "season": 2024,
                "won": 1,
                "minutes": 120,
                "sets_played": 3,
                "tiebreaks_played": 0,
                "aces": 10,
                "double_faults": 2,
                "service_points": 100,
                "first_serves_in": 60,
                "first_serve_points_won": 44,
                "second_serve_points_won": 20,
                "service_games": 10,
                "break_points_saved": 4,
                "break_points_faced": 6,
                "break_points_opportunities": 5,
                "break_points_converted": 1,
                "return_points_total": 100,
                "return_points_won": 49,
                "opponent_service_games": 10,
                "player_rank": 1,
                "player_rank_points": 1000,
                "opponent_rank": 2,
                "opponent_rank_points": 800,
                "total_points_won": 100,
                "total_points_played": 200,
                "total_points_lost": 100,
                "surface_match": 1,
                "hard_match": 1,
                "clay_match": 0,
                "grass_match": 0,
                "surface": "Hard",
            }
        ]
    )

    result = build_player_season_table(player_match)

    assert result.loc[0, "service_hold_rate"] == 0.8
