import importlib
import math
from collections import defaultdict
from config_loader import load_config
from statistics import mean, variance

class GameEngine:
    def __init__(self):
        self.conf = load_config()
        self.NUM_PLAYS_PER_GAME = self.conf["num_plays_per_game"]
        self.NUM_GAMES_PER_MATCH = self.conf["num_games_per_match"]
        self.NO_POINT_THRESHOLD = self.conf["no_point_threshold"]
        self.MAX_SCORE = self.conf["max_score"]

    def play_single_round(self, strategy1, strategy2):
        move1 = strategy1.play()
        move2 = strategy2.play()

        if move1 == move2:
            result = 0
        elif (
            (move1 == "rock" and move2 == "scissors") or
            (move1 == "scissors" and move2 == "paper") or
            (move1 == "paper" and move2 == "rock")
        ):
            result = 1
        else:
            result = 2

        strategy1.handle_moves(move1, move2)
        strategy2.handle_moves(move2, move1)
        return result

    def play_single_game(self, strategy1_class, strategy2_class):
        s1 = strategy1_class()
        s2 = strategy2_class()

        wins1, wins2, draws = 0, 0, 0
        for _ in range(self.NUM_PLAYS_PER_GAME):
            result = self.play_single_round(s1, s2)
            if result == 1:
                wins1 += 1
            elif result == 2:
                wins2 += 1
            else:
                draws += 1

        norm1, norm2 = self.compute_normalized_proportions(wins1, wins2)
        score1 = norm1 * self.MAX_SCORE
        score2 = norm2 * self.MAX_SCORE
        return norm1, norm2, score1, score2, wins1, wins2, draws

    def compute_normalized_proportions(self, wins1, wins2):
        total = wins1 + wins2
        if total == 0 or abs(wins1 - wins2) < total * self.NO_POINT_THRESHOLD:
            return 0.0, 0.0

        prop1 = wins1 / total - 0.5
        prop2 = wins2 / total - 0.5
        norm1 = prop1 / 0.5
        norm2 = prop2 / 0.5
        return norm1, norm2

    def round_robin_tournament(self, strategy_classes):
        total_scores = defaultdict(float)
        score_records = defaultdict(list)
        normalized_records = defaultdict(list)
        games_played = defaultdict(int)
        matchups = []
        detailed_stats = []
        matchup_results = defaultdict(list)

        for i in range(len(strategy_classes)):
            for j in range(i + 1, len(strategy_classes)):
                s1_cls = strategy_classes[i]
                s2_cls = strategy_classes[j]
                s1 = s1_cls()
                s2 = s2_cls()

                sum_norm1, sum_norm2 = 0.0, 0.0
                sum_score1, sum_score2 = 0.0, 0.0
                total_wins1, total_wins2, total_draws = 0, 0, 0

                for _ in range(self.NUM_GAMES_PER_MATCH):
                    norm1, norm2, score1, score2, wins1, wins2, draws = self.play_single_game(s1_cls, s2_cls)
                    sum_norm1 += norm1
                    sum_norm2 += norm2
                    sum_score1 += score1
                    sum_score2 += score2
                    total_wins1 += wins1
                    total_wins2 += wins2
                    total_draws += draws

                avg_score1 = sum_score1 / self.NUM_GAMES_PER_MATCH
                avg_score2 = sum_score2 / self.NUM_GAMES_PER_MATCH
                avg_norm1 = sum_norm1 / self.NUM_GAMES_PER_MATCH
                avg_norm2 = sum_norm2 / self.NUM_GAMES_PER_MATCH

                total_scores[s1_cls.name] += avg_score1
                total_scores[s2_cls.name] += avg_score2
                score_records[s1_cls.name].append(avg_score1)
                score_records[s2_cls.name].append(avg_score2)
                normalized_records[s1_cls.name].append(avg_norm1)
                normalized_records[s2_cls.name].append(avg_norm2)
                games_played[s1_cls.name] += 1
                games_played[s2_cls.name] += 1

                matchup_results[s1_cls.name].append(avg_score1)
                matchup_results[s2_cls.name].append(avg_score2)

                decisive_rounds = total_wins1 + total_wins2 or 1

                print(f"\n--- Match: {s1_cls.name} vs {s2_cls.name} ---")
                print(f"Wins: {total_wins1} / {total_wins2} | Draws: {total_draws}")
                print(f"Winrates: {total_wins1 / decisive_rounds:.2%} / {total_wins2 / decisive_rounds:.2%}")
                print(f"Averaged Score: {avg_score1:.2f} / {avg_score2:.2f}")
                print(f"Normalized Proportion: {avg_norm1:.2f} / {avg_norm2:.2f}")

                matchups.append((s1_cls.name, s2_cls.name, avg_score1, avg_score2))

                detailed_stats.append({
                    "strategies": (s1_cls.name, s2_cls.name),
                    "wins": (total_wins1, total_wins2),
                    "draws": total_draws,
                    "winrates": (
                        total_wins1 / decisive_rounds,
                        total_wins2 / decisive_rounds
                    ),
                    "scores": (avg_score1, avg_score2),
                    "normalized": (avg_norm1, avg_norm2)
                })

        final_avg_scores = {
            name: total_scores[name] / games_played[name] if games_played[name] else 0.0
            for name in total_scores
        }

        score_stats = {}
        for name in score_records:
            scores = score_records[name]
            norms = normalized_records[name]
            n = len(scores)
            avg = final_avg_scores[name]
            var = variance(scores) if n > 1 else 0.0
            std_dev = math.sqrt(var)
            std_error = std_dev / math.sqrt(n) if n > 1 else 0.0
            margin = 1.96 * std_error
            ci_low = avg - margin
            ci_high = avg + margin
            worst_score = min(scores) if scores else 0.0
            win_ratio = sum(1 for s in scores if s > 0) / n if n else 0.0
            dominance_index = sum(1 for s in scores if s > 10) / n if n else 0.0

            score_stats[name] = {
                "average": avg,
                "total": sum(scores) * self.NUM_GAMES_PER_MATCH,
                "variance": var,
                "std_dev": std_dev,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "normalized_avg": mean(norms) if norms else 0.0,
                "worst_score": worst_score,
                "win_ratio": win_ratio,
                "dominance_index": dominance_index,
                "matchups_played": n
            }

        return {
            "final_scores": final_avg_scores,
            "score_stats": score_stats,
            "matchups": matchups,
            "metrics": detailed_stats
        }
