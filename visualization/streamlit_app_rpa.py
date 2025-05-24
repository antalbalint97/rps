# performance_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rpsa_client import RPSAClient
from collections import defaultdict

# --- Config ---
API_KEY = "5229c6e5626944f39dda2d28662a69ee"
BASE_URL = "https://rockpapercode.onespire.hu/api/v1/public"
client = RPSAClient(api_key=API_KEY, base_url=BASE_URL)

# --- UI Header ---
st.title("Strategy Performance Dashboard")

# --- Strategy Selector ---
strategies = client.list_regular_strategies(page=1, per_page=25).data
strategy_options = {s.strategy_name: s.strategy_id for s in strategies}
strategy_name = st.selectbox("Select your strategy:", list(strategy_options.keys()))
strategy_id = strategy_options[strategy_name]

# --- Summary ---
st.header("Strategy Summary")
summary = client.get_strategy_summary(strategy_id=strategy_id)
st.json(summary.model_dump())

# --- Leaderboard ---
st.header("Latest Arena Leaderboard")
arenas = client.list_regular_arenas(page=1, per_page=10).data
latest_arena = next((a for a in arenas if a.number_strategies > 1), None)

if latest_arena:
    leaderboard = client.get_arena_leaderboard(arena_id=latest_arena.id)
    df_leaderboard = pd.DataFrame(leaderboard).sort_values(by="avg_points_per_game", ascending=False)
    df_leaderboard["highlight"] = df_leaderboard["strategy_id"] == strategy_id
    colors = df_leaderboard["highlight"].map(lambda x: "lightblue" if x else "white")

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Name", "Avg PPG", "Games", "Wins", "Losses", "Ties", "Net Score", "Win Rate"],
                    fill_color="paleturquoise",
                    align="left"),
                cells=dict(
                    values=[
                        df_leaderboard["strategy_name"],
                        df_leaderboard["avg_points_per_game"],
                        df_leaderboard["games_played"],
                        df_leaderboard["wins"],
                        df_leaderboard["losses"],
                        df_leaderboard["ties"],
                        df_leaderboard["net_score"],
                        df_leaderboard["win_rate"]],
                    fill_color=[colors],
                    align="left"))])
    st.plotly_chart(fig_table, use_container_width=True)
else:
    st.info("No arena found.")


# --- Head-to-Head ---
st.header("Head-to-Head Performance")
h2h = client.get_strategy_head_to_head(strategy_id=strategy_id)
df_h2h = pd.DataFrame(h2h)
id_to_name = {s.strategy_id: s.strategy_name for s in strategies}
df_h2h["opponent_name"] = df_h2h["opponent_strategy_id"].map(id_to_name).fillna("Unknown")
df_h2h["archetype"] = df_h2h["opponent_name"].map(archetypes).fillna("unknown")

styled_h2h = df_h2h.style.applymap(
    lambda x: 'background-color: #ffcdd2' if isinstance(x, float) and x < 0.3 else
              'background-color: #c8e6c9' if isinstance(x, float) and x > 0.6 else '',
    subset=['win_rate']
)
st.dataframe(styled_h2h)

# --- Dominance Metrics ---
st.header("Dominance Metrics vs Strong Opponents")
all_games = client.list_arena_games(latest_arena.id, page=1, per_page=200).data
dom_records = []

for game in all_games:
    try:
        result = client.get_game_results(game.id)
        my_row, opp_row = None, None
        for r in result:
            if r.strategy_id == strategy_id:
                my_row = r
            else:
                opp_row = r
        if my_row and opp_row:
            diff = my_row.score - opp_row.score
            win_type = ("Blowout Win" if diff > 0.01 else
                        "Close Win" if diff > 0 else
                        "Tie" if diff == 0 else
                        "Blowout Loss" if diff < -0.01 else "Close Loss")
            dom_records.append({
                "opponent_id": opp_row.strategy_id,
                "opponent_name": id_to_name.get(opp_row.strategy_id, "Unknown"),
                "my_score": my_row.score,
                "opp_score": opp_row.score,
                "score_diff": diff,
                "win_type": win_type
            })
    except:
        continue

if dom_records:
    df_dom = pd.DataFrame(dom_records)
    agg_dom = df_dom.groupby("opponent_name").agg(
        avg_my_score=("my_score", "mean"),
        avg_opp_score=("opp_score", "mean"),
        dominance_index=("score_diff", "mean"),
        score_std_dev=("score_diff", "std"),
        game_count=("score_diff", "count")
    ).reset_index()

    win_type_counts = df_dom.groupby(["opponent_name", "win_type"]).size().unstack(fill_value=0).reset_index()
    agg_dom = agg_dom.merge(win_type_counts, on="opponent_name", how="left")

    # Blowout Ratio
    agg_dom["blowout_ratio"] = (agg_dom.get("Blowout Win", 0) + agg_dom.get("Blowout Loss", 0)) / agg_dom["game_count"]

    # Add opponent archetypes
    agg_dom["archetype"] = agg_dom["opponent_name"].map(archetypes).fillna("unknown")

    st.dataframe(agg_dom)
    fig_heat = px.imshow(
        agg_dom.pivot_table(index="opponent_name", values="dominance_index"),
        labels=dict(color="Dominance Index"),
        color_continuous_scale="RdBu")
    st.plotly_chart(fig_heat)

    st.download_button("Download Metrics CSV", agg_dom.to_csv(index=False).encode(), "dominance_metrics.csv")

# --- Adaptivity Curve Analyzer ---
st.header("Adaptivity Curve Analyzer")

from statistics import mean, stdev

@st.cache_data(ttl=3600)
def cached_game_results(game_id):
    return client.get_game_results(game_id)

def get_win_rate_bins(strategy_id, opponent_id):
    try:
        latest_arena = client.list_regular_arenas(page=1, per_page=1).data[0]
        all_games = client.list_arena_games(latest_arena.id, page=1, per_page=200).data
    except Exception as e:
        st.error(f"Failed to load latest arena or games: {e}")
        return {
            "early": 0, "mid": 0, "late": 0,
            "learning_curve": 0, "volatility": 0,
            "adaptivity_label": "unknown"
        }

    bins = {"early": [], "mid": [], "late": []}

    for game in all_games:
        ids = {game.strategy_a_id, game.strategy_b_id}
        if strategy_id in ids and opponent_id in ids:
            try:
                results = client.get_game_results(game.id)
                for r in results:
                    if r.strategy_id == strategy_id:
                        total = r.wins + r.losses + r.ties
                        if total == 0:
                            continue
                        wr = r.wins / total
                        third = total // 3
                        bins["early"].append(wr * third)
                        bins["mid"].append(wr * third)
                        bins["late"].append(wr * (total - 2 * third))
            except:
                continue

    early = mean(bins["early"]) if bins["early"] else 0
    mid = mean(bins["mid"]) if bins["mid"] else 0
    late = mean(bins["late"]) if bins["late"] else 0
    values = [early, mid, late]
    learning_curve = late - early
    volatility = stdev(values) if len(set(values)) > 1 else 0

    if learning_curve > 0.05:
        label = "fast_learner"
    elif volatility > 0.05:
        label = "chaotic"
    else:
        label = "flat"

    return {
        "early": round(early, 3),
        "mid": round(mid, 3),
        "late": round(late, 3),
        "learning_curve": round(learning_curve, 3),
        "volatility": round(volatility, 3),
        "adaptivity_label": label
    }


adaptivity_records = []
for opponent in df_leaderboard["strategy_id"]:
    if opponent == strategy_id:
        continue
    name = df_leaderboard[df_leaderboard["strategy_id"] == opponent]["strategy_name"].values[0]
    bins = get_win_rate_bins(opponent, strategy_id)
    adaptivity_records.append({
        "strategy_name": name,
        "early_winrate": bins["early"],
        "mid_winrate": bins["mid"],
        "late_winrate": bins["late"],
        "learning_curve": bins["learning_curve"],
        "volatility": bins["volatility"],
        "adaptivity_label": bins["adaptivity_label"]
    })

if adaptivity_records:
    df_curve = pd.DataFrame(adaptivity_records).sort_values("learning_curve", ascending=False)
    st.subheader("Opponent Adaptivity Classification")
    st.dataframe(df_curve)

    fig_curve = px.line(
        df_curve.melt(id_vars=["strategy_name", "adaptivity_label"], value_vars=["early_winrate", "mid_winrate", "late_winrate"]),
        x="variable", y="value", color="strategy_name", line_dash="adaptivity_label",
        title="Adaptivity Curves (Early → Late Winrate)"
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    st.download_button("Download Adaptivity Metrics CSV", df_curve.to_csv(index=False).encode(), "adaptivity_curve.csv")

# --- Adaptivity Summary and Meta-Strategy Recommender ---
st.header("Adaptivity Summary and Counter-Strategy Recommender")

# Merge adaptivity labels into h2h data
df_combined = df_h2h.merge(
    df_curve[["strategy_name", "adaptivity_label"]],
    left_on="opponent_name",
    right_on="strategy_name",
    how="left",
    suffixes=("", "_drop")
)

if "strategy_name_drop" in df_combined.columns:
    df_combined = df_combined.drop(columns=["strategy_name_drop"])

# Aggregate summary by adaptivity label
summary_table = df_combined.groupby("adaptivity_label").agg(
    avg_win_rate=("win_rate", "mean"),
    std_win_rate=("win_rate", "std"),
    count=("win_rate", "count")
).reset_index().sort_values("avg_win_rate", ascending=False)

# Color styling
def highlight_adaptivity(val):
    if val == "fast_learner":
        return "background-color: #a5d6a7"
    elif val == "chaotic":
        return "background-color: #ffe082"
    elif val == "flat":
        return "background-color: #ef9a9a"
    else:
        return ""

styled_summary = summary_table.style.applymap(highlight_adaptivity, subset=["adaptivity_label"])
st.subheader("Adaptivity Class Summary")
st.dataframe(styled_summary)

# Recommend counter-strategy type based on opponent archetype success
meta_recommend = df_combined.groupby(["adaptivity_label", "archetype"]).agg(
    avg_win_rate_vs_archetype=("win_rate", "mean"),
    num_games=("win_rate", "count")
).reset_index()

# Choose best archetype per adaptivity class
best_counters = meta_recommend.loc[meta_recommend.groupby("adaptivity_label")["avg_win_rate_vs_archetype"].idxmax()]
best_counters = best_counters.rename(columns={"archetype": "recommended_counter_archetype"})

# Merge back into main curve for training labeling
df_full = df_curve.merge(best_counters[["adaptivity_label", "recommended_counter_archetype"]], on="adaptivity_label", how="left")
df_full = df_full.merge(df_h2h[["opponent_name", "win_rate"]], left_on="strategy_name", right_on="opponent_name", how="left")
df_full["your_strategy"] = strategy_name
df_full = df_full.rename(columns={
    "strategy_name": "opponent_name",
    "win_rate": "your_win_rate_against"
})

st.subheader("Counter Strategy Recommender Table")
st.dataframe(df_full[[
    "opponent_name", "adaptivity_label", "your_win_rate_against",
    "recommended_counter_archetype", "early_winrate",
    "mid_winrate", "late_winrate", "learning_curve", "volatility"
]])

# Export unified training dataset
st.download_button(
    "Download Training Dataset as CSV",
    df_full.to_csv(index=False).encode(),
    file_name="rps_adaptivity_training_data.csv"
)

# --- Cleanup ---
client.close()
