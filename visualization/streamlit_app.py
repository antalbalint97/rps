import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Ensure proper path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import GameEngine
from strategies.random_strategy import RandomStrategy
from strategies.enhanced_strategy import EnhancedStrategy
from strategies.copycat_strategy import CopycatStrategy
from strategies.cycle_strategy import CycleStrategy
from strategies.alwaysrock_strategy import AlwaysRockStrategy
from strategies.frequency_strategy import FrequencyStrategy
from strategies.lastn_strategy import LastNStrategy
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.mirrorbaiter_strategy import MirrorBaiterStrategy
from strategies.fmpnet_strategy import FMPNetStrategy
from strategies.delayedmirror_strategy import DelayedMirrorStrategy
from strategies.noisecounter_strategy import NoiseCounterStrategy
from strategies.entropymax_strategy import EntropyMaximizerStrategy


# Strategy name map
strategy_map = {
    "RandomStrategy": RandomStrategy,
    "EnhancedStrategy": EnhancedStrategy,
    "CopycatStrategy": CopycatStrategy,
    "CycleStrategy": CycleStrategy,
    "AlwaysRockStrategy": AlwaysRockStrategy,
    "FrequencyStrategy": FrequencyStrategy,
    "LatNStrategy": LastNStrategy,
    "MarkovStrategy": MarkovStrategy,
    "QLearningStrategy": QLearningStrategy,
    "MirrorBaiterStrategy": MirrorBaiterStrategy,
    "FMPNetStrategy": FMPNetStrategy,
    "DelayedMirrorStrategy": DelayedMirrorStrategy,
    "NoiseCounterStrategy": NoiseCounterStrategy
}

st.set_page_config(page_title="RPS Arena Dashboard", layout="wide")
st.title("RPS Arena - Strategy Analysis")
with st.expander("How to Interpret Strategy Metrics", expanded=False):
    st.markdown("""
    **Metrics Explained:**
    
    - **Average Score**: Mean performance per game across all matchups. Higher is better.
    - **Total Score**: Cumulative score across all games. Scales with number of matchups.
    - **Variance** & **Standard Deviation**: Show score consistency. Lower means more stable performance.
    - **Confidence Interval (CI)**: Range where the true average likely falls. If the entire CI is above 0, the strategy is likely stronger than average.
    - **Normalized Proportion**: Scaled performance from -1 to 1. Helps compare across strategies regardless of raw scores.
    - **Worst Score**: The weakest performance against a single opponent — useful to assess vulnerabilities.
    - **Win Ratio**: Proportion of matchups where the strategy scored better than the opponent (score > 0). Reflects consistency.
    - **Dominance Index**: Proportion of matchups where the strategy significantly outperformed the opponent (e.g., score > +10). Reflects confident wins, not just narrow victories.

    **Visuals:**
    - **Confidence Interval Chart**: Shows the average score and uncertainty for each strategy.
    - **Dominance vs Win Ratio**: Compares strong wins (dominance) with general consistency (win ratio).

    ✅ Use **Average Score + Dominance Index** to find top-tier strategies.  
    ⚠️ Use **Worst Score + Variance** to spot fragile or inconsistent ones.
    """)

# Sidebar configuration
st.sidebar.header("Arena Configuration")
selected_strategies = st.sidebar.multiselect(
    "Select Strategies",
    options=list(strategy_map.keys()),
    default=list(strategy_map.keys())
)

num_plays_per_game = st.sidebar.slider("Rounds per Game", 10, 2000, 100, step=10)
num_games_per_match = st.sidebar.slider("Games per Matchup", 1, 100, 10)
run_arena = st.sidebar.button("Run Arena")

# Run simulation
if run_arena and len(selected_strategies) >= 2:
    engine = GameEngine()
    engine.NUM_PLAYS_PER_GAME = num_plays_per_game
    engine.NUM_GAMES_PER_MATCH = num_games_per_match

    strategy_classes = [strategy_map[name] for name in selected_strategies]

    with st.spinner("Running tournament..."):
        results = engine.round_robin_tournament(strategy_classes)

    score_stats = results["score_stats"]

   # Replace the score_df creation block with this:
    score_df = pd.DataFrame([
        {
            "Strategy": name,
            "Average Score": data["average"],
            "Total Score": data["total"],
            "Variance": data["variance"],
            "Standard Deviation": data["std_dev"],
            "CI Lower": data["ci_low"],
            "CI Upper": data["ci_high"],
            "Normalized Proportion": data["normalized_avg"],
            "Worst Score": data["worst_score"],
            "Win Ratio": data["win_ratio"],
            "Dominance Index": data["dominance_index"],
            "Matchups Played": data["matchups_played"]
        }
        for name, data in score_stats.items()
    ])
    score_df = score_df.sort_values(by="Average Score", ascending=False)

    # Confidence Intervals using Plotly
    st.subheader("Final Scores with Variance")
    ci_fig = go.Figure()
    ci_fig.add_trace(go.Bar(
        x=score_df["Strategy"],
        y=score_df["Average Score"],
        error_y=dict(
            type='data',
            symmetric=False,
            array=score_df["CI Upper"] - score_df["Average Score"],
            arrayminus=score_df["Average Score"] - score_df["CI Lower"],
            thickness=2,
            width=5
        ),
        marker_color=px.colors.qualitative.Pastel
    ))
    ci_fig.update_layout(
        yaxis_title="Average Score",
        title="Confidence Intervals by Strategy",
        xaxis_tickangle=30,
        bargap=0.4
    )
    st.plotly_chart(ci_fig, use_container_width=True)

    #Dominance Matrix
    st.subheader("Dominance and Win Ratio")

    dom_fig = go.Figure()
    dom_fig.add_trace(go.Bar(
        x=score_df["Strategy"],
        y=score_df["Dominance Index"],
        name="Dominance Index",
        marker_color="indianred"
    ))
    dom_fig.add_trace(go.Bar(
        x=score_df["Strategy"],
        y=score_df["Win Ratio"],
        name="Win Ratio",
        marker_color="seagreen"
    ))
    dom_fig.update_layout(
        barmode='group',
        yaxis_title="Metric Value",
        title="Dominance Index vs Win Ratio",
        xaxis_tickangle=30
    )
    st.plotly_chart(dom_fig, use_container_width=True)

    # Summary table
    st.subheader("Strategy Summary")
    st.dataframe(score_df, use_container_width=True)

    st.download_button(
        label="Download Strategy Summary as CSV",
        data=score_df.to_csv(index=False),
        file_name="strategy_summary.csv",
        mime="text/csv"
    )

    # Matchup breakdown
    st.subheader("Matchup Results")
    matchup_data = []
    for entry in results["metrics"]:
        s1, s2 = entry["strategies"]
        wins1, wins2 = entry["wins"]
        draws = entry["draws"]
        wr1, wr2 = entry["winrates"]
        score1, score2 = entry["scores"]
        norm1, norm2 = entry["normalized"]
        matchup_data.append({
            "Strategy 1": s1,
            "Strategy 2": s2,
            "S1 Wins": wins1,
            "S2 Wins": wins2,
            "Draws": draws,
            "S1 Winrate": f"{wr1:.2%}",
            "S2 Winrate": f"{wr2:.2%}",
            "S1 Final Score": round(score1, 2),
            "S2 Final Score": round(score2, 2),
            "S1 Normalized": round(norm1, 3),
            "S2 Normalized": round(norm2, 3)
        })

    matchup_df = pd.DataFrame(matchup_data)
    st.dataframe(matchup_df, use_container_width=True)

    st.download_button(
        label="Download Matchup Results as CSV",
        data=matchup_df.to_csv(index=False),
        file_name="matchup_results.csv",
        mime="text/csv"
    )

     # Matchup-specific heatmap
    st.subheader("Matchup Heatmap")

    selected_heatmap_strategy = st.selectbox(
        "Select Strategy for Matchup Heatmap",
        options=score_df["Strategy"].unique()
    )

    # Prepare matrix of selected strategy vs others
    filtered = matchup_df[
        (matchup_df["Strategy 1"] == selected_heatmap_strategy) |
        (matchup_df["Strategy 2"] == selected_heatmap_strategy)
    ]

    heatmap_records = []
    for _, row in filtered.iterrows():
        if row["Strategy 1"] == selected_heatmap_strategy:
            opponent = row["Strategy 2"]
            score = row["S1 Final Score"]
        else:
            opponent = row["Strategy 1"]
            score = row["S2 Final Score"]
        heatmap_records.append((selected_heatmap_strategy, opponent, score))

    heatmap_df = pd.DataFrame(heatmap_records, columns=["Strategy", "Opponent", "Score"])
    heatmap_df = heatmap_df.pivot(index="Strategy", columns="Opponent", values="Score")

    # Render heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(len(heatmap_df.columns) * 1.2, 1.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"label": "Avg Score"}, ax=ax)
    ax.set_title(f"Matchup Scores for {selected_heatmap_strategy}", fontsize=12)
    st.pyplot(fig)

else:
    st.info("Choose at least two strategies and press 'Run Arena' to start.")
