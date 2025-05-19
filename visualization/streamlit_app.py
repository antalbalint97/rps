import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

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
from strategies.delayedmirror_strategy import DelayedMirrorStrategy
from strategies.noisecounter_strategy import NoiseCounterStrategy
from strategies.entropymax_strategy import EntropyMaximizerStrategy
from strategies.adaptiveswitcher_strategy import AdaptiveSwitcherStrategy
from strategies.patternhunter_strategy import PatternHunterStrategy
from strategies.bayesian_strategy import BayesianUpdateStrategy
from strategies.fmpnet_strategy import FMPNetStrategy
from strategies.lstm_strategy import LSTMStrategy
from strategies.gru_strategy import GRUStrategy
from strategies.thompsonmeta_strategy import ThompsonMetaStrategy
from strategies.thompsonmetav2_strategy import ThompsonMetaV2
from strategies.thompsonmetav3_strategy import ThompsonMetaV3_Contextual
from strategies.boltzmann_strategy import BoltzmannMetaStrategy
from strategies.thompsonmetav4_strategy import ThompsonMetaV4_Profiled
from strategies.metalearnerv5_strategy import MetaLearnerV5_QController
from strategies.metalearnerv6_strategy import MetaLearnerV6_AdaptiveQ
from strategies.antimetav1_strategy import AntiMetaV1_FalsePattern
from strategies.antimetav2_strategy import AntiMetaV2_MetaPredictor
from strategies.antimetav3_strategy import AntiMetaV3_Deprivation
from strategies.antimetav4_strategy import AntiMetaV4_MirrorDiverge
from strategies.antimetav5_strategy import AntiMetaV5_OverfitPunisher
from strategies.metalearnerv7_strategy import MetaLearnerV7_ShadowQ
from strategies.metalearnerv8_strategy import MetaLearnerV8_HybridDeceptiveQ



# Strategy name map
strategy_map = {
    "RandomStrategy": RandomStrategy,
    "EnhancedStrategy": EnhancedStrategy,
    "CopycatStrategy": CopycatStrategy,
    "CycleStrategy": CycleStrategy,
    "AlwaysRockStrategy": AlwaysRockStrategy,
    "FrequencyStrategy": FrequencyStrategy,
    "LastNStrategy": LastNStrategy,
    "MarkovStrategy": MarkovStrategy,
    "QLearningStrategy": QLearningStrategy,
    "MirrorBaiterStrategy": MirrorBaiterStrategy,
    "DelayedMirrorStrategy": DelayedMirrorStrategy,
    "NoiseCounterStrategy": NoiseCounterStrategy,
    "EntropyMaximizerStrategy": EntropyMaximizerStrategy,
    "AdaptiveSwitcherStrategy": AdaptiveSwitcherStrategy,
    "PatternHunterStrategy": PatternHunterStrategy,
    "BayesianStrategy": BayesianUpdateStrategy,
    "ThompsonMetaStrategy": ThompsonMetaStrategy,
    "ThompsonMetaV2": ThompsonMetaV2,
    "ThompsonMetaV3_Contextual": ThompsonMetaV3_Contextual,
    "BoltzmannMetaStrategy": BoltzmannMetaStrategy,
    "ThompsonMetaV4_Profiled": ThompsonMetaV4_Profiled,
    "MetaLearnerV5_QController": MetaLearnerV5_QController,
    "MetaLearnerV6_AdaptiveQ": MetaLearnerV6_AdaptiveQ,
    "AntiMetaV1_FalsePattern": AntiMetaV1_FalsePattern,
    "AntiMetaV2_MetaPredictor": AntiMetaV2_MetaPredictor,
    "AntiMetaV3_Deprivation": AntiMetaV3_Deprivation,
    "AntiMetaV4_MirrorDiverge": AntiMetaV4_MirrorDiverge,
    "AntiMetaV5_OverfitPunisher": AntiMetaV5_OverfitPunisher,
    "MetaLearnerV7_ShadowQ": MetaLearnerV7_ShadowQ,
    "MetaLearnerV8_HybridDeceptiveQ": MetaLearnerV8_HybridDeceptiveQ

}

strategy_descriptions = {
    "AlwaysRockStrategy": "Always plays rock. A naive baseline, easily countered.",
    "RandomStrategy": "Selects moves completely at random. Unpredictable but unoptimized.",
    "CycleStrategy": "Cycles through rock → paper → scissors in order, regardless of opponent.",
    "CopycatStrategy": "Repeats the opponent’s previous move to mimic patterns or exploit repetition.",
    "LastNStrategy": "Plays the most frequent opponent move from the last N rounds.",
    "FrequencyStrategy": "Tracks global move frequencies and counters the most used one.",
    "MarkovStrategy": "Predicts the next move based on recent transition probabilities (Markov chain).",
    "EnhancedStrategy": "A simple reactive strategy that counters the opponent’s last move. Slightly more adaptive than Copycat.",
    "MirrorBaiterStrategy": "Tries to bait the opponent into repeating, then counters the mimic pattern.",
    "QLearningStrategy": "Reinforcement learning model that adapts over time based on state-action rewards.",
    "DelayedMirrorStrategy": "Copies the opponent’s move with a fixed delay, attempting pattern offset.",
    "NoiseCounterStrategy": "Chooses the optimal counter but adds noise to remain unpredictable.",
    "EntropyMaximizerStrategy": "Aims to maximize uncertainty for the opponent by making moves harder to model.",
    "PatternHunterStrategy": "Detects short-term patterns in opponent’s behavior and counters them directly.",
    "AdaptiveSwitcherStrategy": "Switches between strategies mid-game based on performance metrics.",
    "BayesianStrategy": "Uses Bayesian inference to update and predict the most likely next move.",
    "ThompsonMetaStrategy": "Uses classic Thompson Sampling to dynamically choose between sub-strategies based on their win/loss records. Balances exploration and exploitation by sampling from a Beta distribution. Adapts to unknown or changing opponents over time.",
    "ThompsonMetaV2": "Enhances Thompson Sampling by using score-based rewards (+1/-1/0) instead of binary outcomes. Applies exponential decay to past results, allowing the strategy to remain responsive to recent opponent behavior while discounting outdated performance.",
    "ThompsonMetaV3_Contextual": "Introduces lightweight contextual inference by segmenting opponent behavior into patterns (e.g., last 3 moves). Maintains separate Thompson statistics for each context to select strategies more intelligently based on the opponent's current play style.",
    "BoltzmannMetaStrategy": "Uses softmax (Boltzmann) exploration to probabilistically select among sub-strategies based on recent average performance. Balances exploration and exploitation smoothly, adapting to uncertain environments by assigning higher selection probabilities to better-performing strategies. Decays past results to remain responsive over time",
    "ThompsonMetaV4_Profiled": "Adds lightweight opponent profiling to improve Thompson Sampling. Analyzes the opponent’s move repetition, switching rate, and move bias to assign weights to sub-strategies. Prioritizes those most likely to succeed given current behavioral patterns, improving matchups without sacrificing speed.",
    "MetaLearnerV5_QController": "Uses Q-learning to learn which sub-strategy to deploy based on opponent behavior. Models recent repetition and move bias as discrete states, and updates its action policy based on game outcomes. Learns dynamically over time, enabling robust, state-sensitive adaptation without relying on probabilistic sampling.",
    "MetaLearnerV6_AdaptiveQ": "A reinforcement learning-based meta-strategy that dynamically selects among sub-strategies (Markov, Enhanced, Q-Learning) using a Q-table. It profiles the opponent based on move repetition, bias, and switch rate, mapping these features to discrete states. The strategy adapts over time through Q-learning, learning which sub-strategies perform best in each behavioral context. Balances robustness and adaptability without relying on probabilistic sampling.",
    "AntiMetaV1_FalsePattern": "Periodically injects misleading behavioral patterns (e.g., cycle loops) to bait adaptive opponents into overfitting. Once the opponent commits to a counter-strategy, the model reverts to an optimal meta-learning policy. Designed to exploit pattern-seeking agents and confuse reinforcement learners.",
    "AntiMetaV2_MetaPredictor": "Attempts to classify the opponent’s learning behavior (e.g., reactive, pattern-based, random) based on move transitions and bias. Adjusts sub-strategy preferences accordingly using a Q-learning meta-controller. Designed to mislead or counter adaptive learners by exploiting their underlying decision style",
    "AntiMetaV3_Deprivation": "Implements information deprivation by selectively injecting randomness into its own move sequence to confuse adaptive opponents. Built on top of a profiled Q-learning meta-controller, it adds controlled noise after a threshold of rounds, reducing the opponent's ability to learn or exploit consistent behavior patterns. Especially effective against reinforcement learning or pattern-tracking strategies that rely on clear feedback loops.",
    "AntiMetaV4_MirrorDiverge": "Detects if the opponent is mirroring its previous moves and activates a divergence mode to counter it. Combines Q-learning meta-control with mirror tracking logic, allowing it to break free from exploitable patterns. This strategy is particularly effective against reflective or mimic-based opponents and can revert back to normal operation once the threat is gone.",
    "AntiMetaV5_OverfitPunisher": "Detects when the opponent begins overfitting to its own behavior—especially in reinforcement-learning-based strategies—and injects variation to disrupt prediction. By tracking repeated counterattacks and penalizing static patterns, it adapts to exploit overly reactive opponents while still using a Q-learning meta-controller. Balances deceptive output with strategic robustness.",
    "MetaLearnerV7_ShadowQ": "An advanced Q-learning-based meta-strategy that blends profiling, deception, and adaptive learning. It uses opponent behavior traits (bias, repetition, switching), recent losses, and mirror detection to adjust sub-strategy selection. Injects false patterns and enters evasive high-entropy states under pressure, making it resilient against both deterministic and adaptive AI strategies.",
    "MetaLearnerV8_HybridDeceptiveQ": "This strategy blends Q-learning with Thompson Sampling to select among sub-strategies while injecting deceptive behavior like false patterns and mirror baiting. It adapts based on opponent profiling and recent performance, leveraging both learning paradigms for robust and manipulative meta-play."


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

st.sidebar.markdown("### Strategy Info")
selected = st.sidebar.selectbox("Select a Strategy", list(strategy_map.keys()))
st.sidebar.markdown(strategy_descriptions.get(selected, "No description available."))

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

    # Always show ThompsonMetaV6_AdaptiveQ as baseline (default), fallback to first if not present
    default_baseline = "MetaLearnerV8_HybridDeceptiveQ"
    available_strategies = score_df["Strategy"].unique().tolist()

    selected_heatmap_strategy = st.selectbox(
        "Select Strategy for Matchup Heatmap",
        options=available_strategies,
        index=available_strategies.index(default_baseline) if default_baseline in available_strategies else 0
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
    heatmap_df = heatmap_df[sorted(heatmap_df.columns)]
    # fig, ax = plt.subplots(figsize=(len(heatmap_df.columns) * 1.2, 1.8))
    # sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"label": "Avg Score"}, ax=ax)
    # ax.set_title(f"Matchup Scores for {selected_heatmap_strategy}", fontsize=12)
    # st.pyplot(fig)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=heatmap_df.round(1).astype(str).values,  # Annotate with scores
        hovertemplate="Opponent: %{x}<br>Strategy: %{y}<br>Score: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Avg Score")
    ))

    fig.update_layout(
        title=f"Matchup Scores for {selected_heatmap_strategy}",
        xaxis_title="Opponent",
        yaxis_title="Strategy",
        xaxis_tickangle=45,
        height=400,
        margin=dict(l=100, r=40, t=60, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Choose at least two strategies and press 'Run Arena' to start.")
