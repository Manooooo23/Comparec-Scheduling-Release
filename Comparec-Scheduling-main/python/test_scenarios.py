"""
Balayage paramétrique : test exhaustif de scénarios pour trouver la meilleure
configuration.

Pour chaque combinaison, on enregistre :
  - gain énergie (%)
  - attente moyenne et max (h)
  - nb migrations
  - nb timeouts (fallback sur cluster original)

À la fin, on classe les scénarios selon différents critères pour identifier
le meilleur compromis énergie/attente. Les résultats sont sauvegardés dans
results/ et un graphique synthèse est produit.

Lancement :
    cd python
    python test_scenarios.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

from comparec import ComparecWithDispo, compute_cluster_capacities


SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "workloads_with_recommendations_V2.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")


def run_scenario(data, capacities, name, **kwargs):
    """Lance un scénario et retourne une ligne récap."""
    print(f"  Running {name}...")
    ordo = ComparecWithDispo(data, capacities, **kwargs)
    ordo.run()
    bilan = ordo.bilan()
    return {
        "scenario": name,
        **{k: kwargs.get(k) for k in ["rate_reco", "acceptance_rate", "max_wait_h", "with_backfill"]},
        "gain_energie_%": round(bilan["gain_energie_%"], 2),
        "attente_moy_h": round(bilan["attente_moyenne_h"], 2),
        "attente_max_h": round(bilan["attente_max_h"], 2),
        "duree_orig_moy_h": round(bilan["duree_orig_moyenne_h"], 2),
        "duree_sim_moy_h": round(bilan["duree_sim_moyenne_h"], 2),
        "nb_migrations": int(bilan["nb_migrations"]),
        "nb_timeouts": int(bilan["nb_timeouts_fallback"]),
        "taux_adhesion_%": bilan["taux_adhesion_effectif_%"],
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Chargement des données...")
    data = pd.read_csv(DATA_PATH)
    capacities = compute_cluster_capacities(data)
    print(f"  {len(data)} lignes chargées, capacités : {capacities}\n")

    scenarios = []

    print("=== Variantes max_wait_h pures ===")
    for wait in [1.0, 2.0, 5.0, 10.0]:
        scenarios.append(run_scenario(data, capacities,
                                      f"S4_max_wait_{wait}h",
                                      max_wait_h=wait, seed=SEED))

    print("\n=== max_wait_h + acceptance_rate ===")
    for wait in [1.0, 2.0]:
        for acc in [0.3, 0.5, 0.8]:
            scenarios.append(run_scenario(
                data, capacities, f"acc{int(acc*100)}%_wait{wait}h",
                acceptance_rate=acc, max_wait_h=wait, seed=SEED
            ))

    print("\n=== max_wait_h + rate_reco ===")
    for wait in [1.0, 2.0]:
        for rr in [0.3, 0.5, 0.8]:
            scenarios.append(run_scenario(
                data, capacities, f"rr{int(rr*100)}%_wait{wait}h",
                rate_reco=rr, max_wait_h=wait, seed=SEED
            ))

    print("\n=== Avec backfill ===")
    for wait in [1.0, 2.0]:
        scenarios.append(run_scenario(
            data, capacities, f"backfill_wait{wait}h",
            max_wait_h=wait, with_backfill=True, seed=SEED
        ))

    print("\n=== Combinaisons mixtes ===")
    scenarios.append(run_scenario(data, capacities, "MIX_rr80_acc80_wait2h",
                                  rate_reco=0.8, acceptance_rate=0.8, max_wait_h=2.0, seed=SEED))
    scenarios.append(run_scenario(data, capacities, "with_backfill_wait2h",
                                  rate_reco=1.0, acceptance_rate=1.0, max_wait_h=2.0,
                                  with_backfill=True, seed=SEED))

    df = pd.DataFrame(scenarios)

    print("\n" + "=" * 80)
    print("RÉSULTATS COMPLETS")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP 5 PAR GAIN ÉNERGIE")
    print("=" * 80)
    print(df.nlargest(5, "gain_energie_%")[
        ["scenario", "gain_energie_%", "attente_moy_h", "nb_timeouts"]
    ].to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP 5 PAR ATTENTE MOYENNE LA PLUS FAIBLE")
    print("=" * 80)
    print(df.nsmallest(5, "attente_moy_h")[
        ["scenario", "attente_moy_h", "gain_energie_%", "nb_timeouts"]
    ].to_string(index=False))

    print("\n" + "=" * 80)
    print("MEILLEUR COMPROMIS (score = gain_energie_% - attente_moy_h)")
    print("=" * 80)
    df["score_compromis"] = df["gain_energie_%"] - df["attente_moy_h"]
    print(df.nlargest(5, "score_compromis")[
        ["scenario", "score_compromis", "gain_energie_%", "attente_moy_h"]
    ].to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, "scenarios_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats exportés vers {csv_path}")

    plot_results(df)


def plot_results(df):
    """Produit 4 graphiques pour analyser les résultats des scénarios."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Analyse des scénarios d'ordonnancement", fontsize=14, fontweight="bold")

    # 1. Bar chart : gain énergie par scénario
    df_sorted = df.sort_values("gain_energie_%", ascending=True)
    colors = ["#27ae60" if g >= 40 else "#f39c12" if g >= 25 else "#e74c3c"
              for g in df_sorted["gain_energie_%"]]
    axes[0, 0].barh(df_sorted["scenario"], df_sorted["gain_energie_%"], color=colors)
    axes[0, 0].set_xlabel("Gain énergie (%)")
    axes[0, 0].set_title("Gain énergétique par scénario")
    axes[0, 0].axvline(40, color="gray", linestyle="--", alpha=0.5, label="40%")
    axes[0, 0].legend()

    # 2. Scatter : gain énergie vs attente moyenne (Pareto)
    for wait_val in df["max_wait_h"].dropna().unique():
        sub = df[df["max_wait_h"] == wait_val]
        axes[0, 1].scatter(sub["attente_moy_h"], sub["gain_energie_%"],
                           s=100, alpha=0.7, label=f"max_wait={wait_val}h")
    axes[0, 1].set_xlabel("Attente moyenne (h)")
    axes[0, 1].set_ylabel("Gain énergie (%)")
    axes[0, 1].set_title("Compromis énergie / attente")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    # Annotation du meilleur scénario
    best = df.loc[df["gain_energie_%"].idxmax()]
    axes[0, 1].annotate(best["scenario"],
                        (best["attente_moy_h"], best["gain_energie_%"]),
                        xytext=(10, -10), textcoords="offset points",
                        fontsize=8, color="#c0392b", fontweight="bold")

    # 3. Bar chart : durée moyenne des jobs (originale vs simulée)
    df_dur = df.sort_values("duree_sim_moy_h", ascending=True)
    y = range(len(df_dur))
    height = 0.4
    axes[1, 0].barh([i - height/2 for i in y], df_dur["duree_orig_moy_h"],
                    height, label="Originale", color="#95a5a6", alpha=0.8)
    axes[1, 0].barh([i + height/2 for i in y], df_dur["duree_sim_moy_h"],
                    height, label="Simulée", color="#3498db", alpha=0.8)
    axes[1, 0].set_yticks(list(y))
    axes[1, 0].set_yticklabels(df_dur["scenario"])
    axes[1, 0].set_xlabel("Durée moyenne d'un job (h)")
    axes[1, 0].set_title("Durée moyenne des jobs : originale vs simulée")
    axes[1, 0].legend(loc="lower left")

    # 4. Nombre de timeouts par scénario
    df_to = df.sort_values("nb_timeouts", ascending=True)
    axes[1, 1].barh(df_to["scenario"], df_to["nb_timeouts"], color="#3498db")
    axes[1, 1].set_xlabel("Nombre de jobs en timeout (fallback)")
    axes[1, 1].set_title("Jobs ayant dépassé max_wait_h")

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "scenarios_graphs.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"Graphiques sauvegardés dans {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
