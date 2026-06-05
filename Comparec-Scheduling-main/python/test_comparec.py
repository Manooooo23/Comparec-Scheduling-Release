"""
Tests du simulateur ComparecWithDispo.

Lancement :
    cd python
    python test_comparec.py

Sortie : une ligne par test, OK ou FAIL avec le message d'erreur.
Exit code 0 si tout passe, 1 sinon.

Les tests couvrent :
  - le moteur naïf Comparec (borne haute)
  - le moteur réaliste ComparecWithDispo sans contrainte (scénario 1)
  - la reproductibilité avec une graine fixe
  - le respect strict de max_wait_h (scénario 4)
  - la validation des paramètres hors bornes
  - la cohérence backfill on/off (les chiffres restent du même ordre)
  - la conservation du nombre de jobs
"""

import os
import sys
import traceback
import pandas as pd

from comparec import Comparec, ComparecWithDispo, compute_cluster_capacities


# ---------------------------------------------------------------------------
# Chargement des données (une seule fois pour tous les tests)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "workloads_with_recommendations_V2.csv")
print(f"Chargement du jeu de données : {DATA_PATH}")
DATA = pd.read_csv(DATA_PATH)
CAPS = compute_cluster_capacities(DATA)
NB_JOBS_ATTENDUS = DATA["job_id"].nunique()
print(f"  {NB_JOBS_ATTENDUS} jobs uniques sur {len(DATA)} lignes\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_comparec_borne_haute():
    """Le moteur naïf doit retourner un gain entre 40 et 60 % (borne haute)."""
    ordo = Comparec(DATA)
    ordo.run()
    bilan = ordo.bilan()
    gain = bilan["gain_energie_pourcentage"]
    assert 40 < gain < 60, f"Gain Comparec attendu entre 40 et 60 %, obtenu {gain:.2f}"
    assert bilan["nb_jobs"] == NB_JOBS_ATTENDUS, \
        f"Nombre de jobs incorrect : {bilan['nb_jobs']} vs {NB_JOBS_ATTENDUS}"


def test_dispo_scenario1():
    """Scénario 1 (100 % reco) : gain inférieur à la borne haute mais > 30 %."""
    ordo = ComparecWithDispo(DATA, CAPS)
    ordo.run()
    bilan = ordo.bilan()
    assert bilan["nb_jobs"] == NB_JOBS_ATTENDUS, \
        f"Tous les jobs doivent être planifiés ({bilan['nb_jobs']}/{NB_JOBS_ATTENDUS})"
    assert 30 < bilan["gain_energie_%"] < 50, \
        f"Gain S1 hors fourchette : {bilan['gain_energie_%']:.2f}"
    assert bilan["taux_adhesion_effectif_%"] == 100.0, "S1 devrait avoir 100 % d'adhésion"


def test_reproductibilite_seed():
    """Deux runs avec la même seed doivent donner exactement les mêmes chiffres."""
    o1 = ComparecWithDispo(DATA, CAPS, acceptance_rate=0.5, seed=42)
    o2 = ComparecWithDispo(DATA, CAPS, acceptance_rate=0.5, seed=42)
    b1 = (o1.run(), o1.bilan())[1]
    b2 = (o2.run(), o2.bilan())[1]
    assert b1["gain_energie_%"] == b2["gain_energie_%"], \
        f"Reproductibilité cassée : {b1['gain_energie_%']} vs {b2['gain_energie_%']}"
    assert b1["nb_migrations"] == b2["nb_migrations"], \
        "Nb migrations devrait être identique avec même seed"


def test_reproductibilite_rate_reco():
    """Idem mais sur la sélection rate_reco (qui dépend aussi de la seed)."""
    o1 = ComparecWithDispo(DATA, CAPS, rate_reco=0.6, seed=123)
    o2 = ComparecWithDispo(DATA, CAPS, rate_reco=0.6, seed=123)
    b1 = (o1.run(), o1.bilan())[1]
    b2 = (o2.run(), o2.bilan())[1]
    assert b1["nb_recommandations_recues"] == b2["nb_recommandations_recues"], \
        "rate_reco devrait sélectionner les mêmes jobs avec même seed"


def test_max_wait_h_strict():
    """Avec max_wait_h défini, l'attente max doit être inférieure ou égale au seuil."""
    seuil = 2.0
    ordo = ComparecWithDispo(DATA, CAPS, max_wait_h=seuil)
    ordo.run()
    bilan = ordo.bilan()
    assert bilan["attente_max_h"] <= seuil + 1e-6, \
        f"Attente max {bilan['attente_max_h']:.4f}h dépasse le seuil {seuil}h"
    assert bilan["nb_timeouts_fallback"] > 0, \
        "Avec max_wait=2h on s'attend à au moins quelques timeouts"


def test_validation_parametres():
    """Les rates hors [0, 1] doivent lever une ValueError."""
    cas_invalides = [
        {"acceptance_rate": 1.5},
        {"acceptance_rate": -0.1},
        {"rate_reco": 2.0},
        {"rate_reco": -1.0},
    ]
    for kwargs in cas_invalides:
        try:
            ComparecWithDispo(DATA, CAPS, **kwargs)
            raise AssertionError(f"Pas de ValueError pour {kwargs}")
        except ValueError:
            pass  # comportement attendu


def test_backfill_coherent():
    """Avec backfill activé, le gain énergétique doit rester du même ordre de grandeur."""
    sans = ComparecWithDispo(DATA, CAPS)
    avec = ComparecWithDispo(DATA, CAPS, with_backfill=True)
    sans.run(); avec.run()
    b1 = sans.bilan(); b2 = avec.bilan()
    diff = abs(b1["gain_energie_%"] - b2["gain_energie_%"])
    assert diff < 5.0, \
        f"Backfill change trop le gain (diff = {diff:.2f} pts), suspect"
    assert b1["nb_jobs"] == b2["nb_jobs"], \
        "Backfill ne doit pas changer le nombre de jobs planifiés"


def test_conservation_jobs():
    """Sur tous les scénarios, le nombre de jobs planifiés doit être conservé."""
    configs = [
        {},
        {"acceptance_rate": 0.5, "seed": 1},
        {"max_wait_h": 2.0},
        {"rate_reco": 0.6, "seed": 1},
        {"with_backfill": True},
        {"max_wait_h": 2.0, "with_backfill": True},
    ]
    for kwargs in configs:
        ordo = ComparecWithDispo(DATA, CAPS, **kwargs)
        ordo.run()
        bilan = ordo.bilan()
        assert bilan["nb_jobs"] == NB_JOBS_ATTENDUS, \
            f"Jobs perdus avec {kwargs} : {bilan['nb_jobs']}/{NB_JOBS_ATTENDUS}"


def test_taux_adhesion_correspond():
    """Le taux d'adhésion effectif doit être proche de rate_reco * acceptance_rate."""
    ordo = ComparecWithDispo(DATA, CAPS, rate_reco=0.6, acceptance_rate=0.5, seed=42)
    ordo.run()
    bilan = ordo.bilan()
    attendu = 60 * 50 / 100  # 30 %
    obtenu = bilan["taux_adhesion_effectif_%"]
    assert abs(obtenu - attendu) < 3.0, \
        f"Taux d'adhésion {obtenu}% trop loin de l'attendu {attendu}%"


# ---------------------------------------------------------------------------
# Runner minimaliste
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_comparec_borne_haute,
        test_dispo_scenario1,
        test_reproductibilite_seed,
        test_reproductibilite_rate_reco,
        test_max_wait_h_strict,
        test_validation_parametres,
        test_backfill_coherent,
        test_conservation_jobs,
        test_taux_adhesion_correspond,
    ]

    nb_ok = 0
    nb_ko = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
            nb_ok += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            nb_ko += 1

    print()
    print(f"Résultat : {nb_ok}/{len(tests)} tests passés ({nb_ko} échecs)")
    sys.exit(0 if nb_ko == 0 else 1)


if __name__ == "__main__":
    main()
