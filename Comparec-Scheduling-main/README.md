# Comparec-Scheduling

Simulateur d'ordonnancement de jobs HPC sur plusieurs clusters, avec
recommandations énergétiques pré-calculées. L'objectif est de mesurer
l'économie d'énergie réalisable en redirigeant chaque job vers un cluster
plus économe, sous différents scénarios de comportement utilisateur.

Projet L3, Université Paul Sabatier, 2025/2026.
Encadrant : Eyvaz Ahmadzada (IRIT). Tuteur universitaire : Vincent Dugat.
Auteurs : Birane N'Diaye, Emmanuel Denis, Mathis Fontanie, Tom Tassin.

---

## Structure du projet

```
.
├── data/                                # Jeu de données (Git LFS)
│   └── workloads_with_recommendations_V2.csv
├── python/
│   ├── comparec.py                      # Module principal (classes du simulateur)
│   ├── notebook.ipynb                   # Notebook d'exploration et d'exécution
│   ├── test_comparec.py                 # Tests unitaires
│   └── test_scenarios.py                # Balayage paramétrique
├── results/
│   ├── scenarios_results.csv            # Résultats du balayage
│   └── scenarios_graphs.png             # Graphique synthèse 4-en-1
├── rapport/                             # Rapports CC2 et final
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Git LFS

Le CSV de données est versionné via Git LFS. Sans Git LFS, vous ne récupérerez qu'un pointeur texte au clone.

```bash
brew install git-lfs        # macOS
sudo apt install git-lfs    # Ubuntu/Debian
git lfs install
```

### 2. Cloner le dépôt

```bash
git clone https://github.com/Manooooo23/Comparec-Scheduling.git
cd Comparec-Scheduling
git lfs ls-files            # vérifier que le CSV est bien suivi
```

### 3. Environnement Python

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Utilisation rapide

### Exemple minimal

```python
import pandas as pd
from comparec import ComparecWithDispo, compute_cluster_capacities

# 1. Charger les données
data = pd.read_csv("../data/workloads_with_recommendations_V2.csv")

# 2. Récupérer les capacités par cluster (1 job = 1 node)
capacities = compute_cluster_capacities(data)

# 3. Lancer la simulation
ordo = ComparecWithDispo(data, capacities, acceptance_rate=0.5, seed=42)
resultats = ordo.run()

# 4. Récupérer le bilan global
bilan = ordo.bilan()
print(f"Gain énergie : {bilan['gain_energie_%']:.1f}%")
print(f"Attente moyenne : {bilan['attente_moyenne_h']:.2f}h")

# 5. Bilan par cluster
print(ordo.bilan_par_cluster())
```

### Lancer le notebook complet

```bash
cd python
jupyter lab notebook.ipynb
```

Le notebook se déroule de haut en bas :
1. Chargement du dataset
2. Visualisations exploratoires (distributions, matrice de migration)
3. Borne haute via la classe `Comparec` (sans contrainte de dispo)
4. Capacités des clusters (1 job = 1 node)
5. Exécution des 5 scénarios sans backfill
6. Exécution des 5 scénarios avec backfill
7. Tableau récapitulatif et graphiques comparatifs

### Lancer le balayage paramétrique

```bash
cd python
python test_scenarios.py
```

Teste 20 configurations différentes et écrit les résultats dans `results/scenarios_results.csv` et `results/scenarios_graphs.png`.

### Lancer les tests unitaires

```bash
cd python
python test_comparec.py
```

Vérifie la borne haute, la reproductibilité avec seed, le respect strict de `max_wait_h`, la conservation des jobs et plusieurs autres invariants.

---

## Le simulateur, `comparec.py`

### Classe `Comparec` (moteur naïf)

Applique systématiquement la recommandation de rang 0, sans tenir compte
de la disponibilité des clusters. Sert de **borne haute** : tout scénario
réaliste sera nécessairement en dessous de ce gain.

### Classe `ComparecWithDispo` (moteur événementiel)

Simulation à événements discrets avec une file de priorité (`heapq`).
Trois types d'événements : `SUBMIT`, `FINISH`, `TIMEOUT`.

#### Paramètres

| Paramètre | Type | Défaut | Rôle |
|---|---|---|---|
| `cluster_capacities` | dict | requis | Nombre de nodes par cluster |
| `rate_reco` | float | 1.0 | Fraction des jobs qui *reçoivent* une recommandation |
| `acceptance_rate` | float | 1.0 | Probabilité qu'un job reçu *accepte* la reco |
| `max_wait_h` | float | None | Temps max en file. Au-delà, fallback cluster d'origine |
| `seed` | int | None | Graine du générateur aléatoire |
| `with_backfill` | bool | False | Active l'algo de backfill (style HPC LSF/Slurm) |

Le taux effectif de jobs suivant la reco est donc `rate_reco * acceptance_rate`.

#### Sorties

- `run()` retourne un `DataFrame` avec une ligne par job planifié (cluster choisi, rang, temps d'attente, énergie, etc.)
- `bilan()` retourne un `dict` de métriques globales (gain énergie %, attente moyenne, nb migrations, etc.)
- `bilan_par_cluster()` retourne un `DataFrame` agrégé par cluster

#### Garde-fous

En fin de `run()`, trois assertions vérifient :
1. Tous les clusters sont libérés (pas de fuite de slot)
2. Si `max_wait_h` est défini, l'attente max observée le respecte
3. Le heap auxiliaire reste synchronisé avec le heap principal (vérifié à chaque FINISH)

---

## Scénarios étudiés

| Scénario | Paramètres | Comportement modélisé |
|---|---|---|
| S1 | défaut | Tous les utilisateurs suivent la reco |
| S2 | `acceptance_rate=0.5` | 1 utilisateur sur 2 ignore la reco |
| S3 | `acceptance_rate=0.8` | Adhésion réaliste (80%) |
| S4 | `max_wait_h=2.0` | Hard deadline à 2h, utilisateur impatient |
| S5 | `rate_reco=0.6` | 60% des jobs reçoivent une reco |

Chacun est exécuté **avec et sans backfill**, ce qui donne 10 simulations
comparées dans le notebook.

### Snippets par scénario

```python
# S1, adhésion totale
ordo = ComparecWithDispo(data, capacities)

# S2, adhésion aléatoire 50%
ordo = ComparecWithDispo(data, capacities, acceptance_rate=0.5, seed=42)

# S3, adhésion paramétrable 80%
ordo = ComparecWithDispo(data, capacities, acceptance_rate=0.8, seed=42)

# S4, temps d'attente max 2h
ordo = ComparecWithDispo(data, capacities, max_wait_h=2.0)

# S5, seulement 60% des utilisateurs reçoivent les recos
ordo = ComparecWithDispo(data, capacities, rate_reco=0.6, seed=42)

# Variante avec backfill
ordo = ComparecWithDispo(data, capacities, max_wait_h=2.0, with_backfill=True)
```

---

## Aperçu des résultats

| Scénario | Gain énergie (%) | Attente moy (h) |
|---|---|---|
| Borne haute (sans dispo) | 48,8 | n/a |
| S1 | ~43 | ~20 |
| S2 | ~20 | ~19 |
| S3 | ~33 | ~20 |
| S4 (max_wait=2h) | ~45 | ~0,4 |
| S5 | ~24 | ~19 |

Le scénario 4 (timeout dur à 2h) offre le meilleur compromis :
gain énergétique élevé **et** attente quasi nulle. Le balayage
paramétrique (`test_scenarios.py`) montre que `max_wait_h=1h` fait
encore mieux : 44,7 % de gain pour 14 minutes d'attente moyenne.

---

## Aide-mémoire Git LFS

Sans Git LFS, GitHub bloque les push si un fichier dépasse 100 Mo. Avec LFS, Git ne stocke qu'un petit fichier texte de référence localement, tandis que le contenu réel est hébergé sur des serveurs dédiés.

```bash
git lfs install                           # une seule fois par machine
git clone <repo>                          # clone normal, LFS récupère les vrais fichiers
git lfs ls-files                          # voir les fichiers suivis par LFS

# Ajouter un nouveau type de fichier volumineux
git lfs track "*.csv"
git add .gitattributes
git add <fichier>
git commit -m "Ajout d'un fichier volumineux via LFS"
```

---

# 🚀 Utilisation du simulateur `ComparecWithDispo`

Le module [`python/comparec_with_dispo.py`](python/comparec_with_dispo.py) fournit la classe `ComparecWithDispo`, un simulateur événementiel d'ordonnancement sur cloud avec recommandations.

## Installation des dépendances

```bash
pip install -r requirements.txt
```

## Exemple minimal

```python
import pandas as pd
from comparec_with_dispo import ComparecWithDispo

# 1. Charger les données
data = pd.read_csv("data/workloads_with_recommendations_V2.csv")

# 2. Définir les capacités par cluster (nb de jobs simultanés max)
capacities = {
    "cluster_0": 8,
    "cluster_1": 20,
    "cluster_2": 2,
    "cluster_3": 3,
    "cluster_4": 5,
    "cluster_5": 3,
}

# 3. Lancer la simulation
ordo = ComparecWithDispo(data, capacities, acceptance_rate=0.5, seed=42)
resultats = ordo.run()

# 4. Récupérer le bilan global
bilan = ordo.bilan()
print(f"Gain énergie : {bilan['gain_energie_%']:.1f}%")
print(f"Attente moyenne : {bilan['attente_moyenne_h']:.2f}h")

# 5. Bilan par cluster
print(ordo.bilan_par_cluster())
```

## Paramètres principaux

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `cluster_capacities` | Dict `{cluster_id: nb_jobs_max}` | requis |
| `rate_reco` | Fraction des jobs qui reçoivent une reco (1.0 = tous) | `1.0` |
| `acceptance_rate` | Probabilité de suivre la reco reçue (1.0 = toujours) | `1.0` |
| `max_wait_h` | Temps max en file avant fallback sur cluster original | `None` |
| `seed` | Graine aléatoire pour reproductibilité | `None` |
| `with_backfill` | Active l'algorithme de backfill | `False` |

## Scénarios types

```python
# S1 — Adhésion totale
ordo = ComparecWithDispo(data, capacities)

# S2 — Adhésion aléatoire 50%
ordo = ComparecWithDispo(data, capacities, acceptance_rate=0.5, seed=42)

# S3 — Adhésion paramétrable 80%
ordo = ComparecWithDispo(data, capacities, acceptance_rate=0.8, seed=42)

# S4 — Temps d'attente max 2h
ordo = ComparecWithDispo(data, capacities, max_wait_h=2.0)

# S5 — Seulement 60% des utilisateurs reçoivent les recos
ordo = ComparecWithDispo(data, capacities, rate_reco=0.6, seed=42)

# Variantes avec backfill
ordo = ComparecWithDispo(data, capacities, with_backfill=True)
```

## Sorties

- `ordo.run()` → `DataFrame` avec une ligne par job planifié (cluster choisi, rang, temps d'attente, énergie, etc.)
- `ordo.bilan()` → `dict` de métriques globales (gain énergie %, attente moyenne, nb migrations, etc.)
- `ordo.bilan_par_cluster()` → `DataFrame` agrégé par cluster
