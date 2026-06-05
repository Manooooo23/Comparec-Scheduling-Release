# Ordonnanceur V2 — `ComparecWithDispo`

## Objectif

Simuler de manière réaliste la planification de jobs HPC sur plusieurs clusters, en tenant compte de la **disponibilité des ressources** et en choisissant à chaque instant le cluster le plus économe en énergie parmi ceux qui sont libres.

---

## Principe général : simulation à événements discrets

L'ordonnanceur avance dans le temps en traitant une **file d'événements triés chronologiquement** (via une `heapq`). Il n'y a pas de boucle sur des pas de temps fixes : le temps saute directement d'un événement significatif au suivant.

Deux types d'événements :
- `SUBMIT` — un job est soumis par un utilisateur
- `FINISH` — un job se termine et libère un slot sur son cluster

---

## Structures de données

### `logical_jobs` — regroupement des recommandations

Les données sources contiennent plusieurs lignes par job (une par recommandation de cluster, identifiée par un `rank`). La méthode `_group_jobs()` regroupe ces lignes en **un seul objet job** avec un dictionnaire de recommandations indexé par rang :

```
job = {
    "submission_time": ...,
    "user": ...,
    "orig_cluster": ...,
    "recs": {
        0: { "recommended_cluster": "A", "est_energy_kWh": ..., "est_duration_h": ... },
        1: { "recommended_cluster": "B", ... },
        2: { "recommended_cluster": "C", ... },
    }
}
```

La clé d'unicité est `(user, submission_time, orig_cluster)`.

### `cluster_usage` — occupation courante

`defaultdict(int)` qui compte le nombre de jobs **actuellement en cours** sur chaque cluster. Incrémenté à l'assignation, décrémenté au `FINISH`.

### `capacities` — capacité max par cluster

Dictionnaire `{cluster_name: nb_slots_max}`. Dans la configuration actuelle : **1 job simultané par cluster**. Peut être étendu.

### `waiting` — file d'attente

Liste des jobs qui n'ont pas pu être assignés faute de slot disponible. Retentés à chaque événement `FINISH`.

---

## Algorithme principal — `run()`

```
Initialisation :
  Pour chaque job → insérer un événement SUBMIT(submission_time, job) dans la heap

Boucle principale :
  Dépiler l'événement le plus tôt (time, type, payload)

  Si FINISH(cluster) :
    cluster_usage[cluster] -= 1
    Pour chaque job en attente :
      Tenter _try_assign(job)
      Si succès → _schedule(job, ..., start_time=time)
      Sinon    → remettre en attente

  Si SUBMIT(job) :
    Tenter _try_assign(job)
    Si succès → _schedule(job, ..., start_time=time)
    Sinon    → ajouter à waiting
```

---

## Logique d'assignation — `_try_assign()`

Parcourt les recommandations du job **du rang 0 (meilleur) au rang le plus élevé (fallback)** et retourne la première dont le cluster a encore de la capacité libre :

```python
for rank in sorted(job["recs"].keys()):
    cluster = job["recs"][rank]["recommended_cluster"]
    if cluster_usage[cluster] < capacities[cluster]:
        return rec, rank
return None  # aucun cluster disponible
```

Cela garantit qu'on choisit **toujours le cluster le plus économe en énergie disponible** au moment de l'assignation.

---

## Planification — `_schedule()`

Une fois un couple `(rec, rank)` trouvé :

1. `cluster_usage[cluster] += 1` — réserve le slot
2. `end_time = start_time + timedelta(hours=est_duration_h)`
3. Insère un événement `FINISH(end_time, cluster)` dans la heap
4. Enregistre le résultat dans `self.resultats` avec :
   - `start_time`, `end_time`
   - `rang_choisi` (0 = meilleur, >0 = fallback)
   - `temps_attente_h = (start_time - submission_time)`

---

## Métriques produites — `bilan()` et `bilan_par_cluster()`

| Métrique | Description |
|---|---|
| `gain_energie_kWh` | Énergie économisée vs scénario original |
| `gain_energie_%` | Pourcentage de réduction |
| `attente_moyenne_h` | Temps d'attente moyen avant démarrage |
| `attente_max_h` | Pire cas d'attente |
| `nb_jobs_avec_attente` | Nombre de jobs ayant été mis en file |
| `distribution_rangs` | Fréquence d'utilisation de chaque rang (0 = sans fallback) |
| `nb_migrations` | Jobs assignés sur un cluster différent de l'original |

`bilan_par_cluster()` agrège ces métriques par cluster de destination pour identifier les clusters les plus sollicités ou les plus économes.

---

## Limites actuelles

- La file d'attente `waiting` est parcourue **entièrement** à chaque `FINISH`, ce qui peut être coûteux si beaucoup de jobs attendent simultanément (O(n) par événement).
- L'ordre de reprise des jobs en attente n'est pas garanti (pas de FIFO strict sur la file d'attente).
- La capacité est fixée à 1 job/cluster par défaut, ce qui est conservateur.
