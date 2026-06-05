"""
Simulateur événementiel d'ordonnancement sur cloud avec recommandations.

Usage:
    from comparec_with_dispo import ComparecWithDispo
    import pandas as pd

    data = pd.read_csv("workloads_with_recommendations_V2.csv")
    capacities = {"cluster_0": 8, "cluster_1": 20, ...}
    ordo = ComparecWithDispo(data, capacities, acceptance_rate=0.5, seed=42)
    resultats = ordo.run()
    bilan = ordo.bilan()
"""

from heapq import heappush, heappop, nsmallest
from datetime import timedelta
from collections import defaultdict, deque
from itertools import islice

import numpy as np
import pandas as pd


class ComparecWithDispo:
    """
    Simulateur événementiel d'ordonnancement sur cloud avec recommandations.

    Modèle d'adoption en deux décisions séparées :
      1. rate_reco : proportion de jobs qui reçoivent les recommandations.
         Cette sélection est faite une seule fois au démarrage de la simulation.
      2. acceptance_rate : probabilité qu'un job ayant reçu une recommandation
         l'accepte au moment de sa soumission.

      Si les deux paramètres sont utilisés ensemble, le taux attendu de jobs qui
      suivent réellement une recommandation est donc rate_reco * acceptance_rate.
      Les jobs non sélectionnés par rate_reco, ou qui refusent via acceptance_rate,
      restent sur leur cluster original.

    Paramètres :
      - cluster_capacities : nb de nodes par cluster
      - rate_reco          : fraction de jobs qui reçoivent une recommandation
                             (1.0 = tous, 0.6 = 60%, 0.0 = aucun)
      - acceptance_rate    : probabilité d'accepter la recommandation reçue
                             (1.0 = accepte toujours, 0.5 = accepte une fois sur deux)
      - max_wait_h         : (Scénario 4) temps d'attente max en file. Un événement TIMEOUT
                             dédié est planifié à submission_time + max_wait_h pour chaque
                             job mis en attente : garantit un hard deadline strict.
                             None = pas de limite (comportement scénarios 1-3).
      - seed               : graine aléatoire pour reproductibilité.
      - with_backfill      : active l'algorithme de backfill (les jobs courts peuvent
                             passer devant en file si ça ne retarde pas le job de tête).
    """

    FINISH = 0
    SUBMIT = 1
    TIMEOUT = 2

    def __init__(self, jobs: pd.DataFrame, cluster_capacities: dict,
                 acceptance_rate: float = 1.0, rate_reco: float = 1.0, max_wait_h: float = None,
                 seed: int = None, with_backfill: bool = False):
        self._validate_rate("acceptance_rate", acceptance_rate)
        self._validate_rate("rate_reco", rate_reco)

        self.data = jobs.copy()
        self.data["submission_time"] = pd.to_datetime(self.data["submission_time"])
        self.rng = np.random.RandomState(seed)
        self.capacities = cluster_capacities
        self.resultats = []
        self.acceptance_rate = acceptance_rate
        self.rate_reco = rate_reco
        self.max_wait_h = max_wait_h
        self.with_backfill = with_backfill
        self._finish_times_by_cluster = defaultdict(list)
        self.logical_jobs = self._group_jobs(rate_reco)

    @staticmethod
    def _validate_rate(name, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} doit être compris entre 0.0 et 1.0")

    def _group_jobs(self, rate_reco):
        groups = {}
        for _, row in self.data.iterrows():
            key = (row["job_id"], row["submission_time"], row["orig_cluster"])
            if key not in groups:
                groups[key] = {
                    "submission_time": row["submission_time"],
                    "user": row["user"],
                    "orig_cluster": row["orig_cluster"],
                    "orig_energy_kWh": row["orig_energy_kWh"],
                    "orig_duration_h": row["orig_duration_h"],
                    "receives_recommendation": True,
                    "recommendation_suivie": None,
                    "recs": {},
                }
            groups[key]["recs"][row["rank"]] = {
                "job_id": row["job_id"],
                "recommended_cluster": row["recommended_cluster"],
                "est_energy_kWh": row["est_energy_kWh"],
                "est_duration_h": row["est_duration_h"],
            }

        if rate_reco < 1.0:
            print(f"Simulation de réception des recommandations avec rate_reco = {rate_reco}...")
            nb_jobs_avec_reco = round(len(groups) * rate_reco)
            print(f"  {nb_jobs_avec_reco} jobs recevront une recommandation, {len(groups) - nb_jobs_avec_reco} resteront sans reco.")

            keys = list(groups.keys())
            selected = set()
            if nb_jobs_avec_reco > 0:
                chosen_idx = self.rng.choice(len(keys), size=nb_jobs_avec_reco, replace=False)
                selected = {keys[i] for i in chosen_idx}

            for key in groups:
                groups[key]["receives_recommendation"] = key in selected

        return list(groups.values())

    def _decide_recommendation_acceptance(self, job):
        """Décide une seule fois, à la soumission, si le job suit sa recommandation."""
        if not job["receives_recommendation"]:
            job["recommendation_suivie"] = False
        else:
            job["recommendation_suivie"] = self.rng.random() < self.acceptance_rate

    def _job_follows_recommendation(self, job):
        return bool(job.get("recommendation_suivie"))

    def _try_assign(self, job, cluster_usage):
        """Cherche un cluster disponible pour ce job. Retourne (rec, rank) ou None.
        Méthode pure : aucun effet de bord."""
        if self._job_follows_recommendation(job):
            for rank in sorted(job["recs"].keys()):
                rec = job["recs"][rank]
                cluster = rec["recommended_cluster"]
                if cluster_usage[cluster] < self.capacities.get(cluster, float("inf")):
                    return rec, rank
            return None

        orig = job["orig_cluster"]
        if cluster_usage[orig] < self.capacities.get(orig, float("inf")):
            return self._build_orig_rec(job), -1
        return None

    def _schedule(self, job, rec, rank, start_time, cluster_usage, events, counter,
                  recommendation_suivie, timeout_fallback=False):

        cluster = rec["recommended_cluster"]
        duration = timedelta(hours=rec["est_duration_h"])
        end_time = start_time + duration

        job["_scheduled"] = True
        cluster_usage[cluster] += 1
        heappush(events, (end_time, self.FINISH, counter, cluster))
        heappush(self._finish_times_by_cluster[cluster], end_time)

        self.resultats.append({
            "job_id": rec["job_id"],
            "user": job["user"],
            "submission_time": job["submission_time"],
            "start_time": start_time,
            "end_time": end_time,
            "orig_cluster": job["orig_cluster"],
            "cluster_choisi": cluster,
            "rang_choisi": rank,
            "orig_energy_kWh": job["orig_energy_kWh"],
            "est_energy_kWh": rec["est_energy_kWh"],
            "orig_duration_h": job["orig_duration_h"],
            "est_duration_h": rec["est_duration_h"],
            "temps_attente_h": (start_time - job["submission_time"]).total_seconds() / 3600,
            "reco_recue": job["receives_recommendation"],
            "recommendation_suivie": recommendation_suivie,
            "timeout_fallback": timeout_fallback,
        })

    def _build_orig_rec(self, job):
        """Construit le dict 'rec' correspondant au cluster original du job."""
        for rank in job["recs"]:
            if job["recs"][rank]["recommended_cluster"] == job["orig_cluster"]:
                return job["recs"][rank]
        any_rec = next(iter(job["recs"].values()))
        return {
            "job_id": any_rec["job_id"],
            "recommended_cluster": job["orig_cluster"],
            "est_energy_kWh": job["orig_energy_kWh"],
            "est_duration_h": job["orig_duration_h"],
            "orig_energy_kWh": job["orig_energy_kWh"],
            "orig_duration_h": job["orig_duration_h"],
        }

    def _force_assign_on_timeout(self, job, start_time, cluster_usage, events, counter):
        """Force l'assignation du job sur son cluster original (bypass de la capacité).
        Utilisé uniquement quand max_wait_h est dépassé : c'est le hard deadline du scénario 4."""
        rec_orig = self._build_orig_rec(job)
        self._schedule(job, rec_orig, -1, start_time, cluster_usage, events, counter,
                       recommendation_suivie=False, timeout_fallback=True)

    def _get_backfill_candidates(self, waiting_jobs, reservation_time, cluster_cible, current_time):
        if reservation_time is None or cluster_cible is None:
            return None

        candidates = []
        available_window = reservation_time - current_time

        for wjob in waiting_jobs:
            if wjob.get("_scheduled"):
                continue

            if self._job_follows_recommendation(wjob):
                for rank in sorted(wjob["recs"].keys()):
                    wjob_rec = wjob["recs"][rank]
                    cluster = wjob_rec["recommended_cluster"]
                    duration = timedelta(hours=wjob_rec["est_duration_h"])
                    if duration < available_window and cluster == cluster_cible:
                        candidates.append((wjob, wjob_rec, rank))
                        break
            else:
                duration = timedelta(hours=wjob["orig_duration_h"])
                if duration < available_window and wjob["orig_cluster"] == cluster_cible:
                    candidates.append((wjob, None, -1))

        return candidates or None

    def _compute_reservation(self, job, cluster_usage):
        cluster_cible = None

        if self._job_follows_recommendation(job):
            for rank in sorted(job["recs"].keys()):
                rec = job["recs"][rank]
                cluster = rec["recommended_cluster"]
                if cluster_usage[cluster] >= self.capacities.get(cluster, float("inf")):
                    cluster_cible = cluster
                    break
        else:
            orig = job["orig_cluster"]
            if cluster_usage[orig] >= self.capacities.get(orig, float("inf")):
                cluster_cible = orig

        if cluster_cible is None:
            return None

        capacity = self.capacities.get(cluster_cible, float("inf"))
        usage = cluster_usage[cluster_cible]
        releases_needed = max(1, int(usage - capacity + 1))
        finish_heap = self._finish_times_by_cluster.get(cluster_cible, [])

        if len(finish_heap) < releases_needed:
            return None

        reservation_time = nsmallest(releases_needed, finish_heap)[-1]
        return reservation_time, cluster_cible

    def without_backfill(self, waiting, cluster_usage, events, counter, time):
        still_waiting = deque()

        while waiting:
            wjob = waiting.popleft()
            if wjob.get("_scheduled"):
                continue

            result = self._try_assign(wjob, cluster_usage)
            if result is not None:
                rec, rank = result
                self._schedule(wjob, rec, rank, time, cluster_usage, events, counter,
                               recommendation_suivie=self._job_follows_recommendation(wjob))
                counter += 1
            else:
                still_waiting.append(wjob)

        return still_waiting, counter

    def func_with_backfill(self, waiting, cluster_usage, events, counter, time):
        while waiting and waiting[0].get("_scheduled"):
            waiting.popleft()
        if not waiting:
            return counter

        reservation = self._compute_reservation(waiting[0], cluster_usage)
        if reservation is not None:
            reservation_time, cluster_cible = reservation
            candidates = self._get_backfill_candidates(islice(waiting, 1, None), reservation_time, cluster_cible, time)
            if candidates is not None:
                candidates.sort(key=lambda x: x[1]["est_duration_h"] if x[1] is not None else x[0]["orig_duration_h"])
                for wjob, _, _ in candidates:
                    if wjob.get("_scheduled"):
                        continue
                    result = self._try_assign(wjob, cluster_usage)
                    if result is not None:
                        rec, rank = result
                        self._schedule(wjob, rec, rank, time, cluster_usage, events, counter,
                                       recommendation_suivie=self._job_follows_recommendation(wjob))
                        counter += 1

        if waiting and not waiting[0].get("_scheduled"):
            result = self._try_assign(waiting[0], cluster_usage)
            if result is not None:
                rec, rank = result
                self._schedule(waiting[0], rec, rank, time, cluster_usage, events, counter,
                               recommendation_suivie=self._job_follows_recommendation(waiting[0]))
                counter += 1

        remaining = deque(w for w in waiting if not w.get("_scheduled"))
        waiting.clear()
        waiting.extend(remaining)

        return counter

    def run(self):
        cluster_usage = defaultdict(int)
        events = []
        counter = 0
        waiting = deque()
        self.resultats = []
        self._finish_times_by_cluster = defaultdict(list)

        for job in sorted(self.logical_jobs, key=lambda j: j["submission_time"]):
            heappush(events, (job["submission_time"], self.SUBMIT, counter, job))
            counter += 1

        while events:
            time, etype, _, payload = heappop(events)

            if etype == self.FINISH:
                cluster = payload
                cluster_usage[cluster] -= 1
                if self._finish_times_by_cluster[cluster]:
                    heappop(self._finish_times_by_cluster[cluster])

                if self.with_backfill:
                    counter = self.func_with_backfill(waiting, cluster_usage, events, counter, time)
                else:
                    waiting, counter = self.without_backfill(waiting, cluster_usage, events, counter, time)

            elif etype == self.TIMEOUT:
                job = payload
                if not job.get("_scheduled"):
                    self._force_assign_on_timeout(job, time, cluster_usage, events, counter)
                    counter += 1

            else:  # SUBMIT
                job = payload
                self._decide_recommendation_acceptance(job)

                result = self._try_assign(job, cluster_usage)
                if result is not None:
                    rec, rank = result
                    self._schedule(job, rec, rank, time, cluster_usage, events, counter,
                                   recommendation_suivie=self._job_follows_recommendation(job))
                    counter += 1
                else:
                    waiting.append(job)
                    if self.max_wait_h is not None:
                        timeout_time = job["submission_time"] + timedelta(hours=self.max_wait_h)
                        heappush(events, (timeout_time, self.TIMEOUT, counter, job))
                        counter += 1

        return pd.DataFrame(self.resultats)

    def bilan(self) -> dict:
        df = pd.DataFrame(self.resultats)
        rang_dist = df["rang_choisi"].value_counts().sort_index().to_dict()
        nb_recues = int(df["reco_recue"].sum())
        nb_suivies = int(df["recommendation_suivie"].sum())
        nb_timeouts = int(df["timeout_fallback"].sum()) if "timeout_fallback" in df.columns else 0

        return {
            "nb_jobs": len(df),
            "energie_orig_total_kWh": df["orig_energy_kWh"].sum(),
            "energie_sim_total_kWh": df["est_energy_kWh"].sum(),
            "gain_energie_kWh": df["orig_energy_kWh"].sum() - df["est_energy_kWh"].sum(),
            "gain_energie_%": (1 - df["est_energy_kWh"].sum() / df["orig_energy_kWh"].sum()) * 100,
            "duree_orig_moyenne_h": df["orig_duration_h"].mean(),
            "duree_sim_moyenne_h": df["est_duration_h"].mean(),
            "attente_moyenne_h": df["temps_attente_h"].mean(),
            "attente_max_h": df["temps_attente_h"].max(),
            "nb_jobs_avec_attente": (df["temps_attente_h"] > 0).sum(),
            "nb_migrations": (df["orig_cluster"] != df["cluster_choisi"]).sum(),
            "distribution_rangs": rang_dist,
            "nb_recommandations_recues": nb_recues,
            "nb_recommandations_suivies": nb_suivies,
            "taux_recommandations_recues_%": round(nb_recues / len(df) * 100, 1),
            "taux_acceptation_effectif_%": round(nb_suivies / nb_recues * 100, 1) if nb_recues else 0.0,
            "taux_adhesion_effectif_%": round(nb_suivies / len(df) * 100, 1),
            "nb_timeouts_fallback": nb_timeouts,
        }

    def bilan_par_cluster(self) -> pd.DataFrame:
        df = pd.DataFrame(self.resultats)
        return df.groupby("cluster_choisi").agg(
            nb_jobs=("job_id", "count"),
            energie_orig_kWh=("orig_energy_kWh", "sum"),
            energie_sim_kWh=("est_energy_kWh", "sum"),
            duree_sim_moyenne_h=("est_duration_h", "mean"),
            attente_moyenne_h=("temps_attente_h", "mean"),
            rang_moyen=("rang_choisi", "mean"),
        ).round(2)
