"""Archive-guided search with online, within-run expensive-evaluation screening."""

import numpy as np
import pandas as pd


class PTDLCoupledSurrogateSearch:
    def __init__(self, resolution=6, generations=20, seed=20261015,
                 archive_limit=200, progress_callback=None, use_surrogate=True,
                 candidate_pool=48, proposal_policy="uniform", deduplicate=False):
        if resolution < 1 or generations < 1 or candidate_pool < 4:
            raise ValueError("Positive budget and at least four screening proposals required")
        self.size = (resolution + 1) * (resolution + 2) // 2
        self.generations = generations
        self.rng = np.random.default_rng(seed)
        self.archive_limit = archive_limit
        self.progress_callback = progress_callback
        self.use_surrogate = use_surrogate
        self.candidate_pool = candidate_pool
        if proposal_policy not in ("uniform", "credit", "credit-prior", "local-linear"):
            raise ValueError("Unknown proposal policy")
        if use_surrogate and proposal_policy != "uniform":
            raise ValueError("Experimental proposal policies require online trees disabled")
        self.proposal_policy = proposal_policy
        self.seed = seed
        self.deduplicate = deduplicate
        if deduplicate and (use_surrogate or proposal_policy != "credit-prior"):
            raise ValueError("Duplicate repair is registered with credit-prior search")

    @staticmethod
    def _score(values, weight):
        return np.max(values * weight, axis=-1) + .05 * np.sum(values * weight, axis=-1)

    def run(self, problem):
        from sklearn.ensemble import ExtraTreesRegressor
        from MOEAD多目标联合优化 import (
            OptimizerRun, _history_row, _mopso_archive, generate_simplex_weights,
            non_dominated_sort, crowding_distances,
        )
        rng = self.rng
        if self.deduplicate:
            from 耦合候选去重 import UniqueCoupledProblem
            problem = UniqueCoupledProblem(problem, self.seed, self.size)
        lo, hi = problem.lower_bounds, problem.upper_bounds
        width = np.maximum(hi - lo, 1e-12)
        fixed = hi == lo
        population = problem.sample(self.size, rng)
        evaluations = [problem.evaluate(x) for x in population]
        calls = list(evaluations)
        archive = _mopso_archive(calls, self.archive_limit)
        snapshots = {0: list(archive)}
        history = [_history_row(0, evaluations, calls)]
        operator_credit = np.ones(4)
        weights = np.maximum(generate_simplex_weights(resolution=self._resolution()).to_numpy(float), .03)
        weights /= weights.sum(1, keepdims=True)
        labels = getattr(problem, "decision_labels", ())
        aero = [i for i, name in enumerate(labels)
                if name in getattr(problem, "design_labels", ())[:7]]
        # Geometry, system sizing and phase dispatch move together in one proposal.
        groups = [np.array(aero, int),
                  np.array([i for i in range(len(width)) if i not in aero and i < 11], int),
                  np.array([i for i in range(11, len(width))], int)]
        groups = [g for g in groups if len(g)]
        if self.progress_callback:
            self.progress_callback(0, self.generations)
        for generation in range(1, self.generations + 1):
            feasible = [e for e in calls if e.feasible]
            basis = feasible or calls
            values = np.vstack([e.objectives for e in basis])
            ideal = values.min(0)
            scale = np.maximum(values.max(0) - ideal, 1e-9)
            data_x = (np.vstack([e.decision for e in calls]) - lo) / width
            data_y = (np.vstack([e.objectives for e in calls]) - ideal) / scale
            violations = np.array([e.violation for e in calls])
            model = None
            if self.use_surrogate and generation >= 4:
                # All targets come from this run's counted true evaluations.
                model = ExtraTreesRegressor(
                    n_estimators=64, min_samples_leaf=2, max_features=1.,
                    random_state=int(rng.integers(2**31)), n_jobs=1,
                ).fit(data_x, np.column_stack((data_y, np.log1p(violations))))
            parents = archive or evaluations
            normalized = (np.vstack([e.objectives for e in parents]) - ideal) / scale
            trials = []
            generation_credit = np.zeros(4)
            generation_attempts = np.zeros(4)
            for index in rng.permutation(self.size):
                weight = weights[index]
                leader_index = int(np.argmin(self._score(normalized, weight)))
                if rng.random() < .15:
                    leader_index = int(rng.integers(len(parents)))
                leader = parents[leader_index].decision
                base = evaluations[index].decision
                pool = []
                operator_ids = []
                radius = .12 * (1 - generation / self.generations) + .015
                for k in range(self.candidate_pool):
                    a, b = rng.choice(len(calls), 2, replace=False)
                    delta = calls[a].decision - calls[b].decision
                    if k % 3 == 0:
                        proposal = base + rng.uniform(.3, .9) * (leader - base) + rng.uniform(.2, .7) * delta
                    else:
                        proposal = leader + rng.normal(0., radius, len(width)) * width
                        if k % 3 == 1 and groups:
                            mask = np.zeros(len(width), bool)
                            mask[groups[int(rng.integers(len(groups)))]] = True
                            proposal = np.where(mask, proposal, leader)
                        else:
                            mask = rng.random(len(width)) < .25
                            mask[int(rng.integers(len(width)))] = True
                            proposal = np.where(mask, proposal, leader)
                    proposal = np.clip(proposal, lo, hi)
                    proposal[fixed] = lo[fixed]
                    pool.append(proposal)
                    operator_ids.append(k % 3)
                # Coordinate boundary moves expose sizing and dispatch sensitivities
                # that interpolation around archive leaders alone cannot discover.
                for coordinate in rng.permutation(np.flatnonzero(~fixed)):
                    for boundary in (lo[coordinate], hi[coordinate]):
                        proposal = leader.copy()
                        proposal[coordinate] = boundary
                        pool.append(proposal)
                        operator_ids.append(3)
                pool = np.asarray(pool)
                z = (pool - lo) / width
                if model is None:
                    # Without a fitted screen, evaluate one genuine proposal directly.
                    chosen = int(rng.integers(len(pool)))
                    if self.proposal_policy in ("credit", "credit-prior"):
                        # Credit changes only the sampling mix; every selected
                        # proposal still consumes one true mission evaluation.
                        probabilities = .15 + .4*operator_credit/operator_credit.sum()
                        if self.proposal_policy == "credit-prior":
                            counts = np.bincount(operator_ids, minlength=4).astype(float)
                            prior = counts/counts.sum()
                            learned = prior*operator_credit
                            probabilities = .6*prior + .4*learned/learned.sum()
                        operator = int(rng.choice(4, p=probabilities))
                        options = np.flatnonzero(np.asarray(operator_ids) == operator)
                        if len(options):
                            chosen = int(rng.choice(options))
                    elif (self.proposal_policy == "local-linear" and generation >= 3
                          and rng.random() < .5):
                        anchor = (leader-lo)/width
                        distances = np.linalg.norm(data_x-anchor, axis=1)
                        neighbors = np.argsort(distances)[:min(len(calls), 4*len(width))]
                        local = data_x[neighbors]-anchor
                        locality = np.exp(-distances[neighbors]**2/.5)
                        matrix = np.column_stack((np.ones(len(neighbors)), local))
                        penalty = np.eye(matrix.shape[1])*.03
                        penalty[0, 0] = 1e-6
                        targets = np.column_stack((np.clip(data_y[neighbors], -5, 5),
                                                   np.log1p(violations[neighbors])))
                        coefficients = np.linalg.solve(
                            matrix.T @ (locality[:, None]*matrix)+penalty,
                            matrix.T @ (locality[:, None]*targets))
                        deltas = z-anchor
                        predicted = deltas @ coefficients[1:]
                        predicted[:, :3] += normalized[leader_index]
                        predicted[:, 3] += np.log1p(parents[leader_index].violation)
                        acquisition = (self._score(predicted[:, :3], weight)
                                       +10*np.maximum(0, predicted[:, 3])
                                       +.03*np.linalg.norm(deltas, axis=1))
                        repeated = np.min(np.linalg.norm(
                            z[:, None, :]-data_x[None, :, :], axis=2), axis=1) < 1e-8
                        acquisition[repeated] = np.inf
                        if np.isfinite(acquisition).any():
                            chosen = int(np.argmin(acquisition))
                else:
                    predictions = np.stack([tree.predict(z) for tree in model.estimators_])
                    mean, std = predictions.mean(0), predictions.std(0)
                    exploration = .6 * (1 - generation / self.generations) + .1
                    lower = mean[:, :3] - exploration * std[:, :3]
                    violation = np.maximum(0., mean[:, 3] - .3 * std[:, 3])
                    distance = np.min(np.linalg.norm(z[:, None, :] - data_x[None, :, :], axis=2), axis=1)
                    acquisition = self._score(lower, weight) + 10 * violation - .02 * distance
                    acquisition[distance < 1e-8] = np.inf
                    chosen = int(np.argmin(acquisition))
                child = problem.evaluate(pool[chosen])
                trials.append(child)
                operator = operator_ids[chosen]
                generation_attempts[operator] += 1
                reference = parents[leader_index]
                old_score = float(self._score(normalized[leader_index], weight))
                new_score = float(self._score((child.objectives-ideal)/scale, weight))
                if ((child.feasible and not reference.feasible)
                        or (child.feasible and reference.feasible and new_score < old_score)
                        or (not child.feasible and not reference.feasible
                            and child.violation < reference.violation)):
                    generation_credit[operator] += 1
            calls.extend(trials)
            archive = _mopso_archive(archive + trials, self.archive_limit)
            combined = evaluations + trials
            selected = []
            for front in non_dominated_sort(combined):
                remaining = self.size - len(selected)
                if len(front) <= remaining:
                    selected.extend(front)
                else:
                    crowding = crowding_distances(front, combined)
                    selected.extend(sorted(front, key=lambda i: crowding[i], reverse=True)[:remaining])
                    break
            evaluations = [combined[i] for i in selected]
            history.append(_history_row(generation, evaluations, calls))
            history[-1]["surrogate_screen_active"] = float(model is not None)
            history[-1]["random_proposals_per_evaluation"] = self.candidate_pool
            if self.deduplicate:
                history[-1]["duplicate_repairs"] = problem.repairs
            if self.proposal_policy != "uniform":
                operator_credit = .7*operator_credit + .3*(
                    generation_credit/np.maximum(generation_attempts, 1)+.02)
                history[-1].update({f"proposal_credit_{j}": operator_credit[j] for j in range(4)})
            snapshots[generation] = list(archive)
            if self.progress_callback:
                self.progress_callback(generation, self.generations)
        name = "PTDL-耦合代理辅助" if self.use_surrogate else "PTDL-耦合引导"
        if self.proposal_policy == "credit":
            name = "PTDL-耦合贡献学习"
        elif self.proposal_policy == "credit-prior":
            name = "PTDL-耦合贡献学习保先验"
        elif self.proposal_policy == "local-linear":
            name = "PTDL-耦合局部灵敏度"
        if self.deduplicate:
            name = "PTDL-去重贡献学习"
        return OptimizerRun(name, np.vstack([e.decision for e in evaluations]),
                            evaluations, pd.DataFrame(history), calls, snapshots)

    def _resolution(self):
        return int(round((np.sqrt(8 * self.size + 1) - 3) / 2))
