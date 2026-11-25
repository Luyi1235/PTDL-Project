"""Feasibility-first block search with adaptive DE, PSO and genetic operators."""

import numpy as np
import pandas as pd


class PTDLBlockAdaptive:
    """Keep an exact evaluation budget and learn operator credit from real evaluations."""

    def __init__(self, resolution=6, generations=20, seed=20261015,
                 archive_limit=200, progress_callback=None, block_search=True,
                 adaptive_credit=True):
        if resolution < 1 or generations < 1 or archive_limit < 1:
            raise ValueError("Positive resolution, generations and archive size required")
        self.size = (resolution + 1) * (resolution + 2) // 2
        self.generations = generations
        self.rng = np.random.default_rng(seed)
        self.archive_limit = archive_limit
        self.progress_callback = progress_callback
        self.block_search = block_search
        self.adaptive_credit = adaptive_credit

    def run(self, problem):
        from MOEAD多目标联合优化 import (
            OptimizerRun, _history_row, _mopso_archive, _rank_and_crowding,
            _sbx_and_mutate, non_dominated_sort, crowding_distances,
        )
        rng = self.rng
        positions = problem.sample(self.size, rng)
        evaluations = [problem.evaluate(x) for x in positions]
        all_evaluations = list(evaluations)
        archive = _mopso_archive(evaluations, self.archive_limit)
        archive_history = {0: list(archive)}
        history = [_history_row(0, evaluations, all_evaluations)]
        probabilities = np.full(3, 1 / 3)
        credit = np.full(3, .5)
        velocity = np.zeros_like(positions)
        span = np.maximum(problem.upper_bounds - problem.lower_bounds, 1e-12)
        fixed = problem.upper_bounds == problem.lower_bounds
        labels = getattr(problem, "decision_labels", ())
        geometry = np.array([i for i, name in enumerate(labels)
                             if name in getattr(problem, "design_labels", ())], dtype=int)
        control = np.array([i for i in range(len(span)) if i not in geometry], dtype=int)
        if len(geometry) == 0 or len(control) == 0:
            geometry = np.arange(len(span) // 2)
            control = np.arange(len(span) // 2, len(span))
        if self.progress_callback:
            self.progress_callback(0, self.generations)
        for generation in range(1, self.generations + 1):
            values = np.vstack([e.objectives for e in evaluations])
            ideal = values.min(0)
            scale = np.maximum(values.max(0) - ideal, 1e-12)
            rank, crowding, _ = _rank_and_crowding(evaluations)
            trials, trial_velocities, assignments, rewards = [], [], [], []
            for i, parent in enumerate(evaluations):
                operator = int(rng.choice(3, p=probabilities))
                leader = archive[int(rng.integers(len(archive)))].decision
                if operator == 0:
                    choices = np.delete(np.arange(self.size), i)
                    a, b = rng.choice(choices, 2, replace=False)
                    f = float(rng.uniform(.35, .85))
                    proposal = positions[i] + f * (leader - positions[i]) + f * (positions[a] - positions[b])
                    change = rng.random(len(span)) < .65
                    change[int(rng.integers(len(span)))] = True
                    proposal = np.where(change, proposal, positions[i])
                    new_velocity = proposal - positions[i]
                elif operator == 1:
                    inertia = .8 - .45 * generation / self.generations
                    new_velocity = np.clip(
                        inertia * velocity[i] + 1.6 * rng.random(len(span)) * (leader - positions[i]),
                        -.2 * span, .2 * span)
                    proposal = positions[i] + new_velocity
                else:
                    candidates = rng.integers(0, self.size, 2)
                    a = min(candidates, key=lambda j: (rank[j], -crowding[j]))
                    b = int(rng.integers(self.size))
                    proposal = _sbx_and_mutate(positions[a], positions[b],
                                               problem.lower_bounds, problem.upper_bounds, rng)
                    new_velocity = np.zeros(len(span))
                # Alternate geometry and scheduling updates, with a joint update every third generation.
                if self.block_search and generation % 3:
                    active = geometry if generation % 3 == 1 else control
                    mask = np.zeros(len(span), bool)
                    mask[active] = True
                    proposal = np.where(mask, proposal, positions[i])
                    new_velocity = np.where(mask, new_velocity, 0.)
                # Reflect overshoot before clipping; preserve fixed decisions exactly.
                z = (proposal - problem.lower_bounds) / span
                proposal = problem.lower_bounds + (1 - np.abs(np.mod(z, 2) - 1)) * span
                proposal[fixed] = problem.lower_bounds[fixed]
                child = problem.evaluate(proposal)
                if child.feasible != parent.feasible:
                    reward = float(child.feasible)
                elif not child.feasible:
                    reward = max(0., (parent.violation - child.violation) / max(parent.violation, 1e-12))
                else:
                    weight = rng.dirichlet(np.ones(3))
                    old = np.max(weight * (parent.objectives - ideal) / scale)
                    new = np.max(weight * (child.objectives - ideal) / scale)
                    reward = float(new < old - 1e-12)
                assignments.append(operator)
                rewards.append(reward)
                trials.append(child)
                trial_velocities.append(new_velocity)
            combined = evaluations + trials
            combined_velocity = np.vstack((velocity, trial_velocities))
            selected = []
            for front in non_dominated_sort(combined):
                remaining = self.size - len(selected)
                if len(front) <= remaining:
                    selected.extend(front)
                else:
                    distances = crowding_distances(front, combined)
                    selected.extend(sorted(front, key=lambda j: distances[j], reverse=True)[:remaining])
                    break
            evaluations = [combined[j] for j in selected]
            positions = np.vstack([e.decision for e in evaluations])
            velocity = combined_velocity[selected]
            all_evaluations.extend(trials)
            archive = _mopso_archive(archive + trials, self.archive_limit)
            for operator in range(3):
                observed = [r for a, r in zip(assignments, rewards) if a == operator]
                if observed:
                    credit[operator] = .8 * credit[operator] + .2 * np.mean(observed)
            if self.adaptive_credit:
                probabilities = .1 + .7 * (credit + .05) / np.sum(credit + .05)
            row = _history_row(generation, evaluations, all_evaluations)
            for name, probability, reward in zip(("de", "pso", "ga"), probabilities, credit):
                row[f"operator_{name}_probability"] = probability
                row[f"operator_{name}_credit"] = reward
            row["block_search"] = float(self.block_search)
            history.append(row)
            archive_history[generation] = list(archive)
            if self.progress_callback:
                self.progress_callback(generation, self.generations)
        return OptimizerRun("PTDL-自适应分块", positions, evaluations, pd.DataFrame(history),
                            all_evaluations, archive_history)
