"""Reference-guided search with success-adapted steps and persistent particles."""

import numpy as np
import pandas as pd


class PTDLTaskAdaptiveSearch:
    def __init__(self, resolution=6, generations=20, seed=20261015,
                 archive_limit=200, progress_callback=None, adaptive=True):
        if resolution < 1 or generations < 1:
            raise ValueError("Positive resolution and generations required")
        self.resolution = resolution
        self.generations = generations
        self.size = (resolution + 1) * (resolution + 2) // 2
        self.rng = np.random.default_rng(seed)
        self.archive_limit = archive_limit
        self.progress_callback = progress_callback
        self.adaptive = adaptive

    @staticmethod
    def score(values, weight):
        return np.max(values * weight, axis=-1) + .05 * np.sum(values * weight, axis=-1)

    def run(self, problem):
        from MOEAD多目标联合优化 import (
            OptimizerRun, _history_row, _mopso_archive, generate_simplex_weights,
            non_dominated_sort, crowding_distances,
        )
        rng = self.rng
        lo, hi = problem.lower_bounds, problem.upper_bounds
        width = np.maximum(hi-lo, 1e-12)
        free = np.flatnonzero(hi > lo)
        decisions = problem.sample(self.size, rng)
        particles = [problem.evaluate(x) for x in decisions]
        elite = list(particles)
        calls = list(particles)
        archive = _mopso_archive(calls, self.archive_limit)
        weights = np.maximum(generate_simplex_weights(self.resolution).to_numpy(float), .03)
        weights /= weights.sum(1, keepdims=True)
        velocity = np.zeros((self.size, len(width)))
        credit = np.ones(4)
        radii = np.full(4, .12)
        mean_f, mean_cr = .5, .75
        history = [_history_row(0, elite, calls)]
        snapshots = {0: list(archive)}
        if self.progress_callback:
            self.progress_callback(0, self.generations)
        for generation in range(1, self.generations+1):
            feasible = [e for e in calls if e.feasible]
            basis = feasible or calls
            values = np.vstack([e.objectives for e in basis])
            ideal = values.min(0)
            scale = np.maximum(values.max(0)-ideal, 1e-9)
            parents = archive or elite
            zparents = np.vstack([(e.decision-lo)/width for e in parents])
            normalized = (np.vstack([e.objectives for e in parents])-ideal)/scale
            probability = .15 + .4*credit/credit.sum() if self.adaptive else np.full(4, .25)
            success, attempts = np.zeros(4), np.zeros(4)
            accepted_f, accepted_cr = [], []
            children = []
            for i in rng.permutation(self.size):
                weight = weights[i]
                ranking = np.argsort(self.score(normalized, weight))
                top = ranking[:max(1, min(4, len(ranking)))]
                leader = zparents[int(rng.choice(top))]
                state = (particles[i].decision-lo)/width
                op = int(rng.choice(4, p=probability))
                attempts[op] += 1
                f = float(np.clip(rng.normal(mean_f, .1), .1, .9))
                cr = float(np.clip(rng.normal(mean_cr, .1), .1, 1.))
                radius = radii[op] * (.5 + .5*(1-generation/self.generations))
                if op == 0:
                    a, b = rng.choice(len(calls), 2, replace=False)
                    delta = (calls[a].decision-calls[b].decision)/width
                    proposal = state + f*(leader-state) + f*delta
                    mask = rng.random(len(width)) < cr
                    if len(free):
                        mask[rng.choice(free)] = True
                    proposal = np.where(mask, proposal, state)
                elif op == 1:
                    velocity[i] = (.45*velocity[i]
                                   + rng.random(len(width))*1.3*(leader-state))
                    velocity[i] = np.clip(velocity[i], -.2, .2)
                    proposal = state + velocity[i]
                elif op == 2:
                    proposal = leader.copy()
                    if len(free):
                        # Wing/speed and storage/phase dispatch are coupled blocks.
                        groups = [free[free < 11], free[free >= 11]]
                        group = groups[int(rng.integers(2))]
                        if not len(group):
                            group = free
                        selected = rng.choice(group, size=max(1, min(len(group), 3)), replace=False)
                        proposal[selected] += rng.normal(0., radius, len(selected))
                        if len(width) > 10 and 10 in free and np.any(selected < 7):
                            proposal[10] += rng.normal(0., radius*.5)
                else:
                    proposal = leader.copy()
                    if len(free):
                        coordinate = int(rng.choice(free))
                        proposal[coordinate] = (float(rng.integers(2)) if rng.random() < .3
                                                else proposal[coordinate]+rng.choice((-1., 1.))*radius)
                # Reflection retains variation at active bounds without piling up there.
                proposal = 1.-abs(np.mod(proposal, 2.)-1.)
                decision = lo + np.clip(proposal, 0., 1.)*width
                decision[hi == lo] = lo[hi == lo]
                child = problem.evaluate(decision)
                children.append(child)
                old = particles[i]
                old_score = float(self.score((old.objectives-ideal)/scale, weight))
                new_score = float(self.score((child.objectives-ideal)/scale, weight))
                better = ((child.feasible and not old.feasible)
                          or (child.feasible and old.feasible and new_score < old_score)
                          or (not child.feasible and not old.feasible and child.violation < old.violation))
                if better:
                    particles[i] = child
                    gain = max(1e-3, (old_score-new_score)/max(abs(old_score), 1.))
                    success[op] += min(1., gain)
                    if op == 0:
                        accepted_f.append(f)
                        accepted_cr.append(cr)
            calls.extend(children)
            archive = _mopso_archive(archive+children, self.archive_limit)
            combined = elite+children
            selected = []
            for front in non_dominated_sort(combined):
                remaining = self.size-len(selected)
                if len(front) <= remaining:
                    selected.extend(front)
                else:
                    crowding = crowding_distances(front, combined)
                    selected.extend(sorted(front, key=lambda j: crowding[j], reverse=True)[:remaining])
                    break
            elite = [combined[j] for j in selected]
            if self.adaptive:
                credit = .7*credit + .3*(success/np.maximum(attempts, 1)+.01)
                if accepted_f:
                    mean_f = .8*mean_f + .2*float(np.sum(np.square(accepted_f))/np.sum(accepted_f))
                    mean_cr = .8*mean_cr + .2*float(np.mean(accepted_cr))
                radii = np.clip(radii*np.where(success > 0, 1.08, .92), .015, .25)
            history.append(_history_row(generation, elite, calls))
            history[-1].update({f"operator_{j}_probability": probability[j] for j in range(4)})
            snapshots[generation] = list(archive)
            if self.progress_callback:
                self.progress_callback(generation, self.generations)
        return OptimizerRun("PTDL-任务耦合自适应" if self.adaptive else "PTDL-任务耦合固定",
                            np.vstack([e.decision for e in elite]), elite,
                            pd.DataFrame(history), calls, snapshots)
