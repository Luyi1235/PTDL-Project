"""Repair repeated coupled proposals before one counted mission evaluation."""

import numpy as np


class UniqueCoupledProblem:
    def __init__(self, problem, seed, initial_size):
        self.problem = problem
        self.rng = np.random.default_rng(seed+7919)
        self.initial_size = initial_size
        self.count = 0
        self.repairs = 0
        self.seen = set()

    def __getattr__(self, name):
        return getattr(self.problem, name)

    @staticmethod
    def key(decision):
        return tuple(np.round(decision, 12))

    def evaluate(self, decision):
        candidate = np.asarray(decision, float).copy()
        lo, hi = self.problem.lower_bounds, self.problem.upper_bounds
        free = np.flatnonzero(hi > lo)
        if (self.count >= self.initial_size and self.key(candidate) in self.seen and len(free)):
            width = np.maximum(hi-lo, 1e-12)
            center = (candidate-lo)/width
            for _ in range(32):
                group = free[free < 11] if self.rng.random() < .5 else free[free >= 11]
                if not len(group):
                    group = free
                selected = self.rng.choice(group, min(3, len(group)), replace=False)
                proposal = center.copy()
                proposal[selected] += self.rng.normal(0, .025, len(selected))
                proposal = 1-abs(np.mod(proposal, 2)-1)
                repaired = lo+proposal*width
                repaired[hi == lo] = lo[hi == lo]
                if self.key(repaired) not in self.seen:
                    candidate = repaired
                    self.repairs += 1
                    break
        result = self.problem.evaluate(candidate)
        self.seen.add(self.key(candidate))
        self.count += 1
        return result
