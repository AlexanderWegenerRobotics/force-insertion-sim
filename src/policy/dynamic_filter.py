import numpy as np


class DynamicFilter:
    def __init__(self, alpha=0.9, beta=0.3, dt=0.005):
        self.alpha = alpha
        self.beta  = beta
        self.dt    = dt
        self.reset()

    def reset(self):
        self.F_ff     = np.zeros(6)
        self.F_ff_dot = np.zeros(6)

    def step(self, F_df: np.ndarray, dt: float) -> np.ndarray:
        F_ff_ddot  = self.alpha * (self.beta * (F_df - self.F_ff) - self.F_ff_dot)
        self.F_ff_dot += F_ff_ddot * dt
        self.F_ff     += self.F_ff_dot * dt
        return self.F_ff.copy()