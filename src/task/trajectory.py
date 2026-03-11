import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class TrajectoryPlanner:

    def __init__(self):
        self._p_start = None
        self._p_end = None
        self._slerp = None
        self._T = None
        self._t = 0.0
        self._active = False

    def plan(self, p_start, q_start, p_end, q_end, duration: float) -> None:
        assert duration > 0.0
        self._p_start = np.array(p_start, dtype=float)
        self._p_end = np.array(p_end, dtype=float)
        self._T = duration
        self._t = 0.0
        self._active = True

        r_start = Rotation.from_quat(self._wxyz_to_xyzw(q_start))
        r_end = Rotation.from_quat(self._wxyz_to_xyzw(q_end))
        self._slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_start, r_end]))

    def plan_with_speed(self,p_start,q_start,p_end,q_end,max_speed: float = 0.05,min_duration: float = 1.0) -> None:
        dist = np.linalg.norm(np.array(p_end) - np.array(p_start))
        duration = max(dist / max_speed, min_duration)
        self.plan(p_start, q_start, p_end, q_end, duration)

    def step(self, dt: float) -> dict:
        if not self._active:
            raise RuntimeError("No active trajectory. Call plan() first.")

        tau = self._t / self._T
        s, ds_norm, dds_norm = self._minjerk(tau)
        s = np.clip(0.0, 1.0, s)

        ds = ds_norm / self._T
        dds = dds_norm / self._T ** 2

        delta = self._p_end - self._p_start
        pos = self._p_start + s * delta
        vel = ds * delta
        accel = dds * delta

        quat_xyzw = self._slerp(s).as_quat()

        if self._t > 0.0:
            s_prev, _, _ = self._minjerk(np.clip((self._t - dt) / self._T, 0.0, 1.0))
            q_prev_xyzw = self._slerp(s_prev).as_quat()
            dq = self._quat_multiply(quat_xyzw, self._quat_conjugate(q_prev_xyzw))
            omega = 2.0 * dq[:3] / dt
        else:
            omega = np.zeros(3)

        self._t = min(self._t + dt, self._T)
        done = self._t >= self._T

        if done:
            self._active = False
            pos = self._p_end.copy()
            vel = np.zeros(3)
            accel = np.zeros(3)
            omega = np.zeros(3)
            quat_xyzw = self._slerp(1.0).as_quat()

        return {
            "pos": pos,
            "quat": self._xyzw_to_wxyz(quat_xyzw),
            "vel": vel,
            "omega": omega,
            "accel": accel,
            "done": done,
        }

    def is_done(self) -> bool:
        return not self._active

    @staticmethod
    def _minjerk(tau: float):
        tau = np.clip(tau, 0.0, 1.0)
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        ds = 30*tau**2 - 60*tau**3 + 30*tau**4
        dds = 60*tau - 180*tau**2 + 120*tau**3
        return s, ds, dds

    @staticmethod
    def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
        return np.array([q[1], q[2], q[3], q[0]])

    @staticmethod
    def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
        return np.array([q[3], q[0], q[1], q[2]])

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]])

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])