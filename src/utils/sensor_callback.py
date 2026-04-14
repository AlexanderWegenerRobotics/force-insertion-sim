import numpy as np
from scipy.signal import butter, sosfilt_zi
from scipy.spatial.transform import Rotation


class EMAFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._state = None

    def reset(self):
        self._state = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.copy()
        else:
            self._state = self.alpha * x + (1 - self.alpha) * self._state
        return self._state.copy()


class ButterworthFilter:
    def __init__(self, cutoff_hz: float, fs_hz: float, order: int = 2):
        self.sos = butter(order, cutoff_hz, btype='low', fs=fs_hz, output='sos')
        self._zi = None
        self._n = None

    def reset(self):
        self._zi = None
        self._n = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._zi is None:
            self._n = len(x)
            zi_single = sosfilt_zi(self.sos)
            self._zi = np.stack([zi_single * x[i] for i in range(self._n)], axis=-1)

        out = np.empty(self._n)
        for i in range(self._n):
            y, self._zi[:, :, i] = sosfilt_zi_step(self.sos, x[i], self._zi[:, :, i])
            out[i] = y
        return out


def sosfilt_zi_step(sos, x_scalar, zi):
    n_sections = sos.shape[0]
    x = float(x_scalar)
    zi_out = np.empty_like(zi)
    for s in range(n_sections):
        b0, b1, b2, a0, a1, a2 = sos[s]
        y = b0 * x + zi[s, 0]
        zi_out[s, 0] = b1 * x - a1 * y + zi[s, 1]
        zi_out[s, 1] = b2 * x - a2 * y
        x = y
    return x, zi_out


class SensorCallback:
    def __init__(self, device_name: str, gravity_compensation: bool = False,
                 ee_mass: float = 0.0, filter_type: str = "none",
                 filter_alpha: float = 0.2, cutoff_hz: float = 20.0,
                 fs_hz: float = 200.0, butter_order: int = 2,
                 base_orientation: list = [1, 0, 0, 0]):
        self.device_name = device_name
        self.p = f"{device_name}/"
        self.gravity_compensation = gravity_compensation
        self.ee_mass = ee_mass
        self.filter_type = filter_type
        self.latest = None

        if base_orientation is not None:
            quat_wxyz = base_orientation
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            R_world_base = Rotation.from_quat(quat_xyzw).as_matrix()
            self.R_base_world = R_world_base.T
        else:
            self.R_base_world = np.eye(3)

        self._force_filter = None
        self._torque_filter = None

        if filter_type == "ema":
            self._force_filter = EMAFilter(filter_alpha)
            self._torque_filter = EMAFilter(filter_alpha)
        elif filter_type == "butterworth":
            self._force_filter = ButterworthFilter(cutoff_hz, fs_hz, butter_order)
            self._torque_filter = ButterworthFilter(cutoff_hz, fs_hz, butter_order)

    def reset(self):
        if self._force_filter is not None:
            self._force_filter.reset()
        if self._torque_filter is not None:
            self._torque_filter.reset()

    def __call__(self, mj_model, mj_data):
        p = self.p
        ft_force = mj_data.sensor(f'{p}ft_force').data.copy()
        ft_torque = mj_data.sensor(f'{p}ft_torque').data.copy()

        site_id = mj_model.site(f'{p}ft_sensor').id
        R_world_sensor = mj_data.site_xmat[site_id].reshape(3, 3)

        if self.gravity_compensation:
            g_world = np.array([0, 0, self.ee_mass * 9.81])
            g_sensor = R_world_sensor.T @ g_world
            ft_force -= g_sensor

        R_base_sensor = self.R_base_world @ R_world_sensor
        ft_force = R_base_sensor @ ft_force
        ft_torque = R_base_sensor @ ft_torque

        if self._force_filter is not None:
            ft_force = self._force_filter(ft_force)
            ft_torque = self._torque_filter(ft_torque)

        result = {
            'sensors': {
                'ft_force':    ft_force,
                'ft_torque':   ft_torque,
                'ee_linvel':   mj_data.sensor(f'{p}ee_linvel').data.copy(),
                'ee_angvel':   mj_data.sensor(f'{p}ee_angvel').data.copy(),
                'peg_tip_pos': mj_data.sensor(f'{p}peg_tip_pos').data.copy(),
            }
        }
        self.latest = result
        return result