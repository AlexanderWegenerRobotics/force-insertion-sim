import numpy as np
from simcore.common.robot_kinematics import RobotKinematics


class ObservationBuilder:
    def __init__(self, system, device_name="arm"):
        self.system = system
        self.device_name = device_name
        self.prefix = f"{device_name}/"

    def get(self) -> np.ndarray:
        """
        Build 18-dim observation vector:
        [F_ext(6), F_in(6), xdot(6)]
        """
        sensors = self.system.sim.get_sensor_data()
        state   = self.system.get_state()[self.device_name]

        # F_ext: external wrench at ft_sensor site (6)
        ft_force  = sensors[f'{self.prefix}ft_force']
        ft_torque = sensors[f'{self.prefix}ft_torque']
        F_ext = np.concatenate([ft_force, ft_torque])

        # F_in: internal wrench from Pinocchio (6)
        kin = self.system.ctrl[self.device_name].kin_model
        F_in = kin.get_internal_wrench(state.q, state.qd, state.tau)

        # xdot: EE velocity (6)
        ee_linvel = sensors[f'{self.prefix}ee_linvel']
        ee_angvel = sensors[f'{self.prefix}ee_angvel']
        xdot = np.concatenate([ee_linvel, ee_angvel])

        return np.concatenate([F_ext, F_in, xdot])

    def get_peg_tip_pos(self) -> np.ndarray:
        """Convenience — peg tip world position for state machine logic."""
        sensors = self.system.sim.get_sensor_data()
        return sensors[f'{self.prefix}peg_tip_pos'].copy()

    def get_raw(self) -> dict:
        """Return raw sensor dict for debugging / logging."""
        sensors = self.system.sim.get_sensor_data()
        state   = self.system.get_state()[self.device_name]
        kin     = self.system.ctrl[self.device_name].kin_model
        return {
            'F_ext':     np.concatenate([sensors[f'{self.prefix}ft_force'],
                                         sensors[f'{self.prefix}ft_torque']]),
            'F_in':      kin.get_internal_wrench(state.q, state.qd, state.tau),
            'xdot':      np.concatenate([sensors[f'{self.prefix}ee_linvel'],
                                         sensors[f'{self.prefix}ee_angvel']]),
            'peg_tip':   sensors[f'{self.prefix}peg_tip_pos'],
        }