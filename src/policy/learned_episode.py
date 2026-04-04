import numpy as np
from task.insertion_episode import InsertionEpisode, Phase
from policy.dynamic_filter import DynamicFilter
from simcore.common.pose import Pose
from scipy.spatial.transform import Rotation
import time

class LearnedEpisode(InsertionEpisode):
    def __init__(self, system=None, config=None, policy=None, policy_cfg=None):
        super().__init__(system=system, config=config, collector=None)
        self.policy     = policy
        self.filter     = DynamicFilter(
            alpha=policy_cfg.get("filter_alpha", 0.9),
            beta=policy_cfg.get("filter_beta", 0.3),
            dt=self.dt,
        )

    def reset(self, hole_pos: np.ndarray, hole_quat: np.ndarray) -> None:
        super().reset(hole_pos, hole_quat)
        self.filter.reset()

    def run(self):
        while self.running:
            match self.phase:
                case Phase.IDLE:
                    self.phase = self.system_ready()
                case Phase.APPROACH:
                    self.phase = self.run_approach()
                case Phase.CONTACT:
                    self.phase = self.make_contact()
                case Phase.SEARCH | Phase.INSERT:
                    self.phase = self.run_learned_policy()

            if self.phase in (Phase.FAILED, Phase.DONE):
                self.running = False

    def _get_obs(self, state, sensors) -> np.ndarray:
        q, qd, tau = state.q, state.qd, state.tau
        f_ext      = np.zeros(6)
        f_ext[:3]  = sensors.get(f'{self.prefix}ft_force', np.zeros(3))
        f_ext[3:]  = sensors.get(f'{self.prefix}ft_torque', np.zeros(3))
        f_internal = self.system.ctrl[self.device_name].get_internal_wrench(q, qd, tau)
        ee_vel     = self.system.ctrl[self.device_name].kin_model.get_ee_velocity(q, qd)
        return np.concatenate([f_ext, f_internal, ee_vel]).astype(np.float32)

    def run_learned_policy(self) -> Phase:
        cfg       = self.config.get("episode", {}).get("learned", {})
        timeout   = cfg.get("timeout", 30.0)
        max_steps = int(timeout / self.dt)

        hole_quat     = self.hole_nom_pose.quaternion
        approach_quat = np.array([0, 1, 0, 0])
        R_hole        = Rotation.from_quat([hole_quat[1], hole_quat[2], hole_quat[3], hole_quat[0]])
        R_approach    = Rotation.from_quat([approach_quat[1], approach_quat[2], approach_quat[3], approach_quat[0]])
        q_xyzw        = (R_hole * R_approach).as_quat()
        q_down        = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        x_ref = Pose(
            position=np.array([self.hole_nom_pose.position[0], self.hole_nom_pose.position[1], self.xz0 - 0.02]),
            quaternion=q_down
        )

        state   = self.system.get_state()[self.device_name]
        sensors = self.system.sim.get_sensor_data()
        o_prev  = self._get_obs(state, sensors)

        for _ in range(max_steps):
            state     = self.system.get_state()[self.device_name]
            x_current = self.system.ctrl[self.device_name].get_ee_pose_world(state)
            sensors   = self.system.sim.get_sensor_data()

            if x_current.position[2] - self.peg_offset < self.insertion_success_z:
                return Phase.DONE

            o_curr = self._get_obs(state, sensors)
            start_time = time.time()
            F_df   = self.policy.predict(o_prev, o_curr)
            delta_t = time.time() - start_time
            print(f"Incerence time: {delta_t:.4f}, frequency: {(1.0/delta_t):.2f}")
            t0  = time.time()
            Fff = self.filter.step(F_df, dt=time.time() - t0)

            self.system.set_target(self.device_name, {
                "x":   x_ref,
                "xd":  np.zeros(6),
                "xdd": np.zeros(6),
                "Fff": F_df,
            })

            o_prev = o_curr
            self._tick()

        self.fail_phase = Phase.SEARCH
        return Phase.FAILED
