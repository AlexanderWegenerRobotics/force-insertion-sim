import numpy as np
from task.insertion_episode import InsertionEpisode, Phase
from policy.dynamic_filter import DynamicFilter


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
            F_df   = self.policy.predict(o_prev, o_curr)
            Fff    = self.filter.step(F_df)

            self.system.set_target(self.device_name, {
                "x":   x_current,
                "xd":  np.zeros(6),
                "xdd": np.zeros(6),
                "Fff": Fff,
            })

            o_prev = o_curr
            self._tick()

        self.fail_phase = Phase.SEARCH
        return Phase.FAILED
