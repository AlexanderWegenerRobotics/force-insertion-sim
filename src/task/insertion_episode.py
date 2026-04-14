import time
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from enum import Enum, auto
from simcore.common.pose import Pose
from task.trajectory import TrajectoryPlanner
from data.episode_data_collector import EpisodeDataCollector


class Phase(Enum):
    UNDEFINED = auto()
    IDLE      = auto()
    APPROACH  = auto()
    CONTACT   = auto()
    SEARCH    = auto()
    INSERT    = auto()
    DONE      = auto()
    FAILED    = auto()

class InsertionEpisode:
    def __init__(self, system=None, config=None, collector=None):
        self.system = system
        self.config = config
        self.device_name = config.get("device_name", "arm")
        self.collector = collector
        self.headless = system.headless

        self.phase = Phase.UNDEFINED
        self.fail_phase = Phase.UNDEFINED
        self.running = False

        self.prefix = "arm/"
        self.trajectory = TrajectoryPlanner()

        self.q_init = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
        self.peg_offset = config.get("peg_ee_offset")

        hole_cfg = self.config.get('hole_pose')
        self.hole_nom_pose = Pose(position=hole_cfg.get("pos"), quaternion=hole_cfg.get("quat"))
        self.hole_nom_pose.position[2] += hole_cfg.get("height")
        self.insertion_success_z = self.hole_nom_pose.position[2] - self.config.get("insert_depth")
        self.dt = self.system.get_control_cycle()

        self._sim_time = 0.0

    def _tick(self):
        """Advance one timestep: either step physics synchronously or sleep."""
        if self.headless:
            self.system.step()
        else:
            time.sleep(self.dt)
        self._sim_time += self.dt

    def reset(self, hole_pos, hole_quat):
        self.system.sim.reset_device_state("arm", self.q_init)
        self.system.sim.reset_object_pose("hole", hole_pos, hole_quat)
        self.system.set_controller_mode("arm", "dynamic_impedance")
        self.phase = Phase.IDLE
        self.fail_phase = Phase.UNDEFINED
        self.running = True
        self._sim_time = 0.0
        self._wiggle_t = 0.0
        self._last_Fff = np.zeros(6)
        if hasattr(self.system, 'sensor_cb'):
            self.system.sensor_cb.reset()

    def run(self):
        while self.running:
            match self.phase:
                case Phase.IDLE:
                    self.phase = self.system_ready()
                case Phase.APPROACH:
                    self.phase = self.run_approach()
                case Phase.CONTACT:
                    self.phase = self.make_contact()
                case Phase.SEARCH:
                    self.phase = self.run_search()
                case Phase.INSERT:
                    self.phase = self.run_insert()

            if self.phase in (Phase.FAILED, Phase.DONE):
                self.running = False

    def _collect(self, state, x_current, Fff):
        if self.collector is None:
            return
        q, qd, tau = state.q, state.qd, state.tau

        self.system.sensor_cb(self.system.sim.mj_model, self.system.sim.mj_data)
        sensors = self.system.sensor_cb.latest['sensors']

        f_ext = np.zeros(6)
        f_ext[:3] = sensors['ft_force']
        f_ext[3:] = sensors['ft_torque']
        f_internal = self.system.ctrl[self.device_name].get_internal_wrench(q, qd, tau)
        ee_vel = self.system.ctrl[self.device_name].kin_model.get_ee_velocity(q, qd)
        peg_tip = sensors['peg_tip_pos']
        self.collector.record(f_ext=f_ext, f_internal=f_internal, ee_velocity=ee_vel, Fff=Fff,
                            peg_tip_pos=peg_tip, ee_pose=x_current.as_7d(), mode=1, q=q, sim_time=self._sim_time)

    def run_insert(self) -> Phase:
        cfg     = self.config.get("episode", {}).get("insert", {})
        timeout = cfg.get("timeout", 30.0)
        wig     = cfg.get("wiggle", {})
        a, f, phi, az = np.array(wig.get("a")), np.array(wig.get("f")), np.array(wig.get("phi")), wig.get("az")

        z_window, z_score_thresh = cfg.get("z_window"), cfg.get("z_score_threshold")
        v_drop_thresh, done_thresh = cfg.get("velocity_drop_threshold"), cfg.get("done_threshold")

        n_confirm_unstuck = cfg.get("n_confirm_unstuck", 5)
        n_confirm_aligned = cfg.get("n_confirm_aligned", 10)
        n_confirm_stuck   = cfg.get("n_confirm_stuck", 5)
        speed_filter_len  = cfg.get("speed_filter_window", 10)
        ramp_steps        = cfg.get("ramp_steps", 50)
        blend_rate        = 1.0 / max(ramp_steps, 1)

        state = self.system.get_state()[self.device_name]

        z_buf        = deque(maxlen=z_window)
        speed_buf    = deque(maxlen=speed_filter_len)
        fres_z_prev  = 0.0
        insert_state = "STUCK"
        pending      = None
        confirm_cnt  = 0
        blend        = 0.0
        phase_ramp   = 0
        max_steps    = int(timeout / self.dt)

        for step_i in range(max_steps):
            state     = self.system.get_state()[self.device_name]
            x_current = self.system.ctrl[self.device_name].get_ee_pose_world(state)
            sensors   = self.system.sim.get_sensor_data()

            q, qd, tau   = state.q, state.qd, state.tau
            f_ext    = sensors.get(f'{self.prefix}ft_force', np.zeros(6))
            f_res_z  = self.system.ctrl[self.device_name].get_internal_wrench(q, qd, tau)[2] - f_ext[2]

            z_buf.append(x_current.position[2])
            z_arr = np.array(z_buf)
            z_score = abs((z_arr[-1] - z_arr.mean()) / (z_arr.std() + 1e-9)) if len(z_buf) > 10 else 0.0

            xd = self.system.ctrl[self.device_name].kin_model.get_ee_velocity(state.q, state.qd)
            speed_buf.append(np.linalg.norm(xd[:3]))
            speed = np.mean(speed_buf)

            if x_current.position[2] - self.peg_offset < self.insertion_success_z:
                return Phase.DONE

            Fff_wiggle    = a * np.sin(2 * np.pi * f * self._wiggle_t + phi)
            Fff_wiggle[2] = -az

            if phase_ramp < ramp_steps:
                alpha = phase_ramp / ramp_steps
                Fff_wiggle = (1.0 - alpha) * self._last_Fff + alpha * Fff_wiggle
                phase_ramp += 1

            Fff_push      = np.array([0.0, 0.0, -az, 0.0, 0.0, 0.0])

            if insert_state == "STUCK":
                if z_score > z_score_thresh:
                    if pending == "UNSTUCK":
                        confirm_cnt += 1
                    else:
                        pending = "UNSTUCK"
                        confirm_cnt = 1
                else:
                    pending = None
                    confirm_cnt = 0

                if confirm_cnt >= n_confirm_unstuck:
                    insert_state = "UNSTUCK"
                    pending = None
                    confirm_cnt = 0

            elif insert_state == "UNSTUCK":
                if fres_z_prev < f_res_z and speed < v_drop_thresh:
                    if pending == "ALIGNED":
                        confirm_cnt += 1
                    else:
                        pending = "ALIGNED"
                        confirm_cnt = 1
                elif speed < v_drop_thresh:
                    if pending == "STUCK":
                        confirm_cnt += 1
                    else:
                        pending = "STUCK"
                        confirm_cnt = 1
                else:
                    pending = None
                    confirm_cnt = 0

                if pending == "ALIGNED" and confirm_cnt >= n_confirm_aligned:
                    insert_state = "ALIGNED"
                    pending = None
                    confirm_cnt = 0
                elif pending == "STUCK" and confirm_cnt >= n_confirm_stuck:
                    insert_state = "STUCK"
                    pending = None
                    confirm_cnt = 0

            elif insert_state == "ALIGNED":
                if speed < v_drop_thresh:
                    if pending == "STUCK":
                        confirm_cnt += 1
                    else:
                        pending = "STUCK"
                        confirm_cnt = 1
                else:
                    pending = None
                    confirm_cnt = 0

                if confirm_cnt >= n_confirm_stuck:
                    insert_state = "STUCK"
                    pending = None
                    confirm_cnt = 0

            if insert_state == "ALIGNED":
                blend = min(blend + blend_rate, 1.0)
            else:
                blend = max(blend - blend_rate, 0.0)

            Fff = (1.0 - blend) * Fff_wiggle + blend * Fff_push

            self._collect(state, x_current, Fff)

            self.system.set_target(self.device_name, {
                "x":   x_current,
                "xd":  np.zeros(6),
                "xdd": np.zeros(6),
                "Fff": Fff
            })

            fres_z_prev = f_res_z
            self._wiggle_t += self.dt
            self._tick()
        
        self.fail_phase = self.phase
        return Phase.FAILED

    def run_search(self) -> Phase:
        cfg      = self.config.get("episode", {}).get("search", {})
        epsilon, timeout  = cfg.get("hole_detection_threshold"), cfg.get("timeout", 15.0)
        max_steps = int(timeout / self.dt)

        wig      = cfg.get("wiggle", {})
        a, f, phi, az = np.array(wig.get("a")), np.array(wig.get("f")), np.array(wig.get("phi")), wig.get("az")

        state    = self.system.get_state()[self.device_name]
        hole_quat     = self.hole_nom_pose.quaternion
        approach_quat = np.array([0, 1, 0, 0])

        R_hole        = Rotation.from_quat([hole_quat[1], hole_quat[2], hole_quat[3], hole_quat[0]])
        R_approach    = Rotation.from_quat([approach_quat[1], approach_quat[2], approach_quat[3], approach_quat[0]])
        R_final       = R_hole * R_approach
        q_xyzw        = R_final.as_quat()
        q_down        = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        x_ref = Pose(
            position=np.array([self.hole_nom_pose.position[0], self.hole_nom_pose.position[1], self.xz0 - 0.02]),
            quaternion=q_down
        )

        for _ in range(max_steps):
            state     = self.system.get_state()[self.device_name]
            x_current = self.system.ctrl[self.device_name].get_ee_pose_world(state)
            sensors   = self.system.sim.get_sensor_data()

            Fff       = a * np.sin(2 * np.pi * f * self._wiggle_t + phi)
            Fff[2]    = -az
            self._last_Fff = Fff.copy()

            self._collect(state, x_current, Fff)

            self.system.set_target(self.device_name, {
                "x":   x_ref,
                "xd":  np.zeros(6),
                "xdd": np.zeros(6),
                "Fff": Fff
            })

            if self.xz0 - x_current.position[2] > epsilon:
                return Phase.INSERT

            self._wiggle_t += self.dt
            self._tick()

        self.fail_phase = self.phase
        return Phase.FAILED

    def make_contact(self) -> Phase:
        cfg         = self.config.get("episode", {}).get("contact", {})
        f_threshold = cfg.get("force_threshold", 3.0)
        timeout     = cfg.get("timeout", 5.0)
        f_push      = cfg.get("f_push", 3.0)
        n_confirm   = cfg.get("n_confirm", 5)
        max_steps   = int(timeout / self.dt)
        n_above     = 0

        for _ in range(max_steps):
            state   = self.system.get_state()[self.device_name]
            sensors = self.system.sim.get_sensor_data()

            x_current = self.system.ctrl[self.device_name].get_ee_pose_world(state)
            self.system.set_target(self.device_name, {
                "x":   x_current,
                "xd":  np.zeros(6),
                "xdd": np.zeros(6),
                "Fff": np.array([0.0, 0.0, -f_push, 0.0, 0.0, 0.0])
            })

            f_ext = sensors.get(f'{self.prefix}ft_force', np.zeros(6))
            if abs(f_ext[2]) > f_threshold: n_above += 1
            else: n_above = 0

            if n_above >= n_confirm:
                self.xz0 = x_current.position[2]
                return Phase.SEARCH

            self._tick()

        self.fail_phase = self.phase
        return Phase.FAILED

    def system_ready(self) -> Phase:
        if self.headless:
            return Phase.APPROACH
        while not self.system.sim.running:
            time.sleep(0.1)
        return Phase.APPROACH

    def run_approach(self) -> Phase:
        cfg_app = self.config.get("episode", {}).get("approach")

        state = self.system.get_state()
        current_pose = self.system.ctrl[self.device_name].get_ee_pose_world(state[self.device_name])
        p0, q0 = current_pose.position, current_pose.quaternion

        hole_quat     = self.hole_nom_pose.quaternion
        approach_quat = np.array(cfg_app.get("quat", [0, 1, 0, 0]))

        R_hole     = Rotation.from_quat([hole_quat[1], hole_quat[2], hole_quat[3], hole_quat[0]])
        R_approach = Rotation.from_quat([approach_quat[1], approach_quat[2], approach_quat[3], approach_quat[0]])
        R_final    = R_hole * R_approach
        q_xyzw     = R_final.as_quat()
        q2_nominal = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        nom_pose = Pose(
            position=self.hole_nom_pose.position.copy(),
            quaternion=q2_nominal
        )
        nom_pose.position[2] += cfg_app.get("pos_threshold") + self.peg_offset

        pert = cfg_app.get("pertubation")
        p2, q2 = self._sample_pose(nom_pose=nom_pose,pos_std=pert["pos_std"],angle_std=pert["angle_std"])

        hover_height = cfg_app.get("hover_height", 0.15)
        p_hover = np.array([p2[0], p2[1], p2[2] + hover_height])

        self._execute_segment(p0, q0, p_hover, q2, max_speed=cfg_app.get("speed_transit", 0.2))
        self._execute_segment(p_hover, q2, p2, q2, max_speed=cfg_app.get("speed_descent", 0.05))

        sensors = self.system.sim.get_sensor_data()
        tip = sensors[f'{self.prefix}peg_tip_pos'].copy()
        err = np.linalg.norm(tip - p2)

        self.fail_phase = self.phase
        if err < cfg_app.get("success_threshold"):
            return Phase.CONTACT
        else:
            self.fail_phase = self.phase
            return Phase.FAILED

    def _execute_segment(self, p_start, q_start, p_end, q_end, max_speed) -> None:
        self.trajectory.plan_with_speed(p_start, q_start, p_end, q_end, max_speed=max_speed)

        while not self.trajectory.is_done():
            step = self.trajectory.step(self.dt)
            target_pose = Pose(position=step["pos"], quaternion=step["quat"])
            self.system.set_target(self.device_name, {
                "x": target_pose,
                "xd": np.concatenate([step["vel"], step["omega"]])
            })
            self._tick()

    def _sample_pose(self, nom_pose: Pose, pos_std, angle_std):
        pos_std = np.array(pos_std)
        angle_std = np.deg2rad(np.array(angle_std))

        pos = nom_pose.position + np.random.normal(0, pos_std)

        angles = np.random.normal(0, angle_std)
        delta_rot = Rotation.from_euler('xyz', angles)

        nominal_quat = nom_pose.quaternion
        nominal_rot = Rotation.from_quat([nominal_quat[1], nominal_quat[2], nominal_quat[3], nominal_quat[0]])
        perturbed_rot = delta_rot * nominal_rot
        quat_xyzw = perturbed_rot.as_quat()
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat