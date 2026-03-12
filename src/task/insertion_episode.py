import time
import numpy as np
from scipy.spatial.transform import Rotation
from enum import Enum, auto
from simcore.common.pose import Pose
from task.trajectory import TrajectoryPlanner


class Phase(Enum):
    IDLE     = auto()
    APPROACH = auto()
    CONTACT = auto()
    SEARCH   = auto()
    INSERT   = auto()
    DONE     = auto()
    FAILED   = auto()

class InsertionEpisode:
    def __init__(self, system=None, config=None):
        self.system = system
        self.config = config
        self.device_name = config.get("device_name", "arm")

        self.prefix = "arm/"
        self.trajectory = TrajectoryPlanner()

        self.q_init = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
        self.peg_offset = config.get("peg_ee_offset")

        hole_cfg = self.config.get('hole_pose')
        self.hole_nom_pose = Pose(position=hole_cfg.get("pos"), quaternion=hole_cfg.get("quat"))
        self.hole_nom_pose.position[2] += hole_cfg.get("height")
        self.dt = self.system.get_timestep()

    
    def reset(self, hole_pos: np.ndarray, hole_quat: np.ndarray) -> None:
        self.system.sim.reset_device_state("arm", self.q_init)
        self.system.sim.reset_object_pose("hole", hole_pos, hole_quat)
        self.system.set_controller_mode("arm", "dynamic_impedance")
        self.phase = Phase.IDLE
        self.running = True
    
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
            
            if self.phase == Phase.FAILED or self.phase == Phase.DONE:
                self.running = False
            elif self.phase == Phase.INSERT:
                self.phase = Phase.DONE
                self.running = False

    def run_search(self) -> Phase:
        cfg      = self.config.get("episode", {}).get("search", {})
        epsilon, timeout  = cfg.get("hole_detection_threshold"), cfg.get("timeout", 15.0)
        t_start, t  = time.time(), 0.0

        # wiggle parameters from config
        wig      = cfg.get("wiggle", {})
        a, f, phi, az = np.array(wig.get("a")), np.array(wig.get("f")), np.array(wig.get("phi")), wig.get("az")

        # freeze reference at contact surface
        state    = self.system.get_state()[self.device_name]
        # compose hole orientation with approach quaternion to face downward
        hole_quat     = self.hole_nom_pose.quaternion          # wxyz
        approach_quat = np.array([0, 1, 0, 0])                # wxyz, 180° flip

        R_hole        = Rotation.from_quat([hole_quat[1], hole_quat[2], hole_quat[3], hole_quat[0]])
        R_approach    = Rotation.from_quat([approach_quat[1], approach_quat[2], approach_quat[3], approach_quat[0]])
        R_final       = R_hole * R_approach
        q_xyzw        = R_final.as_quat()
        q_down        = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # back to wxyz

        x_ref = Pose(
            position=np.array([self.hole_nom_pose.position[0],self.hole_nom_pose.position[1], self.xz0-0.02 ]),
            quaternion=q_down
        )

        while True:
            if time.time() - t_start > timeout:
                return Phase.FAILED

            state     = self.system.get_state()[self.device_name]
            x_current = self.system.ctrl[self.device_name].get_ee_pose_world(state)

            # lissajous feedforward wrench
            Fff       = a * np.sin(2 * np.pi * f * t + phi)
            Fff[2]    = -az                                  # constant downward push

            self.system.set_target(self.device_name, {
                "x":   x_ref,
                "xd":  np.zeros(6),
                "xdd": np.zeros(6),
                "Fff": Fff
            })

            # hole entrance: peg drops → z decreases
            if self.xz0 - x_current.position[2] > epsilon:
                return Phase.INSERT

            t        += self.dt
            time.sleep(self.dt)


    def make_contact(self) -> Phase:
        cfg         = self.config.get("episode", {}).get("contact", {})
        f_threshold = cfg.get("force_threshold", 3.0)
        timeout = cfg.get("timeout", 5.0)
        f_push      = cfg.get("f_push", 3.0)
        n_confirm   = cfg.get("n_confirm", 5)
        t_start     = time.time()
        n_above     = 0

        while True:
            if time.time() - t_start > timeout:
                return Phase.FAILED

            state   = self.system.get_state()[self.device_name]
            sensors = self.system.sim.get_sensor_data()

            # track current pose, pure force driven
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

            time.sleep(self.dt)

    def system_ready(self) -> Phase:
        while not self.system.sim.running:
            time.sleep(0.1)
        return Phase.APPROACH

    def run_approach(self) -> Phase:
        cfg_app = self.config.get("episode", {}).get("approach")

        state = self.system.get_state()
        current_pose = self.system.ctrl[self.device_name].get_ee_pose_world(state[self.device_name])
        p0, q0 = current_pose.position, current_pose.quaternion

        # Approach quat is relative to hole frame, compose with hole orientation
        hole_quat = self.hole_nom_pose.quaternion  # wxyz
        approach_quat = np.array(cfg_app.get("quat", [0, 1, 0, 0]))  # wxyz

        # Convert to scipy (xyzw), compose, convert back
        R_hole = Rotation.from_quat([hole_quat[1], hole_quat[2], hole_quat[3], hole_quat[0]])
        R_approach = Rotation.from_quat([approach_quat[1], approach_quat[2], approach_quat[3], approach_quat[0]])
        R_final = R_hole * R_approach
        q_xyzw = R_final.as_quat()
        q2_nominal = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # back to wxyz

        nom_pose = Pose(
            position=self.hole_nom_pose.position.copy(),
            quaternion=q2_nominal
        )
        nom_pose.position[2] += cfg_app.get("pos_threshold") + self.peg_offset

        pert = cfg_app.get("pertubation")
        p2, q2 = self._sample_pose(nom_pose=nom_pose, xy_std=pert["xy_std"], z_std=pert["z_std"], angle_std=pert["angle_std_deg"])

        hover_height = cfg_app.get("hover_height", 0.15)
        p_hover = np.array([p2[0], p2[1], p2[2] + hover_height])

        self._execute_segment(p0, q0, p_hover, q2, max_speed=cfg_app.get("speed_transit", 0.2))
        self._execute_segment(p_hover, q2, p2, q2, max_speed=cfg_app.get("speed_descent", 0.05))

        sensors = self.system.sim.get_sensor_data()
        tip = sensors[f'{self.prefix}peg_tip_pos'].copy()
        tip[2] += self.peg_offset
        err = np.linalg.norm(tip - p2)
        return Phase.CONTACT if err < cfg_app.get("success_threshold") else Phase.FAILED


    def _execute_segment(self, p_start, q_start, p_end, q_end, max_speed) -> None:
        self.trajectory.plan_with_speed(p_start, q_start, p_end, q_end, max_speed=max_speed)

        while not self.trajectory.is_done():
            step = self.trajectory.step(self.dt)
            target_pose = Pose(position=step["pos"], quaternion=step["quat"])
            self.system.set_target(self.device_name, {
                "x": target_pose,
                "xd": np.concatenate([step["vel"], step["omega"]])
            })
            time.sleep(self.dt)

    def _sample_pose(self, nom_pose: Pose, xy_std:0.0, z_std: 0.0, angle_std: 0.0):
        nominal_pos = nom_pose.position
        pos = nominal_pos + np.array([np.random.normal(0, xy_std), np.random.normal(0, xy_std), np.random.normal(0, z_std)])

        angle = np.random.normal(0, np.deg2rad(angle_std))
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        delta_rot = Rotation.from_rotvec(axis * angle)

        nominal_quat = nom_pose.quaternion
        nominal_rot = Rotation.from_quat([nominal_quat[1], nominal_quat[2], nominal_quat[3], nominal_quat[0]])
        perturbed_rot = delta_rot * nominal_rot
        quat_xyzw = perturbed_rot.as_quat()
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat