import time
import yaml
import numpy as np
from simcore import load_yaml, Pose

from policy.learned_episode import LearnedEpisode, Phase
from policy.policy_interface import build_policy


class LearnedTask:
    def __init__(self, system=None, task_cfg=None, policy_cfg=None):
        self.system     = system
        self.task_cfg   = task_cfg
        self.policy_cfg = policy_cfg
        self.headless   = system.headless
        self.N_rep      = task_cfg.get("N_task")

        with open(policy_cfg["norm_stats"], "r") as f:
            norm_stats = yaml.safe_load(f)

        policy = build_policy(policy_cfg, norm_stats)
        self.episode = LearnedEpisode(system=system, config=task_cfg, policy=policy, policy_cfg=policy_cfg)

    def run(self) -> None:
        if not self.headless:
            while not self.system.sim.running:
                time.sleep(0.05)

        hole_cfg      = self.task_cfg.get('hole_pose')
        nom_hole_pose = Pose(position=hole_cfg.get("pos"), quaternion=hole_cfg.get("quat"))
        hole_pert     = hole_cfg.get("pertubation")

        task_start_time = time.time()
        success_cnt     = 0
        results         = []

        for n in range(self.N_rep):
            hole_pos, hole_quat = self.episode._sample_pose(
                nom_pose=nom_hole_pose,
                pos_std=hole_pert.get("pos_std"),
                angle_std=hole_pert.get("angle_std"),
            )

            start_time = time.time()
            self.episode.reset(hole_pos, hole_quat)
            self.episode.run()
            duration = self.episode._sim_time if self.headless else time.time() - start_time

            success    = self.episode.phase == Phase.DONE
            fail_phase = self.episode.fail_phase.name if not success else None
            depth      = self._insertion_depth()

            if success:
                success_cnt += 1

            result = {
                "episode":    n,
                "success":    success,
                "duration":   duration,
                "depth":      depth,
                "hole_pos":   hole_pos.tolist(),
                "hole_quat":  hole_quat.tolist(),
                "fail_phase": fail_phase,
            }
            results.append(result)
            self._log(result)

        elapsed = time.time() - task_start_time
        print(f"Finished {self.N_rep} in {elapsed:.1f}s. Success rate: {success_cnt / self.N_rep * 100:.1f}%")
        self.system.stop()
        return results

    def _insertion_depth(self) -> float:
        state     = self.system.get_state()[self.episode.device_name]
        x_current = self.system.ctrl[self.episode.device_name].get_ee_pose_world(state)
        return float(self.episode.hole_nom_pose.position[2] - x_current.position[2])

    def _log(self, r: dict) -> None:
        status = "SUCCESS" if r["success"] else f"FAILED ({r['fail_phase']})"
        depth  = f"{r['depth']*1000:.1f}mm"
        print(f"Episode {r['episode']+1:03d} | {status} | {r['duration']:.2f}s | depth: {depth} | hole_pos: {np.round(r['hole_pos'], 3)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    global_cfg  = load_yaml("configs/global_config.yaml")
    task_cfg    = load_yaml(global_cfg["task_config"])
    policy_cfg  = load_yaml(global_cfg["policy_config"])

    if args.weights:
        policy_cfg["weights"] = args.weights

    task = LearnedTask(task_cfg=task_cfg, policy_cfg=policy_cfg)
    task.run()
