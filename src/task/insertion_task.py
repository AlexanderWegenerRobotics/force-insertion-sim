import time
import numpy as np
from simcore import load_yaml, Pose

from task.insertion_episode import InsertionEpisode, Phase
from data.episode_data_collector import EpisodeDataCollector

class InsertionTask:
    def __init__(self, system=None, config=None):

        self.system = system
        self.collector = EpisodeDataCollector(config)
        self.episode = InsertionEpisode(system=system, config=config, collector=self.collector)
        self.config = config
        self.headless = system.headless
        self.N_rep = config.get("N_task")

    def run(self) -> None:
        if not self.headless:
            while not self.system.sim.running:
                time.sleep(0.05)

        hole_cfg = self.config.get('hole_pose')
        nom_hole_pose = Pose(position=hole_cfg.get("pos"), quaternion=hole_cfg.get("quat"))
        hole_pert = hole_cfg.get("pertubation")
        hole_pos_std = hole_pert.get("pos_std")
        hole_angle_std = hole_pert.get("angle_std")

        task_start_time = time.time()
        success_cnt = 0

        for n in range(self.N_rep):
            hole_pos, hole_quat = self.episode._sample_pose(nom_pose=nom_hole_pose, pos_std=hole_pos_std, angle_std=hole_angle_std)
            
            self.collector.start_episode()
            start_time = time.time()

            self.episode.reset(hole_pos, hole_quat)
            self.episode.run()

            wall_duration = time.time() - start_time
            sim_duration = self.episode._sim_time

            duration = sim_duration if self.headless else wall_duration

            success = self.episode.phase == Phase.DONE
            fail_phase = self.episode.fail_phase.name if self.episode.phase == Phase.FAILED else None

            if success:
                success_cnt += 1

            if (n + 1) % 50 == 0:
                print(f"[{n+1}/{self.N_rep}] Success rate: {(success_cnt / (n+1)) * 100:.1f}%")

            self.collector.finish_episode(
                success=success,
                hole_pos=hole_pos,
                hole_quat=hole_quat,
                fail_phase=fail_phase,
                sim_duration=duration,
            )

            result = {
                "episode": n,
                "success": success,
                "duration": duration,
                "hole_pos": hole_pos.tolist(),
                "hole_quat": hole_quat.tolist(),
                "fail_phase": fail_phase,
            }
            self._log_episode_result(result)
        
        print(f"Finished {self.N_rep} in {(time.time() - task_start_time):.3f}s. Success ratio is {(success_cnt / self.N_rep)*100}%")

        self.system.stop()

    def _log_episode_result(self, result: dict) -> None:
        status = "SUCCESS" if result["success"] else f"FAILED ({result['fail_phase']})"
        print(f"Episode {(result['episode']+1):03d} | {status} | duration: {result['duration']:.2f}s | hole_pos: {np.round(result['hole_pos'], 3)} | hole_quat: {np.round(result['hole_quat'], 3)}")

if __name__ == "__main__":
    config = load_yaml("configs/global_config.yaml")
    task_cfg = load_yaml(config.get("task_config"))
    task = InsertionTask(config=task_cfg)