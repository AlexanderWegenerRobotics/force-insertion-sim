import time
import numpy as np
from simcore import load_yaml, Pose

#from collection.observation import ObservationBuilder
from task.insertion_episode import InsertionEpisode, Phase
#from collection.collector import DataCollector

class InsertionTask:
    def __init__(self, system=None, config=None):

        self.system = system
        self.episode = InsertionEpisode(system=system, config=config)
        self.config = config
        self.N_rep = config.get("N_task")

    def run(self) -> None:
        while not self.system.sim.running:
            time.sleep(0.05)

        hole_cfg = self.config.get('hole_pose')
        nom_hole_pose = Pose(position=hole_cfg.get("pos"), quaternion=hole_cfg.get("quat"))
        hole_xy_std, hole_z_std = hole_cfg.get("pertubation").get("xy_std"), hole_cfg.get("pertubation").get("z_std")
        hole_angle_std = hole_cfg.get("pertubation").get("angle_std_deg")

        for n in range(self.N_rep):
            hole_pos, hole_quat = self.episode._sample_pose(nom_pose=nom_hole_pose, xy_std=hole_xy_std, z_std=hole_z_std, angle_std=hole_angle_std)
            
            start_time = time.time()
            self.episode.reset(hole_pos, hole_quat)
            self.episode.run()
            duration = time.time() - start_time

            result = {
                "episode": n,
                "success": self.episode.phase == Phase.DONE,
                "duration": duration,
                "hole_pos": hole_pos.tolist(),
                "hole_quat": hole_quat.tolist(),
                "fail_phase": self.episode.phase.name if self.episode.phase == Phase.FAILED else None,
            }
            self._log_episode_result(result)

    def _log_episode_result(self, result: dict) -> None:
        status = "SUCCESS" if result["success"] else f"FAILED ({result['fail_phase']})"
        print(f"Episode {result['episode']:03d} | {status} | duration: {result['duration']:.2f}s | hole_pos: {np.round(result['hole_pos'], 3)}")



if __name__ == "__main__":
    config = load_yaml("configs/global_config.yaml")
    task_cfg = load_yaml(config.get("task_config"))
    task = InsertionTask(config=task_cfg)