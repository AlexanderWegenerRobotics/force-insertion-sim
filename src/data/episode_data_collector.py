import numpy as np
import h5py, yaml, time
from pathlib import Path

class EpisodeDataCollector:
    def __init__(self, config: dict):
        collector_cfg = config.get("data_collector", {})
        self.output_dir = Path(collector_cfg.get("output_dir", "obs/"))
        self.enabled = collector_cfg.get("enabled", True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._index = self._load_index()
        self._reset_buffers()

    def _load_index(self) -> list:
        index_path = self.output_dir / "dataset_index.yaml"
        if index_path.exists():
            with open(index_path, "r") as f:
                return yaml.safe_load(f) or []
        return []

    def _reset_buffers(self):
        self._timestamps = []
        self._f_ext = []
        self._f_internal = []
        self._ee_velocity = []
        self._Fff = []
        self._peg_tip_pos = []
        self._ee_pose = []
        self._mode = []
        self._q = []

    def start_episode(self):
        self._reset_buffers()
        self._t_start = time.time()

    def record(self,
               f_ext: np.ndarray,
               f_internal: np.ndarray,
               ee_velocity: np.ndarray,
               Fff: np.ndarray,
               peg_tip_pos: np.ndarray,
               ee_pose: np.ndarray,
               mode: int,
               q: np.ndarray,
               sim_time: float = None):
        if not self.enabled:
            return
        self._timestamps.append(sim_time if sim_time is not None else time.time())
        self._f_ext.append(f_ext.copy())
        self._f_internal.append(f_internal.copy())
        self._ee_velocity.append(ee_velocity.copy())
        self._Fff.append(Fff.copy())
        self._peg_tip_pos.append(peg_tip_pos.copy())
        self._ee_pose.append(ee_pose.copy())
        self._mode.append(mode)
        self._q.append(q.copy())

    def finish_episode(self, success: bool, hole_pos: np.ndarray, hole_quat: np.ndarray,
                       fail_phase: str = None, sim_duration: float = None):
        if not self.enabled:
            return

        n_steps = len(self._timestamps)
        if n_steps == 0:
            return

        duration = sim_duration if sim_duration is not None else (time.time() - self._t_start)

        episode_id = len(self._index)
        episode_dir = self.output_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        filepath = episode_dir / "episode.h5"
        with h5py.File(filepath, "w") as f:
            meta = f.create_group("meta")
            meta.create_dataset("success",    data=bool(success))
            meta.create_dataset("duration",   data=duration)
            meta.create_dataset("hole_pos",   data=hole_pos)
            meta.create_dataset("hole_quat",  data=hole_quat)
            meta.create_dataset("fail_phase", data=fail_phase or "")
            meta.create_dataset("n_steps",    data=n_steps)

            obs = f.create_group("obs")
            obs.create_dataset("timestamps",  data=np.array(self._timestamps))
            obs.create_dataset("f_ext",       data=np.array(self._f_ext))
            obs.create_dataset("f_internal",  data=np.array(self._f_internal))
            obs.create_dataset("ee_velocity", data=np.array(self._ee_velocity))

            act = f.create_group("action")
            act.create_dataset("Fff", data=np.array(self._Fff))

            dbg = f.create_group("debug")
            dbg.create_dataset("peg_tip_pos", data=np.array(self._peg_tip_pos))
            dbg.create_dataset("ee_pose",     data=np.array(self._ee_pose))
            dbg.create_dataset("mode",        data=np.array(self._mode))
            dbg.create_dataset("q",           data=np.array(self._q))

        entry = {
            "episode_id": episode_id,
            "success":    bool(success),
            "duration":   float(duration),
            "hole_pos":   hole_pos.tolist(),
            "hole_quat":  hole_quat.tolist(),
            "fail_phase": fail_phase or "",
            "n_steps":    n_steps,
            "path": f"episode_{episode_id:04d}/episode.h5",
        }
        self._index.append(entry)
        self._save_index()
        self._reset_buffers()

    def _save_index(self):
        with open(self.output_dir / "dataset_index.yaml", "w") as f:
            yaml.dump(self._index, f)

    @staticmethod
    def load_episode(path: str) -> dict:
        data = {}
        with h5py.File(path, "r") as f:
            for group_name in f.keys():
                group = f[group_name]
                if isinstance(group, h5py.Group):
                    data[group_name] = {k: v[()] for k, v in group.items()}
                else:
                    data[group_name] = group[()]
        return data

    @staticmethod
    def load_index(output_dir: str) -> list:
        index_path = Path(output_dir) / "dataset_index.yaml"
        if not index_path.exists():
            raise FileNotFoundError(f"No dataset index found at {index_path}")
        with open(index_path, "r") as f:
            return yaml.safe_load(f) or []