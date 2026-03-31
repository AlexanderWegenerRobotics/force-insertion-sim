import torch
import numpy as np
from abc import ABC, abstractmethod


class PolicyInterface(ABC):
    @abstractmethod
    def predict(self, o_prev: np.ndarray, o_curr: np.ndarray) -> np.ndarray:
        pass


class DDPMPolicy(PolicyInterface):
    def __init__(self, cfg, device="cpu"):
        try:
            from diffusion.ddpm import NoiseEstimator, DDPMSchedule
        except ImportError:
            raise ImportError("force-insertion-policy not installed. Run: pip install -e /path/to/force-insertion-policy")
        
        self.device = torch.device(device)
        T = cfg.get("diffusion_horizon", 50)
        self.model = NoiseEstimator(hidden_dim=cfg.get("hidden_dim", 512)).to(self.device)
        self.model.load_state_dict(torch.load(cfg["weights"], map_location=self.device))
        self.model.eval()
        self.schedule = DDPMSchedule(T=T).to(self.device)

        self.obs_mean = np.array(cfg["obs_mean"], dtype=np.float32)
        self.obs_std  = np.array(cfg["obs_std"],  dtype=np.float32) + 1e-6
        self.act_mean = np.array(cfg["action_mean"], dtype=np.float32)
        self.act_std  = np.array(cfg["action_std"],  dtype=np.float32) + 1e-6

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

    def predict(self, o_prev: np.ndarray, o_curr: np.ndarray) -> np.ndarray:
        o_prev_n = (o_prev - self.obs_mean) / self.obs_std
        o_curr_n = (o_curr - self.obs_mean) / self.obs_std

        o_prev_t = self._to_tensor(o_prev_n)
        o_curr_t = self._to_tensor(o_curr_n)

        with torch.no_grad():
            a = torch.randn(1, 6, device=self.device)
            for tau in reversed(range(self.schedule.T)):
                tau_t = torch.tensor([tau], device=self.device)
                a = self.schedule.p_sample(self.model, o_prev_t, o_curr_t, a, tau_t)

        action_n = a.squeeze(0).cpu().numpy()
        return action_n * self.act_std + self.act_mean


def build_policy(policy_cfg: dict, norm_stats: dict) -> PolicyInterface:
    policy_cfg = {**policy_cfg, **norm_stats}
    kind = policy_cfg.get("policy", "ddpm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "ddpm":
        return DDPMPolicy(policy_cfg, device=device)
    raise ValueError(f"Unknown policy type: {kind}")
