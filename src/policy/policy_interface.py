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
            raise ImportError("force-insertion-policy not installed.")

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

    def _to_tensor(self, x):
        return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

    def predict(self, o_prev, o_curr):
        o_prev_n = (o_prev - self.obs_mean) / self.obs_std
        o_curr_n = (o_curr - self.obs_mean) / self.obs_std
        o_prev_t = self._to_tensor(o_prev_n)
        o_curr_t = self._to_tensor(o_curr_n)
        with torch.no_grad():
            a = torch.randn(1, 6, device=self.device)
            for tau in reversed(range(self.schedule.T)):
                tau_t = torch.tensor([tau], device=self.device)
                a = self.schedule.p_sample(self.model, o_prev_t, o_curr_t, a, tau_t)
        return a.squeeze(0).cpu().numpy() * self.act_std + self.act_mean


class DDPMPolicyONNX(PolicyInterface):
    """Fast CPU inference using ONNX Runtime — drop-in replacement for DDPMPolicy."""

    def __init__(self, cfg):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2          # tune to your CPU
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            cfg["onnx_weights"],
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        T = cfg.get("diffusion_horizon", 50)
        beta_start = cfg.get("beta_start", 1e-4)
        beta_end   = cfg.get("beta_end",   1e-2)

        # Precompute schedule on CPU as numpy (no torch needed at inference)
        betas     = np.linspace(beta_start, beta_end, T, dtype=np.float32)
        alphas    = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        self.alphas    = alphas
        self.alpha_bar = alpha_bar
        self.sigmas    = np.sqrt(betas)
        self.T         = T

        self.obs_mean = np.array(cfg["obs_mean"], dtype=np.float32)
        self.obs_std  = np.array(cfg["obs_std"],  dtype=np.float32) + 1e-6
        self.act_mean = np.array(cfg["action_mean"], dtype=np.float32)
        self.act_std  = np.array(cfg["action_std"],  dtype=np.float32) + 1e-6

    def _p_sample(self, o_prev, o_curr, a_tau, tau_idx):
        """Single DDPM reverse step, fully in numpy via ONNX."""
        tau = np.array([tau_idx], dtype=np.int64)
        eps_hat = self.sess.run(
            ["eps_hat"],
            {"obs_prev": o_prev, "obs_curr": o_curr, "action_noisy": a_tau, "tau": tau}
        )[0]

        alpha     = self.alphas[tau_idx]
        alpha_bar = self.alpha_bar[tau_idx]
        sigma     = self.sigmas[tau_idx]

        mean = (1.0 / np.sqrt(alpha)) * (
            a_tau - (1 - alpha) / np.sqrt(1 - alpha_bar) * eps_hat
        )
        if tau_idx > 0:
            return mean + sigma * np.random.randn(*a_tau.shape).astype(np.float32)
        return mean

    def predict(self, o_prev, o_curr):
        o_prev_n = ((o_prev - self.obs_mean) / self.obs_std)[None]  # (1, 18)
        o_curr_n = ((o_curr - self.obs_mean) / self.obs_std)[None]

        a = np.random.randn(1, 6).astype(np.float32)
        for tau in reversed(range(self.T)):
            a = self._p_sample(o_prev_n, o_curr_n, a, tau)

        return a.squeeze(0) * self.act_std + self.act_mean


def build_policy(policy_cfg: dict, norm_stats: dict) -> PolicyInterface:
    policy_cfg = {**policy_cfg, **norm_stats}
    kind   = policy_cfg.get("policy", "ddpm")
    use_onnx = policy_cfg.get("use_onnx", False)

    if kind == "ddpm":
        if use_onnx:
            return DDPMPolicyONNX(policy_cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DDPMPolicy(policy_cfg, device=device)

    raise ValueError(f"Unknown policy type: {kind}")