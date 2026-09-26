import math

import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()
        # Inference ODE solver. "euler" (default) = the original single-step
        # update, byte-identical to before. "dpmpp_2m" = DPM-Solver++(2M), a
        # 2nd-order linear-multistep predictor whose only job is to hold quality
        # at fewer NFE (see set_solver). Multistep state is per-instance (one
        # scheduler per stream), reset at build_inference_schedule.
        self.solver = "euler"
        self._ms_prev_x0: torch.Tensor | None = None
        self._ms_prev_h: float | None = None

    def set_solver(self, solver: str) -> None:
        """Select the inference ODE solver: 'euler' or 'dpmpp_2m'.

        For a rectified-flow path (this scheduler), DPM-Solver++(1) is provably
        identical to Euler, so 'dpmpp_2m' differs from 'euler' only by the
        2nd-order multistep extrapolation of the data prediction on interior
        steps — it reaches fine-Euler accuracy in fewer steps. Off (euler) =
        unchanged behavior; the solver only engages when a caller also passes
        the current ``timestep`` to ``step`` (the joint denoise loop does; the
        loop-scope-compiled action fast path does not, so it stays Euler and
        CUDA-graph-capturable).
        """
        if solver not in ("euler", "dpmpp_2m"):
            raise ValueError(f"`solver` must be 'euler' or 'dpmpp_2m', got {solver!r}")
        self.solver = solver

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        # Reset multistep solver state at the start of each inference schedule.
        self._ms_prev_x0 = None
        self._ms_prev_h = None
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def _lam(sigma: float) -> float:
        """Rectified-flow log-SNR lambda(sigma)=log((1-sigma)/sigma), clamped
        away from the {0,1} boundaries where it diverges (used only for the
        multistep step-size ratio, not the main update)."""
        sigma = min(max(sigma, 1e-4), 1.0 - 1e-4)
        return math.log((1.0 - sigma) / sigma)

    def step(
        self,
        model_output: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if self.solver == "euler" or timestep is None:
            # Original Euler update (x_{sigma+delta} = x + v*delta), unchanged.
            if delta.ndim == 0:
                return sample + model_output * delta
            delta = delta.view(-1, *([1] * (sample.ndim - 1)))
            return sample + model_output * delta

        # DPM-Solver++(2M), rectified-flow data-prediction form.
        # sigma_s = current noise fraction; sigma_t = sigma_s + delta (< sigma_s).
        s = float(timestep.reshape(-1)[0].item()) / float(self.num_train_timesteps)
        d = float(delta.reshape(-1)[0].item()) if delta.ndim else float(delta.item())
        t = s + d
        s = max(s, 1e-8)
        # data prediction x0_hat: x_s = x0 + s*v, v = model_output -> x0 = x_s - s*v
        x0 = sample - s * model_output
        h = self._lam(t) - self._lam(s)
        if self._ms_prev_x0 is None or self._ms_prev_h is None or t <= 1e-6:
            # 1st order on the first and final steps == Euler (D = x0_hat).
            D = x0
        else:
            r = self._ms_prev_h / h if h != 0.0 else 0.0
            D = (1.0 + 0.5 / r) * x0 - (0.5 / r) * self._ms_prev_x0 if r > 0.0 else x0
        # x_t = (t/s) x_s - (delta/s) D   (equals Euler exactly when D == x0_hat)
        x_next = (t / s) * sample - (d / s) * D
        self._ms_prev_x0 = x0
        self._ms_prev_h = h
        return x_next


class FlowMapScheduler(WanContinuousFlowMatchScheduler):
    """Inference scheduler for a two-time flow map over the SAME rectified-flow path.

    A K-step flow map takes K large jumps instead of many small Euler steps. Because
    rectified flow gives ``dx/dσ = v``, a jump and an Euler step are the *same* update
    ``x + v·δ`` — so this class reuses the parent's ``_phi`` / ``add_noise`` /
    ``build_inference_schedule`` / ``step`` unchanged. The only new thing a flow map
    needs is that each head evaluation is conditioned on BOTH the start level ``s`` and
    the jump ``Δ = t − s`` (in timestep units); ``build_flow_map_schedule`` returns that
    Δ alongside the usual ``(timesteps, deltas)``.

    Per step k of a K-step rollout (σ runs 1 -> 0):

        v = head(x, timestep=timesteps[k], timestep_delta=timestep_deltas[k])
        x = self.step(v, deltas[k], x)          # parent Euler jump

    ``num_inference_steps=1`` gives a single noise->clean jump; the FM Euler path
    velocity is recovered by querying the head at Δ=0. The map at Δ=0 is
    the identity. FlexPi uses ``FlowMapConfig.inference_nodes`` to enforce a
    shared schedule and training-strip coverage across all generated streams.
    """

    def build_flow_map_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(timesteps, deltas, timestep_deltas)`` for a K-step flow-map rollout.

        ``timesteps`` and ``deltas`` are exactly the parent inference schedule (``deltas``
        stay in σ units for the Euler ``step``). ``timestep_deltas = deltas · N`` is the
        signed jump Δ each head eval embeds (negative, since σ decreases toward clean).
        """
        timesteps, deltas = self.build_inference_schedule(
            num_inference_steps, device, torch.float32, shift_override=shift_override,
        )
        timestep_deltas = deltas * float(self.num_train_timesteps)
        return timesteps, deltas, timestep_deltas
