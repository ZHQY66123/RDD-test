# RecDiff/models/d2_ddim.py
import torch as th
import math

def sample_ddim(diffusion, model, x_T, steps=None, eta=0.0, device=None):
    """
    Deterministic DDIM-like sampler that uses model(x_t, t) -> pred_xstart (x0).
    - diffusion: instance of your RecDiff.models.diffusion_process.DiffusionProcess
                 (we read alphas_cumprod, alphas_cumprod_prev, and helper sqrt arrays from it)
    - model: callable (x_t, t_tensor) -> pred_xstart (same shape as x_t)
    - x_T: initial noisy latent (batch, ...), can be q_sample(...) output or Gaussian noise
    - steps: number of diffusion steps to run (if None, uses diffusion.steps)
    - eta: 0.0 means deterministic DDIM; we keep API but implement deterministic version
    Returns: x_0 estimate (tensor with same shape as x_T)
    """
    if device is None:
        device = x_T.device
    if steps is None:
        steps = diffusion.steps
    assert steps <= diffusion.steps

    # convenience aliases (ensure on same device)
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
    sqrt_recip_alphas_cumprod = diffusion.sqrt_recip_alphas_cumprod.to(device)
    sqrt_recipm1_alphas_cumprod = diffusion.sqrt_recipm1_alphas_cumprod.to(device)

    # we will iterate t = steps-1, steps-2, ..., 0
    x = x_T
    batch = x.shape[0]
    for t_int in range(steps - 1, -1, -1):
        t = th.full((batch,), t_int, dtype=th.long, device=device)
        # model returns pred_xstart (x0) according to RecDiff convention
        pred_xstart = model(x, t)  # shape == x.shape

        # recover eps from pred_xstart using RecDiff diffusion arrays:
        # pred_xstart = sqrt_recip_alphas_cumprod[t] * x - sqrt_recipm1_alphas_cumprod[t] * eps
        # => eps = (sqrt_recip_alphas_cumprod[t] * x - pred_xstart) / sqrt_recipm1_alphas_cumprod[t]
        coef1 = _extract_scalar(sqrt_recip_alphas_cumprod, t_int, device)
        coef2 = _extract_scalar(sqrt_recipm1_alphas_cumprod, t_int, device)
        eps = (coef1 * x - pred_xstart) / (coef2 + 1e-12)

        # compute x_{t-1} following deterministic DDIM update:
        # x_{t-1} = sqrt(alpha_{t-1}) * x0 + sqrt(1 - alpha_{t-1}) * eps
        alpha_prev = _extract_scalar(alphas_cumprod_prev, t_int, device)  # scalar or batch-sized
        sqrt_alpha_prev = th.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev = th.sqrt(1.0 - alpha_prev)

        # ensure shapes broadcast properly: expand to x shape
        sqrt_alpha_prev = _expand_to_shape(sqrt_alpha_prev, x.shape)
        sqrt_one_minus_alpha_prev = _expand_to_shape(sqrt_one_minus_alpha_prev, x.shape)

        x_prev = sqrt_alpha_prev * pred_xstart + sqrt_one_minus_alpha_prev * eps

        # optional: if eta > 0 we would add stochastic term; for MVP we keep eta=0 deterministic
        x = x_prev

    # after loop x is x_0 estimate
    return x

def _extract_scalar(arr_tensor, t_int, device):
    """
    arr_tensor is 1-D tensor of length diffusion.steps on cpu or device.
    Return a tensor shaped (batch, 1, 1,...) or a scalar that broadcasts in arithmetic.
    We'll return a 0-dim tensor (scalar) which will broadcast when multiplied.
    """
    # arr_tensor might be on CPU; move to device
    v = arr_tensor.to(device)
    # index
    scalar = v[t_int]
    return scalar

def _expand_to_shape(scalar, shape):
    """
    Turn scalar (0-dim or 1-dim length batch) into a tensor that broadcasts to 'shape'.
    If scalar is 0-dim, just return scalar (it will broadcast); if 1-dim length=B, reshape to (B,1,1,...)
    """
    if isinstance(scalar, th.Tensor) and scalar.dim() == 0:
        return scalar
    if isinstance(scalar, th.Tensor) and scalar.dim() == 1:
        b = scalar.shape[0]
        # produce shape (B,1,1,...) matching x shape dims
        expand_shape = [b] + [1] * (len(shape) - 1)
        return scalar.view(*expand_shape)
    # fallback: convert to 0-dim tensor
    return th.tensor(scalar).to(next((scalar for scalar in ()), 'cpu'))
