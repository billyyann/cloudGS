#!/usr/bin/env python3
"""Conditional DiT: noise is added to the clear DN image Y.

X, a target-prior heatmap, the camera, and the scattering medium are
conditions. The heatmap is a Gaussian at the target centroid on the same
patch grid as Y. It is not a SAR image. Training loss is L1 on the noise.
Inference is deterministic DDIM. See docs/plan_sar_ir_from_sar_contour.md.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn


def sincos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("token dim must be divisible by 4")
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    n = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(n, device=device, dtype=torch.float32) / n))
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((y.sin(), y.cos(), x.sin(), x.cos()), dim=1)


class Schedule:
    def __init__(self, steps: int = 100):
        betas = torch.linspace(1e-4, 2e-2, steps)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        self.steps = steps
        self.sqrt_ab = ab.sqrt()
        self.sqrt_one_minus_ab = (1.0 - ab).sqrt()

    def to(self, device) -> "Schedule":
        self.sqrt_ab = self.sqrt_ab.to(device)
        self.sqrt_one_minus_ab = self.sqrt_one_minus_ab.to(device)
        return self

    def q_sample(self, y: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_ab[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return a * y + b * eps


def timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    ang = t.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat((ang.sin(), ang.cos()), dim=1)


def modulate(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, h: torch.Tensor, kv: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift_s, scale_s, gate_s, shift_c, scale_c, gate_c, shift_f, scale_f, gate_f = (
            self.ada(t_emb).chunk(9, dim=1)
        )
        s = modulate(self.norm1(h), shift_s, scale_s)
        hs, _ = self.self_attn(s, s, s, need_weights=False)
        h = h + gate_s.unsqueeze(1) * hs
        c = modulate(self.norm2(h), shift_c, scale_c)
        hc, _ = self.cross_attn(c, kv, kv, need_weights=False)
        h = h + gate_c.unsqueeze(1) * hc
        f = modulate(self.norm3(h), shift_f, scale_f)
        h = h + gate_f.unsqueeze(1) * self.ffn(f)
        return h


class SarIrDiT(nn.Module):
    """ε-prediction DiT. Query tokens come from noisy Y. Conditions are K/V."""

    def __init__(
        self,
        image: int = 64,
        patch: int = 16,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        cloud_e: int = 4,
        cloud_d: int = 2,
        atm_m: int = 8,
        atm_k: int = 2,
        cam_dim: int = 13,
    ):
        super().__init__()
        if image % patch != 0:
            raise ValueError("image size must be divisible by patch")
        self.image = image
        self.patch = patch
        self.dim = dim
        self.grid = image // patch
        self.n = self.grid * self.grid
        self.cloud_e = cloud_e
        self.cloud_d = cloud_d
        self.atm_m = atm_m
        self.atm_k = atm_k
        self.z_proj = nn.Linear(patch * patch, dim)
        self.x_proj = nn.Linear(patch * patch, dim)
        self.prior_proj = nn.Linear(patch * patch, dim)
        self.cam_proj = nn.Linear(cam_dim, 3 * dim)
        self.cloud_proj = nn.Linear(cloud_e * cloud_d, dim)
        self.atm_proj = nn.Linear(atm_m, atm_k * dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(DiTBlock(dim, heads) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out_proj = nn.Linear(dim, patch * patch)
        nn.init.zeros_(self.out_ada[-1].weight)
        nn.init.zeros_(self.out_ada[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        p = self.patch
        b, _, h, w = img.shape
        g = h // p
        x = img.reshape(b, 1, g, p, g, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, g * g, p * p)
        return x

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        p = self.patch
        g = self.grid
        b = tokens.shape[0]
        x = tokens.reshape(b, g, g, p, p)
        x = x.permute(0, 3, 1, 4, 2).reshape(b, 1, g * p, g * p)
        return x

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        prior: torch.Tensor,
        camera: torch.Tensor,
        cloud: torch.Tensor,
        atmosphere: torch.Tensor,
    ) -> torch.Tensor:
        pos = sincos_2d(self.grid, self.grid, self.dim, z_t.device)
        q = self.z_proj(self.patchify(z_t)) + pos
        x_tok = self.x_proj(self.patchify(x)) + pos
        prior_tok = self.prior_proj(self.patchify(prior)) + pos
        cam_tok = self.cam_proj(camera).reshape(camera.shape[0], 3, self.dim)
        cloud_tok = self.cloud_proj(cloud.reshape(cloud.shape[0], self.n, -1)) + pos
        atm_tok = self.atm_proj(atmosphere).reshape(atmosphere.shape[0], self.atm_k, self.dim)
        kv = torch.cat((x_tok, prior_tok, cam_tok, cloud_tok, atm_tok), dim=1)
        t_emb = self.t_mlp(timestep_embed(t, self.dim))
        h = q
        for block in self.blocks:
            h = block(h, kv, t_emb)
        shift, scale = self.out_ada(t_emb).chunk(2, dim=1)
        h = modulate(self.out_norm(h), shift, scale)
        return self.unpatchify(self.out_proj(h))


def target_prior_heatmap(
    image: int,
    row: float,
    col: float,
    sigma_row: float,
    sigma_col: float,
) -> torch.Tensor:
    """Gaussian on the image grid. Peak is 1 at the centroid. Not a SAR image."""
    yy = torch.arange(image, dtype=torch.float32)
    xx = torch.arange(image, dtype=torch.float32)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    heat = torch.exp(-0.5 * (((gy - row) / sigma_row) ** 2 + ((gx - col) / sigma_col) ** 2))
    return heat.reshape(1, 1, image, image)


@dataclass
class Pair:
    x: torch.Tensor
    y: torch.Tensor
    prior: torch.Tensor
    camera: torch.Tensor
    cloud: torch.Tensor
    atmosphere: torch.Tensor
    dn_std: float
    center_rc: tuple[float, float]
    sigma_rc: tuple[float, float]
    world_km: tuple[float, float, float]


def make_pair(image: int = 64, patch: int = 16, device="cpu") -> Pair:
    """Two synthetic DN images: Y clear, X the same scene with a cloud veil."""
    g = image // patch
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, image),
        torch.linspace(-1, 1, image),
        indexing="ij",
    )
    target = ((yy - 0.15).abs() < 0.22) & ((xx + 0.05).abs() < 0.18)
    sea = 0.35 + 0.05 * torch.sin(6 * xx) * torch.cos(4 * yy)
    radiance_y = torch.where(target, torch.full_like(sea, 1.6), sea)
    cloud_mask = torch.exp(-((yy + 0.1) ** 2 + (xx - 0.2) ** 2) / 0.35)
    radiance_x = radiance_y * (0.45 + 0.55 * (1 - cloud_mask)) + 0.25 * cloud_mask
    stack = torch.cat((radiance_x.reshape(-1), radiance_y.reshape(-1)))
    offset = 0.0
    scale = float(torch.quantile(stack, 0.995).clamp_min(1e-6))
    dn_x = (radiance_x - offset) / scale
    dn_y = (radiance_y - offset) / scale
    dn_std = float(dn_y.std().clamp_min(1e-6))
    x = (dn_x / dn_std).reshape(1, 1, image, image)
    y = (dn_y / dn_std).reshape(1, 1, image, image)

    gsd = 4.0
    ys, xs = torch.where(target)
    row = float(ys.float().mean())
    col = float(xs.float().mean())
    sigma_row = float(ys.float().std().clamp_min(1.0))
    sigma_col = float(xs.float().std().clamp_min(1.0))
    world_km = (col * gsd / 1000.0, row * gsd / 1000.0, 0.0)
    prior = target_prior_heatmap(image, row, col, sigma_row, sigma_col)
    fx = fy = 0.0096
    cx = cy = (image - 1) / 2
    cam_world = (13275.0, 19725.0, 3022.0)
    pose6 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    camera = torch.tensor(
        [[fx, fy, cx, cy, *cam_world, *pose6]],
        dtype=torch.float32,
    )
    # Per-patch cloud profile (E, D') = (ka, ks) over 4 layers. Atmosphere is 1D.
    cloud = torch.zeros(1, g, g, 4, 2)
    for pr in range(g):
        for pc in range(g):
            r0, c0 = pr * patch, pc * patch
            veil = float(cloud_mask[r0:r0 + patch, c0:c0 + patch].mean())
            cloud[0, pr, pc, :, 0] = 0.02 * veil
            cloud[0, pr, pc, :, 1] = 0.15 * veil
    atmosphere = torch.linspace(0.04, 0.001, 8).reshape(1, 8)
    return Pair(
        x=x.to(device),
        y=y.to(device),
        prior=prior.to(device),
        camera=camera.to(device),
        cloud=cloud.to(device),
        atmosphere=atmosphere.to(device),
        dn_std=dn_std,
        center_rc=(row, col),
        sigma_rc=(sigma_row, sigma_col),
        world_km=world_km,
    )


@torch.no_grad()
def ddim_sample(model: SarIrDiT, pair: Pair, schedule: Schedule, steps: int) -> torch.Tensor:
    device = pair.y.device
    ts = torch.linspace(schedule.steps - 1, 0, steps, device=device).long()
    z = torch.randn_like(pair.y)
    for i, t_cur in enumerate(ts):
        t = t_cur.expand(z.shape[0])
        eps = model(z, t, pair.x, pair.prior, pair.camera, pair.cloud, pair.atmosphere)
        y0 = (z - schedule.sqrt_one_minus_ab[t_cur] * eps) / schedule.sqrt_ab[t_cur].clamp_min(1e-8)
        if i + 1 == len(ts):
            z = y0
        else:
            t_prev = ts[i + 1]
            z = schedule.sqrt_ab[t_prev] * y0 + schedule.sqrt_one_minus_ab[t_prev] * eps
    return z


def save_gray(path: str, img: torch.Tensor) -> None:
    x = img.detach().float().cpu().reshape(img.shape[-2], img.shape[-1])
    lo = float(x.min())
    hi = float(x.max())
    u8 = ((x - lo) / max(hi - lo, 1e-8) * 255).clamp(0, 255).to(torch.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(u8.numpy(), mode="L").save(path)
    except ImportError:
        with open(path.replace(".png", ".pgm"), "wb") as f:
            h, w = u8.shape
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write(u8.numpy().tobytes())


def train_step(model, opt, schedule, pair) -> float:
    b = 4
    y = pair.y.expand(b, -1, -1, -1)
    x = pair.x.expand(b, -1, -1, -1)
    t = torch.randint(0, schedule.steps, (b,), device=y.device)
    eps = torch.randn_like(y)
    z = schedule.q_sample(y, t, eps)
    prior = pair.prior.expand(b, -1, -1, -1)
    cam = pair.camera.expand(b, -1)
    cloud = pair.cloud.expand(b, -1, -1, -1, -1)
    atm = pair.atmosphere.expand(b, -1)
    pred = model(z, t, x, prior, cam, cloud, atm)
    loss = (pred - eps).abs().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image, patch = 64, 16
    pair = make_pair(image, patch, device)
    model = SarIrDiT(image=image, patch=patch, dim=64, depth=2, heads=4).to(device)
    schedule = Schedule(100).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n = model.n
    n_prior = n
    n_cam = 3
    n_med = n + model.atm_k
    losses = []
    for step in range(40):
        losses.append(train_step(model, opt, schedule, pair))
    model.eval()
    y_hat = ddim_sample(model, pair, schedule, steps=8)
    y_hat_dn = y_hat * pair.dn_std
    out = os.path.join(os.path.dirname(__file__), "..", "scenes", "fire_gs", "dit_smoke")
    out = os.path.abspath(out)
    save_gray(os.path.join(out, "X_cloudy_dn.png"), pair.x * pair.dn_std)
    save_gray(os.path.join(out, "Y_clear_dn.png"), pair.y * pair.dn_std)
    save_gray(os.path.join(out, "prior_heatmap.png"), pair.prior)
    stale = os.path.join(out, "sar_9x9.png")
    if os.path.exists(stale):
        os.remove(stale)
    save_gray(os.path.join(out, "Yhat_ddim_dn.png"), y_hat_dn)
    finite = bool(torch.isfinite(y_hat).all() and math.isfinite(losses[-1]))
    print(
        f"device={device}  N={n}  N_prior={n_prior}  N_cam={n_cam}  N_med={n_med}  "
        f"kv={n + n_prior + n_cam + n_med}"
    )
    heat = pair.prior[0, 0]
    patch_mean = heat.reshape(image // patch, patch, image // patch, patch).mean(dim=(1, 3))
    print(
        f"prior center row,col=({pair.center_rc[0]:.2f}, {pair.center_rc[1]:.2f})  "
        f"sigma=({pair.sigma_rc[0]:.2f}, {pair.sigma_rc[1]:.2f}) px  "
        f"world_km=({pair.world_km[0]:.3f}, {pair.world_km[1]:.3f}, {pair.world_km[2]:.3f})"
    )
    print(f"prior min={float(heat.min()):.4f}  max={float(heat.max()):.4f}")
    print("prior patch means:")
    for row in patch_mean.tolist():
        print(" ".join(f"{v:7.4f}" for v in row))
    print(f"loss first={losses[0]:.4f}  last={losses[-1]:.4f}  finite={finite}")
    print(f"Yhat dn mean={float(y_hat_dn.mean()):.4f}  Y dn mean={float((pair.y * pair.dn_std).mean()):.4f}")
    print(f"wrote {out}")
    if not finite or float(heat.max()) <= float(heat.min()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
