"""Property conditioning for classifier-free guidance (CFG).

Mirrors the design of MatterGen's ``mattergen/property_embeddings.py``: each
conditioning property owns a *conditional* embedder plus a learned *null*
(unconditional) vector. For every structure we select either embedding via a
per-field boolean mask. The summed result is a single ``(B, d_model)`` vector
that ``CrystaliteModel`` adds to its time embedding ``t_emb``.

Everything is zero-initialized so that, before any fine-tuning, the conditioner
contributes exactly ``0`` and the model reproduces the unconditional base.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Canonical property name -> spec.
#   kind: "scalar" | "space_group" | "chemical_system"
#   log10: for scalars, standardize log10(value) instead of value (matches
#          MatterGen's StandardScalerTorch(log10_transform=True) for moduli).
SUPPORTED_PROPERTIES: dict[str, dict] = {
    "band_gap": {"kind": "scalar", "log10": False},
    "dft_band_gap": {"kind": "scalar", "log10": False},
    "energy_above_hull": {"kind": "scalar", "log10": False},
    "formation_energy_per_atom": {"kind": "scalar", "log10": False},
    "dft_mag_density": {"kind": "scalar", "log10": False},
    "hhi_score": {"kind": "scalar", "log10": False},
    "dft_bulk_modulus": {"kind": "scalar", "log10": True},
    "ml_bulk_modulus": {"kind": "scalar", "log10": True},
    "dft_shear_modulus": {"kind": "scalar", "log10": True},
    "space_group": {"kind": "space_group"},
    "chemical_system": {"kind": "chemical_system"},
}

_EPS = 1e-8


def scalar_property_names(properties: list[str]) -> list[str]:
    """Subset of ``properties`` whose values are continuous scalars."""
    return [p for p in properties if SUPPORTED_PROPERTIES[p]["kind"] == "scalar"]


class _ScalarEmbedding(nn.Module):
    """Standardize -> sinusoidal Fourier features -> MLP (final layer zero-init).

    Standardization stats live in buffers so they persist in ``state_dict`` and
    are reused deterministically at sampling time.
    """

    def __init__(self, d_model: int, log10: bool, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.log10 = log10
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("std", torch.ones(()))
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        # Zero-init output so the property contributes 0 before fine-tuning.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _fourier(self, x: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=x.device)
            / half
        )
        args = 2 * math.pi * x[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.float().view(-1)
        if self.log10:
            x = torch.log10(x.clamp_min(_EPS))
        x = (x - self.mean) / self.std.clamp_min(_EPS)
        # NaN/missing rows are masked out by the parent; sanitize so the encode
        # never propagates NaN gradients.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mlp(self._fourier(x))


class _SpaceGroupEmbedding(nn.Module):
    """Learned embedding over the 230 space groups (1-indexed input)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(230, d_model)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        idx = torch.nan_to_num(value.float().view(-1), nan=1.0).round().long()
        idx = idx.clamp(1, 230) - 1
        return self.embedding(idx)


class _ChemicalSystemEmbedding(nn.Module):
    """Multi-hot over atomic numbers -> linear projection (zero-init)."""

    def __init__(self, d_model: int, vz: int) -> None:
        super().__init__()
        self.vz = vz
        self.linear = nn.Linear(vz + 1, d_model, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, value: list, device: torch.device) -> torch.Tensor:
        from pymatgen.core.periodic_table import Element

        multi_hot = torch.zeros(len(value), self.vz + 1, device=device)
        for i, sys in enumerate(value):
            if sys is None:
                continue  # masked to null by parent
            symbols = sys.split("-") if isinstance(sys, str) else list(sys)
            for sym in symbols:
                z = Element(sym).Z
                if 0 <= z <= self.vz:
                    multi_hot[i, z] = 1.0
        return self.linear(multi_hot)


class PropertyConditioner(nn.Module):
    """Sum of per-property embeddings, with CFG conditioning dropout.

    Returns a ``(B, d_model)`` tensor to be added to the time embedding. When no
    properties are configured the model should skip instantiating this module.
    """

    def __init__(
        self,
        properties: list[str],
        d_model: int,
        vz: int,
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__()
        unknown = [p for p in properties if p not in SUPPORTED_PROPERTIES]
        if unknown:
            raise ValueError(
                f"Unsupported conditioning properties {unknown}. "
                f"Supported: {sorted(SUPPORTED_PROPERTIES)}"
            )
        self.properties = list(properties)
        self.p_uncond = float(p_uncond)
        self.d_model = d_model

        self.embedders = nn.ModuleDict()
        # One learned null (unconditional) vector per property, zero-init.
        self.null = nn.ParameterDict()
        for name in self.properties:
            kind = SUPPORTED_PROPERTIES[name]["kind"]
            if kind == "scalar":
                self.embedders[name] = _ScalarEmbedding(
                    d_model, log10=SUPPORTED_PROPERTIES[name]["log10"]
                )
            elif kind == "space_group":
                self.embedders[name] = _SpaceGroupEmbedding(d_model)
            elif kind == "chemical_system":
                self.embedders[name] = _ChemicalSystemEmbedding(d_model, vz=vz)
            self.null[name] = nn.Parameter(torch.zeros(d_model))

    def fit_stats(self, stats: dict[str, tuple[float, float]]) -> None:
        """Load precomputed (mean, std) for scalar properties into buffers."""
        for name, (mean, std) in stats.items():
            emb = self.embedders[name] if name in self.embedders else None
            if isinstance(emb, _ScalarEmbedding):
                emb.mean.fill_(float(mean))
                emb.std.fill_(float(std))

    def _present_mask(self, value, batch_size: int, device: torch.device) -> torch.Tensor:
        """(B, 1) bool: True where a usable (non-missing) label exists."""
        if value is None:
            return torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        if isinstance(value, torch.Tensor):
            present = ~torch.isnan(value.float().view(batch_size, -1)).any(dim=1)
            return present.view(batch_size, 1)
        # list-like (chemical_system)
        present = torch.tensor(
            [v is not None for v in value], dtype=torch.bool, device=device
        )
        return present.view(batch_size, 1)

    def forward(
        self,
        cond: dict | None,
        batch_size: int,
        device: torch.device,
        *,
        training: bool,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        total = torch.zeros(batch_size, self.d_model, device=device)
        cond = cond or {}
        for name in self.properties:
            value = cond.get(name)
            present = self._present_mask(value, batch_size, device)

            # Decide which rows use the unconditional (null) embedding.
            use_uncond = ~present
            if force_uncond:
                use_uncond = torch.ones_like(use_uncond)
            elif training and self.p_uncond > 0.0:
                drop = torch.rand(batch_size, 1, device=device) < self.p_uncond
                use_uncond = use_uncond | (present & drop)

            null_emb = self.null[name].expand(batch_size, self.d_model)
            if bool(use_uncond.all()):
                total = total + null_emb
                continue

            embedder = self.embedders[name]
            if isinstance(embedder, _ChemicalSystemEmbedding):
                cond_emb = embedder(value, device)
            else:
                cond_emb = embedder(value)
            total = total + torch.where(use_uncond, null_emb, cond_emb)
        return total
