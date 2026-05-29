import pytest
import torch

from src.models.property_conditioning import PropertyConditioner

D_MODEL = 16
# Core tests use only torch-native embedders (no pymatgen dependency).
PROPS = ["band_gap", "space_group"]


def _make_conditioner(properties=PROPS, p_uncond: float = 0.0) -> PropertyConditioner:
    torch.manual_seed(0)
    return PropertyConditioner(
        properties=properties, d_model=D_MODEL, vz=94, p_uncond=p_uncond
    )


def _full_cond(batch: int) -> dict:
    return {
        "band_gap": torch.full((batch, 1), 2.0),
        "space_group": torch.full((batch, 1), 225.0),
    }


def _randomize(cond: PropertyConditioner) -> None:
    with torch.no_grad():
        for p in cond.null.values():
            p.copy_(torch.randn_like(p))
        for emb in cond.embedders.values():
            for q in emb.parameters():
                q.copy_(torch.randn_like(q))


def test_output_shape_and_zero_init():
    cond = _make_conditioner()
    out = cond.forward(
        _full_cond(4), batch_size=4, device=torch.device("cpu"),
        training=False, force_uncond=False,
    )
    assert out.shape == (4, D_MODEL)
    # Everything is zero-initialized so the conditioner contributes nothing until
    # fine-tuned -> guarantees an adapter starts exactly at the base model.
    assert torch.allclose(out, torch.zeros_like(out))


def test_force_uncond_and_missing_match_null_sum():
    cond = _make_conditioner()
    _randomize(cond)
    null_sum = sum(cond.null[name] for name in PROPS).expand(3, D_MODEL)

    forced = cond.forward(
        _full_cond(3), 3, torch.device("cpu"), training=False, force_uncond=True
    )
    assert torch.allclose(forced, null_sum, atol=1e-5)

    # cond=None -> all properties missing -> also the null sum.
    missing = cond.forward(None, 3, torch.device("cpu"), training=False)
    assert torch.allclose(missing, null_sum, atol=1e-5)


def test_nan_rows_route_to_null():
    cond = _make_conditioner(properties=["band_gap"])
    _randomize(cond)
    out = cond.forward(
        {"band_gap": torch.tensor([[1.0], [float("nan")]])},
        2, torch.device("cpu"), training=False,
    )
    null = cond.null["band_gap"]
    # Row 1 (NaN) must equal the null embedding; row 0 (present) must differ.
    assert torch.allclose(out[1], null, atol=1e-5)
    assert not torch.allclose(out[0], null, atol=1e-5)


def test_chemical_system_embedder():
    pytest.importorskip("pymatgen")
    cond = _make_conditioner(properties=["chemical_system"])
    _randomize(cond)
    out = cond.forward(
        {"chemical_system": ["Li-O", None]},
        2, torch.device("cpu"), training=False,
    )
    null = cond.null["chemical_system"]
    # None entry -> null; present entry -> conditional (differs from null).
    assert torch.allclose(out[1], null, atol=1e-5)
    assert not torch.allclose(out[0], null, atol=1e-5)
