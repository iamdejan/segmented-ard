"""Unit/smoke tests for explain_occlusion numerical invariants.

Run in the project environment with: pixi run python -m pytest test_explain_occlusion.py -q
The tests use a synthetic segmentation network; no BDD100k images are required.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import explain_occlusion as oc


class TinySegmentation(nn.Module):
    """Road-logit output has known dependence on RGB and spatial position."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        weight = torch.ones((h, w), device=x.device, dtype=x.dtype)
        weight[:h // 2, :w // 2] = 4
        positive = (x[:, 0] * weight) + 0.2 * x[:, 1] - x[:, 2]
        negative = torch.zeros_like(positive)
        return torch.stack([positive, negative], dim=1)


@pytest.fixture
def sample() -> tuple[TinySegmentation, torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    model = TinySegmentation().eval()
    image = torch.rand((1, 3, 8, 8))
    gt = torch.zeros((8, 8), dtype=torch.long)
    gt[:2] = 1
    gt[0, 0] = 19
    return model, image, gt


def test_global_logit_matches_single_image_target(sample):
    model, image, _ = sample
    logits = model(image)
    assert torch.allclose(oc.global_mean_logits(logits, 0)[0], logits[0, 0].mean())
    assert oc.make_score_forward(model, 0)(image).shape == (1,)
    assert torch.allclose(oc.make_score_forward(model, 0)(image), logits[:, 0].mean((1, 2)))


def test_gt_score_fixed_mask_and_ignore(sample):
    model, image, gt = sample
    logits = model(image)
    gt_score = oc.gt_region_logits(logits, 0, gt)
    assert torch.allclose(gt_score, logits[0, 0][gt == 0].mean().unsqueeze(0))
    with pytest.raises(ValueError):
        oc.gt_region_logits(logits, 0, torch.ones_like(gt) * 19)


def test_iou_ignore_and_signed_effects():
    gt = torch.tensor([[0, 0, 19], [1, 0, 1]])
    pred = torch.tensor([[0, 1, 0], [0, 0, 1]])
    # TP=2 at (0,0),(1,1); FP=1 at (1,0); FN=1 at (0,1).
    assert oc.target_iou(pred, gt, 0).item() == pytest.approx(0.5)


def test_rgb_tile_and_partial_area(sample):
    _, image, gt = sample
    cam = np.linspace(0, 1, num=64, dtype=np.float32).reshape(8, 8)
    tiles = oc.build_tiles(cam, gt, 0, (3, 3), 0.8)
    assert len(tiles) == 9
    assert sum(t.area for t in tiles) == 64
    assert tiles[-1].area == 4  # Bottom-right partial tile.
    replacement = torch.full_like(image, 0.5)
    orig_gt = gt.clone()
    masked = oc.apply_tiles(image, replacement, tiles, [0, 8])
    assert torch.equal(gt, orig_gt)
    assert torch.equal(masked[..., :3, :3], replacement[..., :3, :3])
    assert torch.equal(masked[..., 6:8, 6:8], replacement[..., 6:8, 6:8])
    assert torch.equal(masked[..., 3:6, 3:6], image[..., 3:6, 3:6])


def test_independent_effect_matches_manual_forward(sample):
    model, image, gt = sample
    baseline = torch.full_like(image, 0.5)
    cam = np.ones((8, 8), dtype=np.float32)
    tiles = oc.build_tiles(cam, gt, 0, (4, 4), 0.8)
    original = oc.infer_metrics(model, image, gt, 0)
    frame = oc.tile_effects(model, image, baseline, gt, 0, tiles, batch_size=2, original=original)
    assert len(frame) == 4
    for tile in tiles:
        modified = oc.apply_tiles(image, baseline, tiles, [tile.index])
        manual = (model(image)[0, 0].mean() - model(modified)[0, 0].mean()).item()
        assert frame.loc[tile.index, "delta_global_logit"] == pytest.approx(manual, abs=1e-6)
    # Values may be positive or negative; they are not clipped to [0,1].
    assert np.isfinite(frame["delta_global_logit"]).all()


def test_deletion_determinism_and_coverage(sample):
    model, image, gt = sample
    cam = np.arange(64, dtype=np.float32).reshape(8, 8) / 63
    tiles = oc.build_tiles(cam, gt, 0, (3, 3), 0.8)
    assert oc.deletion_order(tiles, "random", 17) == oc.deletion_order(tiles, "random", 17)
    assert oc.deletion_order(tiles, "matched_random", 17) == oc.deletion_order(tiles, "matched_random", 17)
    for policy in ("top", "bottom", "random", "matched_random"):
        order = oc.deletion_order(tiles, policy, 17)
        assert sorted(order) == list(range(len(tiles)))
        assert oc.selected_at_budget(tiles, order, 0, 64) == []
        # Whole tiles cover a budget to within one tile's area.
        chosen = oc.selected_at_budget(tiles, order, 0.2, 64)
        area = sum(tiles[i].area for i in chosen)
        assert abs(area / 64 - 0.2) <= max(t.area for t in tiles) / 64
        original = oc.infer_metrics(model, image, gt, 0)
        data, _ = oc.ordered_deletion(model, image, torch.full_like(image, 0.5), gt, 0,
                                      tiles, order, policy, 0, (0.0, 0.2, 0.3), original, 2)
        assert data[0]["occluded_pixels"] == 0
        assert data[0]["delta_global_logit"] == pytest.approx(0.0, abs=1e-6)
        assert all(0 <= r["actual_fraction"] <= 1 for r in data)


def test_numeric_cam_and_saved_robust_tensor(tmp_path: Path, sample):
    _, image, _ = sample
    cam = np.ones((8, 8), dtype=np.float32) * 0.4
    np.save(tmp_path / "cam.npy", cam)
    adversarial = (image + 0.03).clamp(0, 1)
    np.save(tmp_path / "pgd.npy", adversarial.numpy())
    row = {"heatmap_npy_path": "cam.npy", "input_tensor_path": "pgd.npy",
           "input_sha256": oc.tensor_hash(adversarial)}
    assert np.array_equal(oc.load_cam_data(row, tmp_path, (8, 8)), cam)
    assert torch.equal(oc.load_reference_input(row, image, "robust", tmp_path), adversarial)
    with pytest.raises(ValueError, match="Regenerate"):
        oc.load_reference_input({}, image, "robust", tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        oc.load_reference_input({**row, "input_sha256": "bad"}, image, "robust", tmp_path)
    with pytest.raises(ValueError, match="heatmap_npy_path"):
        oc.load_cam_data({}, tmp_path, (8, 8))


def test_manifest_upsert(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    rows = [dict(image_path="a", model_name="m", value=1)]
    oc.save_csv_upsert(path, rows, ("image_path", "model_name"))
    oc.save_csv_upsert(path, [dict(image_path="a", model_name="m", value=2)], ("image_path", "model_name"))
    saved = pd.read_csv(path)
    assert len(saved) == 1
    assert saved.iloc[0]["value"] == 2


def test_gray_blur_and_domain(sample):
    _, image, _ = sample
    assert torch.allclose(oc.replacement_baseline(image, "gray", 5, 1), torch.full_like(image, 0.5))
    blur = oc.replacement_baseline(image, "blur", 5, 1)
    assert blur.shape == image.shape
    assert 0 <= float(blur.min()) <= float(blur.max()) <= 1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        oc.replacement_baseline(image * 3, "gray", 5, 1)


def test_spearman_and_auc(sample):
    model, image, gt = sample
    original = oc.infer_metrics(model, image, gt, 0)
    cam = np.random.default_rng(12).random((8, 8)).astype(np.float32)
    tiles = oc.build_tiles(cam, gt, 0, (4, 4), 0.8)
    patches = oc.tile_effects(model, image, torch.full_like(image, .5), gt, 0, tiles, 2, original)
    result = oc.compute_summary({"model_name": "synthetic"}, original, patches, None, tiles)
    assert "tile_spearman" in result
    assert "top10_tiles_road_interior_fraction" in result


def test_cli_defaults():
    args = oc.parse_arguments(["--checkpoint", "dummy=abc.pt"])
    oc.validate_args(args)
    assert args.patch_size == (32, 32)
    assert args.budgets == oc.DEFAULT_BUDGETS
    assert oc.parse_pair("16x32") == (16, 32)
    with pytest.raises(ValueError):
        oc.validate_args(oc.parse_arguments(["--checkpoint", "dummy=abc.pt", "--stride", "64"]))


def test_optional_captum_integration(sample):
    pytest.importorskip("captum")
    model, image, gt = sample
    baseline = torch.full_like(image, 0.5)
    attr = oc.sliding_occlusion(model, image, 0, gt, (4, 4), (4, 4), baseline, 2)
    assert attr.shape == (8, 8)
    assert np.isfinite(attr).all()
    tiles = oc.build_tiles(attr, gt, 0, (4, 4), 0.8)
    changes = oc.tile_effects(model, image, baseline, gt, 0, tiles, 2, oc.infer_metrics(model, image, gt, 0))
    # No overlapping windows: all locations in a tile have the same Captum score.
    for tile in tiles:
        patch = attr[tile.y0:tile.y1, tile.x0:tile.x1]
        assert np.allclose(patch, changes.loc[tile.index, "delta_global_logit"], atol=1e-5)


def test_end_to_end_deletion_with_synthetic_project(tmp_path: Path, monkeypatch):
    """Exercise CSV alignment, deduplication, artifacts and CLI integration."""
    model, image, gt = TinySegmentation(), torch.rand(1, 3, 8, 8), torch.zeros(8, 8, dtype=torch.long)
    gt[:2] = 1
    (tmp_path / "source.png").touch()
    (tmp_path / "gt.png").touch()
    (tmp_path / "fake.pt").touch()
    np.save(tmp_path / "cam.npy", np.tile(np.linspace(0, 1, 8, dtype=np.float32), (8, 1)))
    np.save(tmp_path / "reference.npy", image.numpy())
    score = float(model(image)[:, 0].mean())
    rows = []
    for i, category in enumerate(("high_iou", "boundary_failure")):
        rows.append(dict(
            category=category, model_name="synthetic", class_id=np.nan,
            image_index=0, image_path="source.png", mask_path="gt.png", rank=i + 1,
        ))
    pd.DataFrame(rows).to_csv(tmp_path / "selected.csv", index=False)
    pd.DataFrame([dict(
        model_name="synthetic", input_mode="clean", target_class_id=0,
        image_path="source.png", mask_path="gt.png", target_score=score,
        heatmap_npy_path="cam.npy", input_tensor_path="reference.npy", input_sha256=oc.tensor_hash(image),
    )]).to_csv(tmp_path / "cam_manifest.csv", index=False)

    project_gradcam = types.ModuleType("explain_gradcam")
    project_gradcam.parse_checkpoint_spec = lambda raw: tuple(raw.split("=", 1))
    project_gradcam.resolve_target_class = lambda raw: 0
    project_gradcam.sanitise_filename = lambda raw: raw
    project_gradcam.load_single_sample = lambda image_path, mask_path, device: (image.clone(), gt.clone())
    def expand(frame, names, max_cases=None):
        frame = frame.copy()
        frame["source_row_index"] = list(range(len(frame)))
        frame["source_model_name"] = frame["model_name"]
        return frame if max_cases is None else frame.head(max_cases)
    project_gradcam.expand_and_filter_cases = expand

    project_evaluate = types.ModuleType("evaluate_performance")
    class Configuration:
        DEVICE = torch.device("cpu")
    project_evaluate.Configuration = Configuration
    project_evaluate.CLASS_NAMES = ["road", "other"]
    project_evaluate.CLASS_COLORS = np.tile(np.array([[128, 64, 128], [0, 0, 142]], dtype=np.uint8), (10, 1))
    project_evaluate.load_trained_model = lambda path, device: TinySegmentation().eval()
    monkeypatch.setitem(sys.modules, "explain_gradcam", project_gradcam)
    monkeypatch.setitem(sys.modules, "evaluate_performance", project_evaluate)

    args = oc.parse_arguments([
        "--project-root", str(tmp_path), "--selected-cases", "selected.csv",
        "--gradcam-manifest", "cam_manifest.csv", "--output-dir", "output",
        "--checkpoint", "synthetic=fake.pt", "--experiment", "deletion",
        "--tile-size", "4", "--random-repeats", "2", "--budgets", "0,0.2,0.3",
    ])
    oc.run_pipeline(args)
    manifest = pd.read_csv(tmp_path / "output" / "occlusion_manifest.csv")
    assert len(manifest) == 1  # Two selected categories are one unique image.
    assert "high_iou;boundary_failure" == manifest.iloc[0].selection_categories
    output = Path(manifest.iloc[0].output_dir)
    assert (output / "tile_effects.csv").is_file()
    assert (output / "deletion.csv").is_file()
    assert (output / "diagnostic.png").is_file()
    assert (tmp_path / "output" / "model_comparison.csv").is_file()
    assert len(pd.read_csv(output / "deletion.csv")) == (1 + 1 + 2 + 2) * 3
    # Run a second time: output manifest must upsert, never duplicate.
    oc.run_pipeline(args)
    assert len(pd.read_csv(tmp_path / "output" / "occlusion_manifest.csv")) == 1
