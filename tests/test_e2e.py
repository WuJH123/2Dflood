from pathlib import Path

from hydrograph.config import ExperimentConfig
from hydrograph.data import UrbanFloodBenchRepository
from hydrograph.adapt import adapt_sparse_city, evaluate_sparse_adapted
from hydrograph.paper_eval import evaluate_checkpoint
from hydrograph.split import make_manifest
from hydrograph.trainer import train


def test_one_epoch_train_and_target_eval(synthetic_root, tmp_path):
    cfg = ExperimentConfig()
    cfg.data.root = str(synthetic_root)
    cfg.data.warmup_steps = 3
    cfg.data.context_steps = 2
    cfg.data.horizon_steps = 2
    cfg.data.window_stride = 4
    cfg.model.hidden_dim = 16
    cfg.model.processor_layers = 1
    cfg.model.token_count = 4
    cfg.model.token_heads = 2
    cfg.model.token_layers = 1
    cfg.model.adapter_rank = 4
    cfg.model.dropout = 0.0
    cfg.train.epochs = 1
    cfg.train.windows_per_event = 1
    cfg.train.val_events_per_epoch = 1
    cfg.train.amp = False
    cfg.train.output_dir = str(tmp_path / "out")
    cfg.train.device = "cpu"
    cfg.train.masked_state_prob = 0.5
    cfg.protocol = "leave_one_city_out"
    cfg.target_model = "Model_3"
    repo = UrbanFloodBenchRepository(synthetic_root)
    refs = repo.list_events("train")
    manifest = make_manifest(refs, cfg.protocol, 7, target_model=cfg.target_model)
    ckpt = train(cfg, repo, manifest)
    assert Path(ckpt).exists()
    metrics = evaluate_checkpoint(ckpt, cfg, repo, manifest.test)
    assert "water_level_2d_rmse" in metrics
    assert metrics["nan_fraction"] == 0.0

    adapt_refs, eval_refs = manifest.test[:1], manifest.test[1:]
    ad = adapt_sparse_city(ckpt, cfg, repo, "Model_3", adapt_refs, 0.25,
                           tmp_path / "adapt", epochs=1, seed=11)
    adapted_metrics = evaluate_sparse_adapted(ckpt, ad, cfg, repo, eval_refs)
    assert "water_level_2d_rmse" in adapted_metrics
    assert adapted_metrics["nan_fraction"] == 0.0
