import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.loaders import create_grouped_splits, generate_mock_finefs_samples
from src.graph.builder import process_and_save_dataset_graphs
from src.models.siamese_gat import SiameseGATv2
from src.training.augmentation import generate_synthetic_negatives
from src.training.losses import AdaptiveTripletMarginLoss
from src.training.triplet_sampler import TripletDataset, partition_by_goe_tiers
from src.utils.config import load_all_configs
from src.utils.hardware import get_compute_device
from src.utils.seed import seed_everything


def train_one_epoch(
    model: SiameseGATv2,
    loader: DataLoader,
    criterion: AdaptiveTripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Run one training epoch on PyG triplets with gradient clipping."""
    model.train()
    totals = {
        "loss": 0.0,
        "triplet_acc": 0.0,
        "margin_satisfied_acc": 0.0,
        "mean_d_ap": 0.0,
        "mean_d_an": 0.0,
    }
    total_triplets = 0

    for batch_A, batch_P, batch_N, batch_alpha in loader:
        bs = int(batch_alpha.numel())
        batch_A = batch_A.to(device)
        batch_P = batch_P.to(device)
        batch_N = batch_N.to(device)
        batch_alpha = batch_alpha.to(device)

        optimizer.zero_grad()
        z_A, z_P, z_N = model.forward_triplet(batch_A, batch_P, batch_N)
        loss, stats = criterion(z_A, z_P, z_N, batch_alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
        optimizer.step()

        total_triplets += bs
        for key in totals:
            totals[key] += float(stats[key]) * bs

    if total_triplets == 0:
        raise RuntimeError("Training DataLoader produced zero triplets.")

    return {key: val / float(total_triplets) for key, val in totals.items()}


@torch.no_grad()
def evaluate_triplets(
    model: SiameseGATv2,
    loader: DataLoader,
    criterion: AdaptiveTripletMarginLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model in eval mode and monitor latent dispersion (anti-collapse)."""
    model.eval()
    totals = {
        "loss": 0.0,
        "triplet_acc": 0.0,
        "margin_satisfied_acc": 0.0,
        "mean_d_ap": 0.0,
        "mean_d_an": 0.0,
    }
    total_triplets = 0
    collected_z_a: List[torch.Tensor] = []

    for batch_A, batch_P, batch_N, batch_alpha in loader:
        bs = int(batch_alpha.numel())
        batch_A = batch_A.to(device)
        batch_P = batch_P.to(device)
        batch_N = batch_N.to(device)
        batch_alpha = batch_alpha.to(device)

        z_A, z_P, z_N = model.forward_triplet(batch_A, batch_P, batch_N)
        _, stats = criterion(z_A, z_P, z_N, batch_alpha)

        total_triplets += bs
        for key in totals:
            totals[key] += float(stats[key]) * bs
        collected_z_a.append(z_A.detach().cpu())

    if total_triplets == 0:
        raise RuntimeError("Validation DataLoader produced zero triplets.")

    all_z = torch.cat(collected_z_a, dim=0)
    embedding_std = float(torch.std(all_z, dim=0).mean().item()) if all_z.shape[0] > 1 else 0.0

    return {
        "val_loss": totals["loss"] / float(total_triplets),
        "val_triplet_acc": totals["triplet_acc"] / float(total_triplets),
        "val_margin_satisfied_acc": totals["margin_satisfied_acc"] / float(total_triplets),
        "val_mean_d_ap": totals["mean_d_ap"] / float(total_triplets),
        "val_mean_d_an": totals["mean_d_an"] / float(total_triplets),
        "embedding_std": embedding_std,
    }


def _should_save_checkpoint(
    val_metrics: Dict[str, float],
    best_val_acc: float,
    best_val_loss: float,
    has_checkpoint: bool,
) -> bool:
    if not has_checkpoint:
        return True
    if val_metrics["embedding_std"] <= 1e-6:
        return False
    if val_metrics["val_triplet_acc"] > best_val_acc + 1e-12:
        return True
    if (
        abs(val_metrics["val_triplet_acc"] - best_val_acc) <= 1e-12
        and val_metrics["val_loss"] < best_val_loss - 1e-12
    ):
        return True
    return False


def run_training_pipeline(
    train_graphs: List[Data],
    val_graphs: List[Data],
    data_cfg: Dict[str, Any],
    graph_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the full training pipeline: train-only augmentation,
    GOE tier partitioning, deterministic validation, early stopping, and saving best_model.pt.
    """
    t_cfg = train_cfg["training"]
    seed = int(t_cfg.get("seed", 42))
    seed_everything(seed)
    device = get_compute_device(prefer_gpu=True)

    batch_size = int(t_cfg.get("batch_size", 32))
    lr = float(t_cfg.get("lr", 1e-3))
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    max_epochs = int(t_cfg.get("max_epochs", 80))
    patience = int(t_cfg.get("early_stopping_patience", 15))
    max_grad_norm = float(t_cfg.get("grad_clip_max_norm", 1.0))
    num_train_triplets = int(t_cfg.get("num_train_triplets_per_epoch", 600))
    num_val_triplets = int(t_cfg.get("num_val_triplets", 200))

    synth_graphs = generate_synthetic_negatives(train_graphs, data_cfg, graph_cfg, train_cfg)
    train_pools, thresholds = partition_by_goe_tiers(
        list(train_graphs) + synth_graphs, train_cfg, fit_thresholds=None
    )
    val_pools, _ = partition_by_goe_tiers(val_graphs, train_cfg, fit_thresholds=thresholds)

    val_ds = TripletDataset(
        val_pools,
        train_cfg,
        num_triplets=num_val_triplets,
        split="val",
        seed=seed,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SiameseGATv2.from_config(train_cfg).to(device)
    loss_cfg = train_cfg.get("loss", {})
    train_criterion = AdaptiveTripletMarginLoss(
        var_reg_weight=float(loss_cfg.get("var_reg_weight", 0.0)),
        var_reg_target=float(loss_cfg.get("var_reg_target", 1.0)),
    )
    eval_criterion = AdaptiveTripletMarginLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ckpt_dir = Path(data_cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    reports_dir = Path(data_cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / "training_history.json"

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        train_ds = TripletDataset(
            train_pools,
            train_cfg,
            num_triplets=num_train_triplets,
            split="train",
            seed=seed + epoch,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        train_metrics = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device, max_grad_norm=max_grad_norm
        )
        val_metrics = evaluate_triplets(model, val_loader, eval_criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_triplet_acc": train_metrics["triplet_acc"],
            "train_margin_satisfied_acc": train_metrics["margin_satisfied_acc"],
            "train_mean_d_ap": train_metrics["mean_d_ap"],
            "train_mean_d_an": train_metrics["mean_d_an"],
            **val_metrics,
        }
        history.append(epoch_record)

        if _should_save_checkpoint(
            val_metrics, best_val_acc, best_val_loss, has_checkpoint=(best_epoch > 0)
        ):
            best_val_acc = float(val_metrics["val_triplet_acc"])
            best_val_loss = float(val_metrics["val_loss"])
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "val_triplet_acc": val_metrics["val_triplet_acc"],
                    "embedding_std": val_metrics["embedding_std"],
                    "goe_thresholds": thresholds,
                    "train_cfg": train_cfg,
                    "graph_cfg": graph_cfg,
                    "data_cfg": data_cfg,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "device": str(device),
        "num_train_real_graphs": len(train_graphs),
        "num_train_synthetic_graphs": len(synth_graphs),
        "num_val_graphs": len(val_graphs),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_triplet_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_embedding_std": history[best_epoch - 1]["embedding_std"] if best_epoch > 0 else 0.0,
        "goe_thresholds": {k: [float(v[0]), float(v[1])] for k, v in thresholds.items()},
        "checkpoint_path": str(ckpt_path),
        "history_path": str(history_path),
        "history": history,
    }


def main() -> None:
    """CLI entrypoint: load graphs from data/processed/ (or generate mock fallback) and start training."""
    cfgs = load_all_configs("configs")
    data_cfg = cfgs["data"]
    graph_cfg = cfgs["graph"]
    train_cfg = cfgs["training"]

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    train_pt = processed_dir / "train_graphs.pt"
    val_pt = processed_dir / "val_graphs.pt"
    expected_in_channels = int(train_cfg["model"]["in_channels"])

    loaded_ok = False
    if train_pt.is_file() and val_pt.is_file():
        train_graphs = torch.load(train_pt, map_location="cpu", weights_only=False)
        val_graphs = torch.load(val_pt, map_location="cpu", weights_only=False)
        if (
            len(train_graphs) > 0
            and train_graphs[0].x.shape[1] == expected_in_channels
            and hasattr(train_graphs[0], "mirrored_to_canonical_leg")
        ):
            loaded_ok = True

    if not loaded_ok:
        from src.graph.builder import main as rebuild_graphs_main

        rebuild_graphs_main()
        train_graphs = torch.load(train_pt, map_location="cpu", weights_only=False)
        val_graphs = torch.load(val_pt, map_location="cpu", weights_only=False)

    summary = run_training_pipeline(train_graphs, val_graphs, data_cfg, graph_cfg, train_cfg)
    compact_summary = {k: v for k, v in summary.items() if k != "history"}
    print(json.dumps(compact_summary, indent=2))


if __name__ == "__main__":
    main()

