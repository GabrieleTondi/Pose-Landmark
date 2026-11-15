import json
from pathlib import Path

import torch
from torch_geometric.nn import GATv2Conv

from src.utils.config import load_all_configs
from src.utils.hardware import get_compute_device, get_hardware_profile
from src.utils.seed import seed_everything


def _sample_simple_edge_index(num_nodes: int, num_edges: int, device: torch.device) -> torch.Tensor:
    """Generate dummy edge_index (2, num_edges) without explicit self-loops."""
    src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long, device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long, device=device)
    dst = torch.where(dst == src, (dst + 1) % num_nodes, dst)
    return torch.stack([src, dst], dim=0)


def main() -> None:
    """Run full smoke test for determinism, YAML configs, device selection, and PyG GATv2Conv."""
    seed_everything(42)
    cfgs = load_all_configs("configs")
    assert set(cfgs.keys()) == {"data", "graph", "training"}

    device = get_compute_device()
    profile = get_hardware_profile()
    profile["selected_compute_device"] = str(device)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    profile_path = reports_dir / "hardware_profile.json"
    with profile_path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)

    x = torch.randn((17, 3), dtype=torch.float32, device=device, requires_grad=True)
    edge_index = _sample_simple_edge_index(num_nodes=17, num_edges=32, device=device)
    edge_attr = torch.randn((32, 5), dtype=torch.float32, device=device)

    conv = GATv2Conv(
        in_channels=3,
        out_channels=32,
        heads=4,
        edge_dim=5,
        add_self_loops=True,
    ).to(device)

    out, (edge_idx_attn, alpha) = conv(
        x, edge_index, edge_attr=edge_attr, return_attention_weights=True
    )

    assert out.shape == (17, 128), f"Expected out shape (17, 128), got {tuple(out.shape)}"
    assert edge_idx_attn.shape == (2, 49), (
        f"Expected edge_idx_attn shape (2, 49), got {tuple(edge_idx_attn.shape)}"
    )
    assert alpha.shape == (49, 4), f"Expected alpha shape (49, 4), got {tuple(alpha.shape)}"

    out.sum().backward()
    assert x.grad is not None, "Gradient with respect to x must not be None."

    print("ENV_CHECK_OK")


if __name__ == "__main__":
    main()

