from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig
from .data import StaticGraph
from .normalization import NormalizationBundle


def _mlp(in_dim: int, out_dim: int, hidden: int, dropout: float, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    d = in_dim
    for _ in range(max(1, layers - 1)):
        modules += [nn.Linear(d, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        d = hidden
    modules.append(nn.Linear(d, out_dim))
    return nn.Sequential(*modules)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    return out


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.dropout(torch.nn.functional.silu(self.down(x))))


class HydraulicTokenMixer(nn.Module):
    """Geometry-agnostic global mixer inspired by physics-token neural operators.

    Nodes are softly assigned to a small set of learned hydraulic-state tokens. Attention
    occurs among tokens rather than all mesh nodes, giving O(NK + K^2) global coupling.
    This is an original implementation and does not copy Transolver source code.
    """

    def __init__(self, dim: int, token_count: int, heads: int, layers: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by token_heads")
        self.token_count = token_count
        self.assign = nn.Linear(dim, token_count)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # assignment [N,K], normalized over nodes for token aggregation
        logits = self.assign(h)
        node_to_token = torch.softmax(logits, dim=-1)
        denom = node_to_token.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
        tokens = (node_to_token.transpose(0, 1) @ h) / denom
        tokens = self.token_encoder(tokens.unsqueeze(0)).squeeze(0)
        back = node_to_token @ tokens
        return self.out_norm(h + back)


class RelationMessage(nn.Module):
    def __init__(self, dim: int, edge_dim: int, edge_dyn_dim: int, dropout: float):
        super().__init__()
        # src, dst, static edge, dynamic edge, hydraulic head difference
        self.net = _mlp(2 * dim + edge_dim + edge_dyn_dim + 1, dim, dim, dropout, layers=3)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
        edge_dynamic: torch.Tensor,
        head_src: torch.Tensor,
        head_dst: torch.Tensor,
        dst_size: int,
    ) -> torch.Tensor:
        s, d = edge_index
        dh = (head_src[s] - head_dst[d]).unsqueeze(-1)
        feats = [h_src[s], h_dst[d], dh]
        if edge_static.shape[-1]:
            feats.insert(2, edge_static)
        if edge_dynamic.shape[-1]:
            feats.insert(-1, edge_dynamic)
        msg = self.net(torch.cat(feats, dim=-1))
        return scatter_sum(msg, d, dst_size)


class CouplingMessage(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = _mlp(2 * dim + 1, dim, dim, dropout, layers=3)

    def forward(self, hs: torch.Tensor, hd: torch.Tensor, idx: torch.Tensor,
                head_s: torch.Tensor, head_d: torch.Tensor, dst_size: int) -> torch.Tensor:
        s, d = idx
        x = torch.cat([hs[s], hd[d], (head_s[s] - head_d[d]).unsqueeze(-1)], dim=-1)
        return scatter_sum(self.net(x), d, dst_size)


class ProcessorBlock(nn.Module):
    def __init__(self, dim: int, edge1_dim: int, edge2_dim: int, dropout: float, adapter_rank: int):
        super().__init__()
        self.m11 = RelationMessage(dim, edge1_dim, 2, dropout)
        self.m22 = RelationMessage(dim, edge2_dim, 2, dropout)
        self.m12 = CouplingMessage(dim, dropout)
        self.m21 = CouplingMessage(dim, dropout)
        self.u1 = nn.GRUCell(dim, dim)
        self.u2 = nn.GRUCell(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.a1 = ResidualAdapter(dim, adapter_rank, dropout)
        self.a2 = ResidualAdapter(dim, adapter_rank, dropout)

    def forward(self, h1, h2, graph: StaticGraph, e1_dyn, e2_dyn, head1, head2, use_coupling: bool = True):
        m1 = self.m11(h1, h1, graph.edge1_index, graph.edge1_static, e1_dyn,
                      head1, head1, graph.n1)
        m2 = self.m22(h2, h2, graph.edge2_index, graph.edge2_static, e2_dyn,
                      head2, head2, graph.n2)
        if use_coupling and graph.coupling_index.shape[1] > 0:
            c12 = self.m12(h1, h2, graph.coupling_index, head1, head2, graph.n2)
            rev = torch.stack([graph.coupling_index[1], graph.coupling_index[0]], dim=0)
            c21 = self.m21(h2, h1, rev, head2, head1, graph.n1)
        else:
            c12 = torch.zeros_like(h2)
            c21 = torch.zeros_like(h1)
        h1n = self.a1(self.n1(self.u1(m1 + c21, h1)))
        h2n = self.a2(self.n2(self.u2(m2 + c12, h2)))
        return h1n, h2n


@dataclass
class StepPrediction:
    node1: torch.Tensor  # [N1,2] next water level, inlet flow
    node2: torch.Tensor  # [N2,3] next rainfall (exogenous), water level, volume
    edge1: torch.Tensor  # [E1,2] next flow, velocity
    edge2: torch.Tensor  # [E2,2]
    hidden1: torch.Tensor
    hidden2: torch.Tensor


class HydroGraphOperator(nn.Module):
    """Heterogeneous 1D-2D hydrodynamic graph operator for UrbanFloodBench."""

    def __init__(self, config: ModelConfig, node1_static_dim: int, node2_static_dim: int,
                 edge1_static_dim: int, edge2_static_dim: int):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        # +1 observation mask channel per node type
        self.enc1 = _mlp(node1_static_dim + 2 + 1, d, d, config.dropout, 3)
        self.enc2 = _mlp(node2_static_dim + 3 + 1, d, d, config.dropout, 3)
        self.blocks = nn.ModuleList([
            ProcessorBlock(d, edge1_static_dim, edge2_static_dim, config.dropout, config.adapter_rank)
            for _ in range(config.processor_layers)
        ])
        self.type1 = nn.Parameter(torch.zeros(1, d))
        self.type2 = nn.Parameter(torch.zeros(1, d))
        self.global_mixer = HydraulicTokenMixer(d, config.token_count, config.token_heads,
                                                config.token_layers, config.dropout)
        self.node1_head = _mlp(d, 2, d, config.dropout, 2)
        self.node2_head = _mlp(d, 2, d, config.dropout, 2)  # water_level, volume only
        flux_in1 = 2 * d + edge1_static_dim
        flux_in2 = 2 * d + edge2_static_dim
        self.edge1_head = _mlp(flux_in1, 2, d, config.dropout, 3)
        self.edge2_head = _mlp(flux_in2, 2, d, config.dropout, 3)
        # Normalization is fitted on source-city training data only and stored in checkpoints.
        for name, dim in [("n1s", node1_static_dim), ("n2s", node2_static_dim),
                          ("e1s", edge1_static_dim), ("e2s", edge2_static_dim),
                          ("n1d", 2), ("n2d", 3), ("e1d", 2), ("e2d", 2)]:
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

    def set_normalization(self, stats: NormalizationBundle) -> None:
        mapping = {
            "n1s": stats.node1_static, "n2s": stats.node2_static,
            "e1s": stats.edge1_static, "e2s": stats.edge2_static,
            "n1d": stats.node1_dynamic, "n2d": stats.node2_dynamic,
            "e1d": stats.edge1_dynamic, "e2d": stats.edge2_dynamic,
        }
        for prefix, st in mapping.items():
            getattr(self, f"{prefix}_mean").copy_(st.mean.to(getattr(self, f"{prefix}_mean").device))
            getattr(self, f"{prefix}_std").copy_(st.std.to(getattr(self, f"{prefix}_std").device))

    @staticmethod
    def _norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean.to(x.device)) / std.to(x.device)

    def _encode(self, graph: StaticGraph, n1: torch.Tensor, n2: torch.Tensor,
                mask1: torch.Tensor, mask2: torch.Tensor):
        s1 = self._norm(graph.node1_static, self.n1s_mean, self.n1s_std)
        s2 = self._norm(graph.node2_static, self.n2s_mean, self.n2s_std)
        n1 = self._norm(n1, self.n1d_mean, self.n1d_std)
        n2 = self._norm(n2, self.n2d_mean, self.n2d_std)
        x1 = torch.cat([s1, n1, mask1[:, None].to(n1.dtype)], dim=-1)
        x2 = torch.cat([s2, n2, mask2[:, None].to(n2.dtype)], dim=-1)
        return self.enc1(x1) + self.type1, self.enc2(x2) + self.type2

    def forward_step(
        self,
        graph: StaticGraph,
        node1: torch.Tensor,
        node2: torch.Tensor,
        edge1: torch.Tensor,
        edge2: torch.Tensor,
        next_rainfall: torch.Tensor,
        obs_mask1: torch.Tensor | None = None,
        obs_mask2: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> StepPrediction:
        device = node1.device
        obs_mask1 = torch.ones(graph.n1, dtype=torch.bool, device=device) if obs_mask1 is None else obs_mask1
        obs_mask2 = torch.ones(graph.n2, dtype=torch.bool, device=device) if obs_mask2 is None else obs_mask2
        # node1/node2 are the model's *available causal state*: observed values where sensors
        # exist and model-propagated values elsewhere. obs_mask marks measurement provenance;
        # callers must never place hidden ground truth at unobserved nodes (audited in rollout/trainer).
        n1_in = node1
        n2_in = node2
        h1e, h2e = self._encode(graph, n1_in, n2_in, obs_mask1, obs_mask2)
        if hidden is None:
            h1, h2 = h1e, h2e
        else:
            # Re-inject current observations/forcing into recurrent state without discarding memory.
            h1, h2 = hidden[0] + h1e, hidden[1] + h2e

        # Water-surface head proxies. 1D water_level is elevation; 2D release convention may be
        # depth or surface elevation. We use the supplied level directly for relation directionality.
        head1 = node1[:, 0]
        head2 = node2[:, 1]
        if not self.config.use_hydraulic_head:
            head1 = torch.zeros_like(head1)
            head2 = torch.zeros_like(head2)
        # Feed normalized edge attributes/signals to message functions while preserving raw graph
        # topology and raw hydraulic heads for directional information.
        graph_msg = StaticGraph(
            graph.model_id, graph.node1_static, graph.node2_static, graph.node1_ids, graph.node2_ids,
            graph.edge1_index, graph.edge2_index, graph.coupling_index,
            self._norm(graph.edge1_static, self.e1s_mean, self.e1s_std) if graph.edge1_static.shape[-1] else graph.edge1_static,
            self._norm(graph.edge2_static, self.e2s_mean, self.e2s_std) if graph.edge2_static.shape[-1] else graph.edge2_static,
            graph.node1_feature_names, graph.node2_feature_names, graph.edge1_feature_names, graph.edge2_feature_names,
            graph.edge1_ids, graph.edge2_ids,
        )
        e1_msg = self._norm(edge1, self.e1d_mean, self.e1d_std)
        e2_msg = self._norm(edge2, self.e2d_mean, self.e2d_std)
        for block in self.blocks:
            h1, h2 = block(h1, h2, graph_msg, e1_msg, e2_msg, head1, head2, self.config.use_coupling)
        if self.config.use_global_tokens:
            both = self.global_mixer(torch.cat([h1, h2], dim=0))
            h1, h2 = both[:graph.n1], both[graph.n1:]

        d1 = self.node1_head(h1) * self.n1d_std.to(h1.device)
        d2 = self.node2_head(h2) * self.n2d_std[1:].to(h2.device)
        pred1 = node1 + d1
        # Physical state bounds are applied out-of-place to preserve autograd versioning.
        pred2 = torch.stack([
            next_rainfall.clamp_min(0),
            node2[:, 1] + d2[:, 0],
            (node2[:, 2] + d2[:, 1]).clamp_min(0),
        ], dim=-1)

        s1, d1i = graph.edge1_index
        s2, d2i = graph.edge2_index
        f1_parts = [h1[s1], h1[d1i]]
        if graph.edge1_static.shape[-1]:
            f1_parts.append(graph_msg.edge1_static)
        f2_parts = [h2[s2], h2[d2i]]
        if graph.edge2_static.shape[-1]:
            f2_parts.append(graph_msg.edge2_static)
        if self.config.use_flux_decoder:
            pred_e1 = edge1 + self.edge1_head(torch.cat(f1_parts, dim=-1)) * self.e1d_std.to(h1.device)
            pred_e2 = edge2 + self.edge2_head(torch.cat(f2_parts, dim=-1)) * self.e2d_std.to(h2.device)
        else:
            pred_e1, pred_e2 = edge1, edge2
        return StepPrediction(pred1, pred2, pred_e1, pred_e2, h1, h2)

    def adapter_parameters(self):
        for name, p in self.named_parameters():
            if ".a1." in name or ".a2." in name or name.endswith("type1") or name.endswith("type2"):
                yield p

    def freeze_for_adaptation(self, train_heads: bool = True) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for p in self.adapter_parameters():
            p.requires_grad = True
        if train_heads:
            for module in [self.node1_head, self.node2_head, self.edge1_head, self.edge2_head]:
                for p in module.parameters():
                    p.requires_grad = True
