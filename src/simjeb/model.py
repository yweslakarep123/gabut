"""MeshGraphNet-lite untuk prediksi field SimJEB (20 channel).

Node feat: [x,y,z (mm×COORD_SCALE), is_surf, is_support, is_load]
Edge feat: [rel_xyz (3), |rel|] dari konektivitas tetrahedral

COORD_SCALE=0.01 (mm→dm) global — bukan normalisasi per-bracket; skala fisik
antar-bracket tetap, hanya conditioning numerik MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import coalesce

COORD_SCALE = 0.01  # mm -> dm, global


def tets_to_edges(tets: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """(Nt,4) -> undirected edge_index (2, E) tanpa duplikat."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = torch.cat([tets[:, [i, j]] for i, j in pairs], dim=0)
    edge_index = edges.t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes)
    return edge_index


def build_graph(
    pos: torch.Tensor,
    tets: torch.Tensor,
    node_surf: torch.Tensor,
    is_support: torch.Tensor,
    is_load: torch.Tensor,
    y: torch.Tensor | None = None,
) -> Data:
    pos_s = pos * COORD_SCALE
    edge_index = tets_to_edges(tets, num_nodes=pos.shape[0])
    src, dst = edge_index
    rel = pos_s[dst] - pos_s[src]
    dist = rel.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([rel, dist], dim=-1)
    x = torch.cat(
        [
            pos_s,
            node_surf.float().unsqueeze(-1),
            is_support.float().unsqueeze(-1),
            is_load.float().unsqueeze(-1),
        ],
        dim=-1,
    )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos_s)
    if y is not None:
        data.y = y
    return data


class EdgeNodeBlock(MessagePassing):
    def __init__(self, hidden: int):
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.norm_e = nn.LayerNorm(hidden)
        self.norm_n = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, e):
        src, dst = edge_index
        e_in = torch.cat([h[src], h[dst], e], dim=-1)
        e = self.norm_e(e + self.edge_mlp(e_in))
        msg = self.propagate(edge_index, h=h, e=e)
        h = self.norm_n(h + self.node_mlp(torch.cat([h, msg], dim=-1)))
        return h, e

    def message(self, h_j, e):
        return e


class MeshGraphNet(nn.Module):
    def __init__(self, hidden: int = 64, n_layers: int = 8, out_dim: int = 20, in_dim: int = 6):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.blocks = nn.ModuleList([EdgeNodeBlock(hidden) for _ in range(n_layers)])
        # decoder melihat h + posisi (skip) supaya field lokal tidak hilang oleh smoothing
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        h = self.node_enc(data.x)
        e = self.edge_enc(data.edge_attr)
        for blk in self.blocks:
            if self.training and h.requires_grad:
                h, e = torch.utils.checkpoint.checkpoint(
                    blk, h, data.edge_index, e, use_reentrant=False
                )
            else:
                h, e = blk(h, data.edge_index, e)
        return self.decoder(torch.cat([h, data.pos], dim=-1))
