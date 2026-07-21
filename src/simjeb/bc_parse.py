"""Parse boundary conditions dari file OptiStruct/Nastran .fem SimJEB.

Struktur (Section 3.4 paper):
- 4x RBE2 (baut): independent = pusat baut (SPC); dependent = node permukaan lubang.
- 1x RBE3 (interface): independent = titik FORCE/MOMENT; dependent = node permukaan clevis.

GRID id 1..N = node volume mesh (indeks 0-based = gid-1). Independent RBE = N+1..N+5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BracketBC:
    bracket_id: int
    bolt_centers: np.ndarray  # (4, 3)
    interface: np.ndarray  # (3,)
    spc_nodes: list[int]
    force_node: int
    support_gids: np.ndarray  # dependent RBE2, 1-based GRID ids
    load_gids: np.ndarray  # dependent RBE3, 1-based GRID ids
    n_bolt_dependents: list[int] = field(default_factory=list)
    n_interface_dependents: int = 0


def _nastran_float(s: str) -> float:
    s = s.strip()
    m = re.match(r"^([+-]?\d*\.?\d+)([+-]\d+)$", s)
    if m:
        s = m.group(1) + "e" + m.group(2)
    return float(s)


def _parse_grid_fixed(line: str) -> tuple[int, np.ndarray] | None:
    if not line.startswith("GRID"):
        return None
    try:
        gid = int(line[8:16])
        xyz = np.array(
            [_nastran_float(line[24:32]), _nastran_float(line[32:40]), _nastran_float(line[40:48])],
            dtype=np.float64,
        )
        return gid, xyz
    except ValueError:
        return None


def _finalize_rbe2(nums: list[int]) -> tuple[int, list[int]] | None:
    # eid, gn, cm, gm1...
    if len(nums) < 4:
        return None
    return nums[1], nums[3:]


def _finalize_rbe3(nums: list[int], n_vol: int) -> tuple[int, list[int]] | None:
    """RBE3: ambil refgrid=nums[1]; dependents = semua int di [1, n_vol]."""
    if len(nums) < 2:
        return None
    indep = nums[1]
    deps = [g for g in nums[2:] if 1 <= g <= n_vol]
    return indep, deps


def parse_fem(path: Path) -> BracketBC:
    path = Path(path)
    bid = int(path.stem)

    grids: dict[int, np.ndarray] = {}
    spc_nodes: list[int] = []
    force_nodes: list[int] = []
    moment_nodes: list[int] = []
    rbe2: list[tuple[int, list[int]]] = []
    rbe3_raw: list[list[int]] = []
    pending: tuple[str, list[int]] | None = None

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("GRID"):
                parsed = _parse_grid_fixed(line)
                if parsed:
                    grids[parsed[0]] = parsed[1]
                continue

            if line.startswith("SPC"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        spc_nodes.append(int(parts[2]))
                    except ValueError:
                        pass
                continue

            if line.startswith("FORCE"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        force_nodes.append(int(parts[2]))
                    except ValueError:
                        pass
                continue

            if line.startswith("MOMENT"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        moment_nodes.append(int(parts[2]))
                    except ValueError:
                        pass
                continue

            cont = pending is not None and (
                line.startswith("+") or (len(line) > 4 and line[:4].strip() == "" and re.search(r"\d", line))
            )
            if cont:
                pending[1].extend(int(x) for x in re.findall(r"-?\d+", line))
                continue

            if pending is not None:
                kind, nums = pending
                if kind == "RBE2":
                    fin = _finalize_rbe2(nums)
                    if fin:
                        rbe2.append(fin)
                else:
                    rbe3_raw.append(nums)
                pending = None

            if line.startswith("RBE2") or line.startswith("RBE3"):
                kind = "RBE2" if line.startswith("RBE2") else "RBE3"
                nums = [int(x) for x in re.findall(r"-?\d+", line[4:])]
                pending = (kind, nums)

        if pending is not None:
            kind, nums = pending
            if kind == "RBE2":
                fin = _finalize_rbe2(nums)
                if fin:
                    rbe2.append(fin)
            else:
                rbe3_raw.append(nums)

    spc_u = sorted(set(spc_nodes))
    load_u = sorted(set(force_nodes + moment_nodes))
    if len(spc_u) != 4:
        raise ValueError(f"{path.name}: harap 4 SPC, dapat {len(spc_u)}: {spc_u}")
    if len(load_u) != 1:
        raise ValueError(f"{path.name}: harap 1 node FORCE/MOMENT, dapat {load_u}")

    # N_vol = max GRID id yang lebih kecil dari independent BC nodes
    indep_ids = set(spc_u) | set(load_u)
    n_vol = min(indep_ids) - 1
    if n_vol < 1:
        raise ValueError(f"{path.name}: gagal infer n_vol dari indep {indep_ids}")

    rbe3: list[tuple[int, list[int]]] = []
    for nums in rbe3_raw:
        fin = _finalize_rbe3(nums, n_vol)
        if fin:
            rbe3.append(fin)

    bolt = np.stack([grids[g] for g in spc_u])
    order = np.lexsort(bolt.T[::-1])
    bolt = bolt[order]
    spc_ordered = [spc_u[i] for i in order]
    interface = grids[load_u[0]]

    support: list[int] = []
    n_bolt_deps = []
    for g in spc_ordered:
        deps = next((d for indep, d in rbe2 if indep == g), [])
        deps = [d for d in deps if 1 <= d <= n_vol]
        n_bolt_deps.append(len(deps))
        support.extend(deps)

    load_deps = next((d for indep, d in rbe3 if indep == load_u[0]), [])
    if not load_deps and rbe3:
        load_deps = max((d for _, d in rbe3), key=len)
    load_deps = [d for d in load_deps if 1 <= d <= n_vol]

    return BracketBC(
        bracket_id=bid,
        bolt_centers=bolt,
        interface=interface,
        spc_nodes=spc_ordered,
        force_node=load_u[0],
        support_gids=np.unique(np.asarray(support, dtype=np.int64)),
        load_gids=np.unique(np.asarray(load_deps, dtype=np.int64)),
        n_bolt_dependents=n_bolt_deps,
        n_interface_dependents=len(load_deps),
    )


def match_bolts_to_template(template: np.ndarray, bolts: np.ndarray) -> np.ndarray:
    """Greedy bipartite matching: kembalikan bolts diurutkan sesuai slot template."""
    D = np.linalg.norm(template[:, None, :] - bolts[None, :, :], axis=2)
    used_t, used_b = set(), set()
    out = np.zeros_like(template)
    for _, r, c in sorted((D[r, c], r, c) for r in range(4) for c in range(4)):
        if r in used_t or c in used_b:
            continue
        used_t.add(r)
        used_b.add(c)
        out[r] = bolts[c]
    return out


def bc_masks(bc: BracketBC, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (N,) untuk node volume: is_support, is_load."""
    is_support = np.zeros(n_nodes, dtype=bool)
    is_load = np.zeros(n_nodes, dtype=bool)
    for g in bc.support_gids:
        idx = int(g) - 1
        if 0 <= idx < n_nodes:
            is_support[idx] = True
    for g in bc.load_gids:
        idx = int(g) - 1
        if 0 <= idx < n_nodes:
            is_load[idx] = True
    return is_support, is_load
