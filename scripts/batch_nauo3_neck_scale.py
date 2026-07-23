#!/usr/bin/env python3
"""Batch NAUO3: deformasi radial lokal di leher (mesh volume) → FEA cantilever.

Pendekatan (a): scale radial di band |y - y0| < L dengan falloff cosine.
Batch penuh N=11 (s=0.90..1.10 step 0.02). Pilot: hanya ekstrem kecuali
--samples override.

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/batch_nauo3_neck_scale.py --samples 0.90,1.10
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fea_nauo3_cantilever import (  # noqa: E402
    G,
    G_MM,
    M_DISTAL,
    M_FOREARM,
    M_PAYLOAD,
    M_SELF,
    M_W1,
    M_W2,
    M_W3,
    X,
    _boundary_faces,
    _tet_volume_sign,
    select_end_faces,
    step1_com_moment,
)

MESH_BASE = ROOT / "reports" / "ur5e_nauo3_volume" / "geometry.npz"
OUT_ROOT = ROOT / "reports" / "ur5e_nauo3_neck_batch"
CCX = Path("/home/daffa/miniforge3/envs/simjeb/bin/ccx")

E_MPA = 70000.0
NU = 0.33

# Band leher (mm) — DIKUNCI untuk definisi batch
# Cosine C1, L=40 mm untuk semua s (termasuk s>1). Sisi s>1 sensitif ke L;
# alternatif falloff diuji lalu ditolak sebagai definisi parameter batch.
Y0_NECK = 262.0
L_BAND = 40.0
L_BAND_CONTRACT = L_BAND
L_BAND_EXPAND = L_BAND
FALLOFF_KIND = "cosine_C1"

# Gate mesh
MAX_ABS_DV_FRAC = 0.05  # |ΔV|/V0
MAX_INVERTED = 0
REACTION_ERR_PCT = 10.0
MIN_SPC_GAP_MM = 20.0

# Hotspot vs band (metadata saja — bukan gate lulus/gagal)
HOTSPOT_CENTER_FRAC = 0.35
HOTSPOT_EDGE_FRAC = 0.70

BASELINE_SIGMA_MAX_MPA = 1.645735221879838


def L_for_s(s: float) -> float:
    return L_BAND


def sigma_for_s(s: float) -> float | None:
    return None


def neck_weight(
    y: np.ndarray,
    y0: float = Y0_NECK,
    L: float = L_BAND,
    kind: str = FALLOFF_KIND,
    sigma: float | None = None,
) -> np.ndarray:
    """Bump 1 di pusat, 0 di luar support.

    Default batch: cosine_C1. Alternatif (eksperimen): gaussian_cut,
    bump_Cinf, smootherstep_C2.
    """
    d = np.abs(y - y0)
    w = np.zeros_like(y, dtype=np.float64)
    if kind == "gaussian_cut":
        if sigma is None:
            sigma = L / 3.0
        inside = d < L
        w[inside] = np.exp(-0.5 * (d[inside] / sigma) ** 2)
        return w
    inside = d < L
    if not np.any(inside):
        return w
    if kind == "cosine_C1":
        w[inside] = 0.5 * (1.0 + np.cos(np.pi * d[inside] / L))
    elif kind == "smootherstep_C2":
        u = 1.0 - d[inside] / L
        w[inside] = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
    elif kind == "bump_Cinf":
        t = np.clip(d[inside] / L, 0.0, 1.0 - 1e-12)
        w[inside] = np.exp(1.0 - 1.0 / (1.0 - t * t))
    else:
        raise ValueError(f"unknown falloff kind: {kind}")
    return w


def deform_neck_radial(
    points: np.ndarray,
    s: float,
    axis_point: np.ndarray,
    axis_u: np.ndarray,
    y0: float = Y0_NECK,
    L: float | None = None,
    falloff_kind: str = FALLOFF_KIND,
    freeze_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Scale radial (⊥ axis_u) dengan faktor efektif 1+(s-1)*w(y)."""
    if L is None:
        L = L_for_s(s)
    sigma = sigma_for_s(s)
    u = axis_u / np.linalg.norm(axis_u)
    w = neck_weight(points[:, 1], y0=y0, L=L, kind=falloff_kind, sigma=sigma)
    if freeze_mask is not None:
        w = w.copy()
        w[freeze_mask] = 0.0
    s_eff = 1.0 + (s - 1.0) * w
    rel = points - axis_point
    along = np.outer(rel @ u, u)
    radial = rel - along
    out = axis_point + along + radial * s_eff[:, None]
    meta = {
        "s": s,
        "y0_mm": y0,
        "L_mm": L,
        "sigma_mm": sigma if falloff_kind == "gaussian_cut" else None,
        "falloff_kind": falloff_kind,
        "band_y_mm": [y0 - L, y0 + L],
        "w_at_cutoff": float(np.exp(-0.5 * (L / sigma) ** 2))
        if falloff_kind == "gaussian_cut" and sigma is not None
        else 0.0,
        "n_nodes_w_gt_0": int((w > 0).sum()),
        "n_nodes_w_gt_0_5": int((w > 0.5).sum()),
        "n_frozen": int(freeze_mask.sum()) if freeze_mask is not None else 0,
        "w_max": float(w.max()),
        "s_eff_min": float(s_eff.min()),
        "s_eff_max": float(s_eff.max()),
        "axis_point_mm": axis_point.tolist(),
        "axis_u": u.tolist(),
    }
    return out, meta


def flip_negative_tets(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, dict]:
    """Flip konektivitas tet ber-volume ≤0 (winding), tanpa menggeser node."""
    tets_o = tets.copy()
    vol = _tet_volume_sign(points, tets_o)
    flip = vol <= 0
    n = int(flip.sum())
    if n:
        tets_o[flip] = tets_o[flip][:, [0, 2, 1, 3]]
    vol2 = _tet_volume_sign(points, tets_o)
    return tets_o, {
        "n_flipped": n,
        "n_still_nonpos": int((vol2 <= 0).sum()),
        "min_vol_before": float(vol.min()),
        "min_vol_after": float(vol2.min()),
    }


def mesh_gates(
    points0: np.ndarray,
    points: np.ndarray,
    tets: np.ndarray,
    surf_faces: np.ndarray | None,
) -> dict:
    vol0 = np.abs(_tet_volume_sign(points0, tets))
    vol = _tet_volume_sign(points, tets)
    V0 = float(vol0.sum())
    V = float(np.abs(vol).sum())
    n_inv = int((vol <= 0).sum())
    dv_frac = (V - V0) / V0 if V0 > 0 else float("nan")

    # Surface connectivity unchanged → euler dari faces baseline tetap valid
    # secara topologi; cek hanya volume/invert + ΔV.
    reasons: list[str] = []
    if n_inv > MAX_INVERTED:
        reasons.append(f"inverted_tets={n_inv}")
    if abs(dv_frac) > MAX_ABS_DV_FRAC:
        reasons.append(f"abs_dV_frac={dv_frac:.4f}>{MAX_ABS_DV_FRAC}")

    return {
        "pass": len(reasons) == 0,
        "fail_reasons": reasons,
        "V0_mm3": V0,
        "V_mm3": V,
        "dV_frac": dv_frac,
        "n_inverted_or_neg": n_inv,
        "n_tets": int(len(tets)),
        "n_nodes": int(len(points)),
    }


def write_cantilever_inp(
    path: Path,
    points: np.ndarray,
    tets: np.ndarray,
    ends: dict,
    dens_t_per_mm3: float,
    F_N: float,
    M_dist_Nmm: np.ndarray,
    grav: np.ndarray,
) -> None:
    n_nodes = len(points)
    ref = n_nodes + 1
    vol = _tet_volume_sign(points, tets)
    tets_w = tets.copy()
    flip = vol < 0
    if flip.any():
        tets_w[flip] = tets_w[flip][:, [0, 2, 1, 3]]
    tet_ids_d, face_ids_d = ends["dist_faces"]
    dc = ends["dist_center"]
    prox = ends["prox_nodes"] + 1

    with path.open("w") as f:
        f.write("*HEADING\nNAUO3 neck-scale cantilever\n")
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"{i}, {x:.8e}, {y:.8e}, {z:.8e}\n")
        f.write(f"{ref}, {dc[0]:.8e}, {dc[1]:.8e}, {dc[2]:.8e}\n")
        f.write("*ELEMENT, TYPE=C3D4, ELSET=Eall\n")
        for i, th in enumerate(tets_w, start=1):
            a, b, c, d = (th + 1).tolist()
            f.write(f"{i}, {a}, {b}, {c}, {d}\n")
        f.write("*NSET, NSET=Nprox\n")
        row: list[str] = []
        for nid in prox:
            row.append(str(nid))
            if len(row) == 16:
                f.write(", ".join(row) + ",\n")
                row = []
        if row:
            f.write(", ".join(row) + ",\n")
        f.write(f"*NSET, NSET=Nref\n{ref},\n")
        f.write("*SURFACE, NAME=Sdist, TYPE=ELEMENT\n")
        for tid, fid in zip(tet_ids_d, face_ids_d):
            f.write(f"{int(tid)+1}, S{int(fid)}\n")
        f.write("*MATERIAL, NAME=AlUR\n*ELASTIC\n")
        f.write(f"{E_MPA}, {NU}\n*DENSITY\n{dens_t_per_mm3:.8e}\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AlUR\n")
        f.write(
            f"*COUPLING, CONSTRAINT NAME=RBE3dist, REF NODE={ref}, SURFACE=Sdist\n"
        )
        f.write("*DISTRIBUTING\n1, 6\n")
        f.write("*BOUNDARY\nNprox, 1, 3\n")
        f.write("*STEP\n*STATIC\n")
        f.write("*DLOAD\n")
        f.write(
            f"Eall, GRAV, {G_MM:.8e}, {grav[0]:.8e}, {grav[1]:.8e}, {grav[2]:.8e}\n"
        )
        f.write("*CLOAD\n")
        f.write(f"{ref}, 1, {F_N * grav[0]:.8e}\n")
        f.write(f"{ref}, 2, {F_N * grav[1]:.8e}\n")
        f.write(f"{ref}, 3, {F_N * grav[2]:.8e}\n")
        f.write(f"{ref}, 4, {M_dist_Nmm[0]:.8e}\n")
        f.write(f"{ref}, 5, {M_dist_Nmm[1]:.8e}\n")
        f.write(f"{ref}, 6, {M_dist_Nmm[2]:.8e}\n")
        f.write("*NODE FILE\nU, RF\n")
        f.write("*EL FILE\nS\n")
        f.write("*NODE PRINT, NSET=Nprox\nRF\n")
        f.write("*END STEP\n")


def _rf_moment_from_forc(
    frd_path: Path, points: np.ndarray, prox_nodes0: np.ndarray, prox_c: np.ndarray
):
    """Parse nodal reaction forces from FRD FORC (or RF) block."""
    num_re = re.compile(r"[+-]?\d+\.\d+E[+-]\d+")
    lines = frd_path.read_text(errors="replace").splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(" -4") and ("FORC" in line or "RF" in line):
            i += 1
            while i < len(lines) and lines[i].startswith(" -5"):
                i += 1
            vals: dict[int, list[float]] = {}
            while i < len(lines) and not lines[i].startswith(" -3"):
                if lines[i].startswith(" -1"):
                    raw = lines[i][3:].lstrip()
                    m = num_re.search(raw)
                    if m:
                        nid = int(raw[: m.start()])
                        nums = [float(x) for x in num_re.findall(raw)]
                        if len(nums) >= 3:
                            vals[nid] = nums[:3]
                i += 1
            if vals:
                blocks.append(vals)
        else:
            i += 1
    if not blocks:
        return None, None, 0
    vals = blocks[-1]
    RF = np.zeros(3)
    M = np.zeros(3)
    n = 0
    for nid0 in prox_nodes0:
        nid = int(nid0) + 1
        if nid not in vals:
            continue
        R = np.asarray(vals[nid][:3], dtype=float)
        r = points[nid0] - prox_c
        RF += R
        M += np.cross(r, R)
        n += 1
    return M / 1000.0, RF, n


def parse_frd_nodal_von_mises(frd_path: Path, n_nodes: int) -> np.ndarray:
    """Parse CalculiX FRD STRESS as nodal (n_nodes entries), return von Mises."""
    num_re = re.compile(r"[+-]?\d+\.\d+E[+-]\d+")
    vm = np.full(n_nodes, np.nan, dtype=np.float64)
    in_stress = False
    n_read = 0
    with frd_path.open("r", errors="replace") as f:
        for line in f:
            if line.startswith(" -4") and "STRESS" in line:
                in_stress = True
                continue
            if not in_stress:
                continue
            if line.startswith(" -5"):
                continue
            if line.startswith(" -3"):
                break
            if not line.startswith(" -1"):
                continue
            body = line[3:].lstrip()
            m = num_re.search(body)
            if not m:
                continue
            nid = int(body[: m.start()])
            nums = [float(x) for x in num_re.findall(body)]
            if len(nums) < 6:
                continue
            sxx, syy, szz, sxy, syz, szx = nums[:6]
            v = math.sqrt(
                0.5
                * (
                    (sxx - syy) ** 2
                    + (syy - szz) ** 2
                    + (szz - sxx) ** 2
                    + 6.0 * (sxy**2 + syz**2 + szx**2)
                )
            )
            if 1 <= nid <= n_nodes:
                vm[nid - 1] = v
                n_read += 1
    if n_read == 0 or not np.isfinite(vm).any():
        raise RuntimeError(f"No STRESS parsed from {frd_path}")
    return vm


def classify_hotspot(
    points: np.ndarray,
    vm: np.ndarray,
    prox: np.ndarray,
    dist: np.ndarray,
    y0: float = Y0_NECK,
    L: float = L_BAND,
    halo_mm: float = 2.0,
) -> dict:
    prox_set = set(prox.tolist())
    dist_set = set(dist.tolist())
    bc = np.unique(np.concatenate([prox, dist]))

    # distances to BC (chunked)
    d_bc = np.full(len(points), np.inf)
    bc_pts = points[bc]
    chunk = 4000
    for a in range(0, len(points), chunk):
        b = min(a + chunk, len(points))
        sub = points[a:b]
        md = np.full(len(sub), np.inf)
        for c0 in range(0, len(bc_pts), 2000):
            c1 = min(c0 + 2000, len(bc_pts))
            d2 = np.sum((sub[:, None, :] - bc_pts[None, c0:c1, :]) ** 2, axis=2)
            md = np.minimum(md, d2.min(axis=1))
        d_bc[a:b] = np.sqrt(md)

    finite = np.isfinite(vm)
    # global max (exclude non-finite)
    imax = int(np.nanargmax(vm))
    vmax = float(vm[imax])

    mask_far = finite & (d_bc > halo_mm)
    imax_far = int(np.flatnonzero(mask_far)[np.argmax(vm[mask_far])])
    vmax_far = float(vm[imax_far])

    def min_d(idx: int, nodes: np.ndarray) -> float:
        return float(np.linalg.norm(points[nodes] - points[idx], axis=1).min())

    d_prox = min_d(imax_far, prox)
    d_dist = min_d(imax_far, dist)
    on_prox = imax_far in prox_set
    on_dist = imax_far in dist_set
    if on_prox:
        bc_class = "ON_SPC_PROXIMAL"
    elif on_dist:
        bc_class = "ON_DISTAL_COUPLING"
    elif min(d_prox, d_dist) < halo_mm:
        bc_class = "NEAR_BC_HALO"
    else:
        bc_class = "IN_BODY_AWAY_FROM_BC"

    y = float(points[imax_far, 1])
    dy = abs(y - y0)
    dy_over_L = dy / L if L > 0 else float("nan")
    if dy_over_L <= HOTSPOT_CENTER_FRAC:
        band_pos = "BAND_CENTER"
    elif dy_over_L >= HOTSPOT_EDGE_FRAC:
        band_pos = "BAND_EDGE"
    elif dy < L:
        band_pos = "BAND_MID"
    else:
        band_pos = "OUTSIDE_BAND"

    qs = (50, 95, 99, 100)
    pct = {f"p{q}": float(np.percentile(vm[finite], q)) for q in qs}
    pct_far = {f"p{q}": float(np.percentile(vm[mask_far], q)) for q in qs}

    return {
        "bc_classification": bc_class,
        "band_position": band_pos,
        "node_0based": imax_far,
        "xyz_mm": points[imax_far].tolist(),
        "y_mm": y,
        "dy_from_y0_mm": dy,
        "dy_over_L": dy_over_L,
        "von_mises_MPa": vmax_far,
        "global_max_MPa": vmax,
        "global_max_node": imax,
        "dist_to_SPC_mm": d_prox,
        "dist_to_distal_mm": d_dist,
        "percentiles_all": pct,
        "percentiles_exclude_2mm_BC": pct_far,
        "thresholds": {
            "center_if_dy_over_L_le": HOTSPOT_CENTER_FRAC,
            "edge_if_dy_over_L_ge": HOTSPOT_EDGE_FRAC,
            "L_mm": L,
            "y0_mm": y0,
        },
    }


def spc_band_clearance_report(
    points: np.ndarray, ends: dict, L: float = L_BAND, y0: float = Y0_NECK
) -> dict:
    yp = points[ends["prox_nodes"], 1]
    yd = points[ends["dist_nodes"], 1]
    band = [y0 - L, y0 + L]
    gap_prox = band[0] - float(yp.max())
    gap_dist = float(yd.min()) - band[1]
    n_overlap = int(((yp >= band[0]) & (yp <= band[1])).sum())
    return {
        "n_prox": int(len(ends["prox_nodes"])),
        "n_dist": int(len(ends["dist_nodes"])),
        "prox_y_mm": {
            "min": float(yp.min()),
            "max": float(yp.max()),
            "mean": float(yp.mean()),
        },
        "dist_y_mm": {
            "min": float(yd.min()),
            "max": float(yd.max()),
            "mean": float(yd.mean()),
        },
        "band_y_mm": band,
        "y0_mm": y0,
        "L_mm": L,
        "gap_prox_max_y_to_band_lo_mm": gap_prox,
        "gap_band_hi_to_dist_min_y_mm": gap_dist,
        "n_SPC_nodes_in_band": n_overlap,
        "safe_clearance": gap_prox > MIN_SPC_GAP_MM and n_overlap == 0 and gap_dist > 0,
    }


def run_one_sample(
    s: float,
    points0: np.ndarray,
    tets: np.ndarray,
    node_surf: np.ndarray,
    surf_faces: np.ndarray,
    dens0_t_mm3: float,
    axis_point: np.ndarray,
    axis_u: np.ndarray,
    ends0_prox_y: dict,
    reuse_fea: bool = False,
) -> dict:
    tag = f"s_{s:.2f}"
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)

    L = L_for_s(s)
    # Freeze SPC/distal nodes dari seleksi di mesh undeformed
    tet_ids0, face_ids0, face_nodes0 = _boundary_faces(tets)
    ends_undeformed = select_end_faces(
        points0, node_surf, tet_ids0, face_ids0, face_nodes0
    )
    freeze = np.zeros(len(points0), dtype=bool)
    freeze[ends_undeformed["prox_nodes"]] = True
    freeze[ends_undeformed["dist_nodes"]] = True
    points, dmeta = deform_neck_radial(
        points0, s, axis_point, axis_u, L=L, freeze_mask=freeze
    )
    tets_s, flip_meta = flip_negative_tets(points, tets)
    dmeta["tet_flip_repair"] = flip_meta
    gate = mesh_gates(points0, points, tets_s, surf_faces)

    # simpan mesh terdeform (volume saja; FEA tidak butuh surf_faces)
    np.savez_compressed(
        out / "geometry.npz",
        vol_points=points,
        vol_tets=tets_s,
        node_surf=node_surf,
    )
    # Note: surf_faces from baseline index into surf_vertices of baseline order;
    # for FEA we use vol mesh only. Keep deform meta.
    (out / "deform_meta.json").write_text(json.dumps({**dmeta, "mesh_gate": gate}, indent=2))

    sample: dict = {
        "s": s,
        "tag": tag,
        "status": "mesh_gate_fail" if not gate["pass"] else "pending_fea",
        "deform": dmeta,
        "mesh_gate": gate,
        "fail_reasons": list(gate["fail_reasons"]),
    }
    if not gate["pass"]:
        (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
        return sample

    # FEA setup on deformed mesh (SPC reselected — should not include deformed neck)
    tet_ids, face_ids, face_nodes = _boundary_faces(tets_s)
    ends = select_end_faces(points, node_surf, tet_ids, face_ids, face_nodes)
    # safety: SPC must still clear band
    clear = spc_band_clearance_report(points, ends, L=L)
    sample["spc_clearance"] = clear
    if not clear["safe_clearance"]:
        sample["status"] = "spc_band_overlap"
        sample["fail_reasons"].append("spc_band_overlap_after_deform")
        (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
        return sample

    u = ends["u"]
    z = np.array([0.0, 0.0, -1.0])
    grav = z - u * np.dot(z, u)
    if np.linalg.norm(grav) < 1e-6:
        grav = np.array([-1.0, 0.0, 0.0])
        grav = grav - u * np.dot(grav, u)
    grav = grav / np.linalg.norm(grav)

    F_N = M_DISTAL * G
    levers = [
        (M_FOREARM, X["forearm_com"] - X["elbow"]),
        (M_W1, X["w1_com"] - X["elbow"]),
        (M_W2, X["w2_com"] - X["elbow"]),
        (M_W3, X["w3_com"] - X["elbow"]),
        (M_PAYLOAD, X["payload"] - X["elbow"]),
    ]
    M_elbow_Nm = sum(m * G * lev for m, lev in levers)
    mdir = np.cross(u, grav)
    mdir = mdir / (np.linalg.norm(mdir) + 1e-30)
    M_dist_Nmm = mdir * (M_elbow_Nm * 1000.0)

    # densitas tetap dari baseline (massa skala dengan volume)
    dens = dens0_t_mm3
    vols = np.abs(_tet_volume_sign(points, tets_s))
    V_mm3 = float(vols.sum())
    dens_kg_m3 = dens / 1e-12
    m_self_eff = dens_kg_m3 * (V_mm3 * 1e-9)

    inp = out / "nauo3_cantilever.inp"
    write_cantilever_inp(inp, points, tets_s, ends, dens, F_N, M_dist_Nmm, grav)

    frd = out / "nauo3_cantilever.frd"
    if reuse_fea and frd.exists():
        sample["ccx_returncode"] = 0
        sample["ccx_reused"] = True
    else:
        proc = subprocess.run(
            [str(CCX), "-i", inp.stem],
            cwd=str(out),
            capture_output=True,
            text=True,
        )
        (out / "ccx_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
        sample["ccx_returncode"] = proc.returncode
        sample["ccx_reused"] = False
        if proc.returncode != 0:
            sample["status"] = "ccx_fail"
            sample["fail_reasons"].append(f"ccx_returncode={proc.returncode}")
            (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
            return sample

    M_fe_Nm, RF_fe, nrf = _rf_moment_from_forc(
        frd, points, ends["prox_nodes"], ends["prox_center"]
    )
    M_fe_norm = float(np.linalg.norm(M_fe_Nm)) if M_fe_Nm is not None else None
    step1 = step1_com_moment()
    M_analytic = step1["M_total_new_Nm"]
    centroid = np.average(points[tets_s].mean(axis=1), weights=vols, axis=0)
    x_com_mm = float(np.dot(centroid - ends["prox_center"], u))
    L_m = ends["L_mm"] / 1000.0
    x_com_m = x_com_mm / 1000.0
    M_fea_pred = m_self_eff * G * x_com_m + F_N * L_m + M_elbow_Nm

    err_a = abs(M_fe_norm - M_analytic) / M_analytic * 100.0 if M_fe_norm else None
    err_f = abs(M_fe_norm - M_fea_pred) / M_fea_pred * 100.0 if M_fe_norm else None
    reaction_pass = (
        M_fe_norm is not None
        and ((err_f is not None and err_f < REACTION_ERR_PCT) or (err_a is not None and err_a < REACTION_ERR_PCT))
    )
    sample["reaction"] = {
        "M_fe_Nm": M_fe_norm,
        "RF_N": RF_fe.tolist() if RF_fe is not None else None,
        "M_analytic_Nm": M_analytic,
        "M_fea_pred_Nm": M_fea_pred,
        "err_vs_analytic_pct": err_a,
        "err_vs_fea_pred_pct": err_f,
        "pass": reaction_pass,
        "n_prox_with_rf": nrf,
        "m_self_eff_kg": m_self_eff,
        "L_mm": ends["L_mm"],
        "x_com_from_prox_m": x_com_m,
    }
    if not reaction_pass:
        sample["status"] = "reaction_check_fail"
        sample["fail_reasons"].append(
            f"reaction_err analytic={err_a}% fea_pred={err_f}% (lim {REACTION_ERR_PCT}%)"
        )
        (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
        return sample

    vm = parse_frd_nodal_von_mises(frd, len(points))
    hot = classify_hotspot(
        points, vm, ends["prox_nodes"], ends["dist_nodes"], y0=Y0_NECK, L=L
    )
    # sigma max di pusat band (dy/L <= 0.35) — selalu dicatat
    cmask = (np.abs(points[:, 1] - Y0_NECK) <= HOTSPOT_CENTER_FRAC * L) & np.isfinite(
        vm
    )
    if cmask.any():
        ic = int(np.flatnonzero(cmask)[np.argmax(vm[cmask])])
        sigma_band_center = {
            "definition": "max von_mises among nodes with |y-y0|/L <= 0.35",
            "dy_over_L_threshold": HOTSPOT_CENTER_FRAC,
            "L_mm": L,
            "sigma_max_MPa": float(vm[cmask].max()),
            "node_0based": ic,
            "xyz_mm": points[ic].tolist(),
            "y_mm": float(points[ic, 1]),
            "dy_over_L": float(abs(points[ic, 1] - Y0_NECK) / L),
        }
    else:
        sigma_band_center = None
    sample["hotspot"] = hot
    sample["sigma_max_MPa"] = hot["von_mises_MPa"]
    sample["sigma_max_band_center"] = sigma_band_center
    sample["vs_old_baseline_MPa"] = {
        "old_baseline": BASELINE_SIGMA_MAX_MPA,
        "delta_MPa": hot["von_mises_MPa"] - BASELINE_SIGMA_MAX_MPA,
        "rel_pct": (hot["von_mises_MPa"] / BASELINE_SIGMA_MAX_MPA - 1.0) * 100.0,
    }

    # band_position = metadata informatif (bukan gate)
    sample["status"] = "ok"
    sample["band_position"] = hot["band_position"]

    (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
    (out / "stress_hotspot.json").write_text(json.dumps(hot, indent=2))
    return sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--samples",
        default="0.90,0.92,0.94,0.96,0.98,1.00,1.02,1.04,1.06,1.08,1.10",
        help="Comma-separated s values (default: full N=11 batch)",
    )
    ap.add_argument(
        "--reuse-fea",
        action="store_true",
        help="If nauo3_cantilever.frd already exists for a sample, skip ccx",
    )
    args = ap.parse_args()
    samples = [float(x) for x in args.samples.split(",") if x.strip()]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    g = np.load(MESH_BASE)
    points0 = np.asarray(g["vol_points"], dtype=np.float64).copy()
    tets = np.asarray(g["vol_tets"], dtype=np.int32)
    node_surf = np.asarray(g["node_surf"], dtype=bool)
    surf_faces = np.asarray(g["surf_faces"], dtype=np.int32)

    tet_ids, face_ids, face_nodes = _boundary_faces(tets)
    ends0 = select_end_faces(points0, node_surf, tet_ids, face_ids, face_nodes)
    clear0 = spc_band_clearance_report(points0, ends0, L=L_BAND)
    (OUT_ROOT / "spc_band_clearance.json").write_text(json.dumps(clear0, indent=2))
    print("=== SPC vs band clearance ===")
    print(json.dumps(clear0, indent=2))
    if not clear0["safe_clearance"]:
        print("STOP: band overlaps SPC — pilih L lebih kecil.")
        return 2

    # Axis: u from end selection; point = volume centroid
    u = ends0["u"]
    axis_point = points0.mean(0)
    # densitas baseline (tetap untuk semua s)
    vols0 = np.abs(_tet_volume_sign(points0, tets))
    V0 = float(vols0.sum())
    dens_kg_m3 = M_SELF / (V0 * 1e-9)
    dens0_t_mm3 = dens_kg_m3 * 1e-12
    print(f"baseline V={V0:.1f} mm3  dens={dens_kg_m3:.1f} kg/m3")
    print(f"falloff={FALLOFF_KIND}  L={L_BAND} mm  (locked for batch)")

    results = []
    for s in samples:
        print(f"\n######## sample s={s:.2f} ########")
        r = run_one_sample(
            s,
            points0,
            tets,
            node_surf,
            surf_faces,
            dens0_t_mm3,
            axis_point,
            u,
            clear0,
            reuse_fea=args.reuse_fea,
        )
        print(
            f"  status={r['status']}  sigma={r.get('sigma_max_MPa')}  "
            f"band={r.get('hotspot', {}).get('band_position')}  "
            f"fail={r.get('fail_reasons')}"
        )
        results.append(r)

    # Compact per-sample rows for batch_summary
    rows = []
    for r in results:
        hot = r.get("hotspot") or {}
        react = r.get("reaction") or {}
        rows.append(
            {
                "s": r["s"],
                "status": r["status"],
                "fail_reasons": r.get("fail_reasons") or [],
                "global_max_MPa": r.get("sigma_max_MPa"),
                "global_max_xyz_mm": hot.get("xyz_mm"),
                "global_max_y_mm": hot.get("y_mm"),
                "band_position": hot.get("band_position"),
                "reaction_pass": react.get("pass"),
                "reaction_M_Nm": react.get("M_fe_Nm"),
                "reaction_err_vs_analytic_pct": react.get("err_vs_analytic_pct"),
                "mesh_gate_pass": (r.get("mesh_gate") or {}).get("pass"),
                "vs_old_baseline_rel_pct": (r.get("vs_old_baseline_MPa") or {}).get(
                    "rel_pct"
                ),
            }
        )

    summary = {
        "approach": "a_mesh_deform_neck_radial",
        "falloff_kind": FALLOFF_KIND,
        "L_mm": L_BAND,
        "L_locked": True,
        "L_lock_note": (
            "L=40 mm cosine C1 dikunci untuk seluruh batch. "
            "Sisi s>1 sensitif terhadap pilihan L (bahu/fillet); "
            "jangan ganti L tanpa redefinisi parameter dataset."
        ),
        "y0_mm": Y0_NECK,
        "qoi": "global_max_von_mises (sah untuk seluruh rentang s, termasuk bahu s>1)",
        "spc_band_clearance": clear0,
        "density_policy": "fixed_from_baseline_s1_undeformed",
        "samples_requested": samples,
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_fail": sum(1 for r in results if r["status"] != "ok"),
        "failures": [
            {"s": r["s"], "status": r["status"], "fail_reasons": r["fail_reasons"]}
            for r in results
            if r["status"] != "ok"
        ],
        "samples": rows,
        "sigma_max_band_center_by_s": {
            f"{r['s']:.2f}": r.get("sigma_max_band_center") for r in results
        },
        "results": results,
        "s1_sanity_check": None,
    }
    # s=1.0 sanity vs old baseline
    for r in results:
        if abs(r["s"] - 1.0) < 1e-9 and r.get("sigma_max_MPa") is not None:
            summary["s1_sanity_check"] = {
                "sigma_max_MPa": r["sigma_max_MPa"],
                "old_baseline_MPa": BASELINE_SIGMA_MAX_MPA,
                "delta_MPa": r["sigma_max_MPa"] - BASELINE_SIGMA_MAX_MPA,
                "rel_pct": (r["sigma_max_MPa"] / BASELINE_SIGMA_MAX_MPA - 1.0) * 100.0,
                "band_position": (r.get("hotspot") or {}).get("band_position"),
                "pass_near_baseline": abs(
                    r["sigma_max_MPa"] - BASELINE_SIGMA_MAX_MPA
                )
                / BASELINE_SIGMA_MAX_MPA
                < 0.05,
            }

    (OUT_ROOT / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(f"n_ok={summary['n_ok']} n_fail={summary['n_fail']}")
    if summary["s1_sanity_check"]:
        print("s=1.0 sanity:", json.dumps(summary["s1_sanity_check"], indent=2))
    print("wrote", OUT_ROOT / "batch_summary.json")
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
