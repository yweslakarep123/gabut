#!/usr/bin/env python3
"""Phase-2 pilot: compose two independent neck radial scales (center + shoulder).

Does NOT modify deform_neck_radial — applies it twice in sequence.
Shoulder (y0, L) taken from existing s>1 hotspot diagnostics, not invented.

Pilot combinations (stop after these four — no full grid/optimizer):
  1. (1.00, 1.00) — identity compose; must match baseline
  2. (0.90, 1.00) — shoulder no-op; must match Phase-1 s=0.90
  3. (0.90, 1.10) — thin center, thicken shoulder
  4. (0.95, 1.05) — moderate version of (3)

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/pilot_nauo3_neck_2d.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from batch_nauo3_neck_scale import (  # noqa: E402
    BASELINE_SIGMA_MAX_MPA,
    L_BAND,
    MESH_BASE,
    MIN_SPC_GAP_MM,
    OUT_ROOT as BATCH_OUT_ROOT,
    Y0_NECK,
    _boundary_faces,
    run_one_sample,
    spc_band_clearance_report,
)
from fea_nauo3_cantilever import (  # noqa: E402
    M_SELF,
    _tet_volume_sign,
    select_end_faces,
)

PILOT_OUT = ROOT / "reports" / "ur5e_nauo3_neck_pilot2d"
PILOT_LOG = PILOT_OUT / "pilot_log.jsonl"
SHOULDER_HOTSPOT_JSON = BATCH_OUT_ROOT / "s_1.10" / "stress_hotspot.json"
SPC_CLEARANCE_JSON = BATCH_OUT_ROOT / "spc_band_clearance.json"
PHASE1_S090_JSON = BATCH_OUT_ROOT / "s_0.90" / "sample_result.json"

# Sanity tolerances (relative)
SIGMA_MATCH_RTOL = 0.02  # 2% — compose identity / Phase-1 replay
VOL_MATCH_RTOL = 1e-4


def load_shoulder_band() -> dict:
    """y0/L for shoulder stage from on-disk diagnostics (not assumed)."""
    if not SHOULDER_HOTSPOT_JSON.is_file():
        raise FileNotFoundError(
            f"missing shoulder hotspot diagnostic: {SHOULDER_HOTSPOT_JSON}"
        )
    hot = json.loads(SHOULDER_HOTSPOT_JSON.read_text())
    y0_bahu = float(hot["y_mm"])

    if SPC_CLEARANCE_JSON.is_file():
        clear0 = json.loads(SPC_CLEARANCE_JSON.read_text())
        prox_ymax = float(clear0["prox_y_mm"]["max"])
    else:
        # Fallback: will be recomputed from mesh in main()
        prox_ymax = None

    # Characteristic half-width = distance hotspot → locked neck center,
    # capped so band_lo clears SPC by the same MIN_SPC_GAP_MM gate.
    L_from_geometry = abs(Y0_NECK - y0_bahu)
    source = {
        "y0_source": str(SHOULDER_HOTSPOT_JSON.relative_to(ROOT)),
        "y0_mm": y0_bahu,
        "hotspot_band_position": hot.get("band_position"),
        "hotspot_von_mises_MPa": hot.get("von_mises_MPa"),
        "L_from_geometry_mm": L_from_geometry,
        "L_geometry_note": (
            "|Y0_NECK - y_hotspot_s1.10|; locked neck band edge is at "
            f"Y0_NECK-L_BAND={Y0_NECK - L_BAND}"
        ),
    }
    if prox_ymax is not None:
        # Gate is strict: gap_prox > MIN_SPC_GAP_MM (not >=). Keep 0.5 mm margin.
        L_spc_cap = y0_bahu - prox_ymax - MIN_SPC_GAP_MM - 0.5
        L_bahu = min(L_from_geometry, L_spc_cap)
        source.update(
            {
                "prox_y_max_mm": prox_ymax,
                "L_spc_cap_mm": L_spc_cap,
                "L_mm": L_bahu,
                "L_choice_note": (
                    "min(L_from_geometry, y0_bahu - prox_ymax - MIN_SPC_GAP_MM - 0.5) "
                    "so shoulder band clears the existing SPC gate (strict >)"
                ),
            }
        )
    else:
        L_bahu = L_from_geometry
        source.update({"L_mm": L_bahu, "L_choice_note": "geometry-only (no SPC file)"})

    if L_bahu <= 0:
        raise RuntimeError(
            f"derived L_bahu={L_bahu} non-positive; shoulder band cannot clear SPC"
        )
    return source


def append_log(row: dict) -> None:
    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    with PILOT_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    shoulder = load_shoulder_band()
    y0_bahu = float(shoulder["y0_mm"])
    L_bahu = float(shoulder["L_mm"])

    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    if PILOT_LOG.exists():
        PILOT_LOG.unlink()

    g = np.load(MESH_BASE)
    points0 = np.asarray(g["vol_points"], dtype=np.float64).copy()
    tets = np.asarray(g["vol_tets"], dtype=np.int32)
    node_surf = np.asarray(g["node_surf"], dtype=bool)
    surf_faces = np.asarray(g["surf_faces"], dtype=np.int32)

    tet_ids, face_ids, face_nodes = _boundary_faces(tets)
    ends0 = select_end_faces(points0, node_surf, tet_ids, face_ids, face_nodes)
    clear0 = spc_band_clearance_report(points0, ends0, L=L_BAND)
    if not clear0["safe_clearance"]:
        print("STOP: baseline SPC/neck-band clearance failed")
        return 2

    # Recompute SPC-safe L with live mesh if JSON was stale
    prox_ymax = float(clear0["prox_y_mm"]["max"])
    L_spc_cap = y0_bahu - prox_ymax - MIN_SPC_GAP_MM - 0.5
    L_from_geometry = abs(Y0_NECK - y0_bahu)
    L_bahu = min(L_from_geometry, L_spc_cap)
    shoulder["prox_y_max_mm"] = prox_ymax
    shoulder["L_spc_cap_mm"] = L_spc_cap
    shoulder["L_mm"] = L_bahu
    (PILOT_OUT / "shoulder_band.json").write_text(json.dumps(shoulder, indent=2))

    vols0 = np.abs(_tet_volume_sign(points0, tets))
    V0 = float(vols0.sum())
    dens_kg_m3 = M_SELF / (V0 * 1e-9)
    dens0_t_mm3 = dens_kg_m3 * 1e-12
    axis_point = points0.mean(0)
    axis_u = ends0["u"]

    phase1_s090 = None
    if PHASE1_S090_JSON.is_file():
        phase1_s090 = json.loads(PHASE1_S090_JSON.read_text())

    print("=== NAUO3 2D compose pilot (center + shoulder) ===")
    print(f"Y0_NECK={Y0_NECK}  L_BAND={L_BAND}")
    print(
        f"Y0_SHOULDER={y0_bahu:.6f}  L_SHOULDER={L_bahu:.6f}  "
        f"(from {shoulder['y0_source']})"
    )
    print(f"baseline V0={V0:.1f} mm3  sigma_allow={BASELINE_SIGMA_MAX_MPA}")
    print(f"log → {PILOT_LOG}")

    pilots = [
        {
            "id": "sanity_identity",
            "s_center": 1.00,
            "s_shoulder": 1.00,
            "note": "compose identity; must match baseline sigma/volume",
        },
        {
            "id": "sanity_phase1_s090",
            "s_center": 0.90,
            "s_shoulder": 1.00,
            "note": "shoulder no-op; must match Phase-1 s=0.90 sigma",
        },
        {
            "id": "thin_center_thick_shoulder",
            "s_center": 0.90,
            "s_shoulder": 1.10,
            "note": "tipiskan tengah, tebalkan bahu",
        },
        {
            "id": "moderate_compose",
            "s_center": 0.95,
            "s_shoulder": 1.05,
            "note": "versi moderat dari kombinasi thin-center/thick-shoulder",
        },
    ]

    rows: list[dict] = []
    for p in pilots:
        sc, ss = float(p["s_center"]), float(p["s_shoulder"])
        tag = f"sc{sc:.2f}_ss{ss:.2f}"
        stages = [
            (sc, Y0_NECK, L_BAND),
            (ss, y0_bahu, L_bahu),
        ]
        print(f"\n######## {p['id']}  {tag}  ########")
        print(f"  stages={stages}")
        t0 = time.perf_counter()
        sample = run_one_sample(
            stages,
            points0,
            tets,
            node_surf,
            surf_faces,
            dens0_t_mm3,
            axis_point,
            axis_u,
            clear0,
            reuse_fea=False,
            tag=tag,
            out_root=PILOT_OUT,
        )
        wall = time.perf_counter() - t0

        gate = sample.get("mesh_gate") or {}
        V = gate.get("V_mm3")
        sigma = sample.get("sigma_max_MPa")
        status = sample.get("status")
        dV_frac = None if V is None else (V - V0) / V0
        mass_below_baseline = bool(V is not None and V < V0 * (1.0 - 1e-12))
        stress_ok = bool(
            sigma is not None and sigma <= BASELINE_SIGMA_MAX_MPA * (1.0 + 1e-12)
        )
        slack_2d = bool(
            status == "ok" and mass_below_baseline and stress_ok
        )

        sanity = {}
        if p["id"] == "sanity_identity" and status == "ok" and sigma is not None:
            sanity = {
                "sigma_vs_baseline_rel": (sigma / BASELINE_SIGMA_MAX_MPA) - 1.0,
                "volume_vs_V0_rel": dV_frac,
                "pass_sigma": abs(sigma - BASELINE_SIGMA_MAX_MPA)
                / BASELINE_SIGMA_MAX_MPA
                <= SIGMA_MATCH_RTOL,
                "pass_volume": abs(dV_frac or 1.0) <= VOL_MATCH_RTOL,
            }
            sanity["pass"] = bool(sanity["pass_sigma"] and sanity["pass_volume"])
        if p["id"] == "sanity_phase1_s090" and status == "ok" and sigma is not None:
            ref = (phase1_s090 or {}).get("sigma_max_MPa")
            if ref is not None:
                sanity = {
                    "phase1_s090_sigma_MPa": ref,
                    "sigma_vs_phase1_rel": (sigma / ref) - 1.0,
                    "pass": abs(sigma - ref) / ref <= SIGMA_MATCH_RTOL,
                }

        row = {
            "id": p["id"],
            "tag": tag,
            "note": p["note"],
            "s_center": sc,
            "s_shoulder": ss,
            "y0_center_mm": Y0_NECK,
            "L_center_mm": L_BAND,
            "y0_shoulder_mm": y0_bahu,
            "L_shoulder_mm": L_bahu,
            "status": status,
            "fail_reasons": sample.get("fail_reasons") or [],
            "sigma_max_MPa": sigma,
            "sigma_allow_MPa": BASELINE_SIGMA_MAX_MPA,
            "V_mm3": V,
            "V0_mm3": V0,
            "dV_frac": dV_frac,
            "mass_below_baseline": mass_below_baseline,
            "stress_within_baseline": stress_ok,
            "slack_2d_candidate": slack_2d,
            "mesh_gate_pass": gate.get("pass"),
            "reaction_pass": (sample.get("reaction") or {}).get("pass"),
            "band_position": sample.get("band_position"),
            "wall_time_s": wall,
            "sanity": sanity,
            "sample_result_path": str((PILOT_OUT / tag / "sample_result.json").relative_to(ROOT)),
        }
        append_log(row)
        rows.append(row)
        print(
            f"  status={status}  sigma={sigma}  V={V}  dV_frac={dV_frac}  "
            f"slack_2d={slack_2d}  wall={wall:.1f}s  sanity={sanity}"
        )

    slack = [r for r in rows if r["slack_2d_candidate"]]
    sanity_rows = [r for r in rows if r["id"].startswith("sanity_")]
    summary = {
        "pilot": "nauo3_neck_2d_compose",
        "n_samples": len(rows),
        "shoulder_band": shoulder,
        "baseline_sigma_max_MPa": BASELINE_SIGMA_MAX_MPA,
        "V0_mm3": V0,
        "sanity_checks": {
            r["id"]: r.get("sanity") for r in sanity_rows
        },
        "slack_2d_found": len(slack) > 0,
        "slack_2d_samples": slack,
        "verdict": (
            "2d_slack_found"
            if slack
            else "no_2d_slack_in_pilot_combinations"
        ),
        "message": (
            (
                f"Found {len(slack)} pilot combination(s) with volume below "
                f"baseline AND sigma_max <= baseline — 2D compose opens slack "
                f"not seen in the 1D family."
            )
            if slack
            else (
                "No pilot combination achieved lower volume than baseline while "
                "keeping sigma_max <= baseline. Within these 4 compose samples, "
                "2D does not yet demonstrate free thinning slack. "
                "Gate failures (if any) are retained in pilot_log.jsonl."
            )
        ),
        "samples": rows,
    }
    (PILOT_OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== PILOT SUMMARY ===")
    print(f"verdict: {summary['verdict']}")
    print(summary["message"])
    for r in rows:
        print(
            f"  {r['tag']}: status={r['status']} sigma={r['sigma_max_MPa']} "
            f"dV_frac={r['dV_frac']} slack={r['slack_2d_candidate']}"
        )
    print(f"wrote {PILOT_LOG}")
    print(f"wrote {PILOT_OUT / 'pilot_summary.json'}")

    # Non-zero exit if identity/phase1 sanity failed hard
    for r in sanity_rows:
        san = r.get("sanity") or {}
        if r["status"] == "ok" and san and not san.get("pass", True):
            print(f"SANITY FAIL: {r['id']} → {san}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
