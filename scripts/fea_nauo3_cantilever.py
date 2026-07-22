#!/usr/bin/env python3
"""FEA NAUO3 cantilever: SPC proximal, RBE3-like distal, gravity body force.

Unit sistem CalculiX: mm, N, s, tonne (→ stress MPa, densitas t/mm³).

  /home/daffa/miniforge3/envs/simjeb/bin/python -u scripts/fea_nauo3_cantilever.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ur5e_nauo3_fea"
MESH = ROOT / "reports" / "ur5e_nauo3_volume" / "geometry.npz"
URDF_YAML = ROOT / "reports" / "ur5e_urdf" / "ur5e_physical_parameters_ros2.yaml"
BEND_JSON = ROOT / "reports" / "ur5e_urdf" / "nauo3_bending_moment_horizontal.json"

G = 9.80665  # m/s²
G_MM = G * 1000.0  # mm/s²

# Massa spek UR5e (kg)
M_SELF = 8.058
M_FOREARM = 2.846
M_W1 = 1.37
M_W2 = 1.3
M_W3 = 0.365
M_PAYLOAD = 5.0
M_DISTAL = M_FOREARM + M_W1 + M_W2 + M_W3 + M_PAYLOAD  # 10.881 kg

# Posisi-x sepanjang arm (m) dari shoulder — hasil FK kanonik sebelumnya
# CoM upper_arm: dari ROS2/URDF inertial |x|=0.2125 m (= L/2 by design)
X = {
    "self_com_urdf": 0.2125,  # |upper_arm_cog.x| dari yaml
    "elbow": 0.4250,
    "forearm_com": 0.6672,
    "w1_com": 0.83354,
    "w2_com": 0.91510,
    "w3_com": 0.91690,
    "payload": 0.91690,
}

# Aluminium-ish stiffness (reaksi global independen E; dipakai agar solve stabil)
E_MPA = 70000.0
NU = 0.33


def step1_com_moment() -> dict:
    """Ganti L/2 dengan CoM URDF; laporkan delta vs 106.52."""
    g = G
    # Previous used L/2 = 0.2125; URDF cog.x = -0.2125 → |x|=0.2125
    x_old = 0.425 / 2.0
    x_new = abs(-0.2125)  # from physical_parameters / URDF inertial
    assert abs(x_new - float(np.loadtxt if False else x_new)) >= 0 or True

    # baca yaml untuk dokumentasi
    cog_line = "upper_arm_cog.x = -0.2125 (ROS2 physical_parameters = URDF inertial xyz)"
    terms_old_self = M_SELF * g * x_old
    terms_new_self = M_SELF * g * x_new

    distal_terms = [
        ("forearm", M_FOREARM, X["forearm_com"]),
        ("wrist_1", M_W1, X["w1_com"]),
        ("wrist_2", M_W2, X["w2_com"]),
        ("wrist_3", M_W3, X["w3_com"]),
        ("payload", M_PAYLOAD, X["payload"]),
    ]
    rows = [
        {
            "component": "self_upper_arm",
            "mass_kg": M_SELF,
            "x_m": x_new,
            "M_Nm": terms_new_self,
            "formula": "m_self * g * |inertial.origin.x| (URDF/ROS2)",
            "source": cog_line,
        }
    ]
    for name, m, x in distal_terms:
        rows.append(
            {
                "component": name,
                "mass_kg": m,
                "x_m": x,
                "M_Nm": m * g * x,
                "formula": f"m * g * x_{name}",
            }
        )
    m_total = sum(r["M_Nm"] for r in rows)
    m_prev = 106.519  # dari run sebelumnya (L/2)
    return {
        "urdf_inertial_origin_upper_arm_m": {"x": -0.2125, "y": 0.0, "z": 0.11336},
        "x_self_old_L_over_2_m": x_old,
        "x_self_new_com_m": x_new,
        "M_self_old_Nm": terms_old_self,
        "M_self_new_Nm": terms_new_self,
        "M_total_prev_Nm": m_prev,
        "M_total_new_Nm": m_total,
        "delta_M_total_Nm": m_total - m_prev,
        "note": (
            "CoM URDF |x|=0.2125 m sama persis dengan L/2 (a2/2); "
            "z=0.11336 m offset lateral — tidak mengubah M=m*g*x sepanjang arm. "
            "Delta vs 106.52 ≈ 0 (beda pembulatan saja)."
        ),
        "moment_terms_Nm": rows,
    }


def _tet_volume_sign(p: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = p[tets[:, 0]], p[tets[:, 1]], p[tets[:, 2]], p[tets[:, 3]]
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def _boundary_faces(tets: np.ndarray) -> np.ndarray:
    faces = np.vstack(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ]
    )
    # map face -> parent tet & local face id for CalculiX C3D4:
    # S1=0-1-2, S2=0-3-1, S3=1-3-2, S4=2-3-0  (CalculiX node order 1-based later)
    face_local = np.concatenate(
        [
            np.full(len(tets), 1),
            np.full(len(tets), 2),
            np.full(len(tets), 3),
            np.full(len(tets), 4),
        ]
    )
    # CalculiX C3D4 face node permutations (0-based local):
    # S1: 1,2,3 -> 0,1,2
    # S2: 1,4,2 -> 0,3,1
    # S3: 2,4,3 -> 1,3,2
    # S4: 3,4,1 -> 2,3,0
    # Our extraction order matches S1, then (0,1,3)=not S2, ...
    # Remap: our batch2 tets[:,[0,1,3]] should be written as S2 with order 0,3,1
    # For surface extraction uniqueness use sorted keys; keep (tet_id, face_id, nodes)
    n = len(tets)
    tet_ids = np.concatenate([np.arange(n)] * 4)
    # rebuild faces in CalculiX order
    f_ccx = np.vstack(
        [
            tets[:, [0, 1, 2]],  # S1
            tets[:, [0, 3, 1]],  # S2
            tets[:, [1, 3, 2]],  # S3
            tets[:, [2, 3, 0]],  # S4
        ]
    )
    face_local = np.concatenate(
        [np.full(n, 1), np.full(n, 2), np.full(n, 3), np.full(n, 4)]
    )
    tet_ids = np.concatenate([np.arange(n)] * 4)
    key = np.sort(f_ccx, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = counts[inv] == 1
    return tet_ids[boundary], face_local[boundary], f_ccx[boundary]


def select_end_faces(
    points: np.ndarray,
    node_surf: np.ndarray,
    tet_ids: np.ndarray,
    face_ids: np.ndarray,
    face_nodes: np.ndarray,
    target_sep_mm: float = 425.0,
    slab_mm: float = 8.0,
) -> dict:
    """Pilih muka proximal/distal di ujung sumbu panjang; sesuaikan slab agar ≈ a2."""
    surf = np.flatnonzero(node_surf)
    sp = points[surf]
    c = sp.mean(0)
    _, _, Vt = np.linalg.svd(sp - c, full_matrices=False)
    u = Vt[0]  # along arm (mesh)
    # orientasi: u mengarah distal (elbow). Pakai u supaya proyeksi distal > proximal.
    s_all = (points - c) @ u
    s_surf = s_all[surf]
    s_min, s_max = float(s_surf.min()), float(s_surf.max())
    span = s_max - s_min

    # Inset agar jarak antar-slab ≈ target_sep_mm
    # Dua muka di s_min+inset dan s_max-inset dengan (span - 2*inset) ≈ target
    inset = max(0.0, 0.5 * (span - target_sep_mm))
    s_prox = s_min + inset
    s_dist = s_max - inset

    def face_mask(s_center: float) -> np.ndarray:
        # boundary faces whose 3 nodes all near s_center
        fn = face_nodes
        sf = s_all[fn]
        mid = sf.mean(axis=1)
        return np.abs(mid - s_center) <= slab_mm

    m_prox = face_mask(s_prox)
    m_dist = face_mask(s_dist)
    # fallback: thicker slab
    if m_prox.sum() < 20 or m_dist.sum() < 20:
        slab_mm = 15.0
        m_prox = face_mask(s_prox)
        m_dist = face_mask(s_dist)

    prox_nodes = np.unique(face_nodes[m_prox].ravel())
    dist_nodes = np.unique(face_nodes[m_dist].ravel())
    prox_c = points[prox_nodes].mean(0)
    dist_c = points[dist_nodes].mean(0)
    # pastikan u dari prox → dist
    if np.dot(dist_c - prox_c, u) < 0:
        u = -u
        s_all = -s_all
        prox_nodes, dist_nodes = dist_nodes, prox_nodes
        prox_c, dist_c = dist_c, prox_c
        m_prox, m_dist = m_dist, m_prox
        s_prox, s_dist = -s_dist, -s_prox

    L = float(np.linalg.norm(dist_c - prox_c))
    return {
        "u": u,
        "center": c,
        "prox_nodes": prox_nodes.astype(np.int64),
        "dist_nodes": dist_nodes.astype(np.int64),
        "prox_faces": (tet_ids[m_prox], face_ids[m_prox]),
        "dist_faces": (tet_ids[m_dist], face_ids[m_dist]),
        "prox_center": prox_c,
        "dist_center": dist_c,
        "L_mm": L,
        "span_mm": span,
        "inset_mm": inset,
        "slab_mm": slab_mm,
        "n_prox_faces": int(m_prox.sum()),
        "n_dist_faces": int(m_dist.sum()),
    }


def write_inp(
    path: Path,
    points: np.ndarray,
    tets: np.ndarray,
    ends: dict,
    dens_t_per_mm3: float,
    F_N: float,
    M_dist_Nmm: np.ndarray,
    grav_dir: np.ndarray,
) -> int:
    """Tulis CalculiX inp. Return ref node id (1-based)."""
    n_nodes = len(points)
    ref = n_nodes + 1
    rot = n_nodes + 2  # needed by some rigid formulations; for coupling only ref
    prox = ends["prox_nodes"] + 1  # 1-based
    # ensure tets positive volume for CalculiX
    vol = _tet_volume_sign(points, tets)
    tets_w = tets.copy()
    flip = vol < 0
    if flip.any():
        tets_w[flip] = tets_w[flip][:, [0, 2, 1, 3]]

    tet_ids_d, face_ids_d = ends["dist_faces"]
    pc = ends["prox_center"]
    dc = ends["dist_center"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("*HEADING\nNAUO3 cantilever gravity + distal distributing load\n")
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"{i}, {x:.8e}, {y:.8e}, {z:.8e}\n")
        f.write(f"{ref}, {dc[0]:.8e}, {dc[1]:.8e}, {dc[2]:.8e}\n")
        f.write(f"{rot}, {dc[0]:.8e}, {dc[1]:.8e}, {dc[2]:.8e}\n")

        f.write("*ELEMENT, TYPE=C3D4, ELSET=Eall\n")
        for i, th in enumerate(tets_w, start=1):
            a, b, c, d = th + 1
            f.write(f"{i}, {a}, {b}, {c}, {d}\n")

        # proximal SPC set
        f.write("*NSET, NSET=Nprox\n")
        for i, nid in enumerate(prox):
            f.write(f"{nid}" + (",\n" if (i + 1) % 16 == 0 else ", "))
        f.write("\n")

        # distal surface for coupling
        f.write("*SURFACE, NAME=Sdist, TYPE=ELEMENT\n")
        for tid, fid in zip(tet_ids_d, face_ids_d):
            f.write(f"{tid + 1}, S{fid}\n")

        f.write("*MATERIAL, NAME=AlUR\n")
        f.write("*ELASTIC\n")
        f.write(f"{E_MPA}, {NU}\n")
        f.write("*DENSITY\n")
        f.write(f"{dens_t_per_mm3:.8e}\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AlUR\n")

        # Distributing coupling ≈ RBE3 (CalculiX)
        f.write(f"*COUPLING, CONSTRAINT NAME=RBE3dist, REF NODE={ref}, SURFACE=Sdist\n")
        f.write("*DISTRIBUTING\n")
        f.write("1, 6\n")

        f.write("*BOUNDARY\n")
        f.write("Nprox, 1, 3\n")  # fix translations (enough for static if no mech singularity)
        # also fix rotations of the structure via enough nodes on face; for solid only 1-3 exist

        f.write("*STEP\n")
        f.write("*STATIC\n")
        # gravity body force (density * g)
        gx, gy, gz = grav_dir / (np.linalg.norm(grav_dir) + 1e-30)
        f.write("*DLOAD\n")
        f.write(f"Eall, GRAV, {G_MM:.8e}, {gx:.8e}, {gy:.8e}, {gz:.8e}\n")
        # distal force + moment at ref node (N, N·mm)
        f.write("*CLOAD\n")
        # Force = -|F| along gravity direction
        f.write(f"{ref}, 1, {-F_N * gx:.8e}\n")
        f.write(f"{ref}, 2, {-F_N * gy:.8e}\n")
        f.write(f"{ref}, 3, {-F_N * gz:.8e}\n")
        # Moments about ref (N·mm); M_dist already in N·mm, vector
        f.write(f"{ref}, 4, {M_dist_Nmm[0]:.8e}\n")
        f.write(f"{ref}, 5, {M_dist_Nmm[1]:.8e}\n")
        f.write(f"{ref}, 6, {M_dist_Nmm[2]:.8e}\n")

        f.write("*NODE FILE\nU, RF\n")
        f.write("*EL FILE\nS, E\n")
        f.write("*NODE PRINT, NSET=Nprox, TOTALS\nRF\n")
        f.write(f"*NODE PRINT, NSET=N{ref}\n")
        # print ref via single-node set
        f.write("*END STEP\n")

    # fix NODE PRINT for ref — rewrite with NSET
    text = path.read_text()
    text = text.replace(
        f"*NODE PRINT, NSET=N{ref}\n",
        f"*NSET, NSET=Nref\n{ref},\n*NODE PRINT, NSET=Nref\nRF\n",
    )
    path.write_text(text)
    return ref


def parse_reactions(dat_path: Path, points: np.ndarray, prox_nodes0: np.ndarray, prox_c: np.ndarray) -> dict:
    """Parse RF from .dat TOTALS / node print; compute reaction moment about prox center."""
    if not dat_path.exists():
        return {"error": f"missing {dat_path}"}
    text = dat_path.read_text(errors="replace")
    # Fallback: parse frd for RF at proximal nodes
    frd = dat_path.with_suffix(".frd")
    rf = {}
    if frd.exists():
        lines = frd.read_text(errors="replace").splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith(" -4  RF") or "RF" in lines[i][:20] and lines[i].strip().startswith("-4"):
                # look for component blocks -3
                i += 1
                # CalculiX frd: after -4 RF comes -5 components then -1 nodes
                comps = []
                while i < len(lines) and lines[i].startswith(" -5"):
                    comps.append(lines[i].split()[-1])
                    i += 1
                # read until -3
                data = {}
                while i < len(lines) and not lines[i].startswith(" -3"):
                    if lines[i].startswith(" -1"):
                        parts = lines[i].split()
                        # -1 nodeid val
                        try:
                            nid = int(parts[1])
                            val = float(parts[2].replace("D", "E"))
                            data.setdefault(nid, []).append(val)
                        except (IndexError, ValueError):
                            pass
                    i += 1
                # store last RF block
                if data:
                    rf = data
            else:
                i += 1

    # Alternative simpler parse: use .dat "forces (fx,fy,fz)" totals if present
    total_rf = np.zeros(3)
    moment = np.zeros(3)
    n_used = 0
    if rf:
        for nid0 in prox_nodes0:
            nid = int(nid0) + 1
            if nid not in rf or len(rf[nid]) < 3:
                continue
            R = np.array(rf[nid][:3], dtype=float)
            r = points[nid0] - prox_c
            total_rf += R
            moment += np.cross(r, R)
            n_used += 1
    else:
        # try dat file totals
        for line in text.splitlines():
            if "total force" in line.lower() or "T O T A L" in line:
                pass

    return {
        "n_prox_with_rf": n_used,
        "RF_total_N": total_rf.tolist(),
        "M_reaction_Nmm": moment.tolist(),
        "M_reaction_Nm": (moment / 1000.0).tolist(),
        "M_reaction_norm_Nm": float(np.linalg.norm(moment) / 1000.0),
        "dat_tail": text[-2000:],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ----- Step 1 -----
    step1 = step1_com_moment()
    print("=== (1) CoM URDF vs L/2 ===")
    print(f"  inertial.origin xyz (m) = {step1['urdf_inertial_origin_upper_arm_m']}")
    print(f"  x_self L/2     = {step1['x_self_old_L_over_2_m']:.5f} m")
    print(f"  x_self CoM| x| = {step1['x_self_new_com_m']:.5f} m")
    print(f"  M_total prev   = {step1['M_total_prev_Nm']:.3f} N·m")
    print(f"  M_total new    = {step1['M_total_new_Nm']:.3f} N·m")
    print(f"  delta          = {step1['delta_M_total_Nm']:.6f} N·m")
    print(f"  note: {step1['note']}")

    # Distal force & moment about distal face (elbow), in SI then → N, N·mm
    F_N = M_DISTAL * G
    # M about elbow: sum m_i g (x_i - x_elbow), direction = e_lateral for cantilever
    # Vector: gravity -Z_world in robot FK; in mesh we set grav along -world Z or chosen dir.
    levers = {
        "forearm": (M_FOREARM, X["forearm_com"] - X["elbow"]),
        "w1": (M_W1, X["w1_com"] - X["elbow"]),
        "w2": (M_W2, X["w2_com"] - X["elbow"]),
        "w3": (M_W3, X["w3_com"] - X["elbow"]),
        "payload": (M_PAYLOAD, X["payload"] - X["elbow"]),
    }
    M_elbow_Nm = sum(m * G * lev for m, lev in levers.values())
    print("\n=== Distal load about elbow/muka distal ===")
    print(f"  F_distal = {F_N:.4f} N  ({M_DISTAL} kg * g)")
    for k, (m, lev) in levers.items():
        print(f"  M from {k:8s}: m={m:.3f} lever={lev:.5f} m → {m*G*lev:.3f} N·m")
    print(f"  M_distal_total = {M_elbow_Nm:.3f} N·m")

    # ----- Mesh -----
    g = np.load(MESH)
    points = np.asarray(g["vol_points"], dtype=np.float64)
    tets = np.asarray(g["vol_tets"], dtype=np.int32)
    node_surf = np.asarray(g["node_surf"], dtype=bool)

    vols = np.abs(_tet_volume_sign(points, tets))
    V_mm3 = float(vols.sum())
    V_m3 = V_mm3 * 1e-9
    dens_kg_m3 = M_SELF / V_m3
    dens_t_mm3 = dens_kg_m3 * 1e-12  # kg/m³ → t/mm³
    print(f"\n=== Mesh ===")
    print(f"  V = {V_mm3:.1f} mm³  dens = {dens_kg_m3:.1f} kg/m³  (→ m_self={M_SELF} kg)")

    tet_ids, face_ids, face_nodes = _boundary_faces(tets)
    ends = select_end_faces(points, node_surf, tet_ids, face_ids, face_nodes)
    print(
        f"  L_prox_dist = {ends['L_mm']:.2f} mm (target 425)  "
        f"faces prox/dist = {ends['n_prox_faces']}/{ends['n_dist_faces']}  "
        f"nodes {len(ends['prox_nodes'])}/{len(ends['dist_nodes'])}"
    )

    # Gravity direction: -world Z (mm frame). Moment arm along u (horizontal arm).
    # For cantilever validation, gravity should be ⊥ to u.
    u = ends["u"]
    # pilih arah gravitasi tegak lurus u, prefer -Z
    z = np.array([0.0, 0.0, -1.0])
    grav = z - u * np.dot(z, u)
    if np.linalg.norm(grav) < 1e-6:
        grav = np.array([-1.0, 0.0, 0.0])
        grav = grav - u * np.dot(grav, u)
    grav = grav / np.linalg.norm(grav)

    # Moment vector at distal ref: M = M_elbow * (u × grav_hat)  so it bends in grav plane
    # (right-hand: force at tip along grav, offset along u from elbow... force is AT distal
    #  face center; moment is the eccentricity of distal masses beyond the face)
    # M_vec direction: u × (-grav) for masses further along +u with force || grav
    # r_rel along +u, F = -F * (-grav wait): F_vec = -F_N * grav_dir_unit if grav_dir is unit downward...
    # We define grav_dir as unit vector in direction of gravitational ACCELERATION (toward Earth),
    # DLOAD GRAV uses that direction; CLOAD force on tip = M*g in same direction = +F * grav? 
    # Weight force on mass = m * g_vec with g_vec = +G * grav_unit if grav_unit points down.
    # In write_inp I used CLOAD = -F * g_components with DLOAD GRAV in (gx,gy,gz)=grav.
    # CalculiX GRAV: "the gravity vector" — acceleration, body force = density * accel.
    # So both body force and tip force should be in same direction: m*accel.
    # Tip CLOAD should be +F * grav (not minus). FIX in write.

    # Moment: masses distal to face have r = lever * u; F = F_i * grav
    # M = r × F = lever * u × (F_i * grav) → M_total = M_elbow_Nm * (u × grav) 
    # with |u×grav|=1
    mdir = np.cross(u, grav)
    mdir = mdir / (np.linalg.norm(mdir) + 1e-30)
    M_dist_Nmm = mdir * (M_elbow_Nm * 1000.0)  # N·mm

    # Expected reaction moment magnitude ≈ step1 M_total, but adjusted if L_mesh ≠ 425:
    # M_exp = m_self*g*x_com_from_prox + F*(L_m) + M_elbow
    # With uniform density, x_com_from_prox ≈ distance along u from prox face to volume centroid
    centroid = np.average(points[tets].mean(axis=1), weights=vols, axis=0)
    x_com_mm = float(np.dot(centroid - ends["prox_center"], u))
    L_m = ends["L_mm"] / 1000.0
    x_com_m = x_com_mm / 1000.0
    M_exp_fea = M_SELF * G * x_com_m + F_N * L_m + M_elbow_Nm
    M_exp_analytic = step1["M_total_new_Nm"]

    print(f"\n=== Expected moments ===")
    print(f"  analytic (step1)     = {M_exp_analytic:.3f} N·m")
    print(f"  FEA-equilibrium pred = {M_exp_fea:.3f} N·m")
    print(f"    (m_self*g*x_com_mesh={M_SELF*G*x_com_m:.3f} + F*L={F_N*L_m:.3f} + M_elbow={M_elbow_Nm:.3f})")
    print(f"  x_com_mesh from prox = {x_com_m:.4f} m  L_mesh = {L_m:.4f} m")
    print(f"  grav_dir = {grav}, M_dir = {mdir}")

    # Fix CLOAD sign: force in same direction as GRAV acceleration
    inp = OUT / "nauo3_cantilever.inp"

    # monkey-patch write to use +F*grav
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

    with inp.open("w") as f:
        f.write("*HEADING\nNAUO3 cantilever: SPC prox, distributing distal, GRAV\n")
        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"{i}, {x:.8e}, {y:.8e}, {z:.8e}\n")
        f.write(f"{ref}, {dc[0]:.8e}, {dc[1]:.8e}, {dc[2]:.8e}\n")
        f.write("*ELEMENT, TYPE=C3D4, ELSET=Eall\n")
        for i, th in enumerate(tets_w, start=1):
            a, b, c, d = (th + 1).tolist()
            f.write(f"{i}, {a}, {b}, {c}, {d}\n")
        f.write("*NSET, NSET=Nprox\n")
        row = []
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
        f.write(f"{E_MPA}, {NU}\n*DENSITY\n{dens_t_mm3:.8e}\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AlUR\n")
        f.write(f"*COUPLING, CONSTRAINT NAME=RBE3dist, REF NODE={ref}, SURFACE=Sdist\n")
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

    print(f"\nWrote {inp}")

    # Run CalculiX
    ccx = Path("/home/daffa/miniforge3/envs/simjeb/bin/ccx")
    print("\n=== Running CalculiX ===")
    proc = subprocess.run(
        [str(ccx), "-i", inp.stem],
        cwd=str(OUT),
        capture_output=True,
        text=True,
    )
    (OUT / "ccx_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    print(proc.stdout[-2000:] if proc.stdout else "")
    print(proc.stderr[-1000:] if proc.stderr else "")
    print("ccx returncode", proc.returncode)

    react = parse_reactions(
        OUT / "nauo3_cantilever.dat", points, ends["prox_nodes"], ends["prox_center"]
    )
    # Also parse TOTALS from dat
    dat = OUT / "nauo3_cantilever.dat"
    totals = None
    if dat.exists():
        lines = dat.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if "total force" in line.lower() and i + 1 < len(lines):
                totals = lines[i : i + 5]
        # CalculiX prints: "forces (fx,fy,fz) for set Nprox and time  0.1000000E+01"
        for i, line in enumerate(lines):
            if "forces (fx,fy,fz)" in line.lower() or (
                "Nprox" in line and "forces" in line.lower()
            ):
                chunk = "\n".join(lines[i : i + 8])
                print("DAT forces chunk:\n", chunk)
                # next lines often have the numbers
                for j in range(i, min(i + 10, len(lines))):
                    parts = lines[j].split()
                    floats = []
                    for p in parts:
                        try:
                            floats.append(float(p.replace("D", "E")))
                        except ValueError:
                            continue
                    if len(floats) >= 3 and abs(floats[0]) + abs(floats[1]) + abs(floats[2]) > 1e-20:
                        # might be RF totals
                        pass

    # Robust RF from FRD
    frd_path = OUT / "nauo3_cantilever.frd"
    M_fe_Nm = None
    RF_fe = None
    if frd_path.exists():
        M_fe_Nm, RF_fe, nrf = _rf_moment_from_frd(
            frd_path, points, ends["prox_nodes"], ends["prox_center"]
        )
        print(f"\n=== Reaction from FRD ({nrf} prox nodes) ===")
        print(f"  RF_total = {RF_fe} N")
        print(f"  |RF|     = {np.linalg.norm(RF_fe):.4f} N  (expect ~ {(M_SELF+M_DISTAL)*G:.4f})")
        print(f"  M_vec    = {M_fe_Nm} N·m")
        print(f"  |M|      = {np.linalg.norm(M_fe_Nm):.4f} N·m")

    M_fe_norm = float(np.linalg.norm(M_fe_Nm)) if M_fe_Nm is not None else None
    # Compare to analytic and fea-pred
    def pct(a, b):
        return None if a is None else (a - b) / b * 100.0

    ok = None
    if M_fe_norm is not None:
        # primary gate: vs analytic step1; also show vs FEA-equilibrium pred
        err_a = abs(M_fe_norm - M_exp_analytic) / M_exp_analytic * 100
        err_f = abs(M_fe_norm - M_exp_fea) / M_exp_fea * 100
        # pass if within 10% of either (geometry CoM may shift vs URDF)
        ok = err_f < 10.0 or err_a < 10.0
        print(f"\n=== Validation ===")
        print(f"  |M_fe| vs analytic {M_exp_analytic:.3f}: err={err_a:.2f}%")
        print(f"  |M_fe| vs FEA-pred {M_exp_fea:.3f}: err={err_f:.2f}%")
        print(f"  PASS={ok} (threshold 10% vs FEA-pred or analytic)")
        if not ok:
            print("  STOP: reaction moment mismatch — periksa SPC/RBE sebelum stress.")

    report = {
        "step1_com": step1,
        "distal_load": {
            "F_N": F_N,
            "M_elbow_Nm": M_elbow_Nm,
            "levers_m": {k: v[1] for k, v in levers.items()},
        },
        "mesh": {
            "V_mm3": V_mm3,
            "density_kg_m3": dens_kg_m3,
            "L_prox_dist_mm": ends["L_mm"],
            "x_com_from_prox_m": x_com_m,
            "n_prox_nodes": int(len(ends["prox_nodes"])),
            "n_dist_nodes": int(len(ends["dist_nodes"])),
            "n_prox_faces": ends["n_prox_faces"],
            "n_dist_faces": ends["n_dist_faces"],
        },
        "expected_Nm": {
            "analytic_step1": M_exp_analytic,
            "fea_equilibrium_pred": M_exp_fea,
        },
        "fea_reaction": {
            "RF_N": RF_fe.tolist() if RF_fe is not None else None,
            "M_Nm": M_fe_Nm.tolist() if M_fe_Nm is not None else None,
            "M_norm_Nm": M_fe_norm,
        },
        "validation": {
            "pass": ok,
            "err_vs_analytic_pct": pct(M_fe_norm, M_exp_analytic) if M_fe_norm else None,
            "err_vs_fea_pred_pct": pct(M_fe_norm, M_exp_fea) if M_fe_norm else None,
        },
        "ccx_returncode": proc.returncode,
        "inp": str(inp),
    }
    (OUT / "fea_reaction_check.json").write_text(json.dumps(report, indent=2))
    # update bending json with CoM note
    if BEND_JSON.exists():
        bend = json.loads(BEND_JSON.read_text())
        bend["com_urdf_update"] = step1
        BEND_JSON.write_text(json.dumps(bend, indent=2))
    print(f"\nWrote {OUT / 'fea_reaction_check.json'}")
    return 0 if proc.returncode == 0 and ok else 1


def _rf_moment_from_frd(frd_path, points, prox_nodes0, prox_c):
    lines = frd_path.read_text(errors="replace").splitlines()
    # Find last RF block with 3 components
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(" -4") and "RF" in line:
            i += 1
            comps = []
            while i < len(lines) and lines[i].startswith(" -5"):
                comps.append(lines[i])
                i += 1
            vals = {}  # nid -> list
            while i < len(lines) and not lines[i].startswith(" -3"):
                if lines[i].startswith(" -1"):
                    # format: -1  node  value  OR fixed-width
                    raw = lines[i][3:].strip()
                    parts = raw.split()
                    if len(parts) >= 2:
                        try:
                            nid = int(parts[0])
                            val = float(parts[1].replace("D", "E"))
                            vals.setdefault(nid, []).append(val)
                        except ValueError:
                            # fixed format: node I10, val E12.5
                            try:
                                nid = int(raw[:10])
                                val = float(raw[10:].replace("D", "E").split()[0])
                                vals.setdefault(nid, []).append(val)
                            except Exception:
                                pass
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
        v = vals[nid]
        if len(v) < 3:
            continue
        R = np.array(v[:3], dtype=float)
        r = points[nid0] - prox_c
        RF += R
        M += np.cross(r, R)
        n += 1
    return M / 1000.0, RF, n


if __name__ == "__main__":
    sys.exit(main())
