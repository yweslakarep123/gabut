"""Fase 1 — inspeksi assembly STEP UR5e (tanpa FEA).

Membaca struktur part/label dari file STEP resmi UR, lalu cek tiap solid:
volume, bounding box, watertight (via tessellation + trimesh), non-manifold.

Tidak mengasumsikan nama link — melaporkan apa yang tertulis di STEP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _require_cq():
    try:
        import cadquery as cq  # noqa: F401
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_COMPSOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepBndLib import BRepBndLib
        from OCP.Bnd import Bnd_Box
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopLoc import TopLoc_Location
        from OCP.BRep import BRep_Tool
        from OCP.TopAbs import TopAbs_FACE
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.gp import gp_Trsf
    except ImportError as e:
        print(
            "CadQuery/OCP belum terpasang. Install dulu:\n"
            "  conda install -c conda-forge cadquery\n"
            f"Detail: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _label_of(shape) -> str:
    """Ambil nama dari TShape bila ada; fallback kosong."""
    try:
        from OCP.TDF import TDF_Label
        # STEP reader names often land on shape via Name attribute in XDE;
        # tanpa XDE, coba Shape.DumpToString header / HashCode saja.
    except Exception:
        pass
    # Tanpa document XDE, nama part sering hilang. Caller pakai STEPCAFControl
    # bila tersedia.
    return ""


def _read_step_with_names(path: Path):
    """Baca STEP via STEPCAF (XDE) supaya nama part tersedia; fallback reader biasa."""
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_LabelSequence
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS

    app = XCAFApp_Application.GetApplication_s()
    # OCP butuh TCollection_ExtendedString, bukan str Python mentah
    fmt = TCollection_ExtendedString("MDTV-XCAF")
    doc = TDocStd_Document(fmt)
    app.NewDocument(fmt, doc)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEPCAF gagal baca {path} (status={status})")
    if not reader.Transfer(doc):
        raise RuntimeError(f"STEPCAF Transfer gagal untuk {path}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    parts: list[tuple[str, object]] = []

    def _name(label) -> str:
        attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
            return str(attr.Get().ToExtString())
        return f"unnamed_{label.Tag()}"

    def _collect(label, prefix: str = ""):
        name = _name(label)
        full = f"{prefix}/{name}" if prefix else name
        shape = shape_tool.GetShape_s(label)
        # Kalau ada komponen anak, telusuri; kalau solid leaf, simpan.
        comps = TDF_LabelSequence()
        has_comps = bool(shape_tool.GetComponents_s(label, comps) and comps.Length() > 0)
        if has_comps:
            for i in range(1, comps.Length() + 1):
                _collect(comps.Value(i), full)
            return
        # Leaf: pecah ke solid individu
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        n_sol = 0
        while exp.More():
            n_sol += 1
            solid = TopoDS.Solid(exp.Current())
            parts.append((f"{full}::solid{n_sol}", solid))
            exp.Next()
        if n_sol == 0:
            # mungkin sudah solid tunggal
            parts.append((full, shape))

    for i in range(1, labels.Length() + 1):
        _collect(labels.Value(i))

    if not parts:
        # Fallback: reader polos tanpa nama
        r = STEPControl_Reader()
        if r.ReadFile(str(path)) != IFSelect_RetDone:
            raise RuntimeError(f"STEPControl_Reader gagal: {path}")
        r.TransferRoots()
        shape = r.OneShape()
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        k = 0
        while exp.More():
            k += 1
            parts.append((f"solid_{k}", TopoDS.Solid(exp.Current())))
            exp.Next()
    return parts


def _brep_bbox_volume(solid):
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    vol = float(props.Mass())
    box = Bnd_Box()
    BRepBndLib.Add_s(solid, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    bbox = {
        "min": [xmin, ymin, zmin],
        "max": [xmax, ymax, zmax],
        "size": [xmax - xmin, ymax - ymin, zmax - zmin],
    }
    return vol, bbox


def tessellate_solid(solid, linear_deflection: float = 0.5, *, clean: bool = True):
    """Tessellate BRep solid → trimesh + quality metrics.

    ``clean=True`` menghapus triangulation lama dulu (wajib sebelum re-tessellate
    dengan deflection berbeda).
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    import trimesh

    vol, bbox = _brep_bbox_volume(solid)
    if clean:
        BRepTools.Clean_s(solid)
    BRepMesh_IncrementalMesh(solid, linear_deflection)

    vertices = []
    faces = []
    v_offset = 0
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, loc)
        if triangulation is None:
            exp.Next()
            continue
        trsf = loc.Transformation()
        n_nodes = triangulation.NbNodes()
        for i in range(1, n_nodes + 1):
            p = triangulation.Node(i)
            p.Transform(trsf)
            vertices.append([p.X(), p.Y(), p.Z()])
        n_tri = triangulation.NbTriangles()
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for i in range(1, n_tri + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            a, b, c = n1 - 1 + v_offset, n2 - 1 + v_offset, n3 - 1 + v_offset
            if reversed_face:
                faces.append([a, c, b])
            else:
                faces.append([a, b, c])
        v_offset += n_nodes
        exp.Next()

    watertight = None
    is_volume = None
    n_vertices = len(vertices)
    n_faces = len(faces)
    euler = None
    issues: list[str] = []
    mesh = None
    if n_faces == 0:
        issues.append("no_tessellation")
    else:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        watertight = bool(mesh.is_watertight)
        is_volume = bool(mesh.is_volume)
        euler = int(mesh.euler_number)
        if not watertight:
            issues.append("not_watertight")
        if not mesh.is_winding_consistent:
            issues.append("inconsistent_winding")
        try:
            if hasattr(mesh, "is_empty") and mesh.is_empty:
                issues.append("empty_mesh")
        except Exception:
            pass

    stats = {
        "volume_mm3": vol,
        "bbox_mm": bbox,
        "n_mesh_vertices": n_vertices,
        "n_mesh_faces": n_faces,
        "watertight": watertight,
        "is_volume": is_volume,
        "euler_number": euler,
        "issues": issues,
        "deflection_mm": linear_deflection,
    }
    return mesh, stats


def _solid_stats(solid, linear_deflection: float = 0.5):
    _, stats = tessellate_solid(solid, linear_deflection, clean=True)
    return stats


def match_parts(
    parts: list[tuple[str, object]],
    *,
    names_substr: list[str] | None = None,
    indices: list[int] | None = None,
) -> list[tuple[int, str, object]]:
    """Filter parts by 1-based index and/or substring di nama (OR)."""
    out: list[tuple[int, str, object]] = []
    for i, (name, solid) in enumerate(parts, 1):
        ok = False
        if indices and i in indices:
            ok = True
        if names_substr:
            low = name.lower()
            if any(s.lower() in low for s in names_substr):
                ok = True
        if names_substr is None and indices is None:
            ok = True
        if ok:
            out.append((i, name, solid))
    return out


def main() -> None:
    _require_cq()
    p = argparse.ArgumentParser(description="Inspeksi STEP UR5e Fase 1")
    p.add_argument("step_path", type=Path, help="path ke .step / .stp")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/ur5e_step_inspection.json"),
        help="output JSON laporan",
    )
    p.add_argument(
        "--deflection",
        type=float,
        default=0.5,
        help="linear deflection tessellation (mm)",
    )
    args = p.parse_args()
    if not args.step_path.exists():
        print(f"File tidak ada: {args.step_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"membaca: {args.step_path} ({args.step_path.stat().st_size/1e6:.1f} MB)")
    parts = _read_step_with_names(args.step_path)
    print(f"ditemukan {len(parts)} solid/leaf part\n")

    rows = []
    for i, (name, solid) in enumerate(parts, 1):
        print(f"[{i}/{len(parts)}] {name} ...", flush=True)
        try:
            stats = _solid_stats(solid, args.deflection)
        except Exception as e:
            stats = {"error": str(e), "issues": ["stats_failed"]}
        row = {"index": i, "name": name, **stats}
        rows.append(row)
        wt = stats.get("watertight")
        vol = stats.get("volume_mm3")
        print(
            f"         watertight={wt}  volume={vol}  issues={stats.get('issues')}"
        )

    report = {
        "source": str(args.step_path.resolve()),
        "n_parts": len(rows),
        "parts": rows,
        "summary": {
            "watertight_true": sum(1 for r in rows if r.get("watertight") is True),
            "watertight_false": sum(1 for r in rows if r.get("watertight") is False),
            "watertight_unknown": sum(1 for r in rows if r.get("watertight") is None),
            "with_errors": sum(1 for r in rows if "error" in r),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n=== ringkasan ===")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nNama part (urut):")
    for r in rows:
        print(f"  {r['index']:3d}. {r['name']}")
    print(f"\ntersimpan: {args.out}")


if __name__ == "__main__":
    main()
