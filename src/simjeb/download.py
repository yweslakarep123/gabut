"""Klien API Harvard Dataverse untuk dataset SimJEB.

Endpoint yang dipakai (publik, tanpa API token):
- Daftar file : GET {SERVER}/api/datasets/:persistentId/versions/:latest/files?persistentId={DOI}
- Unduh file  : GET {SERVER}/api/access/datafile/{id}  (ikuti redirect)

Setiap unduhan penuh diverifikasi terhadap checksum MD5 yang dilaporkan Dataverse.

Unduh selektif per-bracket: file per-bracket dikemas dalam zip besar per tipe
(obj/vtk/csv). Endpoint access datafile me-redirect ke URL presigned S3 yang
mendukung HTTP range request, sehingga member zip individual bisa diekstrak
lewat `remotezip` tanpa mengunduh arsip penuh. (MD5 Dataverse berlaku untuk
arsip utuh, jadi tidak bisa dipakai di jalur ini; integritas per-member
dijamin CRC-32 internal zip.)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

SERVER = "https://dataverse.harvard.edu"
PERSISTENT_ID = "doi:10.7910/DVN/XFUWJG"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# File yang diunduh pada Phase 1 (eksplorasi).
PHASE1_LABELS = [
    "README.txt",
    "all_bracket_metadata.tab",  # diunduh dalam format asli CSV
    "SimJEB_sample_files.zip",
]

CHUNK_SIZE = 1024 * 1024


def list_files() -> list[dict]:
    """Ambil metadata semua file di versi terbaru dataset."""
    url = f"{SERVER}/api/datasets/:persistentId/versions/:latest/files"
    resp = requests.get(url, params={"persistentId": PERSISTENT_ID}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse API error: {payload}")
    return payload["data"]


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(entry: dict, dest_dir: Path = RAW_DIR, original_format: bool = True) -> Path:
    """Unduh satu file berdasarkan entri dari list_files(), dengan verifikasi MD5.

    Untuk file tabular yang dikonversi Dataverse (mis. .tab), original_format=True
    mengunduh file asli yang diunggah (mis. .csv); MD5 yang dilaporkan Dataverse
    berlaku untuk file asli tersebut.
    """
    df = entry["dataFile"]
    file_id = df["id"]
    is_tabular = df.get("tabularData", False)
    use_original = original_format and is_tabular

    filename = df.get("originalFileName") if use_original else df["filename"]
    expected_size = df.get("originalFileSize") if use_original else df["filesize"]
    expected_md5 = df.get("md5")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size == expected_size:
        if expected_md5 and _md5_of(dest) == expected_md5:
            print(f"  sudah ada & MD5 cocok, lewati: {filename}")
            return dest
        if not expected_md5:
            print(f"  sudah ada (ukuran cocok), lewati: {filename}")
            return dest

    # Unduh ke file .part agar bisa di-resume within-file lewat HTTP Range.
    part = dest.with_name(dest.name + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    if expected_size and resume_from >= expected_size:
        resume_from = 0  # .part korup/kebesaran: mulai ulang

    url = resolve_s3_url(file_id, params={"format": "original"} if use_original else None)
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        if resume_from and resp.status_code != 206:
            # Server tidak menghormati Range: mulai dari nol.
            resume_from = 0
        mode = "ab" if resume_from else "wb"
        if resume_from:
            print(f"  resume {filename} dari byte {resume_from:,}")
        total = expected_size or int(resp.headers.get("Content-Length", 0)) or None
        with open(part, mode) as f, tqdm(
            total=total, initial=resume_from, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))

    if expected_size and part.stat().st_size != expected_size:
        raise RuntimeError(
            f"Ukuran tidak cocok untuk {filename}: harap {expected_size:,}, "
            f"dapat {part.stat().st_size:,} (file .part dipertahankan untuk resume)"
        )
    part.replace(dest)

    if expected_md5:
        actual = _md5_of(dest)
        if actual != expected_md5:
            dest.unlink()  # buang hasil korup agar run berikutnya mulai bersih
            raise RuntimeError(
                f"MD5 tidak cocok untuk {filename}: harap {expected_md5}, dapat {actual}"
            )
        print(f"  MD5 OK: {filename}")
    return dest


def resolve_s3_url(file_id: int, params: dict | None = None) -> str:
    """Ikuti redirect Dataverse -> URL presigned S3 (berlaku ~1 jam)."""
    url = f"{SERVER}/api/access/datafile/{file_id}"
    resp = requests.get(url, params=params, allow_redirects=False, timeout=60)
    if resp.status_code not in (301, 302, 303, 307, 308):
        raise RuntimeError(f"Diharapkan redirect untuk datafile {file_id}, dapat {resp.status_code}")
    return resp.headers["Location"]


def open_remote_zip(file_id: int):
    """Buka zip di Dataverse sebagai RemoteZip (baca via HTTP range request)."""
    from remotezip import RemoteZip

    return RemoteZip(resolve_s3_url(file_id))


def extract_members(file_id: int, members: list[str], dest_dir: Path) -> list[Path]:
    """Ekstrak member tertentu dari zip remote tanpa mengunduh arsip penuh.

    Nama member yang tidak ada di arsip dilaporkan lewat KeyError.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with open_remote_zip(file_id) as zf:
        names = set(zf.namelist())
        missing = [m for m in members if m not in names]
        if missing:
            raise KeyError(f"Member tidak ada di zip (id={file_id}): {missing}")
        for m in tqdm(members, desc=f"zip {file_id}", unit="file"):
            zf.extract(m, path=dest_dir)
            out.append(dest_dir / m)
    return out


def download_by_labels(labels: list[str], dest_dir: Path = RAW_DIR) -> list[Path]:
    entries = {e["label"]: e for e in list_files()}
    missing = [lbl for lbl in labels if lbl not in entries]
    if missing:
        raise KeyError(f"Label tidak ditemukan di dataset: {missing}")
    return [download_file(entries[lbl], dest_dir) for lbl in labels]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh file dataset SimJEB dari Harvard Dataverse")
    parser.add_argument("--list", action="store_true", help="tampilkan daftar file di dataset")
    parser.add_argument("--phase1", action="store_true", help="unduh file eksplorasi Phase 1")
    parser.add_argument("--labels", nargs="+", help="unduh file berdasarkan label persis")
    args = parser.parse_args()

    if args.list:
        entries = list_files()
        total = 0
        print(f"{'id':>9}  {'ukuran':>12}  label")
        for e in entries:
            df = e["dataFile"]
            total += df["filesize"]
            print(f"{df['id']:>9}  {df['filesize']:>12,}  {e['label']}")
        print(f"\nTotal: {len(entries)} file, {total / 1e9:.2f} GB")
        return

    if args.phase1:
        paths = download_by_labels(PHASE1_LABELS)
        print("\nSelesai. File tersimpan di:")
        for p in paths:
            print(f"  {p}")
        return

    if args.labels:
        download_by_labels(args.labels)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
