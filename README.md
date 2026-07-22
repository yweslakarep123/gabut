# Jazari — SimJEB Data Pipeline

Pipeline data untuk mengunduh dan mem-parsing **SimJEB** (Simulated Jet Engine Bracket
Dataset) menjadi struktur siap-ML, sebagai proof-of-concept surrogate model yang
memprediksi performa struktural (field stress & displacement) dari geometri.

## Dataset

- **Sumber:** [Harvard Dataverse — doi:10.7910/DVN/XFUWJG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XFUWJG)
- **Project page:** https://simjeb.github.io/
- **Paper:** Whalen, Beyene, Mueller (2021), *SimJEB: Simulated Jet Engine Bracket Dataset*, Computer Graphics Forum.
- **Isi:** 381 bracket hasil desain manusia dari kompetisi "GE Jet Engine Bracket
  Challenge" (GrabCAD, 2013). Tiap bracket punya file CAD (STEP), tetrahedral mesh
  (VTK), surface mesh (OBJ), dan hasil simulasi FEA (stress & displacement) untuk
  4 load case.

## Lisensi dataset (PENTING)

- Dataset SimJEB dilisensikan di bawah **Open Data Commons Attribution License (ODC-By)**:
  https://opendatacommons.org/licenses/by/
- **Desain CAD asli** berasal dari GrabCAD dan berlisensi **non-komersial**
  (GrabCAD terms). Jangan gunakan geometri asli untuk keperluan komersial.
- Beri atribusi ke penulis dataset (Whalen, Beyene, Mueller — MIT) saat memakai data ini.

## Setup

```bash
conda env create -f environment.yml
conda activate simjeb
```

## Struktur proyek

```
jazari/
├── environment.yml        # definisi conda env
├── src/simjeb/
│   ├── download.py        # klien API Harvard Dataverse (list, unduh resumable +
│   │                      #   verifikasi MD5, ekstrak selektif via HTTP range)
│   ├── parse.py           # parser satu bracket (OBJ + VTK + CSV field, tervalidasi)
│   └── build.py           # batch: parse 381 bracket -> data/processed/
├── data/
│   ├── raw/               # unduhan mentah dari Dataverse (zip, csv, txt)
│   │   └── full/          # hasil ekstraksi: {id}.obj, {id}.vtk, {id}field.csv
│   └── processed/         # output siap-ML (lihat di bawah)
└── README.md
```

## Penggunaan

```bash
# daftar semua file di dataset
PYTHONPATH=src python -m simjeb.download --list

# unduh file eksplorasi (README, metadata, sample files)
PYTHONPATH=src python -m simjeb.download --phase1

# unduh zip penuh (resumable; ulangi perintah yang sama jika koneksi putus)
PYTHONPATH=src python -m simjeb.download --labels "SimJEB_surfmesh_(obj).zip" \
    "SimJEB_volmesh_(vtk).zip" "SimJEB_simresults_(csv)_firsthalf.zip" \
    "SimJEB_simresults_(csv)_secondhalf.zip"

# parse semua bracket -> data/processed/
PYTHONPATH=src python -m simjeb.build
```

## Format output (`data/processed/`)

Satu folder per bracket (`{id}/`) berisi dua file NumPy:

- `geometry.npz` — `surf_vertices` (Ns,3) f64, `surf_faces` (Nf,3) i32,
  `vol_points` (N,3) f64, `vol_tets` (Nt,4) i32, `node_surf` (N,) bool.
- `fields.npz` — per load case `lc` di `{ver, hor, dia, tor}` (vertical /
  horizontal / diagonal / torsional): `disp_{lc}` (N,3), `disp_mag_{lc}` (N,),
  `stress_{lc}` (N,) — semuanya f32, satu nilai per node volume mesh.

Ditambah `manifest.csv` (satu baris per bracket: status parse, path, jumlah
node/tet, max stress & displacement per load case, kategori bentuk, dan kolom
split resmi `test_split_0/1/2` dari dataset — pakai ini, jangan buat random
split sendiri agar sebanding dengan benchmark SimJEB/DeepJEB) dan
`failures.log` (traceback per bracket gagal; kosong = semua sukses).

Satuan: mm untuk panjang/displacement, **MPa** untuk stress (von Mises).
Catatan: README dataset menulis "GPa (N/mm²)" yang kontradiktif; verifikasi
numerik memastikan MPa, dan paper (arXiv:2105.03534, Fig. 4) memastikan von Mises.

## Keputusan training / evaluasi MeshGraphNet (split_0)

**Filter train `n_nodes <= 150k` (penyimpangan dari rencana Fase 3).**
Rencana awal: full-graph + gradient checkpointing, dan untuk graf ekstrem
(`N > 300k`) neighbor sampling / turunkan H — *jangan drop sampel*. Di
praktik (VRAM ~8 GB), setelah OOM pada graf besar, keputusan diganti jadi
**filter langsung** lewat `--max-train-nodes 150000` (default di
`src/simjeb/train.py`): 22 bracket di-drop dari `train_fit` (273→251), di
luar id 293 yang sudah dilewati karena `bc.npz` hilang (file `.fem` 0 byte).
Alasan: waktu + kompleksitas implementasi neighbor sampling belum sebanding
untuk PoC ini. Val (30) dan test resmi (77) **tidak** difilter ukuran.

Id train yang di-drop (22): 29, 33, 106, 123, 136, 142, 220, 270, 364, 378,
380, 389, 394, 428, 464, 498, 545, 550, 590, 592, 611, 625.

**Evaluasi test resmi:** dari 77 bracket `test_split_0`, hanya **id 281**
(642k node) yang di-skip saat eval karena CUDA OOM. Empat bracket lain dengan
`n_nodes > 150k` (98, 108, 332, 390) **masuk** ke MAE pooled. Jadi angka di
`reports/mgn_split0_test_mae.csv` = MAE atas **76/77** bracket test (bukan
77 penuh). Klaim “20/20 mengalahkan naive” berlaku untuk subset itu; CSV
sendiri belum mencatat `n_eval` / daftar skip — lihat juga
`reports/train_split0_log.txt` baris `(eval skip OOM: [281])`.

## Geometri robot — UR5e (Graphical Documentation)

Sumber STEP resmi: [Robot Step file — UR5e/UR7e e-Series](https://www.universal-robots.com/download/mechanical-e-series/ur5e/robot-step-file-ur5e-e-series/)
(login akun gratis UR diperlukan). File yang sama dipakai UR5e dan UR7e
(mekanik identik). Ada dua varian STEP (M8 male vs female connector;
serial UR5e ≥ `20245501103` = female).

File ini adalah **Graphical Documentation** UR, tunduk pada
[Terms and Conditions for Use of Graphical Documentation](https://www.universal-robots.com/legal/terms-and-conditions/terms_and_conditions_for_use_of_graphical_documentation.txt)
(v1.01, 2023-09-12). Ringkas poin relevan untuk proyek ini:

- **Diizinkan** (secara umum): simulasi, visualisasi, representasi digital,
  path planning, collision avoidance, dan algoritma untuk produk UR —
  termasuk penggunaan oleh hobi/mahasiswa/peneliti untuk tujuan
  **non-komersial**.
- **Dilarang**: membuat model fisik untuk tujuan komersial; memakai
  dokumentasi untuk menciptakan/memperbaiki produk yang bersaing
  (langsung atau tidak langsung) dengan robot UR / afiliasinya;
  comparative advertising / penggunaan yang merugikan UR.
- **IP derivative works**: karya turunan dari Graphical Documentation
  secara default dimiliki UR; boleh dipakai hanya dalam lingkup Section 1.
- **Share**: boleh bagikan ke pihak ketiga sejauh perlu untuk use yang
  diizinkan, dengan menyertakan salinan T&Cs; publikasi download harus
  dalam paket software yang menyertakan T&Cs.
- **Notice**: tampilkan notice hak cipta bila memungkinkan
  (`© 2023 Universal Robots A/S. Use hereof is subject to … Graphical Documentation`).
- **AS IS**, tanpa warranty; larangan use di nuklir / senjata / avionik
  tertentu (Section 1.5).
- Jika use di luar T&Cs: minta izin khusus ke `legal@universal-robots.com`.

Untuk PoC riset non-komersial (surrogate FEA dari geometri link) umumnya
masuk Section 1.1–1.2 — **jangan** pakai geometri ini untuk produk
komersial atau robot pesaing tanpa izin tertulis.

### Mapping visual & target topologi (NAUO3)

Inspeksi STEP `UR7e.step`: 23 solid dalam 1 assembly; instance `NAUO1`…`NAUO7`
dipetakan visual ke rantai UR5e (shoulder→upper arm→forearm→wrist→flange).
**Fokus pertama: NAUO3 = upper arm** (satu solid, ~4870 cm³).

Cek PNG `reports/ur5e_renders/18_*NAUO3*/{front,side,top,iso}.png`:
**tidak terlihat lubang/bore tembus** sepanjang badan link (ujung housing
terlihat tertutup + logo; silinder tengah tampak solid dari luar). Jadi
meski robot e-Series fisik punya cable routing internal, model Graphical
Documentation ini diperlakukan sebagai **solid genus-0**.

- **Target heal NAUO3:** `watertight=true`, **euler number ≈ 2** (bukan ≈0).
- Euler ≈0 hanya relevan bila nanti ditemukan tunnel tembus (torus / genus-1).
- **Euler 1470 vs 10 bukan kontradiksi** — beda tahap preprocessing:
  - `retessellate_ur_solid` / inspeksi mentah: tessellasi OCCT per-face
    **tanpa weld vertex** → seam antar-face jadi boundary palsu →
    `euler≈1470` (deflection 0,5 mm maupun 0,05 mm; delta=0).
  - `heal_ur_solid` baseline: metrik dihitung **setelah** `_weld`
    (`merge_vertices` + hapus face degenerate/duplikat) → `euler≈10`,
    ~50 boundary edges (seam terbuka kecil yang nyata).
- Eksperimen re-tessellasi deflection 0,5→0,05 mm: euler mentah tetap 1470,
  watertight tetap false → lanjut heal eksplisit (trimesh.repair →
  ShapeFix BRep + drop void mikro → PyMeshLab bila perlu).
- Hasil heal sukses: `reports/ur5e_nauo3_heal/02_shapefix.stl`
  (`watertight=true`, `euler=2`). Delapan void mikro (~0,1–0,3 mm³) di
  BRep sumber dibuang; bukan kanal kabel.

**Geometri heal = solid penuh, tanpa kanal kabel internal.** CAD Graphical
Documentation UR tidak memodelkan cable routing e-Series. FEA / surrogate
dari mesh ini akan cenderung **lebih kaku/kuat** daripada link fisik
sungguhan (yang berongga untuk kabel). Ini pendekatan awal PoC — **bukan**
replikasi mekanik persis UR5e asli.

### Baseline volume mesh NAUO3 & pass optimize Gmsh

Artefak: `reports/ur5e_nauo3_volume/geometry.npz` (+ `mesh_report.json`).
`radius_ratio` memakai rumus yang sudah dikoreksi (`R = ||O−p0||`).

Pass optimize yang aman secara geometri = **Gmsh default** (edge-swap /
relocate interior) pada mesh yang sudah ada — file
`reports/ur5e_nauo3_volume/_opt_default.msh`. Relocate3D / Netgen
**tidak** dipakai sebagai baseline: Relocate3D menggeser surface
(~22 mm, ΔV ≈ −6.5%, tet inverted); Netgen segfault di SwapImprove
(gmsh 4.15.2).

Keputusan apply default → `geometry.npz` final:
jalankan `scripts/apply_default_optimize_if_better.py` (lihat command di
chat). Kriteria: `radius_ratio` p50 **atau** p95 lebih baik, yang lain
tidak memburuk; `|ΔV| < 0.5%`; 0 inverted; hitungan dihedral tajam tidak
naik. Hasil tertulis di
`reports/ur5e_nauo3_volume/optimize_radius_ratio_decision.json`.

Jika skrip memutuskan **keep_pre_optimize**: baseline pra-optimasi tetap
dipakai karena default tidak memperbaiki distribusi `radius_ratio`
(median/p95) meski ΔV ≈ 0 — bukan karena volume/geometri rusak.

### Mapping NAUO (STEP) ↔ link URDF UR5e + massa

Sumber: [artikel DH/dynamics resmi UR](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/)
(tabel **UR5e/UR7e**), disalin ke
`Universal_Robots_ROS2_Description` dan (sejak 2025-03) juga
`ros-industrial/universal_robot`.

#### Kenapa sempat ada dua set massa?

| Set | Asal | Σ 6 link bergerak | Σ + `base_mass=4.0` |
|-----|------|-------------------:|--------------------:|
| **Final (pakai ini)** | Tabel UR5e/UR7e resmi; ROS2 `#187` / ros-industrial `#689` | **17.70 kg** | 21.70 kg |
| Legacy (jangan pakai) | Hampir identik tabel **UR5 CB-series** (bukan e-Series), sempat tertanam di yaml UR5e lama | 16.99 kg | 20.99 kg |

Commit ROS2 `#187` (*Fix masses of robot links*, 2024-09, Felix Exner / UR):
*"this should match the actual specifications"* → link ke artikel DH di atas.
Ros-industrial `#689` (2025-03) menyalin revisi yang sama.
**Bukan** safety factor: angka “ROS2” = spek UR5e; angka “ros-industrial lama”
yang kita unduh sebelumnya = **massa UR5 non-e yang salah tempel**.

Pola vs datasheet lengan 20,6 kg: total legacy lebih dekat (±1,9%) karena
kebetulan; total spek (+`base_mass` 4,0 yang dikomentari *might be incorrect*)
+5,3%. Σ **hanya Link 1–6 dari artikel UR = 17,70 kg** — `base_mass` di URDF
terpisah dan tidak ada di tabel UR5e artikel itu. Untuk lentur NAUO3, base
tidak masuk beban distal → pakai spek link, bukan mengejar total 20,6 kg.

| NAUO | URDF link | mass final (kg) | catatan |
|------|-----------|----------------:|---------|
| NAUO2 | `shoulder_link` | 3.761 | = Link 1 artikel UR |
| **NAUO3** | **`upper_arm_link`** | **8.058** | = Link 2 |
| NAUO4 | `forearm_link` | 2.846 | = Link 3 |
| NAUO5 | `wrist_1_link` | 1.37 | = Link 4; STEP punya 2 solid |
| NAUO6 | `wrist_2_link` | 1.3 | = Link 5 |
| NAUO7 | `wrist_3_link` | 0.365 | = Link 6 |
| (base) | `base_link_inertia` | 4.0 | tidak di tabel UR5e; flag “might be incorrect” |

**NAUO3 lentur — angka final:** `m_self = 8.058 kg`; massa distal link
`forearm+wrist_1/2/3 = 5.881 kg`; **`m_payload = 5 kg` terpisah di tool
tip** (bukan digabung ke m_distal). Detail FK + momen per-komponen:
`reports/ur5e_urdf/nauo3_bending_moment_horizontal.json`
(`M_total ≈ 106.5 N·m` @ pose horizontal kanonik; CoM URDF
`|x|=0.2125` = L/2). FEA reaction-check pertama (CalculiX):
`reports/ur5e_nauo3_fea/fea_reaction_check.json` — `|M|=106.48` vs
analytic 106.52 (**δ ≈ −0.04%**, PASS).

#### Material STEP & von Mises (load case cantilever)

- **Material di STEP: kosong.** Baik `nauo3_healed.step` (OCCT export)
  maupun `UR7e.step` asli (SolidWorks AP214) **tidak** memuat
  `MATERIAL_DESIGNATION` / `MATERIAL_PROPERTY` / nama alloy. Tidak ada
  yang bisa dipakai sebagai yield resmi dari CAD.
- **Lokasi σ_vm max:** node di **badan link**, **bukan** di 634 node SPC
  proksimal maupun permukaan coupling distal
  (`dist_SPC ≈ 92 mm`, `dist_distal ≈ 315 mm`) →
  klasifikasi `IN_BODY_AWAY_FROM_BC` (bukan artefak RBE/SPC ala bracket 175).
- **Percentile σ_vm (MPa, nodal):** p50≈0.14, p95≈0.89, p99≈1.20,
  **max≈1.65**. Detail: `reports/ur5e_nauo3_fea/stress_vonmises_report.json`.
- **Safety factor (ASUMSI, belum terkonfirmasi UR):** yield aluminium cor
  ~**185 MPa** → SF ≈ **112** @ σ_max. Angka ini **hanya untuk riset
  internal**; jangan dipakai keputusan desain/produk sebelum material UR
  diverifikasi. Stress rendah juga karena model Graphical Docs = **solid
  penuh** (tanpa rongga kabel) → lebih kaku/kuat dari link fisik.

#### Offset joint (lengan momen) dari URDF yang sama

`reports/ur5e_urdf/joint_origins_chain.json` — `<joint>/<origin xyz>` (m):

| joint | parent → child | xyz (m) | peran DH |
|-------|----------------|---------|----------|
| `shoulder_pan_joint` | base_inertia → shoulder | `(0, 0, 0.1625)` | d1 |
| `shoulder_lift_joint` | shoulder → upper_arm | `(0, 0, 0)` | hanya RPY |
| `elbow_joint` | upper_arm → forearm | `(-0.425, 0, 0)` | **a2 = panjang NAUO3** |
| `wrist_1_joint` | forearm → wrist_1 | `(-0.3922, 0, 0.1333)` | a3 + d4 |
| `wrist_2_joint` | wrist_1 → wrist_2 | `(0, -0.0997, ≈0)` | d5 |
| `wrist_3_joint` | wrist_2 → wrist_3 | `(0, 0.0996, ≈0)` | d6 |

Untuk momen di root/shoulder-lift NAUO3, lengan utama ke elbow = **0,425 m**;
beban distal digeser lagi sepanjang forearm+wrist sesuai pose.

Ringkasan: `reports/ur5e_urdf/inertial_summary.json`.

# gabut
