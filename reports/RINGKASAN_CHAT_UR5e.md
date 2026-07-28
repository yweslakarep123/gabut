# Ringkasan sesi kerja UR5e / SimJEB

Dokumen ini merangkum apa yang sudah dikerjakan di lima chat terkait pipeline geometri robot UR5e (dari heal STEP sampai batch FEA dan workstream termal).

**Rentang kerja:** ~21–23 Juli 2026  
**Fokus utama:** link NAUO3 (upper arm) → dataset variasi leher; lalu tooling termal + Fase 1 NAUO2 (shoulder).

| Chat | Topik singkat |
|------|----------------|
| [Heal & mesh NAUO3](ee419295-53e1-42c8-a1f3-b427ff96c68c) | Heal watertight, upaya tet-mesh, `.gitignore` |
| [Volume mesh sukses](3860844c-1350-4663-898a-63e012bc9afe) | Perbaiki PLC/overlap → `geometry.npz` |
| [URDF + FEA baseline](ee053667-92d1-424b-bc94-a1479a869901) | Massa, momen, CalculiX, stress |
| [Batch variasi leher](89f386ff-cd9c-4f84-80f3-13e0b5c180dc) | Deform mesh N=11, QoI σ global max |
| [Thermal + NAUO2](f246325a-0120-4548-975f-fb18801a3ce9) | Validasi fin analitik, heal+mesh shoulder |

---

## Alur besar (satu kalimat)

STEP Graphical Docs UR → heal solid genus-0 → tet-mesh SimJEB-compatible → load case cantilever tervalidasi → batch deform leher → mulai workstream termal (tooling + geometri NAUO2).

---

## 1. Heal NAUO3 & upaya volume mesh

**Chat:** [ee419295-53e1-42c8-a1f3-b427ff96c68c](ee419295-53e1-42c8-a1f3-b427ff96c68c)

### Yang dikerjakan

1. **Observasi render** (`front`/`side`): **tidak** ada lubang tembus ujung-ke-ujung → target topologi **watertight + euler ≈ 2** (bukan ≈0).
2. **Heal berurutan** sampai target tercapai:
   - `trimesh.repair` → gagal (belum watertight)
   - **ShapeFix BRep** (drop 8 void mikro + Sewing) → **sukses**: watertight, euler=2  
   - Artefak: `reports/ur5e_nauo3_heal/02_shapefix.stl`
3. Dokumentasi: euler mentah **1470** vs baseline weld **~10** = beda tahap preprocessing (bukan kontradiksi); CAD solid tanpa kanal kabel → FEA akan lebih kaku dari link fisik.
4. **Volume mesh** dicoba berkali-kali (STL→Gmsh, STEP OCC, local size face mikro, backend tess):
   - Hang di surface **1091** (cone mikro ~1.5 mm vs mesh_size 5 mm)
   - Overlap facet 761/765, periodic surface 1274
   - Tess STEP kasar euler=−10 (tidak watertight)
5. User minta job panjang dijalankan sendiri → command detail diberikan; background task di-kill.
6. **`.gitignore`** ditambah agar data besar (`data/raw`, mesh/STL/STEP, renders, checkpoints) tidak ikut push.

### Status akhir chat ini

Heal NAUO3 **selesai**. Tet-mesh **belum** berhasil di chat ini.

---

## 2. Volume mesh NAUO3 berhasil

**Chat:** [3860844c-1350-4663-898a-63e012bc9afe](3860844c-1350-4663-898a-63e012bc9afe)

### Masalah

Backend `tess` + STL heal watertight masih gagal: **overlapping facets** (surface 35/36) saat remesh Gmsh; tetgen `recoversubfaces` gagal; akar = **self-intersecting faces** di surface (~284).

### Solusi yang terbukti

Pipeline: **iso-remesh → dilate-clean self-intersect → MeshOnlyEmpty** (bukan `createGeometry` remesh).

### Hasil

| Metrik | Nilai |
|--------|------:|
| Nodes / tets | ~38.8k / ~192k |
| Watertight / euler / self | true / 2 / 0 |
| Volume | ~4.85×10⁶ mm³ (dekat solid) |
| Inverted tets | 0 |

Artefak utama: `reports/ur5e_nauo3_volume/geometry.npz`  
Catatan: ada sliver (dihedral tajam); cocok sebagai **baseline ML/FEA awal**, belum mesh FEM “industri”.

---

## 3. Kualitas mesh, URDF, load case, FEA pertama

**Chat:** [ee053667-92d1-424b-bc94-a1479a869901](ee053667-92d1-424b-bc94-a1479a869901)

### 3.1 Bug `radius_ratio`

- Formula circumradius salah: pakai `||O||` (ke origin) bukan `||O−p0||` → hampir semua tet terlihat “buruk palsu”.
- Setelah perbaikan: tet regular = **1.0**; NAUO3 median ~**1.25**, `rho>10` ~**6%** (bukan 100%).

### 3.2 Optimize Gmsh

- Relocate3D: dihedral bagus tapi **merusak surface** (ΔV −6.5%).
- Netgen: segfault.
- **Default optimize** diterapkan ke `geometry.npz` (perbaikan kecil, ΔV≈0) — backup `geometry_pre_optimize.npz`.

### 3.3 Massa & kinematika URDF

- Mapping: NAUO2…7 ↔ shoulder → wrist_3.
- Dua set massa sempat membingungkan: angka “ros-industrial lama” = **massa UR5 CB-series salah tempel**, bukan safety factor.
- **Dipakai spek UR5e resmi:** m_self NAUO3 = **8.058 kg**; distal links = **5.881 kg**; payload **5 kg** terpisah di tip.
- Offset joint: a2 = **0.425 m** (`joint_origins_chain.json`).

### 3.4 Momen lentur (pose horizontal)

Pose kanonik `q = [0, 0, 0, −90°, 0, 0]` → tip along-arm ≈ **0.917 m**.

| Suku | M (N·m) |
|------|--------:|
| Self (CoM) | 16.8 |
| Distal links | ~44.8 |
| Payload 5 kg | ~45.0 |
| **Total** | **~106.5** |

### 3.5 FEA CalculiX — reaction check PASS

Setup: SPC proksimal + distributing (RBE3-like) distal + gravity ρ·g.

| | Analytic | FEA | err |
|--|--:|--:|--:|
| \|M\| proximal | 106.52 | 106.48 | **−0.04%** |

### 3.6 Stress von Mises

- Material STEP: **kosong** (tidak ada alloy).
- σ_max ≈ **1.65 MPa** di **badan link** (bukan artefak BC), y≈262 mm.
- SF asumsi Al cor ~185 MPa ≈ **112** (hanya riset internal; model solid penuh lebih kaku dari fisik).

---

## 4. Batch variasi geometri leher (menuju dataset)

**Chat:** [89f386ff-cd9c-4f84-80f3-13e0b5c180dc](89f386ff-cd9c-4f84-80f3-13e0b5c180dc)

### Keputusan desain

| Item | Pilihan |
|------|---------|
| Pendekatan | **(a) deformasi mesh volume lokal** (bukan CadQuery / edit BRep) |
| Parameter | Skala radial leher `s` di sekitar y₀=262 mm |
| Falloff terkunci | **cosine C1, L=40 mm** |
| N | **11** sampel, s = 0.90 … 1.10 step 0.02 |
| Load case | Sama seperti FEA baseline (cantilever + payload 5 kg) |
| QoI | **Global max** von Mises (termasuk bahu s>1) |

### Investigasi penting sebelum batch penuh

1. SPC vs band: **aman** (gap ~48 mm).
2. Pilot s=0.90 bersih (CENTER); s=1.10 awalnya di **BAND_EDGE** → stop.
3. Diagnostik: cosine C1 punya lompatan kurvatura di tepi; setelah uji C2 / bump / Gaussian, hotspot s>1 tetap di bahu → terbukti **efek struktural** (fillet penebalan), bukan artefak numerik acak.
4. Scaling s=1.03 vs 1.10 sebanding |s−1| → mendukung fenomena fisik.
5. Keputusan: terima global max di bahu; **kunci L=40**.

### Hasil batch final

**11/11 OK**, 0 gagal mesh/reaction.  
Ringkasan: `reports/ur5e_nauo3_neck_batch/batch_summary.json`

Tren masuk akal: leher lebih tipis (`s<1`) → σ naik; `s=1.0` sanity ≈ baseline 1.65 MPa.

---

## 5. Workstream Thermal + Fase 1 NAUO2

**Chat:** [f246325a-0120-4548-975f-fb18801a3ce9](f246325a-0120-4548-975f-fb18801a3ce9)

### 5.1 Validasi tooling termal (fin analitik)

Batang Al D=10 mm, L=100 mm; Tb=100 °C; tip adiabatik; konveksi samping.

| | Hasil |
|--|--------|
| Error awal (mesh 2 mm) | max ~**0.30 °C** |
| Penyebab | **keduanya**: kebocoran `*FILM` di rim tip + diskretisasi |
| Setelah `exclude_end_planes` + mesh 1 mm | max ~**0.03 °C** |

Unit CCX: h = 0.01 N/(s·mm·K) (= h_SI/1000).  
Artefak: `reports/thermal_fin_validate/`.

### 5.2 NAUO2 Fase 1 (geometri saja)

| Langkah | Hasil |
|---------|--------|
| Identitas | `#17 NAUO2` = shoulder; boss mounting motor terlihat |
| Heal | ShapeFix sukses (drop 4 void mikro); euler=2 |
| Volume mesh | ~**14k node / 67k tet** @ 5 mm; V_ratio≈0.994 |
| BC termal | **Belum** — tunggu keputusan heat-source surface |

Ringkasan: `reports/ur5e_nauo2_phase1.json`

---

## Artefak kunci (navigasi cepat)

| Artefak | Isi |
|---------|-----|
| `reports/ur5e_nauo3_heal/` | STL heal upper arm |
| `reports/ur5e_nauo3_volume/geometry.npz` | Baseline tet-mesh NAUO3 |
| `reports/ur5e_urdf/` | Inertial, joint origins, momen lentur |
| `reports/ur5e_nauo3_fea/` | CalculiX cantilever + stress report |
| `reports/ur5e_nauo3_neck_batch/` | 11 variasi leher + diagnostik falloff |
| `reports/thermal_fin_validate/` | Validasi solver termal |
| `reports/ur5e_nauo2_heal/`, `…_volume/` | Shoulder siap geometri |
| `README.md` (bagian UR5e) | Keputusan desain & batasan lisensi CAD |

Skrip terkait (contoh): `src/simjeb/heal_ur_solid.py`, `mesh_ur_volume.py`, `scripts/batch_nauo3_neck_scale.py`, `scripts/thermal_fin_analytic_validate.py`, `scripts/mesh_nauo2_volume.py`.

---

## Status saat ini & langkah logis berikutnya

**Selesai / siap pakai**

- [x] Heal + volume mesh NAUO3 (baseline)
- [x] Load case struktural tervalidasi (reaction |M| ≈ analytic)
- [x] Batch N=11 deform leher → bukti pipeline dataset otomatis
- [x] Tooling termal tervalidasi vs analitik
- [x] Fase 1 NAUO2 (heal + mesh)

**Belum**

- [ ] BC / FEA termal pada NAUO2 (perlu keputusan muka heat source)
- [ ] Scale dataset lebih besar / multi-parameter
- [ ] Geometri berongga (cable routing) — sengaja di luar scope PoC saat ini
- [ ] Material alloy resmi UR (STEP kosong; SF masih asumsi)

---

## Catatan proses kolaborasi yang berulang

- Job Gmsh/FEA panjang sering dijalankan **oleh user** (command detail dari agent); agent fokus analisis log + perbaikan kode.
- Debugging memakai siklus hipotesis → instrumentasi → bukti runtime (surface 1091, PLC, falloff kink, rim FILM).
- Banyak keputusan desain dikunci di `README.md` agar tidak hilang antar-sesi.
