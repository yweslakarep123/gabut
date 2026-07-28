# Proposal sampling NAUO3 — revisi final (angka terkoreksi)

Inventaris dasar: `reports/ur5e_nauo3_dataset_manifest.json` (**14** titik unik + 2 duplikat).
Dokumen ini menggantikan hitungan section 4 yang inkonsisten. **Siap dieksekusi** setelah revisi lima poin di bawah — tanpa putaran review lagi untuk perbaikan aritmatika/desain grid ini.

---

## 0. Audit `*_gauss35` (dilakukan sebelum desain grid)

| | |
|---|---|
| Path | `reports/ur5e_nauo3_neck_batch/s_0.90_gauss35/`, `s_1.10_gauss35/` + ringkasan `gauss35_trial.json` |
| Topologi | `n_nodes=38811`, `n_tets=192060`, tet connectivity (sorted) = `geometry.npz` terkunci — **kompatibel** |
| `falloff_kind` | `gaussian_nocut`, `sigma=35` |
| Hotspot s=0.90 | node **3279**, z≈176.17 — **same lobe** dengan cosine/bump |
| Gate | pass; V0 sama dengan baseline terkunci |
| Packaging | ada `.frd`/`.inp`; **tidak** ada `sample_result.json` (hanya `gauss35_trial.json`) |
| API aktif | `gaussian_nocut` **tidak** ada di `neck_weight()` sekarang (hanya `gaussian_cut`, `cosine_C1`, `smootherstep_C2`, `bump_Cinf`) |

**Keputusan reuse:** +2 titik corpus gratis untuk diversitas training (`gaussian_nocut` @ 0.90 & 1.10). **Jangan** alokasikan run baru bertipe `gaussian_nocut` sampai kernel itu dikembalikan ke API. Falloff ketiga yang di-grid di batch ini: `smootherstep_C2` (sudah di kode).

---

## 1. Angka jujur inventaris (tidak berubah)

| | Jumlah |
|---|---|
| Scan mentah | **16** |
| Unik | **14** |
| Duplikat | **2** |
| `.frd` pada unik | **14/14** |
| Bonus corpus (`gauss35`, belum di manifest formal) | **+2** jika diangkat kemudian |

---

## 2. Koreksi aritmatika overlap (grid lama step 0.05)

Grid yang salah dihitung sebelumnya: `s ∈ {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15}` (step **0.05**).

Irisan **persis** dengan lattice Phase-1 step 0.02 `{0.90, 0.92, …, 1.10}`:

| Falloff | Overlap dengan data existing | n |
|---|---|---|
| `cosine_C1` | {0.90, 1.00, 1.10} | **3** (bukan ~8) |
| `bump_Cinf` | {0.90} | **1** |
| Total reuse | | **4** |

Kalau grid 3×7 (=21) dipertahankan apa adanya → run baru = 21 − 4 = **17** (bukan ~12–15). Angka itu sekarang **ditinggalkan** bersama step 0.05.

---

## 3. Desain grid baru: step **0.02**, reuse maksimal

Perpanjang lattice yang sama:  
`s ∈ {0.86, 0.88, 0.90, 0.92, …, 1.10, 1.12, 1.14}` (= 15 nilai).

| Sel | Sudah ada | Run baru |
|---|---|---|
| `cosine_C1` × {0.90 … 1.10} | **11** | **0** |
| `cosine_C1` × {0.86, 0.88, 1.12, 1.14} | 0 | **4** (ekstensi saja) |
| `bump_Cinf` × {0.90} | **1** | 0 |
| `gaussian_nocut` × {0.90, 1.10} | **2** (corpus) | 0 — jangan regenerate |

Tidak ada re-run cosine di axis yang sudah diuji. Tidak ada budget untuk “cek reproduktifitas in-place” (lihat §4).

---

## 4. Yang *tidak* dianggarkan

- Re-run parameter identik untuk “reproduktifitas in-place” — in-place lebih deterministik daripada remesh (tidak ada Delaunay); sudah tersubsumsi temuan `determinism_rerun1_vs_rerun2`.
- Grid bahu / `y0_shoulder` / compose 2D.
- Remesh-per-sample.
- `gaussian_nocut` baru (kernel tidak di API).
- Memadatkan lagi step di dalam `[0.90, 1.10]` untuk cosine.

---

## 5. Tujuan batch (eksplisit) + porsi budget

Batch ini melayani **dua** tujuan; ukuran = jumlah keduanya, bukan “grid yang kelihatan bagus.”

### (b) Primer — konsistensi efek falloff across `s` (~69% budget baru)

Pertanyaan: apakah delta falloff ~1.5% di s=0.90 bertahan di s lain, atau lokal di satu titik?

Uji pada **5 nilai s yang sudah punya cosine_C1**: `{0.90, 0.94, 0.98, 1.00, 1.10}`.

| Kind | s yang diuji | Sudah ada | **FEA baru** |
|---|---|---|---|
| `bump_Cinf` | 5 | 1 (@0.90) | **4** |
| `smootherstep_C2` | 5 | 0 | **5** |
| Subtotal (b) | | | **9** |

### (a) Sekunder — diversitas corpus / lebar `s` (~31% budget baru)

Perpanjang cosine saja di luar interval tertutup:

| Kind | s | **FEA baru** |
|---|---|---|
| `cosine_C1` | 0.86, 0.88, 1.12, 1.14 | **4** |

### Total eksekusi

| | |
|---|---|
| **FEA baru** | **13** |
| Reuse langsung (tidak dijalankan ulang) | 11 cosine + 1 bump + (opsional +2 gauss corpus) |
| Titik unik setelah batch (tanpa mengangkat gauss ke manifest) | 14 + 13 = **27** |
| Jika gauss diangkat ke manifest setelah packaging | 27 + 2 = **29** |

Masih di bawah ambang sanity surrogate (~40–60), tapi cukup untuk (b) terjawab dengan angka dan corpus bertambah ~2× tanpa membuang run ke cosine yang sudah closed.

---

## 6. Label & objektif (dibawa maju, tidak diubah)

- Field nodal penuh dari `.frd` untuk training nanti.
- `sigma_max` = argmax hampir degenerate dua lobe → soft-max / p-norm top-k saat optimisasi gradien.
- Floor ~1% pada label `max` ditoleransi; sinyal desain ~38% dua orde lebih besar.
- Metodologi: **deform in-place** pada `geometry.npz` terkunci saja.

---

## 7. Keputusan eksekusi — selesai

- Jangan training Phase 3 sekarang (27 << 40–60; 29 jika gauss diangkat).
- Jangan ekspansi bahu.
- **Sudah dijalankan:** 13/13 FEA `ok` → `reports/ur5e_nauo3_falloff_s_grid/` (`batch_summary.json`), manifest di-append → **27** unik.

### Purpose (b) — delta vs cosine (hasil, dianotasi same-feature)

| s | bump_Cinf | smootherstep_C2 | same_feature_as_baseline_center (node 3279) |
|---|---|---|---|
| 0.90 | **−1.511%** | −0.409% | true |
| 0.94 | −0.953% | −0.245% | true |
| 0.98 | −0.333% | −0.080% | true |
| 1.00 | 0.000% | 0.000% | true |
| 1.10 | +4.806% | −0.812% | **false** (ketiganya di node 13229, BAND_EDGE) |

Pada s∈{0.90…1.00} perbandingan bersih (node 3279, BAND_CENTER): delta falloff mengecil menuju nol saat s→1 — masuk akal karena semua kernel identik di s=1. Angka +4.806% di s=1.10 **bukan** “efek shape berbalik saat menebal”; itu perbandingan lintas-fitur (bahu vs leher). Jangan di-rata-rata atau di-fit bersama empat titik center sebagai satu tren falloff. Lihat `purpose_b_comparison_note` di `batch_summary.json`.

### Observed σ reversal on shoulder feature for s>1.10 (cosine_C1, same node)

Pada cosine_C1, hotspot node **13229** (BAND_EDGE) untuk ketiga titik berikut — perbandingan same-feature yang sah:

| s | σ_max (MPa) | Δ vs s=1.10 |
|---|---|---|
| 1.10 | 1.4864 | — |
| 1.12 | 1.5414 | **+3.70%** |
| 1.14 | 1.5956 | **+3.52%** (vs 1.12) |

Makin tebal di interval ini, σ_max naik lagi — dua langkah berturut, arah sama, magnitude jauh di atas noise floor remesh ~1.13% yang sudah dikarakterisasi. Itu bertentangan dengan asumsi implisit “lebih tebal = lebih aman” yang mendasari perluasan ke s=1.14. Hipotesis mekanistik yang masuk akal: menebalkan leher menaikkan kekakuan relatif di band pusat dan mendorong lebih banyak momen ke bahu yang tidak ikut diperkuat. **Observed in 3 points, same-node comparison, not yet independently re-verified with an additional point (e.g. s=1.16) to confirm the trend continues** — jangan dinaikkan jadi klaim monoton tanpa titik tambahan. Tidak ada FEA verifikasi di task anotasi ini.
