# Laporan: Scaffold Website Jazari — Nav + Hero

**Tanggal:** 24 Juli 2026  
**Fase:** Phase 0 + Phase 1 (scaffold + Nav + Hero)  
**Sumber brief:** [`jazari-website-cursor-brief.md`](../jazari-website-cursor-brief.md)  
**Lokasi app:** [`website/`](../website/)

---

## Ringkasan

Website marketing/portfolio **Jazari** di-scaffold sebagai aplikasi Next.js di dalam monorepo pipeline Python yang sudah ada. Fase ini berhenti setelah **Nav** dan **Hero** (termasuk background canvas mesh) berfungsi di mode development. Belum ada section Approach, Pipeline, Case Study, Log, Contact, atau halaman `/case-study`.

---

## Keputusan penting sebelum coding

| Keputusan | Pilihan | Alasan |
|-----------|---------|--------|
| Lokasi project | Folder `website/` (bukan root repo) | Root sudah berisi pipeline Python (`src/`, `scripts/`, `data/`, `reports/`) |
| Tailwind | **v4** via `@theme` di `globals.css` | `create-next-app` 2026 default ke Tailwind v4 + `@tailwindcss/postcss`, bukan `tailwind.config.ts` |
| Animasi | Paket `motion` (`motion/react`) | Rebrand Framer Motion; bukan legacy `framer-motion` |
| Font | Geist Sans + Geist Mono bawaan scaffold | Sudah ter-setup di `layout.tsx`; tidak dipasang ulang |
| Content data | Stub typed kosong | Prompt fase ini: jangan isi angka inventory dulu |
| Dark mode | Hanya dark, tanpa toggle | Sesuai brief |

---

## Urutan kerja (bagaimana dibuat)

### 1. Konteks agen — `AGENTS.md`

Dibuat di **root repo** dari Part A brief. Isinya: apa itu Jazari, audience, tone rules, design tokens, inventory angka tervalidasi, dan larangan overclaim.

File: [`AGENTS.md`](../AGENTS.md)

### 2. Environment Node.js

Mesin belum punya `node`/`npx`. Dipasang lewat **nvm** (Node.js LTS v24), lalu dipakai untuk `create-next-app` dan `npm`.

### 3. Scaffold Next.js

```bash
npx create-next-app@latest website \
  --typescript --tailwind --eslint --app \
  --no-src-dir --import-alias "@/*" --use-npm --yes
```

Hasil: Next.js **16.2.11**, React **19**, Tailwind **4**, App Router, tanpa folder `src/`.

Lalu:

```bash
cd website && npm install motion
mkdir -p components/sections content lib
```

### 4. Update `.gitignore` root

Ditambah ignore untuk Node/Next agar `website/node_modules/` dan `website/.next/` tidak ikut commit:

```
website/node_modules/
website/.next/
website/out/
website/.env
website/.env.*
website/.vercel/
```

### 5. Design tokens + fonts

Warna dimasukkan ke [`website/app/globals.css`](../website/app/globals.css) dengan sintaks Tailwind v4:

```css
@theme inline {
  --color-ink-950: #08090b;
  --color-ink-900: #0f1113;
  --color-ink-800: #1a1d21;
  --color-paper-50: #f5f5f3;
  --color-paper-400: #9a9ca3;
  --color-signal-hot: #ff6a3d;
  --color-signal-cold: #3dd6ff;
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);
}
```

[`website/app/layout.tsx`](../website/app/layout.tsx): metadata title **Jazari**, body `bg-ink-950 text-paper-50`, Geist sudah dari scaffold.

Easing bersama di [`website/lib/motion.ts`](../website/lib/motion.ts): `[0.16, 1, 0.3, 1]`, fade + slide-up 12px.

### 6. Content stubs (kosong)

| File | Isi |
|------|-----|
| `content/metrics.ts` | Tipe `Metric` + array `[]` |
| `content/pipeline.ts` | Tipe `PipelineStage` + array `[]` |
| `content/log.ts` | Tipe `LogEntry` + array `[]` |

Siap diisi di fase berikutnya tanpa mengubah layout JSX.

### 7. Nav

File: [`website/components/Nav.tsx`](../website/components/Nav.tsx) (client component)

- Fixed top
- Transparan di atas hero; setelah scroll ~85% tinggi viewport → `bg-ink-900` + border `ink-800`
- Wordmark: `JAZARI` (Geist Mono, letter-spaced)
- Links: Approach · Pipeline · Case Study · Log · Contact → `#approach`, `#pipeline`, `#case-study`, `#log`, `#contact`
- CTA: **Talk to the engineer** → `#contact`
- Mobile: link horizontal scroll (tanpa hamburger overbuilt)

### 8. Hero + MeshCanvas

**Hero** — [`website/components/sections/Hero.tsx`](../website/components/sections/Hero.tsx)

Copy sesuai brief (tidak diubah wording):

- Eyebrow: `PHYSICS-VALIDATED AI FOR ROBOTICS HARDWARE`
- H1: *We don't ship a prediction until it survives contact with the ground truth.*
- Subhead: pipeline CAD → surrogate model
- Primary CTA: **See the validation →** → `#case-study`
- Secondary: **Read the pipeline** → `#pipeline`

Reveal masuk dengan `motion` + `useReducedMotion`.

**MeshCanvas** — [`website/components/MeshCanvas.tsx`](../website/components/MeshCanvas.tsx)

- Grid wireframe low-poly (14×9), garis tipis opacity ~0.2
- Displacement vertex sine-wave lambat
- `requestAnimationFrame`
- Pause jika `document.visibilityState !== 'visible'`
- Skip animasi jika `prefers-reduced-motion: reduce` (frame statis)
- DPR capped ke 2

### 9. Wire-up halaman

[`website/app/page.tsx`](../website/app/page.tsx) hanya:

```tsx
<main>
  <Nav />
  <Hero />
</main>
```

### 10. Verifikasi

- `npx tsc --noEmit` — lulus
- `npm run build` — lulus (route `/` static)
- `npm run dev` — HTTP 200 di `http://localhost:3000`
- Copy kunci (JAZARI, eyebrow, headline, CTA) muncul di HTML

---

## Struktur file yang dihasilkan

```
jazari/
├── AGENTS.md                 # konteks brand/tone untuk agen
├── .gitignore                # + ignore Node/Next di website/
└── website/
    ├── app/
    │   ├── globals.css       # tokens Tailwind v4 + dark base
    │   ├── layout.tsx        # Geist + metadata Jazari
    │   └── page.tsx          # Nav + Hero saja
    ├── components/
    │   ├── Nav.tsx
    │   ├── MeshCanvas.tsx
    │   └── sections/
    │       └── Hero.tsx
    ├── content/
    │   ├── metrics.ts        # stub kosong
    │   ├── pipeline.ts       # stub kosong
    │   └── log.ts            # stub kosong
    ├── lib/
    │   └── motion.ts         # easing / reveal tokens
    └── package.json          # next, react, motion, tailwind v4
```

---

## Yang sengaja belum dibuat

Sesuai scope fase ini, **belum** ada:

- Section §01–§05 (Approach, Pipeline, Case Study home, About, Field Notes)
- Halaman `/case-study`
- Isi angka di `metrics` / `pipeline` / `log`
- Proof strip + animated counters
- Footer

Itu masuk Phase 2–4 di Part C brief.

---

## Cara menjalankan

```bash
cd website
npm run dev
```

Buka: [http://localhost:3000](http://localhost:3000)

---

## Catatan teknis untuk fase berikutnya

1. Token warna sudah di `globals.css` (`@theme`), bukan `tailwind.config.ts`.
2. Import animasi: `import { motion } from "motion/react"`.
3. Anchor nav sudah siap (`#approach`, `#pipeline`, dll.) — section berikutnya cukup pakai `id` yang sama.
4. Isi `content/*.ts` dulu sebelum menulis JSX section, agar copy tidak hardcode di komponen.
