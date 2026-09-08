# video-removal

Hapus objek (orang, benda) dari video lewat CLI. Prompt teks atau klik satu titik.

```
python remove.py in.mp4 -o out.mp4 --prompt "person"
```

## Cara kerjanya

```
video ──ffmpeg──> frames (full res)  ─────────────────────┐
                  frames (work res, 720p) ──┐             │
                                            ▼             │
              Grounding DINO ── box ──> SAM 2 ──> mask/frame
                                            │             │
                            dilate + temporal pad         │
                                            ▼             │
                                      INPAINTER           │
                                   (pluggable backend)    │
                                            │             │
                                            ▼             ▼
                              composite hanya area mask ke full res
                                            │
                                            ▼
                                  ffmpeg + audio asli ──> out.mp4
```

Dua keputusan desain yang menentukan kualitas hasil:

**1. Mask sengaja dilebarkan.** Mask yang pas-pasan meninggalkan bayangan, motion
blur, dan tepi rambut di luar area yang diisi — dan inpainter akan
merekonstruksi sisa itu dengan setia, jadinya "halo" bekas objek. Default
`--dilate 12` plus `--temporal-pad 2`. Kalau masih ada halo, naikkan.

**2. Inpaint di 720p, tempel balik hanya area mask.** Pixel di luar mask tidak
pernah disentuh model, jadi source 4K tetap 4K di seluruh area yang tidak ada
objeknya. Ini yang bikin hasil tajam tanpa perlu GPU 80GB.

## Backend inpainting

`python remove.py --list-backends`

| backend | komersial | VRAM | mask besar | catatan |
|---|---|---|---|---|
| `median` | ya | 0 | 5/5 | **kamera statis saja.** Median temporal per-pixel → background asli, bukan tebakan |
| `propainter` | **TIDAK** (S-Lab 1.0) | ~10GB | 2/5 | rasio kualitas/VRAM terbaik, tapi buyar di mask lebar/lama |
| `external` | tergantung repo | — | 4/5 | jalur upgrade, lihat di bawah |

### Kasusmu: objek besar / lama di layar

Ini titik lemah ProPainter. Ia memakai flow-guided propagation — mencari pixel
asli dari frame lain untuk ditempel. Kalau objeknya besar dan tidak pernah
pindah, **tidak ada frame yang pernah melihat background di baliknya**, jadi
tidak ada yang bisa dipropagasi dan hasilnya melebur jadi blur.

Pipeline akan memperingatkanmu otomatis:

```
[warn] mask covers 18.4% of frame and backend 'propainter' scores 2/5 on large masks.
```

Yang bisa dilakukan, berurutan dari yang paling murah:

1. **Kalau kameranya statis, pakai `--backend median`.** Bukan kompromi — ini
   lebih baik dari model manapun, karena memulihkan pixel background yang
   sesungguhnya. Kalau objeknya sempat bergeser sedikit saja sepanjang klip,
   ini menang telak.
2. **Potong jadi segmen pendek.** ProPainter jauh lebih baik di klip 3-5 detik
   daripada 30 detik.
3. **Naik ke model diffusion** lewat `--backend external` — model diffusion
   meng-*halusinasi* background yang plausible, jadi tidak butuh frame referensi
   sama sekali. Ini juga jalur lisensi komersialmu.

### Upgrade ke backend komersial

`propainter` berlisensi S-Lab 1.0 (non-komersial). Karena statusmu masih
"mungkin komersial nanti", semua backend ada di balik satu interface
([base.py](vremove/inpaint/base.py)) — ganti backend tidak menyentuh pipeline.

Kandidat berlisensi lebih longgar: **DiffuEraser** (Alibaba, ProPainter-prior +
diffusion, kuat di mask besar & video panjang), **MiniMax-Remover**, dan
**VACE / Wan2.1** (kualitas tertinggi, 24-48GB).

Flag CLI tiap repo itu berubah antar rilis, jadi jangan ditebak — pasang
repo-nya di pod, baca README-nya, lalu sambungkan:

```bash
python remove.py in.mp4 -o out.mp4 --prompt "person" \
  --backend external \
  --backend-cmd "python /workspace/DiffuEraser/run.py --input {frames} --mask {masks} --output {out}"
```

Placeholder: `{frames}` `{masks}` `{out}` `{fps}`. Semua `*.png`/`*.jpg` di
bawah `{out}` dikumpulkan urut. Kalau sebuah repo terbukti bagus, promosikan
jadi adapter permanen — contoh polanya ada di
[propainter.py](vremove/inpaint/propainter.py).

⚠️ Lisensi di atas berdasarkan status yang saya tahu, dan bisa berubah. **Cek
sendiri file LICENSE tiap repo sebelum dipakai komersial** — termasuk lisensi
bobot modelnya, yang kadang berbeda dari lisensi kodenya.

## Setup di RunPod

Pakai **Pod**, bukan Serverless, selama tahap ini. Siklus `docker build` → push
→ test di serverless itu 10+ menit sekali coba; tuning parameter mask butuh
puluhan iterasi. Serverless nanti setelah settingnya ketemu.

Pod: template PyTorch, GPU 24GB (4090 / A5000 / L4), disk ≥50GB.

```bash
git clone <repo-ini> /workspace/video-removal && cd /workspace/video-removal
bash scripts/setup_pod.sh
source scripts/env.sh
```

Alur kerja — **selalu cek mask dulu**, jangan langsung inpaint:

```bash
# 1. lihat mask-nya (detik, bukan menit)
python remove.py in.mp4 -o preview.mp4 --prompt "person" --preview

# 2. mask kurang lebar? bocor ke objek lain? tuning
python remove.py in.mp4 -o preview.mp4 --prompt "person" --preview --dilate 20

# 3. baru jalankan yang asli
python remove.py in.mp4 -o out.mp4 --prompt "person" --dilate 20 --work-dir work/
```

`--work-dir` menyimpan frame antara, jadi kalau inpainting gagal kamu tidak
perlu mengulang masking.

Kalau prompt teks meleset, tunjuk manual:

```bash
python remove.py in.mp4 -o out.mp4 --point 640,380 --point 900,200,0
#                                    ↑ ini objeknya   ↑ label 0 = BUKAN objeknya
```

## RunPod Serverless

```
laptop                          R2/S3                  RunPod worker
  │                               │                          │
  ├── upload in.mp4 ─────────────>│                          │
  ├── sign GET(in) + PUT(out)     │                          │
  ├── POST /run {urls, prompt} ───────────────────────────── >│
  │<── job id                                                 ├─ GET in.mp4
  ├── poll /status/{id} ───────────────────────────────────── ├─ mask → inpaint
  │<── 40% inpainting ...                                     ├─ PUT out.mp4
  │<── COMPLETED + stats                                      │
  ├── download out.mp4 <──────────│                          │
```

Worker tidak pernah memegang kredensial bucket — hanya URL bertanda tangan yang
kedaluwarsa. File: [worker/handler.py](worker/handler.py),
[worker/storage.py](worker/storage.py), [Dockerfile](Dockerfile),
klien [client/submit.py](client/submit.py).

### Build & deploy

Dua jalur. **Yang direkomendasikan: biarkan RunPod build dari GitHub** — kamu
hanya push ~60KB kode, dan seluruh unduhan besar (base image 3GB + bobot 2,2GB)
terjadi di datacenter mereka, bukan lewat koneksi rumahmu.

```bash
git push                       # itu saja
```

Lalu di RunPod → Serverless → New Endpoint, pilih sumber **GitHub repo**,
hubungkan akun GitHub-mu, pilih repo ini, dan set **Dockerfile path = `Dockerfile`**
(di root — memang ditaruh di situ supaya build context-nya benar).

Alternatif, build lokal lalu push image sendiri:

```bash
docker build -t <user>/video-removal:v1 .
docker push <user>/video-removal:v1     # ~2,2GB upload; layer base biasanya di-mount
```

Setting endpoint:

| setting | nilai | alasan |
|---|---|---|
| GPU | 24GB (4090 / L4 / A5000) | ProPainter ~10GB + SAM 2 |
| Container image | `<user>/video-removal:v1` | |
| **FlashBoot** | **on** | memangkas cold start |
| Idle timeout | 5-10 detik | worker nganggur tetap dibayar |
| Execution timeout | 900 detik+ | job video itu menit, bukan detik |
| Max workers | 1-2 dulu | pagar biaya saat masih eksperimen |
| Network volume | **jangan** | mengunci endpoint ke satu datacenter → ketersediaan GPU anjlok saat ramai. Bobot sudah di-bake ke image |

Env opsional di endpoint: `MAX_INPUT_SECONDS` (default 60), `MAX_INPUT_MB`
(300), `MAX_WORK_RES` (1080), `ALLOW_BACKEND_CMD` (default off).

### Pakai

```bash
pip install -r client/requirements.txt
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
export S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
       S3_BUCKET=... S3_ACCESS_KEY_ID=... S3_SECRET_ACCESS_KEY=...

# selalu preview dulu -- ini murah
python client/submit.py in.mp4 -o preview.mp4 --prompt "person" --preview
python client/submit.py in.mp4 -o out.mp4     --prompt "person" --dilate 20
```

Belum mau setup R2? `--inline` mengirim base64 lewat payload (klip <8MB saja),
cukup untuk membuktikan endpoint hidup.

**Tes pertama setelah deploy** — buktikan plumbing-nya dulu, tanpa melibatkan
model sama sekali:

```bash
python client/submit.py short.mp4 -o out.mp4 --box 400,200,700,600 \
       --backend median --inline
```

Kalau ini berhasil, berarti download → proses → upload → polling semuanya jalan.
Baru setelah itu error apa pun yang muncul pasti berasal dari model, bukan dari
infrastruktur — dan itu memangkas ruang debug drastis.

Ctrl-C saat polling akan **cancel job**-nya, bukan sekadar keluar — job yang
ditinggalkan tetap membakar detik GPU.

### Keputusan yang sudah dikunci di kode

- **Model dimuat sekali saat import**, bukan per job
  ([handler.py:47](worker/handler.py#L47)). Worker hangat melewati ~15 detik
  loading yang kalau tidak, dibayar tiap request.
- **Async `/run` + polling**, bukan `/runsync` — koneksi HTTP sinkron akan
  timeout jauh sebelum video selesai.
- **Cap input** durasi & ukuran. Endpoint serverless akan dengan senang hati
  menagihmu untuk klip 40 menit yang ter-submit tidak sengaja.
- **CUDA OOM → `refresh_worker`** ([handler.py:169](worker/handler.py#L169)).
  Worker yang sudah OOM biasanya rusak permanen; tanpa ini ia akan menggagalkan
  semua job berikutnya.
- **SAM 2 offload ke CPU** (`offload_video_to_cpu`). Tanpa ini SAM 2 menahan
  seluruh klip di VRAM dan video panjang OOM.

Ongkos kasar: kelas 24GB ≈ $0.0003-0.0005/detik. Klip 10 detik @720p ≈ 2-4 menit
proses ≈ beberapa sen. Cek pricing terkini, angkanya sering berubah.

## Status

Diuji di WSL2 (Ubuntu 24.04, Python 3.12, ffmpeg 6.1) dengan klip sintetis
640x360 25fps 4 detik bersuara.

**Sudah terverifikasi jalan:**

| bagian | bukti |
|---|---|
| syntax seluruh repo | `compileall` bersih |
| `video_io` extract/encode/remux | 100 frame masuk → 100 keluar, 640x360 & audio utuh |
| `static_box_masks` + `postprocess_masks` | coverage 24.5% sesuai hitungan |
| `median` jalur fallback | "24.49% occluded in every frame" → TELEA |
| `median` jalur sebenarnya | mask bergerak → objek hilang **6400/6400 → 0 pixel**, plate merekonstruksi background |
| `composite_hires` | pixel di luar mask drift mean 0.785 / max 23 = persis lantai H.264 |
| CLI end-to-end | `--box` + `--backend median` |

**Belum diuji sama sekali:** SAM 2, Grounding DINO, ProPainter, Docker build,
`worker/handler.py`, `client/submit.py`. Semua itu butuh pod.

### Satu default yang terbukti salah dan sudah diperbaiki

Frame full-res dulu diekstrak sebagai JPEG q2. Diukur pada pixel **di luar**
mask — yang justru menjadi seluruh alasan adanya composite hi-res:

| intermediate | mean error | max error |
|---|---|---|
| JPEG q2 | 1.72 | 91 |
| PNG | 0.785 | 23 |

PNG sekarang jadi default. Ongkosnya cuma ~2x scratch disk (klip 30 detik
1080p: 200MB vs 104MB pada footage sintetis; footage asli lebih besar, kira-kira
2-4MB/frame). `--jpeg-frames` untuk kembali ke perilaku lama kalau disk mepet.

### Urutan verifikasi berikutnya

Langkah 1 dan 2 sudah saya jalankan; ulangi kalau kamu mengubah kode.

```bash
# di WSL -- venv wajib, Ubuntu 24.04 memblokir pip ke system python (PEP 668)
python3 -m venv ~/venvs/vremove
~/venvs/vremove/bin/pip install numpy opencv-python-headless Pillow tqdm

# 1. syntax, tanpa GPU, tanpa dependency berat
~/venvs/vremove/bin/python -m compileall -q remove.py vremove worker client

# 2. seluruh pipeline TANPA model apa pun, tanpa GPU, tanpa download.
#    --box = rectangle statis, --backend median = tidak pakai neural net.
#    Menguji: extract -> mask shaping -> inpaint -> composite -> encode + audio.
~/venvs/vremove/bin/python remove.py clip.mp4 -o out.mp4 \
    --box 400,200,700,600 --backend median

# 3. baru masking (SAM 2) + inpainting -- butuh torch, di pod
```

Langkah 2 penting: kalau ada bug di plumbing — nama file frame, urutan mask,
remux audio, aritmetika composite — kamu menemukannya di laptop dalam hitungan
detik, bukan setelah membakar 10 menit GPU.

Titik gagal yang paling mungkin, berurutan:

- **Flag `inference_propainter.py`** — diambil dari CLI ProPainter tapi bisa
  bergeser antar commit. Adapter mencetak command lengkapnya sebelum jalan, jadi
  mudah dikoreksi.
- **URL rilis bobot ProPainter** di [fetch_weights.sh](scripts/fetch_weights.sh)
  — kalau 404, tag-nya pindah; bump `PP_TAG`. Build tidak gagal, cuma cold start
  jadi lambat karena ProPainter mengunduh sendiri saat job pertama.
- **`post_process_grounded_object_detection`** ganti nama argumen (`threshold` vs
  `box_threshold`) di transformers ~4.51. Sudah di-handle try/except, belum diuji.
- **Path config SAM 2** (`configs/sam2.1/sam2.1_hiera_l.yaml`) di-resolve Hydra
  dari dalam package `sam2`, bukan relatif ke cwd.
- **Tag base image** `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` — bump kalau
  driver RunPod sudah lebih maju.

Ukuran image: base `-runtime` 3.0GB terkompresi + layer milik kita ~2.2GB
(bobot SAM 2 900MB, ProPainter ~320MB, cache Grounding DINO ~700MB, deps
~270MB). Yang benar-benar kamu upload saat push adalah ~2.2GB itu — layer base
biasanya di-*mount* dari repo publik Docker Hub, bukan diunggah ulang.

Push berikutnya cepat karena hanya layer kode yang berubah — itu sebabnya
`COPY vremove/` diletakkan paling akhir di Dockerfile.
