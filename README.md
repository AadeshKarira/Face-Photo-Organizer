# Photo Organizer — Group Photos by Face Identity

Automatically scans a directory of photos, detects faces, clusters them by identity, and copies/moves each image into a `Person_N` folder.

---

## How it works

| Phase | What happens |
|-------|-------------|
| **Scan** | Recursively finds all `.jpg/.jpeg/.png/.webp` files |
| **Detect** | InsightFace `buffalo_l` detects faces and extracts 512-d ArcFace embeddings |
| **Cluster** | DBSCAN groups embeddings by cosine similarity — no need to specify the number of people |
| **Organise** | Images are copied (or moved) into `Person_1/`, `Person_2/`, … folders |
| **Report** | A summary table + `report.json` is written to the output directory |

Images containing **multiple faces** are placed in every matching person's folder.

---

## Requirements

- Python 3.11+
- On first run, InsightFace automatically downloads the `buffalo_l` model pack (~300 MB) to `~/.insightface/models/`.

---

## Installation

```bash
# 1. Clone / download the project
cd photo_organizer

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU acceleration** (optional): replace `onnxruntime` with `onnxruntime-gpu` in
> `requirements.txt` and ensure CUDA + cuDNN are installed.  No code changes needed.

---

## Usage

```
python main.py --input <source_dir> --output <dest_dir> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | *(required)* | Source directory to scan recursively |
| `--output`, `-o` | *(required)* | Root output directory |
| `--threshold`, `-t` | `0.40` | Cosine similarity threshold for "same person".  Range `(0, 1)`. Higher = stricter. |
| `--move` | `False` | Move files instead of copying |
| `--det-size` | `640` | Detector input size in pixels (square).  Larger = better recall on small faces, slower. |
| `--det-thresh` | `0.5` | Minimum face-detection confidence score |
| `--verbose`, `-v` | `False` | Enable DEBUG logging |

### Examples

```bash
# Basic copy run
python main.py --input ~/Pictures --output ~/Sorted

# Stricter identity matching + move originals
python main.py --input ~/Pictures --output ~/Sorted --threshold 0.50 --move

# Lenient matching (useful for low-quality / varied-angle photos)
python main.py --input ~/Pictures --output ~/Sorted --threshold 0.30
```

---

## Output structure

```
Sorted/
├── Person_1/
│   ├── IMG_0042.jpg
│   └── IMG_0117.jpg
├── Person_2/
│   └── DSC_1234.jpg
├── Unknown/          ← faces DBSCAN couldn't cluster (very rare appearances)
│   └── …
├── report.json       ← machine-readable summary
└── photo_organizer.log
```

---

## Tuning `--threshold`

| Value | Effect |
|-------|--------|
| `0.25–0.35` | Lenient — might merge different people in large collections |
| `0.40–0.45` | **Recommended default** — good balance for varied lighting / angles |
| `0.50–0.60` | Strict — safer for look-alike siblings; may split one person across folders |

---

## Sample report output

```
────────────────────────────────────────────────────
  PHOTO ORGANIZER  —  SUMMARY REPORT
────────────────────────────────────────────────────
  Run completed at : 2026-05-31T14:22:07
  Elapsed time     : 142.3s
  Images scanned   : 1 200
  Faces detected   : 2 847
  Unique people    : 14
  Unassigned imgs  : 3

  Person               Images
  ------               ------
  Person_1                312
  Person_2                278
  Person_3                201
  …
────────────────────────────────────────────────────
```

---

## Performance notes

- **Memory**: embeddings are stored as `float32` arrays.  512 floats × 4 bytes = 2 KB per face.  1 million faces ≈ 2 GB RAM.  For very large collections consider processing in batches (the clustering step is the bottleneck).
- **Speed**: on a modern CPU, expect ~1–3 images/second.  With a GPU, ~10–30 images/second.
- **First run**: InsightFace downloads ~300 MB of model weights on the first run.

---

## License

MIT
