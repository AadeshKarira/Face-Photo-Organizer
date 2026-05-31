# Photo Organizer Commands

## Setup

```powershell
cd C:\Users\aades\OneDrive\Desktop\pet\photo_organizer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python.exe main.py --input "C:\Users\aades\OneDrive\Desktop\pet\test-images" --output "C:\Users\aades\OneDrive\Desktop\pet\test-images-output"
```

## Options

- `--input <dir>`: source directory to scan recursively for images.
- `--output <dir>`: destination directory where `Person_N` folders will be created.
- `--threshold <float>`: similarity threshold for grouping faces; higher values are stricter.
- `--move`: move files instead of copying them.
- `--det-size <px>`: face detector input size.
- `--det-thresh <float>`: minimum face detection confidence.
- `--verbose` or `-v`: enable debug logging.
