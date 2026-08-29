import os
import uuid
import shutil
from pathlib import Path

TEMP_DIR = Path("E:/Hackverse2k26/backend/temp")

def init_temp_dir():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

def get_temp_path(suffix: str) -> Path:
    init_temp_dir()
    return TEMP_DIR / f"{uuid.uuid4()}{suffix}"

def cleanup_file(path: str | Path):
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    except Exception as e:
        print(f"Error cleaning up {path}: {e}")
