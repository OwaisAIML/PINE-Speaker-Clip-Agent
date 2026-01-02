# audio/extract.py
import subprocess
import sys
from pathlib import Path

def extract_audio(video_path, wav_path):
    video_path = Path(video_path)
    wav_path = Path(wav_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path)
    ], check=True)

    print(f"[OK] Audio extracted to {wav_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract.py <video_path> <output_wav>")
        sys.exit(1)

    extract_audio(sys.argv[1], sys.argv[2])
