import subprocess
from pathlib import Path


def extract_clip(video_path, start, end, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if end <= start:
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", f"{start:.2f}",
        "-to", f"{end:.2f}",
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.0",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(cmd, check=True)
