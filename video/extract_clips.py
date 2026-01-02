import json
import subprocess
from pathlib import Path


def extract_clips(
    video_path: str,
    segments_json: str,
    output_dir: str = "storage/clips"
):
    video_path = Path(video_path)
    segments_json = Path(segments_json)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not segments_json.exists():
        raise FileNotFoundError(f"Segments JSON not found: {segments_json}")

    with open(segments_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    clip_count = 0

    for speaker, segments in data.items():
        speaker_dir = output_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segments, start=1):
            start = round(float(seg["start"]), 2)
            end = round(float(seg["end"]), 2)

            # Skip invalid or zero-length segments
            if end <= start:
                continue

            out_file = speaker_dir / f"{speaker}_clip_{i:03d}.mp4"

            cmd = [
                "ffmpeg", "-y",

                # Fast & accurate seek
                "-ss", str(start),
                "-to", str(end),
                "-i", str(video_path),

                # Map video + audio safely (audio optional)
                "-map", "0:v:0",
                "-map", "0:a?",

                # Video encoding (editor-safe)
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",

                # Audio encoding
                "-c:a", "aac",
                "-b:a", "128k",

                # MP4 fast start (important)
                "-movflags", "+faststart",

                str(out_file)
            ]

            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True
                )
                clip_count += 1
                print(f"[OK] {out_file}")

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to extract clip: {out_file}")
                print(e.stderr.decode(errors="ignore"))

    print(f"\n✅ Extracted {clip_count} clips successfully")
