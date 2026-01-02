import json
from pathlib import Path

PADDING = 1.5  # seconds

def expand_segments(
    input_json: str,
    output_json: str,
    video_duration: float
):
    input_json = Path(input_json)
    output_json = Path(output_json)

    with open(input_json, "r") as f:
        data = json.load(f)

    expanded = {}

    for speaker, segments in data.items():
        expanded[speaker] = []

        for seg in segments:
            start = max(0.0, seg["start"] - PADDING)
            end = min(video_duration, seg["end"] + PADDING)

            expanded[speaker].append({
                "start": round(start, 2),
                "end": round(end, 2)
            })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(expanded, f, indent=2)

    print(f"Segments expanded successfully → {output_json}")
