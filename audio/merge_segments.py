import json
from pathlib import Path

MIN_DURATION = 0.4      # seconds
MAX_GAP = 0.4           # seconds


def merge_segments(
    diarization_json: str,
    output_json: str
):
    diarization_json = Path(diarization_json)
    output_json = Path(output_json)

    with open(diarization_json, "r") as f:
        data = json.load(f)

    merged = {}

    for speaker, segments in data.items():
        merged_segments = []
        current = None

        for seg in segments:
            start = seg["start"]
            end = seg["end"]

            if current is None:
                current = {"start": start, "end": end}
                continue

            gap = start - current["end"]

            if gap <= MAX_GAP:
                # merge
                current["end"] = end
            else:
                # finalize previous
                if current["end"] - current["start"] >= MIN_DURATION:
                    merged_segments.append(current)
                current = {"start": start, "end": end}

        # last segment
        if current and (current["end"] - current["start"] >= MIN_DURATION):
            merged_segments.append(current)

        if merged_segments:
            merged[speaker] = merged_segments

    with open(output_json, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Segments merged successfully → {output_json}")
