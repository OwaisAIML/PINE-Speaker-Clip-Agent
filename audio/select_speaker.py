import json
from pathlib import Path


def select_target_speaker(
    segments_json: str,
    target_speaker: str,
    output_json: str = "storage/target_segments.json"
):
    segments_json = Path(segments_json)
    output_json = Path(output_json)

    with open(segments_json, "r") as f:
        segments = json.load(f)

    if target_speaker not in segments:
        raise ValueError(f"Speaker {target_speaker} not found")

    target_segments = segments[target_speaker]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(
            {
                "speaker": target_speaker,
                "segments": target_segments
            },
            f,
            indent=2
        )

    return target_segments
