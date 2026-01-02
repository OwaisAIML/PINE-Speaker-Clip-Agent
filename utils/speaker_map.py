def build_person_map(speakers: dict) -> dict:
    """
    Convert diarization speaker IDs into user-friendly Person_X labels.

    Input:
        {
          "SPEAKER_00": [...],
          "SPEAKER_01": [...],
          ...
        }

    Output:
        {
          "Person_1": "SPEAKER_00",
          "Person_2": "SPEAKER_01",
          ...
        }
    """
    person_map = {}

    for idx, speaker_id in enumerate(speakers.keys(), start=1):
        person_map[f"Person_{idx}"] = speaker_id

    return person_map
def print_speaker_summary(speakers: dict, person_map: dict):
    print("\n=== DETECTED SPEAKERS ===\n")

    for person, speaker_id in person_map.items():
        segments = speakers[speaker_id]
        total_time = sum(end - start for start, end in segments)

        print(f"{person}")
        print(f"  Internal ID : {speaker_id}")
        print(f"  Segments    : {len(segments)}")
        print(f"  Total Time  : {total_time:.2f} seconds\n")
