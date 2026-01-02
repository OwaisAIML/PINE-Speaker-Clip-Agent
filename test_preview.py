import json

with open("storage/diarization.json") as f:
    data = json.load(f)

for spk, segments in data.items():
    total = sum(s["end"] - s["start"] for s in segments)
    print(f"{spk}: {len(segments)} segments | {total:.2f}s total")
