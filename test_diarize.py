from audio.diarize import run_diarization

AUDIO_PATH = "data/test.wav"

speakers = run_diarization(AUDIO_PATH)

print("\n=== SPEAKER SUMMARY ===\n")

for i, (speaker, segments) in enumerate(speakers.items(), start=1):
    total = sum(end - start for start, end in segments)
    print(f"Person_{i}")
    print(f"  Internal ID: {speaker}")
    print(f"  Segments: {len(segments)}")
    print(f"  Total Time: {total:.2f} seconds\n")
