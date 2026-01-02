from audio.diarize import run_diarization
from utils.speaker_map import build_person_map, print_speaker_summary

AUDIO_PATH = "data/test.wav"

speakers = run_diarization(
    AUDIO_PATH,
    min_speakers=4,
    max_speakers=10
)


person_map = build_person_map(speakers)

print_speaker_summary(speakers, person_map)
