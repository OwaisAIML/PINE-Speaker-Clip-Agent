from pyannote.audio import Pipeline


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization"
)

def run_diarization(
    audio_path: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    """
    Run diarization with optional speaker constraints.
    """

    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.setdefault(speaker, []).append(
            (round(turn.start, 2), round(turn.end, 2))
        )

    return speakers
