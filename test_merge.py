from audio.merge_segments import merge_segments

merge_segments(
    diarization_json="storage/diarization.json",
    output_json="storage/segments_final.json"
)
