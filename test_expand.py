from audio.expand_segments import expand_segments

expand_segments(
    input_json="storage/segments_final.json",
    output_json="storage/segments_expanded.json",
    video_duration=64.5  # your video duration
)
