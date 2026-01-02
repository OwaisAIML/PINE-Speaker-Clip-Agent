import streamlit as st
from pathlib import Path

from audio.extract import extract_audio
from audio.diarize import run_diarization
from utils.speaker_map import build_person_map
from extract.extract_all_speakers import extract_clip


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="PINE Speaker Clip Agent",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================
# Styles
# =========================
st.markdown("""
<style>
.title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    padding: 1rem;
    border-radius: 12px;
    background: #f8f9fb;
    margin-bottom: 0.8rem;
    border-left: 5px solid #2575fc;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<div class="title">🎙️ PINE Speaker Clip Agent</div>', unsafe_allow_html=True)
st.caption("AI-powered speaker diarization & clip extraction")

# =========================
# Paths
# =========================
STORAGE_DIR = Path("storage")
VIDEO_PATH = STORAGE_DIR / "video.mp4"
AUDIO_PATH = STORAGE_DIR / "audio.wav"
CLIPS_DIR = Path("clips")

# =========================
# Upload
# =========================
st.subheader("📤 Upload Video")

uploaded_file = st.file_uploader(
    "Upload MP4 / MOV / MKV",
    type=["mp4", "mov", "mkv"]
)

if uploaded_file:
    STORAGE_DIR.mkdir(exist_ok=True)
    with open(VIDEO_PATH, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Video uploaded")

# =========================
# Analyze
# =========================
st.subheader("🔍 Analyze Speakers")

if st.button("Analyze") and VIDEO_PATH.exists():
    with st.spinner("Extracting audio..."):
        extract_audio(str(VIDEO_PATH), str(AUDIO_PATH))

    with st.spinner("Running diarization..."):
        speakers = run_diarization(str(AUDIO_PATH))
        person_map = build_person_map(speakers)

    st.session_state["speakers"] = speakers
    st.session_state["person_map"] = person_map
    st.success("🎉 Speakers detected")

# =========================
# Display speakers
# =========================
if "person_map" in st.session_state:
    st.subheader("🧑‍🤝‍🧑 Speakers")

    selected = []

    for person, speaker_id in st.session_state["person_map"].items():
        segments = st.session_state["speakers"][speaker_id]
        total_time = sum(e - s for s, e in segments)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                f"**{person}**  \n"
                f"🎬 Clips: {len(segments)}  \n"
                f"⏱️ {total_time:.1f}s"
            )

        with col2:
            if st.checkbox("Select", key=person, value=True):
                selected.append((person, speaker_id))

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Extract clips
# =========================
if "speakers" in st.session_state and st.button("🚀 Extract Clips"):
    CLIPS_DIR.mkdir(exist_ok=True)

    with st.spinner("Extracting clips..."):
        for person, speaker_id in selected:
            person_dir = CLIPS_DIR / person
            segments = st.session_state["speakers"][speaker_id]

            for i, (start, end) in enumerate(segments, 1):
                out = person_dir / f"{person}_clip_{i:03d}.mp4"
                extract_clip(VIDEO_PATH, start, end, out)

    st.success("✅ Clips extracted")

st.caption("Built with ❤️ — PINE Digital Systems")
