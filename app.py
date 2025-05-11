import streamlit as st
import whisper
import ffmpeg
import os
import tempfile
import shutil
from datetime import timedelta

# Streamlit page configuration
st.set_page_config(page_title="Auto Subtitle Generator", page_icon="🎥", layout="centered")

# Title and description
st.title("Auto Subtitle Generator")
st.markdown("Upload a video to generate subtitles using OpenAI's Whisper AI. Download as an SRT file or get a video with embedded subtitles.")

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

# Option to choose output type
output_type = st.radio("Output Preference", ["Download SRT File", "Video with Embedded Subtitles"])

# Process button
if st.button("Generate Subtitles"):
    if uploaded_file is not None:
        with st.spinner("Processing video... This may take a few minutes."):
            try:
                # Save uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                    tmp_video.write(uploaded_file.read())
                    video_path = tmp_video.name

                # Load Whisper Large-v3 model
                model = whisper.load_model("large-v3")

                # Transcribe video
                result = model.transcribe(video_path, verbose=False)

                # Generate SRT file
                srt_content = ""
                for i, segment in enumerate(result["segments"], 1):
                    start = timedelta(seconds=segment["start"])
                    end = timedelta(seconds=segment["end"])
                    text = segment["text"].strip()
                    srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

                # Save SRT to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
                    tmp_srt.write(srt_content.encode("utf-8"))
                    srt_path = tmp_srt.name

                if output_type == "Download SRT File":
                    # Provide SRT file for download
                    with open(srt_path, "rb") as f:
                        st.download_button(
                            label="Download SRT File",
                            data=f,
                            file_name="subtitles.srt",
                            mime="text/plain"
                        )
                    st.success("SRT file generated! Click to download.")

                else:
                    # Embed subtitles into video
                    output_video = os.path.join(tempfile.gettempdir(), "output_with_subtitles.mp4")
                    try:
                        stream = ffmpeg.input(video_path)
                        stream = ffmpeg.output(
                            stream,
                            output_video,
                            vf=f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&Hffffff&,OutlineColour=&H000000&,BorderStyle=3'",
                            c="copy",
                            y=None
                        )
                        ffmpeg.run(stream, overwrite_output=True)

                        # Provide video for download
                        with open(output_video, "rb") as f:
                            st.download_button(
                                label="Download Video with Subtitles",
                                data=f,
                                file_name="video_with_subtitles.mp4",
                                mime="video/mp4"
                            )
                        st.success("Video with embedded subtitles generated! Click to download.")

                        # Clean up output video
                        os.remove(output_video)
                    except ffmpeg.Error as e:
                        st.error(f"Error embedding subtitles: {e.stderr.decode()}")
                        raise

                # Clean up temporary files
                os.remove(video_path)
                os.remove(srt_path)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload a video file first.")

# Footer
st.markdown("---")
st.markdown("Powered by OpenAI Whisper Large-v3 and FFmpeg. Created for easy subtitle generation.")