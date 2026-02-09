# from youtube_transcript_api import YouTubeTranscriptApi

# def get_transcript(video_id: str) -> str:
#     try:
#         transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        
#         # Combine all text segments
#         full_transcript = " ".join([segment.text for segment in transcript_list])
        
#         return full_transcript
        
#     except Exception as e:
#         raise RuntimeError(f"could not fetch transcript: {str(e)}")


# checking whether it runs on render
import subprocess
import tempfile
import os


def get_transcript(video_id: str) -> str:
    """
    Fetch YouTube transcript using yt-dlp (cloud-safe)
    """

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmp:

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "-o", f"{tmp}/%(id)s.%(ext)s",
            url
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError("Failed to fetch subtitles from YouTube")

        # Find downloaded VTT
        files = os.listdir(tmp)

        vtt_files = [f for f in files if f.endswith(".vtt")]

        if not vtt_files:
            raise RuntimeError("No subtitles available for this video")

        vtt_path = os.path.join(tmp, vtt_files[0])

        return parse_vtt(vtt_path)


def parse_vtt(path: str) -> str:
    """
    Convert VTT to plain text
    """

    lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("WEBVTT"):
                continue

            if "-->" in line:
                continue

            if line.isdigit():
                continue

            lines.append(line)

    return " ".join(lines)
