from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from dotenv import load_dotenv
import os

load_dotenv()


def get_transcript(video_id: str) -> str:
    try:
        pass = os.getenv("PASS")
        ytt_api =  YouTubeTranscriptApi(
                                proxy_config=WebshareProxyConfig(
                                    proxy_username="stskfsev,
                                    proxy_password=PASS
                                )
                            )

        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        
        # Combine all text segments
        full_transcript = " ".join([segment.text for segment in transcript_list])
        
        return full_transcript
        
    except Exception as e:
        raise RuntimeError(f"could not fetch transcript: {str(e)}")


