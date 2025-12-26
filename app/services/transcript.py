from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        
        # Combine all text segments
        full_transcript = " ".join([segment.text for segment in transcript_list])
        
        return full_transcript
        
    except Exception as e:
        raise RuntimeError(f"could not fetch transcript: {str(e)}")