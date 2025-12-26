from pydantic import  BaseModel
from typing import List

class IngestRequest(BaseModel):
    video_id : str

class QueryRequest(BaseModel):
    query: str
    k : int = 5    