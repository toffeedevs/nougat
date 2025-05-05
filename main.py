import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable, NoTranscriptFound

load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Nougat: Question Synthesis"}


class TextObject(BaseModel):
    text: str


@app.post("/nougat/tftext")
async def tftext(to: TextObject):
    context = to.text

    instruction = f"""
    Based on the following context, write as many true/false questions as possible in a structured JSON format.
    Each question should have a "question" field and an "answer" field (true or false). Each question must also be specific.
    Additionally, you should add a "rationale" field that explains the answer to the question. 

    TEXT:
    {context}

    Return only valid JSON with no explanations or additional text.
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [
                {
                    "role": "user",
                    "content": instruction
                }
            ],
            "response_format": {"type": "json_object"}
        })
    )

    response_json = response.json()
    completion_text = response_json["choices"][0]["message"]["content"]

    true_false_questions = json.loads(completion_text)
    return {"questions": true_false_questions}


@app.post("/nougat/mcqtext")
async def mcqtext(to: TextObject):
    context = to.text

    instruction = f"""
    Based on the following context, write as many multiple choice questions as possible in a structured JSON format.
    Each question should have a "question" field, a "choices" field of four potential answers, and an "answer" field which is one of the choices. Each question must also be specific.
    Additionally, you should add a "rationale" field that explains the answer to the question. 

    TEXT:
    {context}

    Return only valid JSON with no explanations or additional text.
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [
                {
                    "role": "user",
                    "content": instruction
                }
            ],
            "response_format": {"type": "json_object"}
        })
    )

    response_json = response.json()
    completion_text = response_json["choices"][0]["message"]["content"]

    mcq = json.loads(completion_text)
    return {"questions": mcq}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/nougat/fitb")
async def mcqtext(to: TextObject):
    context = to.text

    instruction = f"""
    Based on the following context, write as many fill-in-the-blank questions as possible in a structured JSON format. Ensure that the items that are
    being asked to be filled in are pertinent proper nouns that are important in the context. Each question should have a "question" field and an "answer" field which is one of the choices. Each question must also be specific.
    Additionally, you should add a "rationale" field that explains the answer to the question. 

    TEXT:
    {context}

    Return only valid JSON with no explanations or additional text.
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [
                {
                    "role": "user",
                    "content": instruction
                }
            ],
            "response_format": {"type": "json_object"}
        })
    )

    response_json = response.json()
    print(response_json)
    completion_text = response_json["choices"][0]["message"]["content"]

    mcq = json.loads(completion_text)
    return {"questions": mcq}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def transcriptify(to: TextObject):
    # More robust way to extract video ID from URL
    import re
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", to.text)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    video_id = match.group(1)

    try:
        raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable")
    except TranscriptsDisabled:
        raise HTTPException(status_code=403, detail="Transcripts are disabled for this video")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except Exception as e:
        # Catch-all for other issues like network problems on Vercel
        raise HTTPException(status_code=500, detail=f"Transcript fetch error: {str(e)}")

    result = "".join(snippet["text"] for snippet in raw_transcript)

    instruction = f"""
       Based on the following YouTube video transcript, clear up the text to be coherent while maintaining meaning

       TEXT:
       {result[:8000]}

       Return only the text.
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        })
    )

    response_json = response.json()
    refined_transcript = response_json["choices"][0]["message"]["content"]
    return {"transcript": refined_transcript}