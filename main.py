import json
import os
import re
import requests
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    source_document: str
    focus_areas: List[str]
    sample_questions: Optional[List[str]] = []
    difficulty: str  # "Easy" or "Hard"

class TextObject(BaseModel):
    text: str

class FeynmanObject(BaseModel):
    term: str
    text: str
    response: str

# Helper to call OpenRouter API
def call_openrouter(prompt):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        })
    )
    response_json = response.json()
    return json.loads(response_json["choices"][0]["message"]["content"])

@app.get("/")
async def root():
    return {"message": "Nougat: Question Synthesis"}

@app.post("/nougat/mcqtext")
async def mcqtext(qr: QuestionRequest):
    instruction = f"""
    Using the details below, generate as many multiple-choice questions in JSON format as possible.
    Each should include:
    - "source_document"
    - "area_of_focus"
    - "question"
    - "choices" (list of 4)
    - "answer" (one of the choices)
    - "rationale" (must directly cite a certain line from the text in the format, enclosing the certain line in quotes")
    - "difficulty" ("Easy", "Medium", or "Hard")

    Only return valid JSON.

    SOURCE DOCUMENT:
    {qr.source_document}

    AREAS OF FOCUS:
    {qr.focus_areas}

    DIFFICULTY:
    {qr.difficulty}

    SAMPLE QUESTIONS (if any):
    {qr.sample_questions if qr.sample_questions else 'None'}
    """
    result = call_openrouter(instruction)
    return {"questions": result}

@app.post("/nougat/tftext")
async def tftext(qr: QuestionRequest):
    instruction = f"""
    Using the details below, generate true/false questions in JSON format.
    Each should include:
    - "source_document"
    - "area_of_focus"
    - "question"
    - "answer" (true or false)
    - "rationale" (must directly cite a certain line from the text in the format, enclosing the certain line in quotes")
    - "difficulty" ("Easy" or "Hard")

    Only return valid JSON.

    SOURCE DOCUMENT:
    {qr.source_document}

    AREAS OF FOCUS:
    {qr.focus_areas}

    DIFFICULTY:
    {qr.difficulty}

    SAMPLE QUESTIONS (if any):
    {qr.sample_questions if qr.sample_questions else 'None'}

    """
    result = call_openrouter(instruction)
    return {"questions": result}

@app.post("/nougat/fitb")
async def fitb(qr: QuestionRequest):
    instruction = f"""
    Using the details below, generate fill-in-the-blank questions in JSON format.
    Each should include:
    - "source_document"
    - "question" (sentence with blank)
    - "answer" (correct word/phrase)
    - "rationale" 
    - "difficulty" ("Easy" or "Hard")

    Use key proper nouns or facts from the text.

    SOURCE DOCUMENT:
    {qr.source_document}

    AREAS OF FOCUS:
    {qr.focus_areas}

    DIFFICULTY:
    {qr.difficulty}

    SAMPLE QUESTIONS (if any):
    {qr.sample_questions if qr.sample_questions else 'None'}

    """
    result = call_openrouter(instruction)
    return {"questions": result}

@app.post("/nougat/cards")
async def cards(qr: QuestionRequest):
    instruction = f"""
    From the following, create flashcards in JSON format:
    - "front" (main term)
    - "back": {{
        "definition": (short explanation),
        "fill_in_blank": (sentence with blank)
      }}

    Do not invent any content not in the text.

    SOURCE DOCUMENT:
    {qr.source_document}

    AREAS OF FOCUS:
    {qr.focus_areas}

    DIFFICULTY:
    {qr.difficulty}

    """
    result = call_openrouter(instruction)
    return {"cards": result}

@app.post("/nougat/keyterms")
async def keyterms(to: TextObject):
    instruction = f"""
    Extract key terms and important proper nouns from the following text.

    MAIN TEXT:
    {to.text}

    Only return valid JSON.
    """
    result = call_openrouter(instruction)
    return {"terms": result}

@app.post("/nougat/feynman")
async def feynman(fo: FeynmanObject):
    instruction = f"""
    Evaluate the user's explanation using the Feynman technique.
    Return:
    {{
      "clarity": (score /10),
      "accuracy": (score /10),
      "completeness": (score /10),
      "feedback": "specific improvement points"
    }}

    TERM:
    {fo.term}

    TEXT:
    {fo.text}

    USER RESPONSE:
    {fo.response}

    Only return valid JSON.
    """
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [{"role": "user", "content": instruction}]
        })
    )
    raw_content = response.json()["choices"][0]["message"]["content"]
    cleaned = re.sub(r"```(?:json)?", "", raw_content).replace("```", "").strip()
    result = json.loads(cleaned)
    return result

@app.post("/nougat/transcriptify")
async def transcriptify(to: TextObject):
    video_id = to.text.split("=")[1]
    ytt_api = YouTubeTranscriptApi(proxy_config=WebshareProxyConfig(
        proxy_username="wxskymzk",
        proxy_password="n1hv3xukxsfh",
    ))

    raw_transcript = ytt_api.fetch(video_id)
    combined_text = "".join(snippet.text for snippet in raw_transcript)

    instruction = f"""
    Clean and organize the following YouTube transcript into coherent, readable text while maintaining the original meaning.

    TRANSCRIPT:
    {combined_text}

    Only return the cleaned text.
    """
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [{"role": "user", "content": instruction}]
        })
    )
    refined_text = response.json()["choices"][0]["message"]["content"]
    return {"transcript": refined_text}
