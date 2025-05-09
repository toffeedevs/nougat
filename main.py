import json
import os
import requests
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    source_document: str                   # e.g., title or identifier of the document
    focus_areas: List[str]                 # specific areas/topics to focus on
    sample_questions: Optional[List[str]]  # optional pre-given sample questions
    difficulty: str                        # "Easy" or "Hard"
    text: str                              # the main context to generate questions from

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

@app.post("/nougat/mcqtext")
async def mcqtext(qr: QuestionRequest):
    instruction = f"""
    Using the following details, generate multiple-choice questions in JSON format.
    Each question object must include:
    - "source_document": (title or identifier),
    - "area_of_focus": (specific topic),
    - "question": (MCQ stem),
    - "choices": [list of 4 choices],
    - "answer": (one of the choices),
    - "rationale": (why this is the correct answer),
    - "difficulty": ("Easy" or "Hard").

    Only return valid JSON, no extra explanations.

    SOURCE DOCUMENT:
    {qr.source_document}

    AREAS OF FOCUS:
    {', '.join(qr.focus_areas)}

    DIFFICULTY:
    {qr.difficulty}

    SAMPLE QUESTIONS (optional, for style reference):
    {qr.sample_questions if qr.sample_questions else 'None'}

    MAIN TEXT:
    {qr.text}
    """

    result = call_openrouter(instruction)
    return {"questions": result}

@app.post("/nougat/tftext")
async def tftext(qr: QuestionRequest):
    instruction = f"""
    Using the following details, generate true/false questions in JSON format.
    Each question object must include:
    - "source_document": (title or identifier),
    - "area_of_focus": (specific topic),
    - "question": (true/false statement),
    - "answer": (true or false),
    - "rationale": (why this is the correct answer),
    - "difficulty": ("Easy" or "Hard").

    Only return valid JSON, no extra explanations.

    SOURCE DOCUMENT:
    {qr.source_document}

    AREAS OF FOCUS:
    {', '.join(qr.focus_areas)}

    DIFFICULTY:
    {qr.difficulty}

    SAMPLE QUESTIONS (optional, for style reference):
    {qr.sample_questions if qr.sample_questions else 'None'}

    MAIN TEXT:
    {qr.text}
    """

    result = call_openrouter(instruction)
    return {"questions": result}
