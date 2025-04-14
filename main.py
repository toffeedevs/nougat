import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from dotenv import load_dotenv

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

