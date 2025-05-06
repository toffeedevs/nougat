import json
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Nougat: Question Synthesis"}


class TextObject(BaseModel):
    text: str

class FeynmanObject(BaseModel):
    term: str
    text: str
    response: str

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


@app.post("/nougat/transcriptify")
async def transcriptify(to: TextObject):
    id = to.text.split("=")[1]
    ytt_api = YouTubeTranscriptApi(proxy_config=WebshareProxyConfig(
        proxy_username="wxskymzk",
        proxy_password="n1hv3xukxsfh",
    ))

    raw_transcript = ytt_api.fetch(id)
    result = []
    for snippet in raw_transcript:
        result.append(snippet.text)

    filtered_transcript = "".join(result)
    instruction = f"""
       Based on the following YouTube video transcript, clear up the text to be coherent while maintaining meaning

       TEXT:
       {filtered_transcript}

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
            ],
        })
    )

    response_json = response.json()
    refined_transcript = response_json["choices"][0]["message"]["content"]
    return {"transcript": refined_transcript}


@app.post("/nougat/cards")
async def cards(to: TextObject):
    context = to.text

    instruction = f"""
    Based on the following context, write as many flashcard relationships as possible in a structured JSON format. Ensure that 
    each term chosen as the "front" of a flashcard is meaningful in the context of the text. For each flashcard, the "back" should be an object containing:
    - "definition": a concise explanation of the term,
    - "fill in the blank": information from the text VERBATIM where the key term needs to be substituted in. Have short excerpts, and substitute the term with a "___".
    
    DO NOT ADD ANY ADDITIONAL INFORMATION NOT PRESENTED IN THE TEXT.
    
    Return the result as a JSON array of objects, where each object has the keys "front" and "back". Do not include any extra text outside of the JSON structure.
    
    Context:
    {context}

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

    cards = json.loads(completion_text)
    return {"questions": cards}

@app.post("/nougat/keyterms")
async def keyterms(to: TextObject):
    context = to.text

    instruction = f"""
        Based on the following context, extract as many keywords relating to concepts discussed in the text as possible. These must be important proper nouns in 
        the considering the provided context. 

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
    return {"terms":completion_text}

@app.post("/nougat/feynman")
async def feynman(fo: FeynmanObject):
    term = fo.term
    text = fo.text
    response = fo.response

    instruction = f"""
        You are an expert in the Feynman technique for active recall.
        Evaluate the quality of a user's explanation based on the provided term, source text, and their response.
        Out of ten, provide a clarity score, accuracy score, and completeness score. Additionally, add a "feedback" section
        that describes how the response could be improved WITH SPECIFICS FROM THE SOURCE. 

        TERM:
        {term}

        TEXT:
        {text}

        RESPONSE:
        {response}

        Return your evaluation as **valid JSON only**, with no additional commentary or explanation.
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
    return {"scores":completion_text}



