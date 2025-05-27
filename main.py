import json
import os
import re
import tempfile
from http.client import HTTPException
from typing import List, Optional

import requests
from anki_export import ApkgReader
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
    number_of_questions: int
    source_document: str
    focus_areas: List[str]
    sample_questions: Optional[List[str]] = []
    difficulty: str


class ChatBotRequest(BaseModel):
    summary: str
    text: str
    question: str


class TextObject(BaseModel):
    text: str


class FeynmanObject(BaseModel):
    term: str
    text: str
    response: str


class AnkiUrlRequest(BaseModel):
    url: str


# Helper to call OpenRouter API
def call_openrouter(prompt):
    try:
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
        response.raise_for_status()
        response_json = response.json()
        return json.loads(response_json["choices"][0]["message"]["content"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Nougat: Question Synthesis"}


@app.post("/nougat/mcqtext")
async def mcqtext(qr: QuestionRequest):
    try:
        instruction = f"""
        Using the details below, generate STRICTLY {qr.number_of_questions} many multiple-choice questions in JSON format as possible. The questions must be presented in the same order the source material was presented. 
        Each should include:
        - \"source_document\"
        - \"area_of_focus\"
        - \"question\"
        - \"choices\" (list of 4)
        - \"answer\" (one of the choices)
        - \"rationale\" (must directly QUOTE a certain line from the text in the format")
        Examples of rationales include: 
        {"The source text stated that \"...\""}
        - \"difficulty\" (\"Easy\", \"Medium\", or \"Hard\")

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCQ generation error: {str(e)}")


@app.post("/nougat/tftext")
async def tftext(qr: QuestionRequest):
    try:
        instruction = f"""
        Using the details below, generate STRICTLY {qr.number_of_questions} true/false questions in JSON format. The questions must be presented in the same order the source material was presented. 
        Each should include:
        - \"source_document\"
        - \"area_of_focus\"
        - \"question\"
        - \"answer\" (true or false)
        - \"rationale\" (must directly QUOTE a certain line from the text in the format")
        Examples of rationales include: 
        {"The source text stated that \"...\""}
        - \"difficulty\" (\"Easy\", \"Medium\", or \"Hard\")

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"True/False generation error: {str(e)}")


@app.post("/nougat/fitb")
async def fitb(qr: QuestionRequest):
    try:
        instruction = f"""
        Using the details below, generate STRICTLY {qr.number_of_questions} fill-in-the-blank questions in JSON format.
        Each should include:
        - \"source_document\"
        - \"question\" (sentence with blank)
        - \"answer\" (correct word/phrase)
        - \"rationale\" (must directly cite a certain line from the text in the format, enclosing the certain line in quotes")
        - \"difficulty\" (\"Easy\" or \"Hard\")

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fill-in-the-blank generation error: {str(e)}")


@app.post("/nougat/cards")
async def cards(qr: QuestionRequest):
    try:
        instruction = f"""
        From the following, create flashcards in JSON format:
        - \"front\" (main term)
        - \"back\": {{
            \"definition\": (short explanation),
            \"fill_in_blank\": (sentence with blank)
            \"citation\": (excerpt from the text where the flashcard was made)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Card generation error: {str(e)}")


@app.post("/nougat/keyterms")
async def keyterms(to: TextObject):
    try:
        instruction = f"""
        Based on the following context, create as many Feynman technique questions from keywords relating to concepts discussed in the text as possible. These must be important proper nouns in 
        the considering the provided context. 

        Examples of questions include: \"Explain how the electron proton chain works to a 5 year old.\"

        TEXT:
        {to.text}

        Return only valid JSON with no explanations or additional text.
        """
        result = call_openrouter(instruction)
        return {"terms": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyterm generation error: {str(e)}")


@app.post("/nougat/feynman")
async def feynman(fo: FeynmanObject):
    try:
        instruction = f"""
        Evaluate the user's explanation using the Feynman technique. Pretend you're a tutor and provide constructive criticism and a sample response. Do not have extremely formal language. 
        {{
          \"clarity\": (score /10),
          \"accuracy\": (score /10),
          \"completeness\": (score /10),
          \"feedback\": \"specific improvement points\" from the text. 
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
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        cleaned = re.sub(r"```(?:json)?", "", raw_content).replace("```", "").strip()
        result = json.loads(cleaned)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feynman evaluation error: {str(e)}")


@app.post("/nougat/transcriptify")
async def transcriptify(to: TextObject):
    try:
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
        response.raise_for_status()
        refined_text = response.json()["choices"][0]["message"]["content"]
        return {"transcript": refined_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcriptify error: {str(e)}")


@app.post("/nougat/import-anki")
async def import_anki_from_url(req: AnkiUrlRequest):
    try:
        response = requests.get(req.url)
        response.raise_for_status()
        temp_path = tempfile.mktemp(suffix=".apkg")
        with open(temp_path, "wb") as f:
            f.write(response.content)

        result = []
        reader = ApkgReader(temp_path)
        for note in reader.notes:
            record = note['data']['record']
            front = record.get('Front', '')
            back = record.get('Back', '')
            result.append({"front": front, "back": back})
        return {"cards": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anki import error: {str(e)}")


@app.post("/nougat/chatbot")
async def chatbot(c: ChatBotRequest):
    try:
        instruction = f"""
        You are a tutor who is an expert in whatever domain the text and question provided below is. Your job is
        to closely follow the user's queries and address them specifically. Provide responses without overly flowery 
        language and use the summary below to help be specific in the responses.  
        
        If ever asked to provide questions, provide them INDIVIDUALLY unless specified otherwise. 
        
        Keep your conversation fluid, and be helpful to the user. 
        
        SUMMARY:
        {c.summary}
        
        TEXT:
        {c.text}

        QUESTION:
        {c.question}

        Only return the response.
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
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        return {"result": raw_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot functionality failed")

@app.post("/chatbot/summarize")
async def summarize(t: TextObject):
    try:
        instruction = summary_prompt = f"""
        You are a summarization assistant helping distill a multi-turn chat conversation into a clear, concise summary that can be used as context for future chatbot interactions.
        
        Summarize the chat history below with the following goals:
        - **Preserve all key points** shared by the user, including their questions, goals, and challenges.
        - Highlight any important technical details, domain-specific content, or user preferences.
        - Maintain any assumptions, constraints, or stylistic instructions expressed by the user.
        - Structure the summary so it can effectively inform a chatbot’s behavior and understanding in future conversations.
        - WRITE RELEVANT ASPECTS OF THE CHAT VERBATIM. 
        - Write in the third person using full, clear sentences.
        - Be concise, but do **not omit important insights or instructions**.
        - Maintain accounts of what entity said what (chatbot vs. user)
        - Also include what the bot should do given the summary, keep being proactive and moving the bot along.
        
        CHAT HISTORY:
        {t.text}
        
        Return only the cleaned summary.
        """
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": instruction}]
            })
        )
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        return {"result": raw_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot functionality failed")
