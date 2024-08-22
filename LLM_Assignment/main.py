from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from typing import Dict
from pydantic import BaseModel
import os
from PyPDF2 import PdfReader
from docx import Document

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")


os.makedirs("temp", exist_ok=True)

def extract_text_from_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text

    else:  
        with open(file_path, "r") as f:
            return f.read()

class SummarizeRequest(BaseModel):
    file_name: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    if file.content_type not in ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    file_path = f"temp/{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    return {"filename": file.filename, "message": "File uploaded successfully"}

@app.post("/summarize")
async def summarize_document(request: SummarizeRequest) -> Dict[str, str]:
    
    file_path = f"temp/{request.file_name}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    try:
        text = extract_text_from_file(file_path)
        summary = summarizer(text, max_length=100, min_length=25, do_sample=True)
        print(summary)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize file: {str(e)}")
