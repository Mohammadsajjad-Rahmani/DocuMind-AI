from fastapi import FastAPI, UploadFile, File, HTTPException
from schemas import DocumentInput, SummaryResponse, QueryInput, QueryResponse
from rag_engine import doc_engine
import pdfplumber
import io

app = FastAPI(title="DocuMind AI API", version="1.0.0")

@app.post("/extract-text")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="فرمت فایل باید PDF باشد.")
    
    try:
        content = await file.read()
        extracted_text = ""
        
        # استفاده از pdfplumber جهت استخراج دقیق متن و جدول‌ها
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        
        extracted_text = extracted_text.strip()
        
        if not extracted_text:
            raise HTTPException(
                status_code=422, 
                detail="متنی از این فایل استخراج نشد. احتمالاً فایل اسکن‌شده (تصویری) است."
            )
            
        return {"filename": file.filename, "text": extracted_text}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش فایل: {str(e)}")

@app.post("/summarize", response_model=SummaryResponse)
def summarize_document(data: DocumentInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="متن ورودی خالی است.")
    
    summary, total_chunks = doc_engine.generate_summary(data.text)
    return SummaryResponse(summary=summary, total_chunks=total_chunks)

@app.post("/query", response_model=QueryResponse)
def query_document(data: QueryInput):
    if not data.document_text.strip() or not data.question.strip():
        raise HTTPException(status_code=400, detail="متن مستند یا سوال نمی‌تواند خالی باشد.")
    
    answer, relevant_chunks = doc_engine.answer_question(data.question, data.document_text)
    return QueryResponse(answer=answer, relevant_chunks=relevant_chunks)