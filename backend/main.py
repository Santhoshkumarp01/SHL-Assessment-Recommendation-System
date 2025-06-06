from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from rag import SentenceTransformerEmbeddings, load_assessments, recommend_assessments
from langchain_community.vectorstores import FAISS
import os


app = FastAPI(title="SHL Assessment Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://shl-assessment-recommendation-system-1-yzvw.onrender.com",
        "http://localhost:8501",  # For local Streamlit testing
        "*"  # Temporary for debugging; remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store (load once at startup)
documents = load_assessments("assessments.csv")
# Create the embeddings model
embeddings = SentenceTransformerEmbeddings()
# Create the vector store using the documents and embeddings
vector_store = FAISS.from_documents(documents, embeddings)

class QueryInput(BaseModel):
    query: str
    max_duration: Optional[int] = None

class Recommendation(BaseModel):
    assessment_name: str
    url: str
    remote_support: str
    adaptive_support: str
    duration: int
    test_type: List[str]
    description: str

class RecommendationResponse(BaseModel):
    status: str
    recommendations: List[Recommendation]

@app.get("/")
async def root():
    return {"message": "Welcome to the SHL Recommender API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(input: QueryInput):
    try:
        recommendations = recommend_assessments(input.query, vector_store, input.max_duration, documents=documents)
        return {
            "status": "success",
            "recommendations": [
                {
                    "assessment_name": r["assessment_name"],
                    "url": r["url"],
                    "remote_support": r["remote_support"],
                    "adaptive_support": r["adaptive_support"],
                    "duration": r["duration"],
                    "test_type": [r["test_type"]],
                    "description": r["description"]
                }
                for r in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)