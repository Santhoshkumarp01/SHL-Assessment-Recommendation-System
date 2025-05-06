SHL Assessment Recommendation System

Overview
This project is a Retrieval-Augmented Generation (RAG) system designed to recommend SHL assessments based on job role queries and maximum test duration. It uses a hybrid search approach combining vector-based similarity (FAISS, SentenceTransformers) and keyword matching, with a Streamlit frontend and FastAPI backend.

Frontend: Streamlit UI for user queries and results display.
Backend: FastAPI server for processing queries and returning recommendations.
Evaluation: Achieves Mean Recall@3: 1.00 and MAP@3: 1.00 across 7 test cases.

Live Demo
Webapp: https://shl-assessment-recommendation-system-qx2f.onrender.com
API Endpoint: https://shl-assessment-recommendation-backend-4tnr.onrender.com/recommend

Features
Input: Job role query (e.g., "Hiring Java developers, max 40 minutes") and max duration.
Output: Top 3 SHL assessments with details (name, URL, duration, etc.).

Tech Stack:
Frontend: Streamlit, Pandas, Requests
Backend: FastAPI, LangChain, FAISS, SentenceTransformers, Scikit-learn
Data: assessments.csv for assessment metadata
Approach: Hybrid search with query augmentation and assessment boosting for accuracy.

Setup Instructions
Clone the Repository:
git clone https://github.com/Santhoshkumarp01/SHL-Assessment-Recommendation-System.git
cd SHL-Assessment-Recommendation-System

Install Dependencies:
pip install -r requirements.txt

Requirements:
streamlit==1.35.0
requests==2.31.0
pandas==2.2.2
fastapi==0.111.0
uvicorn==0.29.0
langchain==0.0.353
langchain-community==0.0.20
sentence-transformers==2.2.2
faiss-cpu==1.7.4
scikit-learn==1.4.2
huggingface-hub==0.30.2

Run Backend:
uvicorn main:app --host 0.0.0.0 --port 8000
Access: http://localhost:8000/health and http://localhost:8000/recommend

Run Frontend:
streamlit run ui.py
Access: http://localhost:8501

Evaluate:
python evaluate.py
Outputs: Test case metrics (Recall@3, AP@3).

Deployment
Frontend: Deployed on Render (free tier).
URL: https://shl-assessment-recommendation-system-qx2f.onrender.com
Command: streamlit run ui.py --server.port $PORT
Backend: Deployed on Render (free tier).
URL: https://shl-assessment-recommendation-backend-4tnr.onrender.com
Command: uvicorn main:app --host 0.0.0.0 --port $PORT
Note: Render’s free tier may have ~30–60 second spin-up delays.

Evaluation Results
Mean Recall@3: 1.00
MAP@3: 1.00

Test Cases (7 total, all with Recall@3 = 1.00, AP@3 = 1.00):
Java Developer (40 mins)
Sales Role (1 hour)
COO in China (1 hour)
Content Writer (English, SEO)
ICICI Bank Admin (30–40 mins)
QA Engineer (1 hour)
Technical + Branding + Management (90 mins)

Files
ui.py: Streamlit frontend for user interface.
main.py: FastAPI backend with /health and /recommend endpoints.
rag.py: RAG pipeline (data loading, vector store, hybrid search).
evaluate.py: Evaluation script for test cases.
assessments.csv: Assessment metadata.
requirements.txt: Dependencies.

Challenges Overcome
Improved Test 6 (QA Engineer) from Recall@3 = 0.33 to 1.00 via query augmentation.
Fixed deployment issues (e.g., langchain errors, port binding) on Render.
Ensured frontend-backend connectivity with CORS and correct URLs.

Submission Details
GitHub: https://github.com/Santhoshkumarp01/SHL-Assessment-Recommendation-System
Document: https://docs.google.com/document/d/1uY3VHQ7khT6-8ptkPODPtUFvRxXreis3/edit?usp=drive_link&rtpof=true&sd=true
Author: SanthoshKumar P