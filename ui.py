import streamlit as st
import requests
import pandas as pd

# Custom CSS for attractive UI
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; padding: 20px; }
    h1 { color: #1e3a8a; font-family: 'Arial', sans-serif; text-align: center; }
    .stTextInput > div > input { border: 2px solid #1e3a8a; border-radius: 5px; padding: 10px; }
    .stNumberInput > div > input { border: 2px solid #1e3a8a; border-radius: 5px; padding: 10px; }
    .stButton > button { background-color: #1e3a8a; color: white; border-radius: 5px; padding: 10px 20px; display: block; margin: 0 auto; }
    .stTable { background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background-color: #1e3a8a; color: white; }
    tr:hover { background-color: #f5f5f5; }
    a { color: #1e3a8a; text-decoration: none; }
    a:hover { text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

st.title("SHL Assessment Recommendation Tool")
query = st.text_input("Enter job role or query", placeholder="e.g., Java developer, 40 mins")
max_duration = st.number_input("Max duration (minutes)", min_value=10, max_value=120, value=40)

if st.button("Get Recommendations"):
    try:
        response = requests.post(
            "https://shl-assessment-recommendation-backend-4tnr.onrender.com/recommend",
            json={"query": query, "max_duration": max_duration}
        )
        data = response.json()
        if data["status"] == "success":
            df = pd.DataFrame(data["recommendations"])
            df["url"] = df["url"].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
            st.write("### Recommended Assessments")
            st.markdown(
                df[["assessment_name", "url", "remote_support", "adaptive_support", "duration", "test_type", "description"]].to_html(escape=False),
                unsafe_allow_html=True
            )
        else:
            st.error(data["message"])
    except Exception as e:
        st.error(f"Error: {str(e)}")