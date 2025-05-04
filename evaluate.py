import pandas as pd
from rag import load_assessments, build_vector_store, recommend_assessments

# Test dataset (Pages 4–10)
test_data = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "expected": [
            "Automata - Fix (New)", "Core Java (Entry Level) (New)", "Java 8 (New)",
            "Core Java (Advanced Level) (New)", "Agile Software Development",
            "Technology Professional 8.0 Job Focused Assessment", "Computer Science (New)"
        ],
        "max_duration": 40
    },
    {
        "query": "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options",
        "expected": [
            "Entry Level Sales 7.1 (International)", "Entry Level Sales Sift Out 7.1",
            "Entry Level Sales Solution", "Sales Representative Solution",
            "Sales Support Specialist Solution", "Technical Sales Associate Solution",
            "SVAR - Spoken English (Indian Accent) (New)", "Sales & Service Phone Solution",
            "Sales & Service Phone Simulation", "English Comprehension (New)"
        ],
        "max_duration": 60
    },
    {
        "query": "I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour",
        "expected": [
            "Motivation Questionnaire MQM5", "Global Skills Assessment",
            "Graduate 8.0 Job Focused Assessment"
        ],
        "max_duration": 90
    },
    {
        "query": "Content Writer required, expert in English and SEO.",
        "expected": [
            "Drupal (New)", "Search Engine Optimization (New)",
            "Administrative Professional - Short Form", "Entry Level Sales Sift Out 7.1",
            "General Entry Level - Data Entry 7.0 Solution"
        ],
        "max_duration": 60
    },
    {
        "query": "ICICI Bank Assistant Admin, Experience required 0-2 years, test should be 30-40 mins long",
        "expected": [
            "Administrative Professional - Short Form", "Verify - Numerical Ability",
            "Financial Professional - Short Form", "Bank Administrative Assistant - Short Form",
            "General Entry Level - Data Entry 7.0 Solution",
            "Basic Computer Literacy (Windows 10) (New)", "Verify - Verbal Ability - Next Generation"
        ],
        "max_duration": 40
    },
    {
        "query": "Find me 1 hour long assessment for QA Engineer with skills in Automata Selenium, JavaScript, SQL Server, Manual Testing",
        "expected": [
            "Automata Selenium", "Automata - Fix (New)", "Automata Front End",
            "JavaScript (New)", "HTML/CSS (New)", "HTML5 (New)", "CSS3 (New)",
            "Selenium (New)", "SQL Server (New)", "Automata - SQL (New)", "Manual Testing (New)"
        ],
        "max_duration": 60
    },
    {
        "query": "Job description with technical skills, branding focus, and people management, duration at most 90 mins",
        "expected": [
            "Motivation Questionnaire MQM5", "Occupational Personality Questionnaire (OPQ32r)",
            "Global Skills Assessment", "Graduate 8.0 Job Focused Assessment",
            "SHL Verify Interactive - Inductive Reasoning"
        ],
        "max_duration": 90
    }
]

def recall_at_k(predicted: list, expected: list, k: int = 3) -> float:
    predicted = predicted[:k]
    num_relevant = len([p for p in predicted if p in expected])
    return num_relevant / min(k, len(expected)) if expected else 0

def ap_at_k(predicted: list, expected: list, k: int = 3) -> float:
    score = 0
    relevant_count = 0
    for i, p in enumerate(predicted[:k], 1):
        if p in expected:
            relevant_count += 1
            score += relevant_count / i
    return score / min(k, len(expected)) if expected else 0

# Evaluate metrics
def evaluate_metrics():
    documents = load_assessments("assessments.csv")
    vector_store = build_vector_store(documents)
    recall_scores = []
    ap_scores = []
    
    for i, test in enumerate(test_data):
        recommendations = recommend_assessments(
            test["query"], 
            vector_store, 
            max_duration=test["max_duration"],
            documents=documents
        )
        predicted = [r["assessment_name"] for r in recommendations]
        
        # Calculate metrics
        recall = recall_at_k(predicted, test["expected"])
        ap = ap_at_k(predicted, test["expected"])
        
        recall_scores.append(recall)
        ap_scores.append(ap)
        
        # Print individual test results for debugging
        print(f"Test {i+1}: Recall@3 = {recall:.2f}, AP@3 = {ap:.2f}")
        print(f"  Query: {test['query']}")
        print(f"  Predicted (top 3): {predicted[:3]}")
        print(f"  Expected: {test['expected'][:3]} (+ {len(test['expected'])-3 if len(test['expected'])>3 else 0} more)")

    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_ap = sum(ap_scores) / len(ap_scores)
    print(f"\nMean Recall@3: {mean_recall:.2f}")
    print(f"MAP@3: {mean_ap:.2f}")
    return mean_recall, mean_ap

if __name__ == "__main__":
    mean_recall, mean_ap = evaluate_metrics()