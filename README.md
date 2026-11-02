# PhishGuard-AI
# PhishGuard AI: Real-Time Phishing & Malicious URL Detection Engine

An offline Machine Learning system built with **Streamlit** and **Scikit-Learn** that inspects lexical anomalies, structural characteristics, and Shannon entropy in URLs to identify phishing attacks in real time.

## Key Features
- **Deterministic Cybersecurity Feature Extraction:** Extracts Shannon entropy, IP presence, subdomain depth, HTTPS status, and deceptive keyword flags.
- **Machine Learning Benchmark:** Trains and evaluates Random Forest vs. Logistic Regression classifiers with instant metric reports (Accuracy, Precision, Recall, F1-Score).
- **Interactive Visualizations:** Confusion matrix heatmap and Gini feature importance rankings.
- **Zero API Keys:** Works completely offline without external cloud endpoints or network tunnels.

## Setup & Running
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd PhishGuard-AI
