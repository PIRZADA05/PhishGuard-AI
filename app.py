import math
import re
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ PhishGuard AI: Malicious URL Detection Engine")
st.caption("Cybersecurity & AI Laboratory Experiment | Offline Scikit-Learn Pipeline")

# --- 1. Security Feature Extraction Engine ---
def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy to detect algorithmically generated / randomized URLs."""
    if not text:
        return 0.0
    entropy = 0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return float(round(entropy, 3))

def extract_url_features(url: str) -> dict:
    """Extract lexical and structural cybersecurity signals from raw URLs."""
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        url_with_scheme = "http://" + url
    else:
        url_with_scheme = url

    parsed = urlparse(url_with_scheme)
    hostname = parsed.netloc.lower()
    path = parsed.path

    # IP address check in hostname
    has_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$", hostname) else 0

    # Suspicious keywords
    suspicious_terms = ["secure", "login", "verify", "update", "banking", "free", "account", "signin", "token", "auth"]
    has_suspicious_word = 1 if any(w in url.lower() for w in suspicious_terms) else 0

    return {
        "url_length": len(url),
        "entropy": calculate_entropy(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_at": url.count("@"),
        "num_subdomains": max(0, hostname.count(".") - 1),
        "has_ip_address": has_ip,
        "has_suspicious_word": has_suspicious_word,
        "is_https": 1 if url.lower().startswith("https://") else 0
    }

# --- 2. Ground-Truth Training Dataset ---
@st.cache_data
def load_cybersecurity_dataset():
    """Generates a balanced dataset of legitimate and realistic phishing attack vectors."""
    legitimate_urls = [
        "https://www.google.com/search?q=machine+learning",
        "https://github.com/scikit-learn/scikit-learn",
        "https://www.wikipedia.org/wiki/Artificial_intelligence",
        "https://docs.python.org/3/library/urllib.parse.html",
        "https://stackoverflow.com/questions/tagged/python",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.reddit.com/r/MachineLearning/",
        "https://scholar.google.com/citations?user=123",
        "https://aws.amazon.com/console/",
        "https://www.microsoft.com/en-us/windows",
        "https://developer.mozilla.org/en-US/docs/Web",
        "https://www.coursera.org/learn/machine-learning",
        "https://www.kaggle.com/datasets",
        "https://www.linkedin.com/feed/",
        "https://openai.com/research/"
    ] * 20  # Replicate for distribution

    phishing_urls = [
        "http://192.168.1.55/update-your-bank-account-now.html",
        "http://secure-paypal-login-verify-account.tk/signin",
        "http://appleid.apple.com.verify-billing-info.ru/auth",
        "http://microsoft-support-recovery-session-0912.xyz/login.php",
        "http://free-giftcards-reward-token.ga/claim-now",
        "http://banking-service-auth-security-check.cf/login",
        "http://google-drive-shared-file-download-viewer.ml/share",
        "http://netflix-billing-issue-confirm-card.xyz/signin",
        "http://amazon-prime-unusual-activity-alert.net.co/account",
        "http://104.244.42.1/webscr?cmd=_login-run",
        "http://facebook-security-verification-portal-39.biz/auth",
        "http://chase-bank-verify-identity-credentials.xyz/home",
        "http://instagram-blue-tick-free-verify.tk/badge",
        "http://secure-webmail-corporate-account-update.info/portal",
        "http://crypto-airdrop-reward-connect-wallet.pw/claim"
    ] * 20

    data = []
    for u in legitimate_urls:
        feats = extract_url_features(u)
        feats["label"] = 0  # Legitimate
        data.append(feats)

    for u in phishing_urls:
        feats = extract_url_features(u)
        feats["label"] = 1  # Phishing
        data.append(feats)

    return pd.DataFrame(data)

df = load_cybersecurity_dataset()

# Feature columns
feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols]
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Sidebar - Model Selection & Hyperparameters
st.sidebar.header("⚙️ Model Configuration")
algorithm = st.sidebar.selectbox("Classifier Algorithm:", ["Random Forest", "Logistic Regression"])

if algorithm == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators (Trees):", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("Max Tree Depth:", 2, 20, 8)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
else:
    c_val = st.sidebar.slider("Regularization Strength (C):", 0.01, 10.0, 1.0, 0.1)
    model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)

# Model Training
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Dashboard Tabs
tab_scan, tab_metrics, tab_features, tab_data = st.tabs([
    "🔍 Real-Time URL Scanner",
    "📊 Model Performance & Matrix",
    "📈 Feature Importance",
    "🗃️ Training Dataset Inspector"
])

# --- TAB 1: Real-Time URL Scanner ---
with tab_scan:
    st.subheader("Interactive Phishing & Anomaly Triage")
    sample_urls = {
        "Custom Input": "",
        "Suspicious Bank Phish": "http://192.168.1.100/chase-bank-verify-account-security.html",
        "Urgent Account Update": "http://secure-paypal-update-credentials.tk/signin.php",
        "Legitimate Academic Portal": "https://docs.python.org/3/library/index.html",
        "Legitimate Developer Repo": "https://github.com/scikit-learn/scikit-learn"
    }

    selected_sample = st.selectbox("Quick Test Samples:", list(sample_urls.keys()))
    default_text = sample_urls[selected_sample] if selected_sample != "Custom Input" else ""

    target_url = st.text_input("Enter URL to Inspect:", value=default_text, placeholder="e.g. http://my-suspicious-bank-login.xyz/token")

    if st.button("🚀 Analyze Threat Level", type="primary"):
        if not target_url.strip():
            st.warning("Please enter a valid URL.")
        else:
            raw_feats = extract_url_features(target_url)
            input_df = pd.DataFrame([raw_feats])
            
            prob = model.predict_proba(input_df)[0][1]
            threat_percent = int(prob * 100)
            is_phish = prob >= 0.5

            st.divider()
            col_verdict, col_prob = st.columns([2, 1])

            with col_verdict:
                if is_phish:
                    st.error(f"🚨 **MALICIOUS / PHISHING THREAT DETECTED**")
                    st.markdown(f"The structural signals strongly correlate with deceptive credential harvesting.")
                else:
                    st.success(f"✅ **LEGITIMATE / SAFE URL**")
                    st.markdown("No significant malicious heuristics or entropy anomalies detected.")

            with col_prob:
                st.metric("Phishing Probability", f"{threat_percent}%")

            st.write("#### Extracted Threat Signals")
            sig_col1, sig_col2, sig_col3 = st.columns(3)
            sig_col1.metric("Shannon Entropy", raw_feats["entropy"], help="Higher entropy (>4.0) indicates random/algorithmically generated strings.")
            sig_col2.metric("URL Length", raw_feats["url_length"])
            sig_col3.metric("IP Address In Domain", "Yes (High Risk)" if raw_feats["has_ip_address"] else "No")

            with st.expander("View Full Extracted Feature Vector"):
                st.json(raw_feats)

# --- TAB 2: Metrics & Confusion Matrix ---
with tab_metrics:
    st.subheader("Model Evaluation Metrics")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Precision", f"{prec * 100:.2f}%")
    m3.metric("Recall", f"{rec * 100:.2f}%")
    m4.metric("F1-Score", f"{f1 * 100:.2f}%")

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_labels = ["Legitimate (0)", "Phishing (1)"]
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Ground Truth Class"),
        x=cm_labels,
        y=cm_labels,
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# --- TAB 3: Feature Importance ---
with tab_features:
    st.subheader("Cybersecurity Feature Significance")
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Gini Feature Importance Ranking (Random Forest)",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance chart is available when Random Forest is selected.")

# --- TAB 4: Dataset Inspector ---
with tab_data:
    st.subheader("Training Corpus & Extracted Vectors")
    st.dataframe(df, use_container_width=True)
    st.caption("Binary Label: 0 = Legitimate, 1 = Phishing")
