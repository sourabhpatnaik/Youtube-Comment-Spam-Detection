import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import base64

# ── FORCE DOWNLOAD NLTK DATA AT RUNTIME ──
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# ── CENTERED LAYOUT TO KEEP CARDS SNUG ──
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── HIGH-VISIBLE BLUISH-GREY BACKGROUND ENGINE ──
svg_bg = """
<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
    <defs>
        <linearGradient id="bg-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#1e293b" />
            <stop offset="50%" stop-color="#111827" />
            <stop offset="100%" stop-color="#0f172a" />
        </linearGradient>
        <linearGradient id="yt-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#ff0000" stop-opacity="0.25" />
            <stop offset="100%" stop-color="#cc0000" stop-opacity="0.1" />
        </linearGradient>
    </defs>
    
    <rect width="1920" height="1080" fill="url(#bg-grad)" />
    
    <g stroke="#384252" stroke-opacity="0.25" stroke-width="1.2">
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" />
        </pattern>
        <rect width="1920" height="1080" fill="url(#grid)" />
    </g>

    <g stroke="#38bdf8" stroke-opacity="0.35" stroke-width="1.5" fill="#38bdf8" fill-opacity="0.6">
        <circle cx="200" cy="150" r="5" />
        <circle cx="350" cy="220" r="4" />
        <circle cx="150" cy="400" r="6" />
        <circle cx="1700" cy="180" r="5" />
        <circle cx="1800" cy="350" r="4" />
        <circle cx="1650" cy="600" r="6" />
        
        <line x1="200" y1="150" x2="350" y2="220" />
        <line x1="200" y1="150" x2="150" y2="400" />
        <line x1="1700" y1="180" x2="1800" y2="350" />
        <line x1="1800" y1="350" x2="1650" y2="600" />
    </g>

    <g transform="translate(1450, 80)">
        <rect width="320" height="226" rx="70" fill="url(#yt-grad)" stroke="#ff3333" stroke-opacity="0.35" stroke-width="2.5"/>
        <polygon points="130,73 130,153 210,113" fill="#ffffff" fill-opacity="0.6" />
    </g>

    <g transform="translate(100, 700) scale(0.6)">
        <rect width="320" height="226" rx="70" fill="url(#yt-grad)" stroke="#ff3333" stroke-opacity="0.3" stroke-width="2.5"/>
        <polygon points="130,73 130,153 210,113" fill="#ffffff" fill-opacity="0.5" />
    </g>
    
    <circle cx="1500" cy="850" r="80" fill="none" stroke="#38bdf8" stroke-opacity="0.2" stroke-width="1.5" stroke-dasharray="6,4" />
    <circle cx="1500" cy="850" r="40" fill="none" stroke="#38bdf8" stroke-opacity="0.15" stroke-width="1" />
    <line x1="1500" y1="750" x2="1500" y2="950" stroke="#38bdf8" stroke-opacity="0.15" stroke-width="1" />
    <line x1="1400" y1="850" x2="1600" y2="850" stroke="#38bdf8" stroke-opacity="0.15" stroke-width="1" />
</svg>
"""
b64_svg = base64.b64encode(svg_bg.encode()).decode()

# ── Master Layout Stylesheet ──
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{ 
    background-image: url("data:image/svg+xml;base64,{b64_svg}") !important;
    background-size: cover !important;
    background-position: center center !important;
    background-attachment: fixed !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
    background-color: transparent !important;
}}
[data-testid="stHeader"] div {{
    color: #ffffff !important;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(15, 23, 42, 0.45) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}}
[data-testid="stSidebar"] h3 {{
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] strong {{
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255, 255, 255, 0.1) !important;
}}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
    display: none !important;
}}
[data-testid="stSidebar"] div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: column !important;
    gap: 0.6rem !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    box-sizing: border-box !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stRadioCircle"],
[data-testid="stSidebar"] div[role="radiogroup"] label [class*="StyledRadio"],
[data-testid="stSidebar"] div[role="radiogroup"] label div[class*="RadioElement"],
[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"],
[data-testid="stSidebar"] div[role="radiogroup"] label [class*="st-"]::before,
[data-testid="stSidebar"] div[role="radiogroup"] label [class*="st-"]::after {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    position: absolute !important;
    visibility: hidden !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label [class*="st-"] {{
    margin-left: 0 !important;
    padding-left: 0 !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {{
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    text-align: left !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover div[data-testid="stMarkdownContainer"] p {{
    color: #ffffff !important;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input[type="radio"]:checked) {{
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: #6c63ff !important;
    box-shadow: 0 0 15px rgba(108, 99, 255, 0.2) !important;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input[type="radio"]:checked) div[data-testid="stMarkdownContainer"] p {{
    color: #ffffff !important;
}}

.main .block-container {{
    padding: 2rem 1.5rem !important;
    max-width: 820px !important;
    margin: 0 auto !important;
}}

.hero-card {{
    background: rgba(30, 41, 59, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
}}
.hero-icon {{
    background: linear-gradient(135deg, #6c63ff, #4f46e5);
    border-radius: 14px;
    width: 60px; height: 60px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem; flex-shrink: 0;
}}
.hero-title {{ font-size: 1.8rem; font-weight: 700; color: #ffffff; margin: 0; }}
.hero-subtitle {{ font-size: 0.88rem; color: #94a3b8; margin-top: 0.3rem; line-height: 1.4; }}

.unified-card {{
    background: rgba(30, 41, 59, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px !important;
    padding: 1.8rem 2rem !important;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 1.5rem !important;
}}
.section-title {{ color: #818cf8; font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; margin-top: 0; }}
.card-paragraph {{ color: #cbd5e1 !important; font-size: 0.95rem !important; line-height: 1.6 !important; margin: 0; }}

.stTextArea textarea {{
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    background: rgba(15, 23, 42, 0.6) !important;
    color: #ffffff !important;
}}
.stTextArea textarea:focus {{
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
}}

div[data-testid="stForm"] {{
    background: rgba(30, 41, 59, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px !important;
    padding: 1.8rem 2rem !important;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 1.5rem !important;
}}

div[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(135deg, #6c63ff, #4f46e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}}
div[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stFormSubmitButton"] button:hover {{
    color: #ffffff !important;
    opacity: 0.92 !important;
}}

div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stFormSubmitButton"] button {{
    background: rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}}
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stFormSubmitButton"] button:hover {{
    background: rgba(255, 255, 255, 0.18) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}}

.result-not-spam {{
    background: rgba(16, 185, 129, 0.15); border: 1.5px solid rgba(16, 185, 129, 0.4);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    margin-top: 0.5rem;
}}
.result-spam {{
    background: rgba(239, 68, 68, 0.15); border: 1.5px solid rgba(239, 68, 68, 0.4);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    margin-top: 0.5rem;
}}
.result-icon-green {{
    background: #10b981; border-radius: 50%;
    width: 48px; height: 48px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
}}
.result-icon-red {{
    background: #ef4444; border-radius: 50%;
    width: 48px; height: 48px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
}}
.result-text-green {{ color: #34d399; font-size: 1.3rem; font-weight: 700; }}
.result-text-red   {{ color: #f87171; font-size: 1.3rem; font-weight: 700; }}
.result-subtext {{ font-size: 0.9rem; color: #94a3b8; margin-top: 2px; }}
.result-shield {{ margin-left: auto; font-size: 2rem; }}

.stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem; margin-top: 1rem;
}}
.stat-card {{
    background: rgba(30, 41, 59, 0.65) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px;
    padding: 1.1rem 0.8rem 0.8rem 0.8rem;
    box-shadow: 0 10px 20px -10px rgba(0, 0, 0, 0.5); overflow: hidden;
}}
.stat-top {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem; }}
.stat-icon {{
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; flex-shrink: 0;
}}
.stat-icon-blue   {{ background: rgba(59, 130, 246, 0.15); }}
.stat-icon-green  {{ background: rgba(16, 185, 129, 0.15); }}
.stat-icon-red    {{ background: rgba(239, 68, 68, 0.15); }}
.stat-icon-purple {{ background: rgba(139, 92, 246, 0.15); }}
.stat-value {{ font-size: 1.35rem; font-weight: 700; color: #ffffff; line-height: 1; }}
.stat-label {{ font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }}
.stat-bar {{ height: 4px; border-radius: 2px; margin-top: 0.8rem; width: 100%; }}
.bar-blue   {{ background: #3b82f6; }}
.bar-green  {{ background: #10b981; }}
.bar-red    {{ background: #ef4444; }}
.bar-purple {{ background: #8b5cf6; }}

.doc-section {{ margin-bottom: 2rem; }}
.doc-divider {{ border: none; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 1.5rem 0; }}
.doc-list {{ margin: 0.5rem 0 0.5rem 1.2rem; padding: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; }}
.doc-list li {{ margin-bottom: 0.4rem; }}
.doc-subtitle {{ font-weight: 700; color: #ffffff; font-size: 1rem; margin-top: 0.8rem; margin-bottom: 0.4rem; }}
.workflow-step {{ background: rgba(15, 23, 42, 0.4); border: 1.5px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 0.5rem 1rem; display: inline-block; font-weight: 600; color: #a5b4fc; margin: 0.3rem 0; font-size: 0.9rem; }}
.workflow-arrow {{ color: #64748b; font-weight: bold; margin: 0.2rem 0 0.2rem 1.5rem; font-size: 1.1rem; }}
.tech-tag {{ display: inline-block; background: rgba(59, 130, 246, 0.15); color: #38bdf8; font-weight: 600; font-size: 0.85rem; padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem; border: 1px solid rgba(56, 189, 248, 0.3); }}

.dataset-table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.95rem; }}
.dataset-table th {{ background-color: rgba(15, 23, 42, 0.4); color: #818cf8; font-weight: 700; text-align: left; padding: 10px 16px; border-bottom: 2px solid rgba(255, 255, 255, 0.1); }}
.dataset-table td {{ padding: 12px 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); color: #cbd5e1; }}
.dataset-table tr:last-child td {{ border-bottom: none; }}
.badge-green {{ background-color: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 600; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem; }}
.badge-red {{ background-color: rgba(239, 68, 68, 0.2); color: #f87171; font-weight: 600; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem; }}

.footer {{
    text-align: center; padding: 1.5rem 0 0.5rem 0;
    font-size: 0.82rem; color: #64748b;
    border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 2rem;
}}
.footer span {{ color: #6c63ff; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ── Safe Load Model Bundle (Throws visible errors if file missing) ──
@st.cache_resource
def load_model():
    # ADJUST THIS STRING TO MATCH YOUR EXACT FILE FILENAME AND EXTENSION ON GITHUB
    return joblib.load("Model/yt_spam_detection_model.pkl")

model_data = load_model()
classifier_model = model_data['model']
vectorizer = model_data['vectorizer']

# ── Text Preprocessing Pipeline ──
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    # Normalize curly apostrophes from browsers/smart keyboards down to straight ones
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ── Sidebar Layout ──
with st.sidebar:
    st.markdown("### 🛡️ App Navigation")
    st.write("")
    
    page_selection = st.radio(
        "nav_selector",
        options=["🏠   Home / Predict", "ℹ️   About Project", "🗄️   Dataset Info"],
        label_visibility="collapsed"
    )
    
    st.write("---")
    st.markdown("**Project Details**")
    st.write("Detects comments pattern metrics using an engineered machine learning configuration framework.")

# ── Routing Pages Logic Matrix ──
if page_selection == "🏠   Home / Predict":
    
    st.markdown("""
    <div class="hero-card">
        <div style="display:flex;align-items:center;gap:1.2rem;">
            <div class="hero-icon">🛡️</div>
            <div>
                <div class="hero-title">Youtube Comment Spam Detection</div>
                <div class="hero-subtitle">
                    Detect whether a YouTube comment is Spam or Not Spam using<br>
                    Machine Learning
                </div>
            </div>
        </div>
        <div style="font-size:4rem;">✉️</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        if "comment_input" not in st.session_state:
            st.session_state.comment_input = ""

        def clear_text():
            st.session_state.comment_input = ""

        with st.form(key="input_form", clear_on_submit=False):
            st.markdown('<div class="section-title">Enter a Comment to Check</div>', unsafe_allow_html=True)
            
            comment = st.text_area("", placeholder="Type or paste a comment here...",
                                   value=st.session_state.comment_input,
                                   key="comment_input",
                                   height=110, label_visibility="collapsed")
            
            col_space, col_clear, col_btn = st.columns([3.6, 1.7, 1.7])
            with col_clear:
                clear_clicked = st.form_submit_button("🧹   Clear", use_container_width=True, on_click=clear_text)
            with col_btn:
                predict_clicked = st.form_submit_button("✈️   Predict", use_container_width=True)

        st.markdown("""
            <style>
            div[data-testid="stForm"] {
                margin-bottom: 1.5rem !important;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stFormSubmitButton"]:first-child button {
                background: rgba(255, 255, 255, 0.1) !important;
                color: #cbd5e1 !important;
                border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stFormSubmitButton"]:first-child button:hover {
                background: rgba(255, 255, 255, 0.18) !important;
                color: #ffffff !important;
            }
            </style>
        """, unsafe_allow_html=True)

    result_alert = ""
    if predict_clicked and comment.strip():
        # Preprocess and pass array to vectorizer transform stream
        cleaned_input = clean_text(comment)
        vectorized_input = vectorizer.transform([cleaned_input])
        pred = classifier_model.predict(vectorized_input)[0]
        
        if pred == 0:
            result_alert = """<div class="result-not-spam"><div class="result-icon-green">✅</div><div><div class="result-text-green">Not Spam</div><div class="result-subtext">This comment is predicted as Not Spam.</div></div><div class="result-shield">🛡️</div></div>"""
        else:
            result_alert = """<div class="result-spam"><div class="result-icon-red">⚠️</div><div><div class="result-text-red">Spam</div><div class="result-subtext">This comment is predicted as Spam.</div></div><div class="result-shield">🚨</div></div>"""
    elif predict_clicked:
        result_alert = """<div style="color:#f87171;font-size:0.9rem;">⚠️ Please enter a comment before clicking Predict.</div>"""
    else:
        result_alert = """<div style="color:#94a3b8;font-size:0.9rem;">Enter a comment above and click <strong>Predict</strong>.</div>"""

    st.markdown(f"""
    <div class="unified-card">
        <div class="section-title">Prediction Result</div>
        {result_alert}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-top">
                <div class="stat-icon stat-icon-blue">💬</div>
                <div>
                    <div class="stat-value">1901</div>
                    <div class="stat-label">Total Comments</div>
                </div>
            </div>
            <div class="stat-bar bar-blue"></div>
        </div>
        <div class="stat-card">
            <div class="stat-top">
                <div class="stat-icon stat-icon-green">✅</div>
                <div>
                    <div class="stat-value">943</div>
                    <div class="stat-label">Not Spam</div>
                </div>
            </div>
            <div class="stat-bar bar-green"></div>
        </div>
        <div class="stat-card">
            <div class="stat-top">
                <div class="stat-icon stat-icon-red">⚠️</div>
                <div>
                    <div class="stat-value">958</div>
                    <div class="stat-label">Spam</div>
                </div>
            </div>
            <div class="stat-bar bar-red"></div>
        </div>
        <div class="stat-card">
            <div class="stat-top">
                <div class="stat-icon stat-icon-purple">🎯</div>
                <div>
                    <div class="stat-value">89.07%</div>
                    <div class="stat-label">Model Accuracy</div>
                </div>
            </div>
            <div class="stat-bar bar-purple"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif "About" in page_selection:
    st.markdown("""<div class="unified-card"><div class="section-title">🛡️ ABOUT THE MODEL</div><div class="doc-section"><p class="card-paragraph">This project is a Machine Learning based YouTube Spam Detection System designed to classify user comments as Spam or Not Spam. The application uses Natural Language Processing (NLP) techniques to analyze comment text and predict whether a comment contains spam content.</p></div><div class="doc-divider"></div><div class="section-title">📊 DATASET</div><div class="doc-section"><p class="card-paragraph">The model was trained on a YouTube Comments Dataset containing both spam and non-spam comments.</p><div class="doc-subtitle">Target Classes:</div><ul class="doc-list"><li><strong>0</strong> → Not Spam</li><li><strong>1</strong> → Spam</li></ul><p class="card-paragraph" style="margin-top:0.5rem;">The dataset contains comments collected from multiple YouTube videos and includes examples of promotional, misleading, and genuine user comments.</p></div><div class="doc-divider"></div><div class="section-title">🧹 TEXT PREPROCESSING</div><div class="doc-section"><p class="card-paragraph">Before training the model, the text data was cleaned and transformed using the following steps:</p><ul class="doc-list" style="list-style-type: decimal;"><li><strong>Lowercase Conversion:</strong> All text was converted to lowercase.</li><li><strong>Special Character Removal:</strong> Symbols, punctuation marks, and unwanted characters were removed.</li><li><strong>Stopword Removal:</strong> Common words such as "the", "is", "are", and "and" were removed.</li><li><strong>Lemmatization:</strong> Words were converted to their root form to reduce vocabulary size.</li></ul><p class="card-paragraph" style="margin-top:0.5rem;">These preprocessing steps help improve model performance by reducing noise in the data.</p></div><div class="doc-divider"></div><div class="section-title">🔤 TF-IDF VECTORIZATION</div><div class="doc-section"><p class="card-paragraph">TF-IDF (Term Frequency - Inverse Document Frequency) was used to convert text into numerical features.</p><div class="doc-subtitle">TF-IDF Features:</div><ul class="doc-list"><li>Assigns higher importance to meaningful words.</li><li>Reduces the importance of frequently occurring words.</li><li>Converts text into machine-readable vectors.</li></ul><p class="card-paragraph" style="margin-top:0.5rem;">This allows the machine learning model to identify important patterns within comments.</p></div><div class="doc-divider"></div><div class="section-title">🤖 MACHINE LEARNING MODEL</div><div class="doc-section"><p class="card-paragraph">The project uses the <strong>Multinomial Naive Bayes</strong> algorithm for classification.</p><div class="doc-subtitle">Reasons for choosing Multinomial Naive Bayes:</div><ul class="doc-list"><li>Fast and efficient.</li><li>Works well with text classification tasks.</li><li>Performs effectively with TF-IDF features.</li><li>Commonly used in spam filtering applications.</li></ul><p class="card-paragraph" style="margin-top:0.5rem;">The model learns patterns from training data and predicts whether a new comment is Spam or Not Spam.</p></div><div class="doc-divider"></div><div class="section-title">⚙️ MODEL WORKFLOW</div><div class="doc-section" style="text-align: left; padding-left: 0.5rem;"><div class="workflow-step">User Comment</div><br><div class="workflow-arrow">↓</div><div class="workflow-step">Text Cleaning</div><br><div class="workflow-arrow">↓</div><div class="workflow-step">Stopword Removal</div><br><div class="workflow-arrow">↓</div><div class="workflow-step">Lemmatization</div><br><div class="workflow-arrow">↓</div><div class="workflow-step">TF-IDF Vectorization</div><br><div class="workflow-arrow">↓</div><div class="workflow-step">Multinomial Naive Bayes</div><br><div class="workflow-arrow">↓</div><div class="workflow-step" style="background: #1e1b4b; border-color: #6c63ff;">Spam / Not Spam Prediction</div></div><div class="doc-divider"></div><div class="section-title">🛠️ TECHNOLOGIES USED</div><div class="doc-section"><span class="tech-tag">Python</span><span class="tech-tag">Pandas</span><span class="tech-tag">NumPy</span><span class="tech-tag">NLTK</span><span class="tech-tag">Scikit-Learn</span><span class="tech-tag">TF-IDF Vectorizer</span><span class="tech-tag">Multinomial Naive Bayes</span><span class="tech-tag">Streamlit</span></div><div class="doc-divider"></div><div class="section-title">🎯 PROJECT OBJECTIVE</div><div class="doc-section"><p class="card-paragraph">The objective of this project is to demonstrate how Machine Learning and Natural Language Processing can be used to automatically identify spam comments and assist in content moderation on online platforms such as YouTube.</p></div></div>""", unsafe_allow_html=True)

elif "Dataset" in page_selection:
    st.markdown("""<div class="unified-card"><div class="section-title">🗄️ DATASET METRICS INFORMATION</div><div class="doc-section"><p class="card-paragraph">The training source contains tokenized text records parsed sequentially from multiple popular public channels. The content distributions display a healthy, balanced splitting across both target class parameters to safeguard the Multinomial Naive Bayes model from developing class bias.</p></div><div class="doc-divider"></div><div class="section-title">📈 SUMMARY MATRIX STATS</div><table class="dataset-table"><thead><tr><th>Metric Attribute</th><th>Value Breakdown</th></tr></thead><tbody><tr><td><strong>Total Record Parameters</strong></td><td>1,901 text strings</td></tr><tr><td><strong>Features Pipeline Extraction</strong></td><td>TF-IDF Matrix Weights</td></tr><tr><td><strong>Language Focus</strong></td><td>English (En-US Standard)</td></tr></tbody></table><div class="doc-divider"></div><div class="section-title">⚖️ TARGET FEATURE DISTRIBUTION</div><table class="dataset-table"><thead><tr><th>Class Identifier</th><th>Label Definition</th><th>Row Count</th><th>Percentage Split</th></tr></thead><tbody><tr><td><code>0</code></td><td><span class="badge-green">Not Spam</span></td><td>943 rows</td><td>49.61%</td></tr><tr><td><code>1</code></td><td><span class="badge-red">Spam</span></td><td>958 rows</td><td>50.39%</td></tr></tbody></table></div>""", unsafe_allow_html=True)

# Centralized App Footer 
st.markdown("""
<div class="footer">
    <span>Spam Detection App</span> &nbsp;|&nbsp; Built with Streamlit 🎈
</div>
""", unsafe_allow_html=True)
