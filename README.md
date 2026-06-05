# 🛡️ YouTube Comment Spam Detection

A Machine Learning powered web application that classifies YouTube comments as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier — deployed via an interactive Streamlit UI.

---

## 🚀 Live Demo

> Run locally using the steps in the [Getting Started](#-getting-started) section.

---

## 📸 App Preview

| Home / Predict | About Project | Dataset Info |
|---|---|---|
| Comment input form with real-time prediction | NLP pipeline & model documentation | Dataset metrics and class distribution |

---

## 📌 Project Overview

YouTube is flooded with spam comments — promotional links, bot messages, and misleading content. This project automates spam detection by:

- Applying an NLP text preprocessing pipeline to clean raw comments
- Transforming text into numerical features using **TF-IDF Vectorization**
- Classifying comments using **Multinomial Naive Bayes**
- Serving predictions through a glassmorphism-styled **Streamlit** web app

---

## 🧠 ML Pipeline

```
User Comment → Lowercasing → Special Char Removal → Stopword Removal
     → Lemmatization → TF-IDF Vectorization → Multinomial Naive Bayes → Prediction
```

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| Source | YouTube Comments Dataset (CSV) |
| Total Records | 1,956 rows (1,901 after deduplication) |
| Features used | `CONTENT` (comment text) |
| Target column | `CLASS` — `0` = Not Spam, `1` = Spam |
| Class balance | 943 Not Spam (49.61%) / 958 Spam (50.39%) |
| Dropped columns | `DATE`, `COMMENT_ID`, `VIDEO_NAME` |

The dataset is **nearly perfectly balanced**, so no resampling was needed.

---

## 🔍 Exploratory Data Analysis

- Spam comments average **~9 words**; legitimate comments average **~20 words**
- Class distribution visualized via bar chart
- Comment length distribution compared across both classes using overlapping histograms

---

## 🧹 Text Preprocessing

Each comment goes through the following pipeline before vectorization:

1. **Lowercase Conversion** — Normalizes text casing
2. **Special Character Removal** — Strips punctuation, numbers, symbols using regex
3. **Stopword Removal** — Removes common English words (`nltk.corpus.stopwords`)
4. **Lemmatization** — Reduces words to their root form using `WordNetLemmatizer`

---

## 🤖 Model

| Detail | Value |
|---|---|
| Algorithm | Multinomial Naive Bayes |
| Vectorizer | TF-IDF (`TfidfVectorizer`) |
| Train/Test Split | 75% / 25% |
| Accuracy | **89.07%** |
| Export format | `joblib` (saved as `yt_spam_detection_model`) |

---

## 🖥️ Streamlit App Features

- **Home / Predict** — Paste any comment and get an instant Spam / Not Spam prediction
- **About Project** — Full documentation of the NLP pipeline, model logic, and workflow
- **Dataset Info** — Summary stats and class distribution table
- Custom glassmorphism dark UI with SVG animated background
- Dataset stats dashboard (total comments, spam count, accuracy)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-4EA94B?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## 📁 Project Structure

```
youtube-spam-detection/
│
├── Youtube_Spam_Detection.ipynb   # EDA, preprocessing & model training notebook
├── app.py                         # Streamlit web application
├── yt_spam_detection_model        # Exported model + vectorizer (joblib)
├── Youtube-Spam-Dataset.csv       # Dataset
└── README.md
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/youtube-spam-detection.git
cd youtube-spam-detection
```

### 2. Install dependencies
```bash
pip install streamlit scikit-learn pandas numpy nltk joblib
```

### 3. Train the model (if not already exported)
Run all cells in `Youtube_Spam_Detection.ipynb` to generate `yt_spam_detection_model`.

### 4. Launch the app
```bash
streamlit run app.py
```

---

## 📈 Results

The Multinomial Naive Bayes model achieved **89.07% accuracy** on the test set, demonstrating that even a lightweight NLP pipeline can effectively detect spam patterns in short-form text like YouTube comments.

---

## 🙋 Author

**Soro**  
Data Science Student | Innomatics Research Labs  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)
