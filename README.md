<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap" rel="stylesheet">


<div align="center">

```
███╗   ███╗ ██████╗  ██████╗ ██████╗     ██╗     ███████╗███╗   ██╗███████╗
████╗ ████║██╔═══██╗██╔═══██╗██╔══██╗    ██║     ██╔════╝████╗  ██║██╔════╝
██╔████╔██║██║   ██║██║   ██║██║  ██║    ██║     █████╗  ██╔██╗ ██║███████╗
██║╚██╔╝██║██║   ██║██║   ██║██║  ██║    ██║     ██╔══╝  ██║╚██╗██║╚════██║
██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██████╔╝    ███████╗███████╗██║ ╚████║███████║
╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═════╝     ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

<h3>🔍 Brand Sentiment Intelligence Platform</h3>

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Vercel-black?style=for-the-badge)](https://brand-sentiment-silk.vercel.app/)
[![API Server](https://img.shields.io/badge/🤗_API_Server-HuggingFace-yellow?style=for-the-badge)](https://subhankarmannayfy-sentiment-analysis.hf.space)
[![API Docs](https://img.shields.io/badge/📖_API_Docs-Swagger-green?style=for-the-badge)](https://subhankarmannayfy-sentiment-analysis.hf.space/docs)
[![License](https://img.shields.io/badge/License-GPL_v3.0-blue?style=for-the-badge)](LICENSE)

<p><em>Multi-Model · Real-Time · Full-Stack · Zomato Brand Intelligence</em></p>

</div>

---

## ✦ What is Mood Lens?

> **Mood Lens** is a production-grade sentiment analysis platform purpose-built for **brand monitoring** — starting with **Zomato**. It doesn't rely on a single model. Instead, it runs **4 parallel sentiment engines** and aggregates their predictions to deliver richer, more accurate insights from customer reviews and social media text.

---

## ⚡ Quick Links

| Resource | URL |
|----------|-----|
| 🌐 **Frontend App** | [brand-sentiment-silk.vercel.app](https://brand-sentiment-silk.vercel.app/) |
| 🤗 **API Server** | [subhankarmannayfy-sentiment-analysis.hf.space](https://subhankarmannayfy-sentiment-analysis.hf.space) |
| 📖 **API Docs (Swagger)** | [.hf.space/docs](https://subhankarmannayfy-sentiment-analysis.hf.space/docs) |

---

## 🧠 The 4-Model Engine

Mood Lens runs **four sentiment models in parallel** — from classical ML to state-of-the-art transformers:

```
Input Text
    │
    ├──▶  [ Model 1 ]  RoBERTa Transformer     ──▶ Sentiment + Confidence
    ├──▶  [ Model 2 ]  Classical ML (sklearn)  ──▶ Sentiment + Confidence
    ├──▶  [ Model 3 ]  Classical ML (sklearn)  ──▶ Sentiment + Confidence
    └──▶  [ Model 4 ]  Ensemble / Hybrid       ──▶ Sentiment + Confidence
                │
                ▼
         Aggregated Prediction
      [ Positive / Negative / Neutral ]
```

| # | Model | Type | Strength |
|---|-------|------|----------|
| 🔵 **1** | **RoBERTa** | Transformer (Deep Learning) | Highest contextual accuracy |
| 🟢 **2** | **Classical ML** | scikit-learn | Fast inference, lightweight |
| 🟡 **3** | **Classical ML** | scikit-learn | Domain-specific training |
| 🔴 **4** | **Ensemble** | Hybrid | Aggregated robustness |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER BROWSER                        │
│              brand-sentiment-silk.vercel.app            │
│                   [ React Frontend ]                    │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP Request
                         ▼
┌─────────────────────────────────────────────────────────┐
│               HUGGING FACE SPACES                       │
│     subhankarmannayfy-sentiment-analysis.hf.space       │
│                  [ FastAPI Server ]                     │
│                                                         │
│   POST /predict          GET /docs (Swagger UI)         │
│   GET  /health           GET /openapi.json              │
│                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐    │
│   │ RoBERTa  │  │ Model 2  │  │ Model 3  │  │  M4  │    │
│   └──────────┘  └──────────┘  └──────────┘  └──────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 🔌 API Reference

**Base URL:** `https://subhankarmannayfy-sentiment-analysis.hf.space`

**Interactive Docs:** [`/docs`](https://subhankarmannayfy-sentiment-analysis.hf.space/docs)

### `POST /predict`

Analyze sentiment of input text using all 4 models.

```bash
curl -X POST https://subhankarmannayfy-sentiment-analysis.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Zomato delivery was super fast and food was amazing!"}'
```

**Response:**
```json
{
  "text": "Zomato delivery was super fast and food was amazing!",
  "predictions": {
    "roberta":  { "label": "POSITIVE", "score": 0.97 },
    "model_2":  { "label": "POSITIVE", "score": 0.91 },
    "model_3":  { "label": "POSITIVE", "score": 0.88 },
    "ensemble": { "label": "POSITIVE", "score": 0.93 }
  },
  "final_sentiment": "POSITIVE",
  "confidence": 0.93
}
```

### `GET /health`

Check API server status.

```bash
curl https://subhankarmannayfy-sentiment-analysis.hf.space/health
```

---

## 📂 Project Structure

```
brand_monitoring/
│
├── 🖥️  backend/                         ← FastAPI Server (HuggingFace Spaces)
│   ├── app.py                           # API entry point & route definitions
│   ├── roberta_predict.py               # RoBERTa transformer inference engine
│   ├── requirements.txt                 # Python dependencies
│   └── Dockerfile                       # Container configuration
│
├── 🎨  frontend/                        ← React UI (Vercel)
│   ├── public/
│   │   ├── index.html
│   │   └── manifest.json
│   └── src/
│       ├── App.js                       # Root component
│       ├── App.css
│       ├── index.js
│       └── index.css
│
└── 🔬  python/                          ← ML Training Pipeline
    ├── data_raw/                        # Original raw datasets
    ├── data_processed/                  # Cleaned & vectorized data
    └── models/                          # Serialized models (.pkl / .pt)
```

---

## 🛠️ Tech Stack

```
Frontend    │  React.js          →  UI & sentiment display
Backend     │  FastAPI (Python)  →  REST API & model orchestration
Transformer │  RoBERTa           →  Deep learning sentiment model
ML Engine   │  scikit-learn      →  Classical classification models
NLP         │  TF-IDF            →  Text vectorization
Data        │  pandas            →  Preprocessing & ETL
Infra       │  Docker            →  Backend containerization
Deploy      │  HuggingFace       →  API server hosting
Deploy      │  Vercel            →  Frontend hosting
```

---

## ⚙️ Local Development

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd brand_monitoring
```

### 2. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
# → Server: http://localhost:8000
# → Docs:   http://localhost:8000/docs
```

Or with Docker:

```bash
cd backend
docker build -t mood-lens-api .
docker run -p 8000:8000 mood-lens-api
```

### 3. Frontend (React)

```bash
cd frontend
npm install
npm start
# → App: http://localhost:3000
```

> ⚠️ For local dev, update the API base URL in your React app to `http://localhost:8000` instead of the HuggingFace Space URL.

---

## 🧪 ML Pipeline (Training)

```bash
cd python

# Activate virtual environment
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Run training pipeline
python preprocess.py            # Clean & process raw data
python vectorize.py             # TF-IDF vectorization
python train.py                 # Train & save models → /models
```

---

## 🚀 Deployment

| Layer | Platform | URL |
|-------|----------|-----|
| **Frontend** | Vercel | [brand-sentiment-silk.vercel.app](https://brand-sentiment-silk.vercel.app/) |
| **API Backend** | HuggingFace Spaces (Docker) | [subhankarmannayfy-sentiment-analysis.hf.space](https://subhankarmannayfy-sentiment-analysis.hf.space) |
| **API Docs** | Swagger UI | [.hf.space/docs](https://subhankarmannayfy-sentiment-analysis.hf.space/docs) |

---

## 📜 License

Licensed under **GNU GPL v3.0** — Free to use, modify, and distribute with attribution.

---

## 🤝 Contributing

```bash
# 1. Fork the repo
# 2. Create your branch
git checkout -b feature/my-feature

# 3. Commit changes
git commit -m "feat: add my feature"

# 4. Push and open a PR
git push origin feature/my-feature
```

---

<div align="center">

**Mood Lens** — *See how the world feels about your brand*

<p>Made with ❤️ by <a href="https://subhankarmannayfy-sentiment-analysis.hf.space">Mood Lens</a></p>

</div>