
# ⚡ Twitter/X Sentiment Analysis

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg" alt="Twitter Blue Logo" width="100" height="100">
  <h3>High-Performance Sentiment Extraction Engine</h3>
  <p><i>Streamlined · Reproducible · Full-Stack · Dependency-Free</i></p>
</div>

---

## 🚀 Introduction

**Twitter/X Sentiment Analysis** is a full-stack framework designed to deliver **fast and reliable sentiment insights** from tweet datasets. It combines a **React frontend**, **Node.js backend**, and a **Python-based ML engine** to provide real-time sentiment classification.

Whether you're conducting **academic research**, **competitive brand monitoring**, or building a **data-driven product**, this tool converts raw tweet text into **actionable sentiment signals**.

---

## ✨ Key Features

- 🎯 **Tri-Class Sentiment Detection** → Predicts **Positive**, **Negative**, and **Neutral** sentiments.
- 🧠 **Python ML Engine** → Lightweight, fast, and dependency-free.
- 🌐 **Full-Stack Integration** → React UI + Express API + Python backend.
- 🔒 **Privacy-First Design** → Works with custom CSV datasets (no Twitter API required).
- 🔓 **Open Source** → Licensed under **GNU GPL v3.0**.

---

## 📂 Project Structure

```plaintext
brand_monitoring/
├── frontend/                            # React-based UI
│   ├── src/
│   │   └── components/                  # Input form, sentiment display
│   ├── public/
│   └── package.json                     # Frontend dependencies
│
├── backend/                             # Node.js + Express server
│   ├── index.js                         # API entry point
│   └── routes/
│       └── sentimentRoute.js            # Receives input, invokes Python script
│
├── python/                              # Sentiment Analysis Engine
│   ├── Data/
│   │   └── training.1600000.processed.csv
│   ├── models/
│   │   └── v1/                          # Serialized ML models (.pkl)
│   ├── analyzer.py                      # Real-time prediction logic
│   ├── Data_set_Scrap.py                # ETL: Scraping & Cleaning
│   ├── date_vectorize.py                # TF-IDF Vectorization
│   ├── pkl_viewer.py                    # Model diagnostics
│   └── Train_Model.py                   # Training & persistence
└── .vscode/                             # IDE configuration
```

---

## 🔄 Data Flow

1. **Frontend (React)**  
   - User enters tweet text via input form  
   - Form triggers API call to backend

2. **Backend (Node.js + Express)**  
   - Receives input via REST endpoint  
   - Spawns Python process (`analyzer.py`) using `child_process` or `python-shell`  
   - Sends input to Python script and receives sentiment output

3. **Python Engine**  
   - `analyzer.py` loads trained model  
   - Vectorizes input and predicts sentiment  
   - Returns result to backend

4. **Frontend**  
   - Displays sentiment label (Positive / Negative / Neutral) to user

---

## 🛠️ Tech Stack

| Layer           | Technology       | Purpose                              |
|----------------|------------------|--------------------------------------|
| **Frontend**    | React            | UI for input and result display      |
| **Backend**     | Node.js + Express| API routing and Python bridge        |
| **ML Engine**   | Python 3.x       | Sentiment analysis and prediction    |
| **Data**        | pandas           | Data manipulation                    |
| **ML**          | scikit-learn     | Classification algorithms            |
| **NLP**         | TfidfVectorizer  | Text vectorization                   |
| **Persistence** | joblib           | Model serialization                  |

---

## ⚙️ Setup Instructions

### 1️⃣ Install Python Dependencies

```bash
pip install pandas scikit-learn joblib
```

### 2️⃣ Install Node.js Dependencies

```bash
cd backend
npm install
```

### 3️⃣ Install React Frontend Dependencies

```bash
cd frontend
npm install
```

---

## 🚦 Workflow Pipeline

### 🧪 Model Training (Python)

```bash
python Data_set_Scrap.py       # Scrape & clean data
python date_vectorize.py       # TF-IDF vectorization
python Train_Model.py          # Train and save model
```

### 🔍 Inference & Diagnostics

```bash
python analyzer.py             # Real-time sentiment prediction
python pkl_viewer.py           # Model inspection
```

### 🌐 Full-Stack Execution

1. Start backend server:

```bash
cd backend
node index.js
```

2. Start frontend:

```bash
cd frontend
npm start
```

---

## 📜 License & Data Policy

- Licensed under **GNU GPL v3.0** → Free to use, modify, and distribute.
- Uses **pre-cleaned datasets** → No raw Twitter API data, compliant with X/Twitter’s developer policy.
- Logo image licensed under **Creative Commons Attribution-Share Alike 4.0 International**.

---

## 🤝 Contributing

We welcome contributions from developers, researchers, and enthusiasts!

1. Fork the repository  
2. Create your feature branch → `git checkout -b feature/AmazingFeature`  
3. Commit your changes → `git commit -m 'Add AmazingFeature'`  
4. Push to the branch → `git push origin feature/AmazingFeature`  
5. Open a Pull Request

---

## 📬 Contact

For **enterprise inquiries**, **research collaboration**, or **technical support**:

- Open a GitHub Issue  
- Contact the maintainer directly via email or LinkedIn  

---

Let me know if you'd like me to help write the backend API code or React component next!
