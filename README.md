# 🛡️ Bias Audit Dashboard

[![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Firebase](https://img.shields.io/badge/Hosting-Firebase-FFCA28?style=flat-square&logo=firebase)](https://firebase.google.com/)
[![Hugging Face](https://img.shields.io/badge/AI-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/)
[![Gemma](https://img.shields.io/badge/Model-Gemma--2b-blue?style=flat-square)](https://ai.google.dev/gemma)

> **Empowering AI Fairness.** A comprehensive tool designed for the **Google Solution Challenge** to detect, visualize, and mitigate bias in Large Language Models.

---

## 🚀 Live Demo

- **🌐 Frontend Dashboard**: [https://audit-ai-gsc-a3594.web.app](https://audit-ai-gsc-a3594.web.app)
- **🧠 AI Backend API**: [https://pundiradhiraj-bias-audit-backend.hf.space](https://pundiradhiraj-bias-audit-backend.hf.space)

---

## 📖 Overview

As AI models become more integrated into society, identifying and fixing algorithmic bias is critical. The **Bias Audit Dashboard** provides an end-to-end workflow for developers and researchers to audit models like **Gemma-2b** for demographic disparities and harmful stereotypes.

### 🌍 UN Sustainable Development Goals
This project directly contributes to:
- **Goal 10: Reduced Inequalities** - By ensuring AI systems treat all individuals fairly regardless of gender, race, or background.
- **Goal 16: Peace, Justice, and Strong Institutions** - By promoting transparency and accountability in automated decision-making.

---

## ✨ Key Features

### 📊 1. Measure (Audit Visualization)
Visualize bias across multiple axes (Gender, Race, Religion) using high-fidelity scatter plots and heatmaps. Identify exactly where the model deviates from fairness.

### 🚩 2. Flag (Counterfactual Probing)
Test individual prompts using **Counterfactual Probing**. Swap protected attributes (e.g., "He" to "She") and see if the model's toxicity score or response changes unfairly.

### 🛠️ 3. Fix (Mitigation Strategy)
Compare "Baseline" model outputs against "Redacted" or "Debiased" versions. Track improvement in real-time as you apply mitigation techniques.

---

## 🏗️ System Architecture

| Layer | Component | Function |
| :--- | :--- | :--- |
| **Frontend** | React 18 / Vite | High-performance SPA with real-time state management. |
| **Delivery** | Firebase Hosting | Global CDN distribution with automated SSL & edge caching. |
| **API Gateway** | FastAPI / Docker | Asynchronous bridge managing cross-origin AI inference. |
| **AI Core** | Google Gemma-2b | transformer-based LLM optimized for bias detection. |
| **Inference** | BitsAndBytes NF4 | 4-bit NormalFloat quantization for efficient cloud compute. |

### 🛠️ The Audit Workflow
Instead of traditional static testing, the system performs **Dynamic Counterfactual Probing**. For every user input, the backend generates parallel streams—one for the baseline and one for the counterfactual (e.g., swapping gender/race tokens). It then calculates the **Statistical Parity Difference** in real-time, visualizing the bias delta in your dashboard.

---

## 🛠️ Technology Stack

- **Frontend**: React 18, Vite, Lucide Icons, Modern Vanilla CSS.
- **Backend**: FastAPI, Uvicorn.
- **AI Engine**: Google Gemma-2b, Hugging Face Transformers.
- **Mitigation**: 4-bit Quantization (bitsandbytes) for efficient inference.
- **Deployment**: Firebase Hosting (Frontend), Hugging Face Spaces (Backend Docker).

---

## 💻 Local Development

### Prerequisites
- Node.js (v18+)
- Python 3.10+
- Hugging Face API Token

### Setup Frontend
```bash
npm install
npm run dev
```

### Setup Backend
```bash
cd auditai-backend
pip install -r requirements.txt
python app/main.py
```

---

## 🤝 Contributing
Built for the **Google Solution Challenge**. Contributions are welcome! Please feel free to open an issue or submit a pull request.

---

## 📜 License
MIT License. 
