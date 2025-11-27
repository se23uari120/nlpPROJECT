# 🎓 Semantic Reviewer Recommendation System

### **An NLP-based approach to automating academic peer review assignment using SciBERT & SPECTER.**

## 📌 Project Overview
The volume of scientific submissions to conferences is growing exponentially, making manual reviewer assignment inefficient and prone to bias. This project utilizes **Natural Language Processing (NLP)** to automate the matching process.

Instead of simple keyword matching, this system uses **Deep Learning (SPECTER)** to understand the *semantic context* of a paper's abstract and matches it with the most qualified experts from a reviewer pool.

## 🚀 Key Features
* **Semantic Matching:** Uses `allenai/specter` (BERT for scientific papers) to generate high-dimensional embeddings .
* **Conflict of Interest Filtering:** Automatically detects and filters reviewers based on authorship history.
* **Fail-Safe Architecture:** Includes a robust data loader that generates synthetic expert profiles if external datasets are unavailable.
* **Interactive Dashboard:** A user-friendly Web UI built with **Streamlit** for real-time inference and visualization.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **ML Core:** Hugging Face `sentence-transformers`, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Dataset:** PeerRead (AllenAI)

---

## 📂 Project Structure
```bash
Reviewer_AI_Project/
│
├── app.py                   # The Frontend Web Application (Streamlit)
├── nlpmainacademic.py       # The Backend Logic & Data Generation Script
├── reviewer_profiles.csv    # The Database of Experts (Generated automatically)
├── README.md                # Project Documentation
└── requirements.txt         # List of dependencies
