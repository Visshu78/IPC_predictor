# ⚖️ IPC Predictor – Multilingual Legal AI Assistant  

## 📌 Overview  
This project is an **AI-powered legal assistant** that predicts the most relevant **Indian Penal Code (IPC) sections** from a user-provided crime description.  
It supports **10+ Indian languages**, automatically translates queries to English, and provides detailed legal information including **offense, punishment, cognizability, and bailability**.  

---

## ✨ Key Features  
- 🔍 **Hybrid IPC Prediction** → Combines **Sentence-BERT embeddings + TF-IDF keyword matching**  
- 📊 **Severity Calibration** → Aligns **crime severity vs punishment severity** for more realistic predictions  
- 🌍 **Multilingual Support** → Handles queries in **English, Hindi, Telugu, Tamil, Malayalam, Kannada, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu** using **fastText + Meta NLLB-200**  
- ⚡ **Real-Time Predictions** → Precomputed embeddings and vector caching for fast results  
- 🖥️ **Command-line Interface** → Simple CLI to test and explore predictions  

---

## 🛠️ Tech Stack  
- **Programming:** Python  
- **Libraries:** Sentence-BERT, scikit-learn, spaCy, TensorFlow, Hugging Face Transformers, fastText  
- **Models:**  
  - SentenceTransformer: `all-mpnet-base-v2`  
  - Translation: `facebook/nllb-200-distilled-600M`  
  - Language Detection: `fastText lid.176.bin`  

---

## 📂 Project Structure 
IPC_predictor/
│── ipc.py # Core IPC prediction engine (hybrid + severity calibration)
│── translator.py # Multilingual support (language detection + translation)
│── new.py # CLI interface with confidence filtering
│── test_import.py # Basic import test
│── test_fixed_translator.py # Debugging translation issues
│── debug_translation.py # Translation test script
│── merged.json # IPC dataset (sections, punishments, details)
│── ipc_embeddings.pkl # Precomputed embeddings
│── ipc_tfidf.pkl # Precomputed TF-IDF vectors
│── README.md # Project documentation


## 🚀 Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/Visshu78/IPC_predictor.git
cd IPC_predictor

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python new.py

python translator.py

```

## 📈 Future Enhancements

✅ Web/Streamlit interface for user-friendly demo
✅ Evaluation on labeled dataset for accuracy benchmarks
✅ Expand IPC dataset with more detailed case descriptions



## Myself

#Vishal Dhawal

- 🎓 B.Tech CSE @ IIIT Kottayam | Minor @ IIT Mandi
- 🔬 Interests: Machine Learning, Computer Vision, NLP, Human-Computer Interaction
- 🌐 GitHub: @Visshu78

## My Partner

#Soumallya Sarkar

- 🎓 B.Tech CSE @ IIIT Kottayam 
- 🔬 Interests: Machine Learning, Computer Vision, NLP, Computer Networking, Encryption
- 🌐 GitHub: @soumallyasarkar
