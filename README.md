# Mental Care Support System
**An AI-driven intent classification system with deterministic critical intent detection.**

## 📌 Project Overview
This project is a specialized mental care support chatbot designed to categorize user inquiries into **distinct intents and map them to context-aware responses.** The system utilizes a **Deep Learning architecture (Bi-LSTM)** paired with a deterministic intent detection-response layer to ensure high-stakes queries receive pre-validated, safe responses.

## 📊 Data Source
The model was trained on four specialized mental health datasets from Hugging Face:
* `heliosbrahma/mental_health_chatbot_dataset`
* `nbertagnolli/counsel-chat`
* `Indulekha/mental-health`
* `alexandreteles/mental-health-conversational-data`


### Key Features:
* **Contextual Understanding:** Uses a Bidirectional LSTM (Bi-LSTM) to analyze user sentiment from both directions of a sentence.
* **Critical Intent Detection:** A safety feature that overrides AI predictions when high-risk language (e.g., suicide intent) is detected.
* **Modular Architecture:** Clean separation between the data pipeline, model training, and the user interface.

---

## 🏗️ System Architecture
The system is divided into four core modules to ensure scalability and ease of maintenance:

1.  **Data Pipeline (`data_preprocessing.py`):** Handles label mapping, text cleaning, and tokenization for the intent schema from 4 distinct Hugging Face datasets.
2.  **Model Training (`model_training.py`):** Defines the Bi-LSTM architecture and handles the training process using the prepared dataset.
3.  **Inference Engine (`chat_response.py`):** The "Brain" of the system, managing model predictions and **deterministic safety overrides**.
4.  **Interface (`main.py`):** A responsive GUI built with Gradio for real-time user interaction.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* TensorFlow, NumPy, Pandas, Scikit-learn, Gradio, datasets

### Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/enyin-yap/mental-care-support-system.git
   cd mental-care-support-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model:**
   ```bash
   python model_training.py
   ```

4. **Launch the Chatbot:**
   ```bash
   python main.py
   ```
---

## 📊 Evaluation & Audit
To maintain transparency of the system's capabilities:
* **`check_intent.py`**: A dedicated script used to verify the total intent coverage of the trained model, ensuring all categories are correctly recognized and accessible.

---

## 🎥 Project Demonstration
A full walkthrough of the system's functionality, including its ability to handle both general conversational queries and critical safety triggers, is available in the media folder.
* **Watch the Demo:** `media/demo_video.mp4`

---

## 📚 References 
To maintain readability, only primary resources are listed here. Detailed citations for academic papers and API documentation can be found within the comments of the Python files.

* **Primary Datasets:** Aggregated from [Hugging Face](https://huggingface.co/).
* **Deep Learning Framework:** [TensorFlow & Keras](https://www.tensorflow.org/).
* **NLP Processing:** [NLTK](https://www.nltk.org/).

---

## 👥 Project Team & Tasks
* **Team Leader:** Nik Zhi An - Model Architecture
* **Team Member:** Nicholas Ng Yan Zhe - Response Handling
* **Team Member:** Yap En Yin - Datasets Preprocessing 
* **Team Member:** Ong Shi Ning - Deployment and GUI

---

## ⚠️ Disclaimer
**This is a student research project and is not intended to provide professional medical advice, diagnosis, or treatment.** The responses generated are for demonstration purposes only. Users experiencing a mental health crisis should not rely on this application and are encouraged to contact professional emergency services or local carelines immediately.




   