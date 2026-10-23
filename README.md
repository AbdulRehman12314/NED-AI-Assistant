# 🤖 NED-AI-Assistant

An AI-powered assistant built with **Streamlit** designed to streamline information retrieval, admissions guidance, and assistant workflows for **NED University**. This project utilizes an embedded knowledge base coupled with vector searches to deliver highly accurate answers.

---

## 🚀 Features
* **Intelligent Query Resolution:** Answers admission, department, and general administrative questions dynamically.
* **Vector Search Database:** Integrated ChromaDB infrastructure (`ned_admissions_db`) for fast and reliable context retrieval.
* **Clean UI/UX:** Interactive chat interface powered seamlessly by Streamlit.
* **Secure Configurations:** Environment management using `.env` handling for API integrations.

---

## 📁 Repository Structure
```text
NED-Ai-assistant/
├── ned_admissions_db/       # ChromaDB vector database directory
├── app.py                  # Primary Streamlit execution entry point
├── NED_Ai_Assistant.ipynb  # Development notebook / experimentation pipelines
├── requirements.txt        # Managed project package dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Local Setup Instructions

Follow these steps to run the Streamlit application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com
cd NED-AI-Assistant
```

### 2. Configure a Virtual Environment
Create and activate a virtual environment to manage dependencies locally:
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Ensure all application requirements are configured correctly:
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and append your required secrets/credentials:
```text
OPENAI_API_KEY=your_api_key_here
# Add other necessary database or API environment variables below
```

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
