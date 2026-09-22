# 🤖 Context-Aware Q&A Assistant

A production-ready conversational AI application built with **LangChain (LCEL)**, **Streamlit**, and **Context Engineering** techniques. It features automated token budget management, dynamic role-based personas, structured output parsing, and a seamless switch between **Google Gemini (Cloud)** and **Ollama (Local Offline)**.

---

## ✨ Key Features

* **🧠 Context Engineering (Token Window Management):**
  * Tracks chat history token count in real time using `tiktoken`.
  * Automatically condenses and summarizes older conversation turns when total context passes a configured budget ($N$ tokens) while keeping recent turns intact.

* **🎭 Dynamic Role-Based Personas:**
  * **Developer:** Delivers code snippets, technical logic, architecture patterns, and edge cases.
  * **Manager:** Provides high-level strategic summaries, project timelines, business impact, and budget trade-offs.

* **⚙️ Single Config Switch (Cloud vs. Local Model):**
  * Instantly switch between **Google Gemini API** (`gemini-1.5-flash`, `gemini-2.0-flash`) and local offline LLMs via **Ollama** (`llama3`).

* **📐 Structured Output Enforcement:**
  * Employs Pydantic schemas via `.with_structured_output()` to guarantee structured answers containing primary responses and step-by-step reasoning logic.

* **📊 Live Token Metrics:**
  * Sidebar progress bar displays live token counts and notifies users when context auto-summarization is triggered.

---

## 🏗️ Architecture Flow

```text
[ User Input ]
      │
      ▼
┌──────────────────────────────┐
│  Streamlit Frontend & State  │ ◄── [ Role Selector & Model Provider Switcher ]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Context Manager (tiktoken) │
│  Exceeds Token Budget?       │
│  ├── YES ──► Summarize Old Turns via Background LLM
│  └── NO  ──► Retain Original History
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Dynamic System Prompt       │ (Injected based on Role: Developer / Manager)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  LCEL Execution Chain        │ (ChatPromptTemplate | Model with Pydantic Schema)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Formatted Streamlit Output  │ (Answer + Reasoning Logic + Token Metric Update)
└──────────────────────────────┘
```
📁 Project Structure
Plaintext
context_qa_assistant/
│
├── .env                  # API Keys (Excluded from Git)
├── .env.example          # Template for environment variables
├── .gitignore            # Git ignore file
├── config.py             # Global configuration & Model Factory
├── utils/
│   ├── __init__.py       # Package marker
│   ├── token_manager.py  # Token counter & history summarizer logic
│   └── prompts.py        # Role-based dynamic system prompts
├── app.py                # Main Streamlit interface & LCEL chain
├── requirements.txt      # Dependency manifest
└── README.md             # Project documentation
🛠️ Tech Stack
Frontend: Streamlit

Orchestration Framework: LangChain (LCEL - LangChain Expression Language)

LLM Providers: Google Generative AI (langchain-google-genai), Ollama (langchain-ollama)

Token Tracking: tiktoken

Data Validation: Pydantic v2

Environment Management: python-dotenv

🚀 Quickstart Guide
1. Clone the Repository
Bash
git clone [https://github.com/your-username/context-qa-assistant.git](https://github.com/your-username/context-qa-assistant.git)
cd context-qa-assistant
2. Set Up Virtual Environment & Dependencies
Bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\activate

# Install dependencies without caching
pip install --no-cache-dir -r requirements.txt
3. Environment Configuration
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY="your_gemini_api_key_here"
4. Run the Application
streamlit run app.py

Application Screenshots
![Application screenshot 1](screenshots/Screenshot%202026-09-22%20213812.png)
![Application screenshot 2](screenshots/Screenshot%202026-09-22%20213349.png)
![Application screenshot 3](screenshots/Screenshot%202026-09-22%20213335.png)
![Application screenshot 4](screenshots/Screenshot%202026-09-22%20211223.png)