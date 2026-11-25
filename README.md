# ⚡ LLM Pipeline: End-to-End AI Engineering Dashboard

A complete, production-grade, modular, and beginner-friendly Large Language Model (LLM) Pipeline application built with **Python**, **Streamlit**, and the **OpenAI Python SDK**.

This project demonstrates how real-world LLM systems ingest raw user input, sanitize and validate text, build structured prompts, execute live API calls with latency and token telemetry, post-process model responses, and generate structured analytical reports.

---

## 🌟 Key Features

- **True 5-Stage Architecture**: Clear separation of concerns spanning Input Sanitization, Prompt Construction, Live LLM Inference, Response Processing, and Output Packaging.
- **Visual Execution Dashboard**: Live interactive Streamlit UI with real-time status indicators (`⏳ Pending`, `🔄 Running`, `✅ Completed`, `❌ Failed`) for every pipeline stage.
- **Multi-Provider OpenAI Compatibility**: Native support for **OpenAI** (`gpt-4o`, `gpt-4o-mini`), **Groq** (`llama-3.3-70b-versatile`), **DeepSeek** (`deepseek-chat`), **Together AI**, and **Local Ollama** (`llama3`, `mistral`, `qwen`).
- **Prompt Engineering Studio**: Customizable output styles (*Concise*, *Detailed*, *Bullet Points*, *Beginner-Friendly*, *Technical*, *Creative*) and personas (*Software Engineer*, *Educator*, *Executive PM*, *Research Scientist*).
- **Comprehensive Telemetry**: Round-trip latency measurement per stage, tokens per second speed, prompt/completion/total token tracking, and reading time estimation.
- **Interactive Inspectability**: Tabbed interface featuring Original Input vs Output comparisons, intermediate stage artifact inspectors (JSON), live bash execution logs, clipboard copying, and `.txt`/`.json` file downloads.
- **Robust Error Handling**: Graceful trapping and diagnostic messaging for invalid API keys, rate limits, network timeouts, and token overflow.
- **100% Tested**: Complete unit and integration test suite with `pytest`.

---

## 🏗️ Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LLM PIPELINE WORKFLOW                           │
└────────────────────────────────────────────────────────────────────────┘

  [ Raw User Input ]
          │
          ▼
  ┌─────────────────────────────────┐
  │ Stage 1: Input Processing       │
  │ • Unicode normalization (NFKC)  │
  │ • Control character removal     │
  │ • Whitespace & newline collapse │
  │ • Min/max boundary validation   │
  └─────────────────────────────────┘
          │ (Cleaned Text + Input Stats)
          ▼
  ┌─────────────────────────────────┐
  │ Stage 2: Prompt Engineering     │
  │ • Persona injection             │
  │ • Output style & tone rules     │
  │ • Delimited user request frame  │
  │ • Chat completion messages list │
  └─────────────────────────────────┘
          │ (System + User Prompt Messages)
          ▼
  ┌─────────────────────────────────┐
  │ Stage 3: LLM Integration        │
  │ • OpenAI / Provider endpoint    │
  │ • Hyperparameters (temp/tokens) │
  │ • Precision latency measurement │
  │ • Rate limit & exception guard  │
  └─────────────────────────────────┘
          │ (Raw LLM Response + Token Usage)
          ▼
  ┌─────────────────────────────────┐
  │ Stage 4: Response Processing    │
  │ • Sanitization & line format    │
  │ • Sentence & code block count   │
  │ • Key takeaway extraction       │
  │ • Tokens/sec & reading time     │
  └─────────────────────────────────┘
          │ (Processed Insights + Analytics)
          ▼
  ┌─────────────────────────────────┐
  │ Stage 5: Output Generation      │
  │ • Execution bundle aggregation  │
  │ • Input vs output comparison    │
  │ • Export text/JSON reports      │
  │ • Streamlit visual rendering    │
  └─────────────────────────────────┘
```

---

## 📂 Project Structure

```text
llm-pipeline/
├── app.py                      # Streamlit frontend dashboard & interactive UI
├── config/
│   ├── __init__.py             # Config package exports
│   └── settings.py             # AppConfig, style templates, model registry, constants
├── pipeline/
│   ├── __init__.py             # Pipeline package exports
│   ├── input_processor.py      # Stage 1: Validation, sanitization, input metrics
│   ├── prompt_builder.py       # Stage 2: Persona & style prompt templating
│   ├── llm_client.py           # Stage 3: OpenAI-compatible API client & error handling
│   ├── response_processor.py   # Stage 4: Output cleaning, key point & metric extraction
│   ├── output_handler.py       # Stage 5: Bundle compilation, text & JSON serialization
│   └── pipeline_runner.py      # Orchestrator: coordinates generator step-by-step
├── tests/
│   ├── test_input_processor.py # Unit tests for Stage 1
│   ├── test_prompt_builder.py  # Unit tests for Stage 2
│   ├── test_llm_client.py      # Unit tests for Stage 3
│   ├── test_response_processor.py # Unit tests for Stage 4
│   ├── test_output_handler.py  # Unit tests for Stage 5
│   └── test_pipeline_runner.py # Integration tests for complete pipeline
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules (secrets, venv, caches)
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Detailed project documentation
```

---

## 💻 Tech Stack

- **Language**: Python 3.10+ (tested on Python 3.14)
- **Frontend / Dashboard**: Streamlit
- **LLM Integration**: OpenAI Python SDK (`openai>=1.30.0`)
- **Environment Management**: `python-dotenv`
- **Testing**: `pytest`

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher installed on your machine.
- An API key for an LLM provider (OpenAI, Groq, DeepSeek, Together AI, or local Ollama).

### 2. Clone or Navigate to the Repository
```bash
cd "llm pipeline"
```

### 3. Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Key
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and add your API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

*(Note: You can also enter or override your API key directly within the Streamlit sidebar UI at runtime.)*

#### Using Alternative Providers (Groq, Ollama, DeepSeek, Together AI)
Because the pipeline uses the universal OpenAI client specification, you can switch providers simply by configuring the Base URL:

- **Groq**:
  ```env
  OPENAI_API_KEY=gsk_your_groq_key
  OPENAI_BASE_URL=https://api.groq.com/openai/v1
  DEFAULT_MODEL=llama-3.3-70b-versatile
  ```
- **Local Ollama** (No API key needed):
  ```env
  OPENAI_API_KEY=ollama
  OPENAI_BASE_URL=http://localhost:11434/v1
  DEFAULT_MODEL=llama3
  ```
- **DeepSeek**:
  ```env
  OPENAI_API_KEY=sk_your_deepseek_key
  OPENAI_BASE_URL=https://api.deepseek.com/v1
  DEFAULT_MODEL=deepseek-chat
  ```

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

The application will open automatically in your default browser at:
`http://localhost:8501`

---

## 🧪 Running the Test Suite

Run all unit and integration tests with detailed output:
```bash
pytest -v
```

All 29 tests covering all five stages, edge cases, error handlers, and end-to-end integration will execute.

---

## 📖 Detailed Explanation of the 5 Pipeline Stages

### Stage 1: Input Processing (`pipeline/input_processor.py`)
- **Objective**: Prevent garbage-in, garbage-out by validating and normalizing user text before it touches downstream logic or APIs.
- **Operations**:
  1. *Unicode Normalization*: Converts compatibility characters into standard forms using `unicodedata.normalize('NFKC')`.
  2. *Control Character Stripping*: Cleans unprintable non-text bytes while preserving standard line feeds and tabs.
  3. *Whitespace Compression*: Collapses redundant horizontal spaces within lines while preserving paragraph boundaries.
  4. *Constraint Validation*: Enforces minimum and maximum character thresholds and guards against whitespace-only submissions.
  5. *Token & Metric Estimation*: Computes character count, word count, and preliminary token estimations.

### Stage 2: Prompt Engineering (`pipeline/prompt_builder.py`)
- **Objective**: Transform raw text into an optimized instruction payload for conversational LLMs.
- **Operations**:
  1. *Persona Assignment*: Selects an expert persona (e.g., Senior Software Engineer, Patient Educator).
  2. *Style Injection*: Appends explicit structural guidelines (e.g., bulleted lists, concise summaries, technical depth).
  3. *Delimited Framing*: Wraps user inputs in markdown blocks (```) to guard against prompt injection.
  4. *Message Formatting*: Structures output as standardized chat messages: `[{"role": "system", ...}, {"role": "user", ...}]`.

### Stage 3: LLM Integration (`pipeline/llm_client.py`)
- **Objective**: Execute reliable, observable inference across OpenAI or compatible provider APIs.
- **Operations**:
  1. *API Communication*: Invokes the `chat.completions.create` endpoint with configurable `temperature` and `max_tokens`.
  2. *Latency Instrumentation*: Uses high-resolution timers (`time.perf_counter()`) to capture exact round-trip network and generation time.
  3. *Token Telemetry*: Captures `prompt_tokens`, `completion_tokens`, and `total_tokens`.
  4. *Error Resilience*: Catches and surfaces distinct exceptions: `AuthenticationError`, `RateLimitError`, `APIConnectionError`, and `BadRequestError`.

### Stage 4: Response Processing (`pipeline/response_processor.py`)
- **Objective**: Clean, structure, and derive actionable analytics from raw model text.
- **Operations**:
  1. *Sanitization*: Trims trailing whitespace and standardizes double line breaks.
  2. *Empty Response Protection*: Flags empty or invalid model generations before presentation.
  3. *Structural Parsing*: Automatically detects code blocks (````...````) and extracts key bullet points or numbered takeaways.
  4. *Performance Metrics*: Calculates generation throughput (`tokens / second`) and human reading time estimate (`words / 200 wpm`).

### Stage 5: Output Generation (`pipeline/output_handler.py`)
- **Objective**: Compile and package artifacts for user display, copying, and downstream storage.
- **Operations**:
  1. *Bundle Compilation*: Combines original input, sanitized text, chat messages, token stats, stage latencies, and final text into a unified `PipelineExecutionBundle`.
  2. *Comparison Presentation*: Displays original input alongside processed model response.
  3. *Export Formats*: Generates both human-readable text audit reports and machine-readable JSON files for download.

---

## 💡 Example Walkthrough

### Example Input:
> *"What is quantum computing, how do qubits differ from classical bits, and what are its most promising real-world applications in the next decade?"*

### Configuration Selected:
- **Output Style**: `Bullet Points & Summary`
- **Persona**: `Senior Software Engineer`
- **Model**: `gpt-4o-mini`

### Pipeline Execution Summary:
- **Stage 1 (Input Processing)**: Cleaned in `0.001s` (Words: 24, Chars: 154, Est. Tokens: 31)
- **Stage 2 (Prompt Engineering)**: Built in `0.001s` (Prompt Length: 492 chars)
- **Stage 3 (LLM Integration)**: API Response in `1.84s` (Prompt Tokens: 95, Completion Tokens: 210, Total: 305)
- **Stage 4 (Response Processing)**: Processed in `0.001s` (Speed: 114.1 tokens/sec, Extracted 3 Key Points)
- **Stage 5 (Output Generation)**: Packaged in `0.001s` (Total pipeline time: `1.843s`)

### Example Final Output:
> **Core Concepts**
> - **Quantum Computing**: A computing paradigm that leverages the principles of quantum mechanics to solve specialized computational problems intractable for classical computers.
> - **Classical Bits vs. Qubits**: Classical bits exist deterministically as either `0` or `1`. Qubits can exist in a linear combination of states (superposition) and interact via entanglement.
>
> **Top Real-World Applications (Next Decade)**
> - **Materials Science & Chemistry**: Simulating molecular structures for catalyst design and battery chemistry.
> - **Cryptanalysis & Post-Quantum Security**: Auditing cryptographic resilience and migrating to lattice-based cryptography.
> - **Complex Optimization**: Portfolio risk modeling, supply chain routing, and combinatorial optimization.
>
> **Summary Takeaway**: Quantum computing complements classical computing by offering polynomial or exponential speedups for specialized quantum-simulation and optimization algorithms.

---

## 🔒 Security Best Practices

- **Zero Hardcoded Secrets**: Secrets are never hardcoded. All credentials are read from `.env` or temporary session input.
- **Git Ignore Safeguards**: `.env` and credential files are explicitly excluded via `.gitignore`.
- **Delimited Inputs**: Prompts use boundary delimiters to minimize prompt injection risks.

---

## 📄 License
MIT License. Built for educational and real-world AI engineering applications.
