# Learning Studio

Learning Studio turns course notes into a study workspace. Ask a question, open the source behind the answer, and try a practice set. The app grades your answers, updates the difficulty, and schedules a review based on your result.

## Try it locally

Use Python 3.12 or newer. From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn learning_app.main:app --host 127.0.0.1 --port 8026
```

Open a private demo or create an account. The included machine learning, Python, and SQL courses have notes and practice questions. Try asking “How does gradient descent work?” and open the cited excerpt. Use **Materials** to add your own notes and reviewed questions. [Open Learning Studio](http://127.0.0.1:8026).

## Retrieval and practice

A LangGraph workflow searches the selected course with BM25, then returns cited excerpts or abstains when it finds insufficient support. Reimporting identical material into a course reuses the existing source. In local mode, the tutor quotes relevant passages. OpenAI mode uses those passages to generate an explanation and checks the returned source IDs against the retrieved set.

FastAPI serves quizzes and grades submissions with answer keys stored in SQLite. Grading and progress updates commit in one transaction. Repeating a submission returns its original result. Difficulty uses accuracy over the last five practice sets and stays easy until at least six questions have been graded. From there, 60% selects medium and 85% selects hard. The latest practice score schedules a review after one, three, or seven days.

## Use a model and export reviewed examples

Copy `.env.example` to `.env`, then set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL`. In this mode, the Responses API receives the question, retrieved excerpts, difficulty, and up to four recent turns. The app caps the response at 1,000 output tokens.

Use **Export history** to download course and assessment data. **Export reviewed examples** creates a JSONL file from answers you marked helpful, with cited excerpts included as context. The separate command-line workflow validates and splits those examples before you can submit a training job. [Fine-tuning guide](docs/training.md).

## Verify behavior

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe evaluate.py --output evaluation-results.json
```

The 13 tests cover account and course isolation, citations, abstention, grading, duplicate submissions, imports, and training data preparation. In the included evaluation, the expected source ranked first on 12 of 12 queries and the tutor abstained on all three unsupported or wrong-course queries. The evaluation measures the included course notes. The live model adapter was tested with a simulated API, and the repository includes the training workflow without trained weights.

SQLite data lives in `data/learning.sqlite3` by default. Set `PROJECT26_DB` to choose a different file. FastAPI serves the request schemas at [localhost API docs](http://127.0.0.1:8026/docs).
