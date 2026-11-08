"""Learning Studio HTTP boundary. Run with uvicorn learning_app.main:app."""
import json
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .learning import Learning
from .store import Store, now, uid

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env", override=False)
COOKIE = "learning_session"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Register(StrictModel):
    name: str = Field(min_length=2, max_length=60)
    email: str = Field(min_length=5, max_length=180, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(min_length=10, max_length=128)


class Login(StrictModel):
    email: str = Field(min_length=1, max_length=180)
    password: str = Field(min_length=1, max_length=128)


class Ask(StrictModel):
    question: str = Field(min_length=3, max_length=2000)


class CourseCreate(StrictModel):
    title: str = Field(min_length=3, max_length=100)
    description: str = Field(default="Your personal course library.", max_length=300)
    source_title: str = Field(min_length=3, max_length=140)
    content: str = Field(min_length=80, max_length=100000)


class Material(StrictModel):
    title: str = Field(min_length=3, max_length=140)
    content: str = Field(min_length=80, max_length=100000)


class Question(StrictModel):
    topic: str = Field(min_length=2, max_length=60)
    difficulty: str = Field(pattern="^(easy|medium|hard)$")
    prompt: str = Field(min_length=8, max_length=1000)
    options: list[str] = Field(min_length=2, max_length=6)
    correct: int = Field(ge=0, le=5, strict=True)
    explanation: str = Field(min_length=10, max_length=2000)

    @field_validator("options")
    @classmethod
    def validate_options(cls, values):
        clean = [value.strip() for value in values]
        if any(not item or len(item) > 400 for item in clean) or len(set(clean)) != len(clean):
            raise ValueError("Options must be distinct, nonempty, and at most 400 characters.")
        return clean


class Answers(StrictModel):
    answers: dict[str, Annotated[int, Field(strict=True, ge=0, le=5)]] = Field(min_length=1, max_length=6)


class Feedback(StrictModel):
    helpful: bool


class TrainingExport(StrictModel):
    consent: bool


def create_app(db_path=None):
    app = FastAPI(title="Learning Studio", description="Course-grounded tutoring with assessed practice.", version="1.0.0")
    store = Store(db_path or os.getenv("PROJECT26_DB", str(BASE / "data/learning.sqlite3")))
    learning = Learning(store)
    app.state.store = store
    app.state.learning = learning
    attempts = defaultdict(deque)

    @app.middleware("http")
    async def security(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            # Same-origin cookies plus an Origin check protect browser mutations.
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-site request rejected."}, status_code=403)
            if origin:
                expected = urlparse(str(request.base_url))
                actual = urlparse(origin)
                if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
                    return JSONResponse({"detail": "Origin does not match this application."}, status_code=403)
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > 200000):
                return JSONResponse({"detail": "Request body is too large."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def user(request: Request):
        account = store.user_for_token(request.cookies.get(COOKIE))
        if not account:
            raise HTTPException(401, "Sign in to continue.")
        return account

    def course_owned(user_id, course_id):
        row = learning.course(user_id, course_id)
        if not row:
            raise HTTPException(404, "Course not found.")
        return row

    def rate_limit(request):
        address = request.client.host if request.client else "local"
        bucket = attempts[address]
        timestamp = time.monotonic()
        while bucket and timestamp - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= 12:
            raise HTTPException(429, "Too many sign-in attempts. Try again in a minute.")
        bucket.append(timestamp)

    def session(response, request, account):
        store.logout(request.cookies.get(COOKIE))
        response.set_cookie(COOKIE, store.issue_session(account["id"]), httponly=True, secure=request.url.scheme == "https", samesite="strict", max_age=604800, path="/")
        return {"user": account, "provider": learning.provider}

    @app.get("/health")
    def health():
        return {"status": "ok", "app": "learning", "provider": learning.provider}

    @app.get("/api/auth/me")
    def me(account=Depends(user)):
        return {"user": account, "provider": learning.provider}

    @app.post("/api/auth/register", status_code=201)
    def register(payload: Register, request: Request, response: Response):
        rate_limit(request)
        try:
            account = store.register(payload.name, payload.email, payload.password)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(409, "An account with that email already exists.") from exc
        return session(response, request, account)

    @app.post("/api/auth/login")
    def login(payload: Login, request: Request, response: Response):
        rate_limit(request)
        account = store.login(payload.email, payload.password)
        if not account:
            raise HTTPException(401, "Email or password is incorrect.")
        return session(response, request, account)

    @app.post("/api/auth/demo", status_code=201)
    def demo(request: Request, response: Response):
        rate_limit(request)
        account = store.register("Curious learner", f"demo-{uid()}@local.invalid", secrets.token_urlsafe(32), True)
        return session(response, request, account)

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        store.logout(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/")
        return {"signed_out": True}

    @app.get("/api/courses")
    def courses(account=Depends(user)):
        return {"courses": learning.courses(account["id"])}

    @app.post("/api/courses", status_code=201)
    def create_course(payload: CourseCreate, account=Depends(user)):
        course_id = uid()
        with store.connect() as db:
            db.execute("INSERT INTO courses VALUES(?,?,?,?,?,?,?,?)", (course_id, account["id"], "custom-" + course_id, payload.title, payload.description, "YOUR LIBRARY", "sage", now()))
            ingestion = store.ingest(db, course_id, payload.source_title, payload.content)
        return {"course": learning.course(account["id"], course_id), "ingestion": ingestion}

    @app.get("/api/courses/{course_id}")
    def course_detail(course_id: str, account=Depends(user)):
        return {"course": course_owned(account["id"], course_id), "progress": learning.progress(account["id"], course_id), "materials": learning.materials(course_id), "history": learning.history(account["id"], course_id)}

    @app.post("/api/courses/{course_id}/materials", status_code=201)
    def add_material(course_id: str, payload: Material, account=Depends(user)):
        course_owned(account["id"], course_id)
        with store.connect() as db:
            return store.ingest(db, course_id, payload.title, payload.content)

    @app.post("/api/courses/{course_id}/questions", status_code=201)
    def add_question(course_id: str, payload: Question, account=Depends(user)):
        course_owned(account["id"], course_id)
        if payload.correct >= len(payload.options):
            raise HTTPException(422, "Correct answer must refer to one of the supplied options.")
        question_id = uid()
        with store.connect() as db:
            db.execute("INSERT INTO question_bank VALUES(?,?,?,?,?,?,?,?)", (question_id, course_id, payload.topic, payload.difficulty, payload.prompt, json.dumps(payload.options), payload.correct, payload.explanation))
        return {"id": question_id, "added": True}

    @app.post("/api/courses/{course_id}/ask")
    def ask(course_id: str, payload: Ask, account=Depends(user)):
        course_owned(account["id"], course_id)
        try:
            return learning.ask(account["id"], course_id, payload.question)
        except ValueError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/api/answers/{answer_id}/feedback")
    def feedback(answer_id: str, payload: Feedback, account=Depends(user)):
        with store.connect() as db:
            updated = db.execute("UPDATE conversations SET helpful=? WHERE id=? AND owner_id=? AND abstained=0", (int(payload.helpful), answer_id, account["id"])).rowcount
        if not updated:
            raise HTTPException(404, "Answer not found or cannot be rated.")
        return {"saved": True}

    @app.post("/api/courses/{course_id}/quizzes", status_code=201)
    def create_quiz(course_id: str, account=Depends(user)):
        course_owned(account["id"], course_id)
        try:
            return learning.quiz(account["id"], course_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/quizzes/{quiz_id}/submit")
    def submit_quiz(quiz_id: str, payload: Answers, account=Depends(user)):
        try:
            return learning.submit(account["id"], quiz_id, payload.answers)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/export")
    def export(account=Depends(user)):
        data = {"exported_at": now(), "user": account, "courses": []}
        for course in learning.courses(account["id"]):
            data["courses"].append({"course": course, "materials": learning.materials(course["id"]), "progress": learning.progress(account["id"], course["id"]), "history": learning.history(account["id"], course["id"])})
        return JSONResponse(data, headers={"Content-Disposition": 'attachment; filename="learning-history.json"'})

    @app.post("/api/export/training")
    def export_training(payload: TrainingExport, account=Depends(user)):
        if not payload.consent:
            raise HTTPException(422, "Explicit consent is required to export your reviewed conversations.")
        with store.connect() as db:
            rows = db.execute("SELECT question,answer,citations FROM conversations WHERE owner_id=? AND helpful=1 AND abstained=0 ORDER BY rowid", (account["id"],)).fetchall()
        records = []
        for row in rows:
            citations = json.loads(row["citations"])
            excerpts = "\n\n".join(f"[{source['number']}] {source['source']}\n{source['excerpt']}" for source in citations)
            records.append({"messages": [
                {"role": "system", "content": "Answer only using the supplied course material. Cite the numbered sources. If unsupported, say so."},
                {"role": "user", "content": f"Course material:\n{excerpts}\n\nQuestion: {row['question']}"},
                {"role": "assistant", "content": row["answer"]},
            ]})
        return Response("\n".join(json.dumps(record) for record in records) + ("\n" if records else ""), media_type="application/x-ndjson", headers={"Content-Disposition": 'attachment; filename="reviewed-training-examples.jsonl"'})

    app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(BASE / "static/index.html")

    return app


app = create_app()
