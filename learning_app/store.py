"""SQLite ownership, authentication, and durable learning records."""
import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .data import COURSES


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return secrets.token_hex(12)


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000).hex()
    return f"{salt}:{digest}"


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                    password_hash TEXT NOT NULL, created_at TEXT NOT NULL, demo INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), expires_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS courses (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id), slug TEXT NOT NULL,
                    title TEXT NOT NULL, description TEXT NOT NULL, category TEXT NOT NULL, color TEXT NOT NULL,
                    created_at TEXT NOT NULL, UNIQUE(owner_id, slug));
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id), title TEXT NOT NULL,
                    digest TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(course_id, digest));
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                    course_id TEXT NOT NULL REFERENCES courses(id), source TEXT NOT NULL, text TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS question_bank (
                    id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id), topic TEXT NOT NULL,
                    difficulty TEXT NOT NULL, prompt TEXT NOT NULL, options TEXT NOT NULL,
                    correct INTEGER NOT NULL, explanation TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id), course_id TEXT NOT NULL,
                    question TEXT NOT NULL, answer TEXT NOT NULL, citations TEXT NOT NULL, mode TEXT NOT NULL,
                    abstained INTEGER NOT NULL, latency_ms INTEGER NOT NULL, usage TEXT NOT NULL,
                    helpful INTEGER, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS quizzes (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id), course_id TEXT NOT NULL,
                    questions TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, quiz_id TEXT UNIQUE NOT NULL REFERENCES quizzes(id), owner_id TEXT NOT NULL,
                    course_id TEXT NOT NULL, correct INTEGER NOT NULL, total INTEGER NOT NULL,
                    score REAL NOT NULL, topics TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS progress (
                    owner_id TEXT NOT NULL, course_id TEXT NOT NULL, total_correct INTEGER NOT NULL DEFAULT 0,
                    total_questions INTEGER NOT NULL DEFAULT 0, difficulty TEXT NOT NULL DEFAULT 'easy',
                    mastery REAL NOT NULL DEFAULT 0, next_review TEXT, updated_at TEXT,
                    PRIMARY KEY(owner_id, course_id));
                CREATE INDEX IF NOT EXISTS chunks_course ON chunks(course_id);
                CREATE INDEX IF NOT EXISTS history_owner ON conversations(owner_id, course_id, created_at);
                CREATE INDEX IF NOT EXISTS attempts_owner ON attempts(owner_id, course_id, created_at);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def user_for_token(self, token):
        if not token:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as db:
            row = db.execute("SELECT users.id,users.name,users.email,users.demo FROM users JOIN sessions ON users.id=sessions.user_id WHERE token_hash=? AND expires_at>?", (digest, now())).fetchone()
            return dict(row) if row else None

    def register(self, name, email, password, demo=False):
        user_id = uid()
        hashed = password_hash(password)
        with self.connect() as db:
            db.execute("INSERT INTO users VALUES(?,?,?,?,?,?)", (user_id, email.lower(), name, hashed, now(), int(demo)))
            self.seed(db, user_id)
        return {"id": user_id, "name": name, "email": email.lower(), "demo": int(demo)}

    def login(self, email, password):
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
        # Always perform the hash to avoid a cheap user-enumeration timing difference.
        stored = row["password_hash"] if row else "00000000000000000000000000000000:invalid"
        candidate = password_hash(password, stored.split(":")[0])
        if not row or not hmac.compare_digest(candidate, stored):
            return None
        return {key: row[key] for key in ("id", "name", "email", "demo")}

    def issue_session(self, user_id):
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at<?", (now(),))
            db.execute("INSERT INTO sessions VALUES(?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), user_id, (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()))
        return token

    def logout(self, token):
        if token:
            with self.connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def seed(self, db, owner_id):
        for spec in COURSES:
            course_id = uid()
            db.execute("INSERT INTO courses VALUES(?,?,?,?,?,?,?,?)", (course_id, owner_id, spec["slug"], spec["title"], spec["description"], spec["category"], spec["color"], now()))
            for title, content in spec["documents"]:
                self.ingest(db, course_id, title, content)
            for topic, difficulty, prompt, options, correct, explanation in spec["questions"]:
                db.execute("INSERT INTO question_bank VALUES(?,?,?,?,?,?,?,?)", (uid(), course_id, topic, difficulty, prompt, json.dumps(options), correct, explanation))

    def ingest(self, db, course_id, title, content):
        # Course ID participates in every stable identifier, keeping sources owner-scoped.
        digest = hashlib.sha256(content.strip().encode()).hexdigest()
        document_id = hashlib.sha256(f"{course_id}:{digest}".encode()).hexdigest()[:32]
        exists = db.execute("SELECT id FROM documents WHERE course_id=? AND digest=?", (course_id, digest)).fetchone()
        if exists:
            return {"document_id": exists["id"], "added": False, "chunks": 0}
        db.execute("INSERT INTO documents VALUES(?,?,?,?,?)", (document_id, course_id, title, digest, now()))
        blocks = []
        for paragraph in content.strip().split("\n\n"):
            paragraph = paragraph.strip()
            while len(paragraph) > 1200:
                cut = paragraph.rfind(" ", 500, 1200)
                cut = cut if cut >= 500 else 1200
                blocks.append(paragraph[:cut])
                paragraph = paragraph[cut:].strip()
            if paragraph:
                blocks.append(paragraph)
        for i, text in enumerate(blocks):
            chunk_id = hashlib.sha256(f"{document_id}:{i}:{text}".encode()).hexdigest()[:24]
            db.execute("INSERT INTO chunks VALUES(?,?,?,?,?)", (chunk_id, document_id, course_id, title, text))
        return {"document_id": document_id, "added": True, "chunks": len(blocks)}
