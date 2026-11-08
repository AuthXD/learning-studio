import json

import pytest
from fastapi.testclient import TestClient

from learning_app.main import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    return create_app(tmp_path / "learning.sqlite3")


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        response = client.post("/api/auth/register", json={"name": "Test Learner", "email": "learner@example.com", "password": "really-good-password"})
        assert response.status_code == 201, response.text
        yield client


def courses(client):
    return client.get("/api/courses").json()["courses"]


def quiz_keys(app, quiz_id):
    with app.state.store.connect() as db:
        row = db.execute("SELECT questions FROM quizzes WHERE id=?", (quiz_id,)).fetchone()
    return {question["id"]: question["correct"] for question in json.loads(row["questions"])}


def test_authentication_private_accounts_and_logout(app, client):
    own = courses(client)[0]["id"]
    assert client.get("/api/auth/me").status_code == 200
    with TestClient(app) as stranger:
        assert stranger.get("/api/courses").status_code == 401
        assert stranger.post("/api/auth/demo").status_code == 201
        assert stranger.get(f"/api/courses/{own}").status_code == 404
        assert stranger.post(f"/api/courses/{own}/ask", json={"question": "What is overfitting?"}).status_code == 404
        assert stranger.post(f"/api/courses/{own}/quizzes").status_code == 404
    client.post("/api/auth/logout")
    assert client.get("/api/courses").status_code == 401
    assert client.post("/api/auth/login", json={"email": "learner@example.com", "password": "wrong"}).status_code == 401
    result = client.post("/api/auth/login", json={"email": "learner@example.com", "password": "really-good-password"})
    assert result.status_code == 200
    assert "HttpOnly" in result.headers["set-cookie"]
    assert "SameSite=strict" in result.headers["set-cookie"]
    with app.state.store.connect() as db:
        user = db.execute("SELECT password_hash FROM users WHERE email='learner@example.com'").fetchone()
        assert "really-good-password" not in user["password_hash"]
        assert len(user["password_hash"].split(":")) == 2


def test_retrieval_citations_history_and_abstention(client):
    catalog = courses(client)
    ml, python = catalog[0], catalog[1]
    response = client.post(f"/api/courses/{ml['id']}/ask", json={"question": "How does gradient descent update weights?"})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["mode"] == "local_extract"
    assert "learning rate" in answer["answer"]
    assert answer["citations"][0]["source"] == "Gradient descent"
    detail = client.get(f"/api/courses/{ml['id']}").json()
    source_ids = {source["id"] for source in detail["materials"]}
    assert all(item["id"] in source_ids for item in answer["citations"])
    assert detail["history"][0]["id"] == answer["id"]
    unsupported = client.post(f"/api/courses/{python['id']}/ask", json={"question": "What is gradient descent?"}).json()
    assert unsupported["abstained"] is True
    assert unsupported["citations"] == []
    assert client.post(f"/api/courses/{ml['id']}/ask", json={"question": "Who won the intergalactic jellyfish election?"}).json()["abstained"] is True


def test_quiz_first_submission_and_duplicate_are_transactional(app, client):
    course_id = courses(client)[0]["id"]
    quiz = client.post(f"/api/courses/{course_id}/quizzes").json()
    assert all("correct" not in q and "explanation" not in q for q in quiz["questions"])
    answers = quiz_keys(app, quiz["id"])
    response = client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": answers})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["correct"] == 3
    assert result["score"] == 1
    assert result["review_days"] == 7
    assert all(q["explanation"] for q in result["feedback"])
    assert result["difficulty"] == "easy", "Need six graded questions before level promotion"
    replay = client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": answers}).json()
    assert replay["replayed"] is True
    assert replay["id"] == result["id"]
    progress = client.get(f"/api/courses/{course_id}").json()["progress"]
    assert progress["total_questions"] == 3
    assert progress["quiz_count"] == 1
    second = client.post(f"/api/courses/{course_id}/quizzes").json()
    client.post(f"/api/quizzes/{second['id']}/submit", json={"answers": quiz_keys(app, second["id"])})
    progress = client.get(f"/api/courses/{course_id}").json()["progress"]
    assert progress["difficulty"] == "hard"
    assert progress["total_questions"] == 6
    assert progress["mastery"] == 1


def test_forged_score_invalid_choice_and_attempt_ownership(app, client):
    course_id = courses(client)[0]["id"]
    quiz = client.post(f"/api/courses/{course_id}/quizzes").json()
    answers = quiz_keys(app, quiz["id"])
    assert client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": answers, "quiz_score": 1}).status_code == 422
    invalid = dict(answers)
    invalid[next(iter(invalid))] = 5
    assert client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": invalid}).status_code == 422
    invalid[next(iter(invalid))] = True
    assert client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": invalid}).status_code == 422
    with TestClient(app) as stranger:
        stranger.post("/api/auth/demo")
        assert stranger.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": answers}).status_code == 404
    wrong = {key: (value + 1) % 4 for key, value in answers.items()}
    result = client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": wrong}).json()
    assert result["correct"] == 0
    assert result["review_days"] == 1


def test_material_import_idempotency_custom_question_and_citations(client):
    content = "Distributed clocks measure time across computers. Logical clocks track causal order without requiring synchronized wall clocks. Lamport timestamps increase on local events and on receiving messages."
    created = client.post("/api/courses", json={"title": "Distributed systems", "source_title": "Logical clocks", "content": content})
    assert created.status_code == 201, created.text
    course_id = created.json()["course"]["id"]
    imported = client.post(f"/api/courses/{course_id}/materials", json={"title": "Duplicate title", "content": content}).json()
    assert imported["added"] is False
    assert len(client.get(f"/api/courses/{course_id}").json()["materials"]) == 1
    answer = client.post(f"/api/courses/{course_id}/ask", json={"question": "How do logical clocks track causal order?"}).json()
    assert "causal order" in answer["answer"]
    assert answer["citations"][0]["source"] == "Logical clocks"
    assert client.post(f"/api/courses/{course_id}/quizzes").status_code == 409
    question = {"topic": "clocks", "difficulty": "easy", "prompt": "What do logical clocks track?", "options": ["Causal order", "Room temperature"], "correct": 0, "explanation": "Logical clocks track causal order across events."}
    assert client.post(f"/api/courses/{course_id}/questions", json=question).status_code == 201
    quiz = client.post(f"/api/courses/{course_id}/quizzes").json()
    assert len(quiz["questions"]) == 1
    assert client.post(f"/api/courses/{course_id}/questions", json={**question, "correct": 5}).status_code == 422


def test_origin_checks_and_xss_inert_text(client):
    course_id = courses(client)[0]["id"]
    assert client.post(f"/api/courses/{course_id}/ask", json={"question": "What is overfitting?"}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/auth/demo", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    html = client.get("/")
    assert "script-src 'self'" in html.headers["content-security-policy"]
    assert html.headers["x-content-type-options"] == "nosniff"
    assert client.post(f"/api/courses/{course_id}/ask", json={"question": "What is overfitting?"}, headers={"Origin": "http://testserver"}).status_code == 200


def test_training_export_requires_consent_and_positive_review(client):
    course_id = courses(client)[0]["id"]
    answer = client.post(f"/api/courses/{course_id}/ask", json={"question": "What is overfitting?"}).json()
    assert client.post("/api/export/training", json={"consent": False}).status_code == 422
    assert client.post("/api/export/training", json={"consent": True}).text == ""
    assert client.post(f"/api/answers/{answer['id']}/feedback", json={"helpful": True}).status_code == 200
    output = client.post("/api/export/training", json={"consent": True})
    record = json.loads(output.text.strip())
    assert "Question: What is overfitting?" in record["messages"][1]["content"]
    assert "Course material:" in record["messages"][1]["content"]
    assert "Overfitting occurs" in record["messages"][1]["content"]
    exported = client.get("/api/export").json()
    assert "password_hash" not in json.dumps(exported)
    assert len(exported["courses"]) == 3


def test_invalid_model_citation_and_empty_supported_abstention(app, client, monkeypatch):
    course_id = courses(client)[0]["id"]
    app.state.learning.provider = "openai"
    monkeypatch.setattr(app.state.learning, "_openai", lambda state: ("Invented answer", ["foreign-source"], {}))
    response = client.post(f"/api/courses/{course_id}/ask", json={"question": "What is overfitting?"})
    assert response.status_code == 502
    assert client.get(f"/api/courses/{course_id}").json()["history"] == []
    monkeypatch.setattr(app.state.learning, "_openai", lambda state: ("", [], {"total_tokens": 12}))
    response = client.post(f"/api/courses/{course_id}/ask", json={"question": "What is overfitting?"})
    assert response.status_code == 200
    assert response.json()["abstained"] is True


def test_provider_nonobject_output_is_rejected(app, client, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"output": [{"content": [{"type": "output_text", "text": "[]"}]}]}
    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            assert kwargs["json"]["store"] is False
            return FakeResponse()
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr("learning_app.learning.httpx.Client", FakeClient)
    app.state.learning.provider = "openai"
    course_id = courses(client)[0]["id"]
    response = client.post(f"/api/courses/{course_id}/ask", json={"question": "What is overfitting?"})
    assert response.status_code == 502
