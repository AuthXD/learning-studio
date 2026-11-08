FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY learning_app ./learning_app
COPY static ./static
ENV PROJECT26_DB=/app/data/learning.sqlite3
EXPOSE 8026
CMD ["python", "-m", "uvicorn", "learning_app.main:app", "--host", "0.0.0.0", "--port", "8026"]
