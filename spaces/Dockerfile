FROM python:3.11-slim

WORKDIR /app
COPY . /app

EXPOSE 7860
CMD ["python", "-m", "http.server", "7860", "--bind", "0.0.0.0", "--directory", "/app"]
