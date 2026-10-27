FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY shopops ./shopops
COPY data ./data
COPY context ./context
COPY evals ./evals
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8100 8200
CMD ["uvicorn", "shopops.api:app", "--host", "0.0.0.0", "--port", "8200"]
