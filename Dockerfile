FROM python:3.12-slim

WORKDIR /app

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY app ./app
COPY src ./src
COPY model ./model
COPY data/reference ./data/reference

EXPOSE 8000

CMD ["python", "-m", "app.wsgi"]