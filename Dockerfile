FROM python:3.10-slim

WORKDIR /app

# copy requirements first
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy full project
COPY . .

# run server with proper app reference
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
