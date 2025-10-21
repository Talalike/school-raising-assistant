# 1. Base image
FROM python:3.12-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy production dependencies only
COPY requirements.prod.txt .

# 4. Install only required packages
RUN pip install --no-cache-dir -r requirements.prod.txt

# 5. Copy only the necessary source code
COPY main_api.py .
COPY pipeline/ pipeline/
COPY utils/ utils/
COPY embeddings/ embeddings/

# 6. Expose the port required by Cloud Run
EXPOSE 8080

# 7. Start the FastAPI app with Uvicorn
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8080"]

