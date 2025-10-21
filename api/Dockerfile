# -------- Runtime image --------
FROM python:3.11-slim

WORKDIR /app

# 1) Install Python deps
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# 2) Copy application code (everything in api/)
COPY . .

# 3) Expose port and start FastAPI
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
