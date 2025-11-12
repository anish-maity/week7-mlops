FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all app files
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8100

# Run the FastAPI app
CMD ["uvicorn", "iris_log_app:app", "--host", "0.0.0.0", "--port", "8100"]

