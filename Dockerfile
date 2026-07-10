FROM python:3.12-slim

# Prevent python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the default port (Render will inject $PORT)
EXPOSE 8000

# Start the FastAPI server using the dynamically injected PORT from Render
CMD sh -c "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
