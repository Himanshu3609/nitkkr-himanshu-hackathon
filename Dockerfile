# Use Debian Bookworm (Stable LTS) Python 3.11 image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Set working directory inside container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies directly from pre-built wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and files
COPY src/ ./src/
COPY data/ ./data/
COPY docs/ ./docs/
COPY README.md LICENSE ./

# Expose Streamlit default port
EXPOSE 8501

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "src/module_a_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
