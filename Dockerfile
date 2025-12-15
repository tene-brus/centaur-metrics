FROM python:3.12-slim

WORKDIR /app

# Install uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY app/ ./app/

# Ensure data directory exists (app expects this at runtime)
RUN mkdir -p /app/app/data

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app
CMD ["uv", "run", "streamlit", "run", "app/main.py"]
