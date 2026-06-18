FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY faithfulai/ ./faithfulai/
COPY server/ ./server/
COPY .env .

# Install dependencies
RUN pip install --no-cache-dir -e ".[api]"

# Expose port
EXPOSE 8100

# Run the API
CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8100"]
