FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set default values for environment variables
ENV ANTHROPIC_MODEL="claude-3-opus-20240229" \
    TEMPERATURE="0.15" \
    MAX_TOKENS="4096" \
    MAX_RESPONSE_SIZE_MB="100"

CMD ["python", "chatbot.py"]