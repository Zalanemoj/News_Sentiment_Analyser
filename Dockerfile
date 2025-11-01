FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8080

# ... (Previous Dockerfile content) ...

# Command to run the Streamlit application.
# CRITICAL FIX: Disabling CORS and XSRF protection for Cloud Run compatibility.
CMD ["streamlit", "run", "Strimlit.py", "--server.port", "8080", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
