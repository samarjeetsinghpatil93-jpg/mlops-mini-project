
FROM python:3.14

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY flask_app/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

# Copy Flask application
COPY flask_app/ ./flask_app/

# Copy model artifacts
COPY models/vectorizer.pkl ./models/vectorizer.pkl

# Expose Flask port
EXPOSE 5000

# Run Flask as a Python package
CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app.app"]
