#!/bin/bash
# Download spaCy model if not already present
python -m spacy download en_core_web_lg 2>/dev/null || python -m spacy download en_core_web_sm
# Start the app
uvicorn backend.main:app --host 0.0.0.0 --port 8000
