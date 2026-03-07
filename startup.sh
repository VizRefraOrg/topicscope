#!/bin/bash
# Try to use en_core_web_lg if available (pre-installed during build)
# Otherwise download en_core_web_sm as fallback
python -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null || \
python -m spacy download en_core_web_sm

uvicorn backend.main:app --host 0.0.0.0 --port 8000
