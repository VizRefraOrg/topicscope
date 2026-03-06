# TopicScope

**Textual Data Visual Analysis Platform**

TopicScope analyses unstructured text and produces interactive geographic-map-style visualisations of discovered topics. Peaks represent dominant topics, proximity encodes semantic similarity, and elevation reflects relevance.

## Architecture

- **Backend:** FastAPI (Python 3.11) 
- **NLP:** Azure AI Language (NER + Entity Linking)
- **Embeddings:** Azure OpenAI text-embedding-3-small
- **Frontend:** Next.js + Three.js (Phase 2)
- **Database:** PostgreSQL on Azure (Phase 2)

## Local Development

```bash
# Clone
git clone https://github.com/VizRefraOrg/topicscope.git
cd topicscope

# Create .env from template
cp .env.example .env
# Edit .env with your Azure keys

# Install dependencies
pip install -r backend/requirements.txt

# Run
uvicorn backend.main:app --reload --port 8000
```

## API Usage

```bash
curl -X POST http://localhost:8000/api/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here..."}'
```

## Deployment

Pushes to `main` branch auto-deploy to Azure via GitHub Actions.
