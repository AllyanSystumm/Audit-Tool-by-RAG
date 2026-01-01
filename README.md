

**Project Title**
Audit Tool by RAG

**Project Summary**
RAG‑powered verification checkpoint generator for audit/process documents. Ingests PDFs, DOCX, TXT, XLSX. Indexes text in a Chroma vector database. Optionally uses BM25. Generates verification checkpoints using an LLM (Ollama or HuggingFace). Backend is a FastAPI service exposing endpoints for ingestion, generation, and management.

**Quick Start Setup**
Create a Python virtual environment, install dependencies from requirements file, configure environment variables in a `.env` file, and run the backend service on localhost.

**Required Environment Variables**
HF_TOKEN for HuggingFace inference (optional if using Ollama), OLLAMA_API_KEY for Ollama, CHROMA_DB_PATH for vector database location, BACKEND_HOST for host, BACKEND_PORT for port, EMBEDDING_MODEL for default embedding model, LLM_MODEL and DEFAULT_LLM for LLM selection, API_KEY_ENABLED and API_KEYS for API key protection, RATE_LIMIT settings for rate limiting, MAX_UPLOAD_SIZE_MB for upload limits. Config.py contains defaults and extra options.

**Core API Endpoints**
GET slash basic info; GET health for service and components status; GET health/llm for LLM health; GET metrics for runtime metrics and embedding cache statistics; POST upload for file upload and text extraction; POST ingest‑documents to validate and store vectors; POST generate‑checkpoints to generate verification checkpoints with parameters for RAG usage, number of checkpoints, ingest to knowledge base, process type, and LLM model; GET llm‑models to list available LLM models; GET database‑info for vector database stats and embedding model info; DELETE reset‑database to clear the vector database and reset BM25 index.

**Architecture and Components**
Backend folders include ingestion with DocumentProcessor, SemanticChunker and EmbeddingGenerator; retrieval with VectorStore (Chroma), BM25Retriever, HybridRetriever; generation with LLMClient supporting Ollama and HuggingFace and CheckpointGenerator; middleware covering API key authentication, rate limiting and logging; data uploads folder for temporary file storage; vector_db for Chroma DB files. A frontend may be present for UI.

**File Constraints and Validation**
Allowed file extensions include docx, pdf, xlsx, xls, xml, pptx, and txt. Filename validation and sanitization enforced with defaults and limits controlled via configuration.

**Security and Secrets**
`.env` file is ignored by git. Keep secrets out of version control. Required tokens must be set for LLM features. If secrets are accidentally committed, rotate immediately, remove them from history, and force push cleans.

**Testing**
Unit tests can be run using pytest within backend tests folder.

**Development Notes**
Config.py centralizes defaults and runtime settings. To add new LLM providers or models, update LLM_MODELS and implement clients. Uploaded files are ephemeral by default, with options to persist upload files via settings.

**Troubleshooting**
LLM health endpoint may return service unavailable if tokens are missing or invalid. Push attempts to GitHub may be blocked by secret scanning requiring removal of secrets before push. Vector database path issues may require verification of directory existence and write permissions.

**Contributing**
Fork the repository, create feature branches, add tests, and open pull requests. Follow config defaults and security guidance.

