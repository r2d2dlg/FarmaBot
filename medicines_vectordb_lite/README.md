# Lightweight Vector Database for FarmaBot

This directory contains a minimal vector database structure for FarmaBot that's compatible with GitHub.

## About This Database

- This is a lightweight placeholder that maintains compatibility with FarmaBot
- It's designed to be small enough to commit to GitHub (<5MB)
- It contains basic info for the most common medicines
- When using with Gemini models, no OpenAI API key is required

## Included Medicines

The database includes basic information for common medicines like:
- Acetaminophen
- Ibuprofen
- Aspirin
- Amoxicillin
- (and others defined in essential_medicines.json)

## Setup Instructions

To use this database with FarmaBot:

1. Ensure your MODEL in .env is set to a Gemini model
2. Update the following files to point to this directory:

In core/bot.py and services/medicine_service.py, change:
```python
persist_directory="medicines_vectordb",
```

To:
```python
persist_directory="medicines_vectordb_lite", 
```

## Full Database Generation

If you need the complete medicine database with actual embeddings:

1. Set your OpenAI API key in .env
2. Run: `python scripts/generate_vector_db.py`
