# Customer Support Message Analysis API

A FastAPI-based application that manages customer support messages, analyses customer queries using an LLM, retrieves similar historical messages, and generates recommendations.

## Features

- Create and retrieve customer support messages.
- Generate unique IDs and timestamps for messages.
- Analyse customer messages using an LLM.
- Identify sentiment, intent, priority, and summary.
- Retrieve the top 2 similar historical messages using TF-IDF and cosine similarity.
- Generate recommendations using the current message and retrieved historical context.
- Handle invalid requests, missing messages, and API errors.

## Tech Stack

- Python
- FastAPI
- Pydantic
- Google Gemini API
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Uvicorn

## Project Structure

```text
project/
│
├── main.py
├── schemas.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

**Mac/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

## Run the Application

```bash
uvicorn main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/messages` | Create a customer message |
| GET | `/messages` | Retrieve all messages |
| POST | `/messages/{id}/analyze` | Analyse a customer message |
| POST | `/messages/{id}/recommend` | Retrieve similar messages and generate recommendations |

## Similarity Retrieval

The application uses TF-IDF to convert historical customer messages into numerical vectors and cosine similarity to compare them with the current message.

A similarity threshold of 0.30 is used to filter retrieved messages. The system returns up to two matching historical messages.

## Notes

- Messages are currently stored in memory and are cleared when the application restarts.
- Similarity retrieval uses lexical matching, so unrelated messages can occasionally be retrieved.
- LLM functionality requires a valid Gemini API key.