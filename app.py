
from fastapi import FastAPI, status, HTTPException
from dotenv import load_dotenv
from datetime import datetime, timezone
import uuid

from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIError
)
from schemas import MessageCreate

import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI()

SIMILARITY_THRESHOLD = 0.30


gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

messages = []

@app.post("/messages", status_code=status.HTTP_201_CREATED)
def create_message(request: MessageCreate):

    new_message = {
        "id": str(uuid.uuid4()),
        "customer": request.customer,
        "message": request.message,
        "created_at": datetime.now(timezone.utc).strftime("%d/%m/%Y")
    }

    messages.append(new_message)

    return new_message

@app.get("/messages", status_code=status.HTTP_200_OK)
def get_messages():

    return messages


@app.post("/messages/{id}/analyze")
def analyze_message(id: str):

    result = None

    for message in messages:
        if message["id"] == id:
            result = message
            break

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Message not found"
        )

    prompt = f"""
    You are a customer support message analyzer.

    Analyze the following customer message and return
    ONLY a valid JSON object with these four fields:

    1. summary: A short summary of the issue.
    2. sentiment: positive, neutral, or negative.
    3. intent: The customer's main purpose.
    4. priority: low, medium, or high.

    All four fields are mandatory.

    Customer message:
    {result["message"]}
    """

    try:
        response = gemini_client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        analysis = json.loads(
            response.choices[0].message.content
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Invalid JSON response from Gemini"
        )

    except AuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="Gemini authentication failed"
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out"
        )

    except APIError:
        raise HTTPException(
            status_code=502,
            detail="Gemini API failed"
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

    return {
        "message_id": id,
        "analysis": analysis
    }

def retrieve_similar_messages(current_message_id: str, top_k: int = 2):

    historical_messages = [
        message for message in messages
        if message["id"] != current_message_id
    ]

    if not historical_messages:
        return []

    valid_messages = [
        (message, message["message"])
        for message in historical_messages
        if message["message"].strip()
    ]

    if not valid_messages:
        return []

    historical_messages = [
        item[0] for item in valid_messages
    ]

    documents = [
        item[1] for item in valid_messages
    ]

    try:
        vectorizer = TfidfVectorizer()

        vectorized_documents = vectorizer.fit_transform(documents)

        current_message = next(
            message["message"]
            for message in messages
            if message["id"] == current_message_id
        )

        vectorized_current_message = vectorizer.transform(
            [current_message]
        )

        similarity_scores = cosine_similarity(
            vectorized_current_message,
            vectorized_documents
        )[0]

        print("\nSimilarity scores:")

        for index, score in enumerate(similarity_scores):
            print(
                historical_messages[index]["customer"],
                round(float(score), 4)
            )

    except ValueError:
        return []

    ranked_indices = similarity_scores.argsort()[::-1]

    similar_messages = []

    for index in ranked_indices:

        if similarity_scores[index] < SIMILARITY_THRESHOLD:
            continue

        similar_messages.append({
            "id": historical_messages[index]["id"],
            "customer": historical_messages[index]["customer"],
            "message": historical_messages[index]["message"],
            "created_at": historical_messages[index]["created_at"]
        })

        if len(similar_messages) == top_k:
            break

    return similar_messages

@app.post("/messages/{id}/recommend")
def recommend_solution(id: str):

    current_message = None

    for message in messages:

        if message["id"] == id:
            current_message = message
            break

    if current_message is None:
        raise HTTPException(
            status_code=404,
            detail="Message not found"
        )

    similar_messages = retrieve_similar_messages(id)

    prompt = f"""
    You are an expert customer support recommendation assistant.

    Analyze the current customer issue and the historical
    similar messages provided below.

    Current customer message:
    {current_message["message"]}

    Similar historical messages:
    {json.dumps(similar_messages, indent=2)}

    Your task:

    1. Understand the current customer's problem.
    2. Examine the historical similar messages.
    3. Identify any useful solutions or patterns.
    4. Recommend the most appropriate action to resolve
       the current customer's problem.

    IMPORTANT:

    - Use ONLY the current message and the provided
      historical messages.
    - Do not invent historical cases or solutions.
    - If no similar historical messages are available,
      provide a recommendation based only on the
      current customer message.
    - Keep the recommendation practical and clear.

    Return your recommendation as plain text.
    """

    try:

        response = gemini_client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        recommendation = response.choices[0].message.content

    except AuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="Gemini authentication failed"
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out"
        )

    except APIError:
        raise HTTPException(
            status_code=502,
            detail="Gemini API failed"
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

    return {
        "message_id": id,
        "similar_messages": similar_messages,
        "recommendation": recommendation
    }