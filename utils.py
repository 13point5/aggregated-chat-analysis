from pydantic import BaseModel, Field
from typing import List
import numpy as np


class CriteraFeedback(BaseModel):
    strengths: List[str] = Field(
        ..., description="Concise list of strengths for this criteria"
    )
    weaknesses: List[str] = Field(
        ..., description="Concise list of weaknesses for this criteria"
    )
    suggestions: List[str] = Field(
        ..., description="Concise list of suggestions for improvement this criteria"
    )


class Evaluation(BaseModel):
    introduction: CriteraFeedback = Field(
        ...,
        description="Analysis of the introduction of the essay with these criteria: 1. Clarity of thesis statement, 2. Engagement and relevance of opening statements",
    )
    structure: CriteraFeedback = Field(
        ...,
        description="Analasis of the structure of the essay's body with these criteria: 1. Organization and clarity of paragraphs, 2. Logical flow of ideas",
    )
    argumentation: CriteraFeedback = Field(
        ...,
        description="Analysis of the argumentation of the essay with these criteria: 1. Strength and clarity of arguments, 2. Use of critical reasoning",
    )
    evidence: CriteraFeedback = Field(
        ...,
        description="Analysis of the evidence used in the essay with these criteria: 1. Relevance and quality of evidence, 2. Use of citations and references",
    )
    conclusion: CriteraFeedback = Field(
        ...,
        description="Analysis of the conclusion of the essay with these criteria: 1. Restatement of thesis, 2. Summary of main points, 3. Closing statements",
    )


def get_structured_feedback(instructor_client, chat_history) -> List[Evaluation]:
    result = []

    for msg in chat_history:
        if msg["role"] == "user":
            continue

        result.append(
            instructor_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_model=Evaluation,
                messages=[
                    {
                        "role": "system",
                        "content": "You are analysing the feedback provided to an 8th grade student on their essay. Extract concise feedback on each criteria. If there are no strengths, weaknesses or suggestions for a criteria then leave it empty",
                    },
                    {"role": "user", "content": chat_history[1]["content"]},
                ],
            )
        )

    return result


def get_embeddings(openai_client, texts: List[str]):
    result = openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    embeddings = [e.embedding for e in result.data]

    return embeddings


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def cosine_similarity_matrix(embeddings):
    # Normalize the embeddings to unit length
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Compute the cosine similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    return similarity_matrix


def dynamic_threshold(similarity_matrix):
    # Exclude the diagonal elements and flatten the array
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    # Calculate a dynamic threshold: e.g., one standard deviation above the mean
    mean_similarity = np.mean(similarities)
    std_dev = np.std(similarities)
    threshold = mean_similarity + std_dev

    return threshold


def remove_duplicates(openai_client, strings: List[str]) -> List[str]:
    strings = [s.strip() for s in strings if s.strip() != "None"]

    n = len(strings)
    keep = [True] * n

    if n == 0:
        return []

    embeddings = get_embeddings(openai_client, strings)
    similarity_matrix = cosine_similarity_matrix(embeddings)
    threshold = dynamic_threshold(similarity_matrix)

    # Compute pairwise cosine similarity and filter duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if keep[j] and cosine_similarity(embeddings[i], embeddings[j]) > threshold:
                keep[j] = False

    # Filter out duplicates
    unique_strings = [strings[i] for i in range(n) if keep[i]]
    return unique_strings
