import openai  # Requires `openai` Python package (!!!!)
import numpy as np

def summarize_intent_from_confidences(centroids, confidences, labels, model="gpt-4", top_n=3):
    """
    Generate a natural language summary of the robot's intent given centroids, confidences, and labels.
    
    Args:
        centroids (list of tuples): List of (x, y, z) positions for each object.
        confidences (list of floats): List of confidence scores for each object.
        labels (list of str): Human-readable labels for each object.
        model (str): OpenAI model to use (default: "gpt-4").
        top_n (int): How many top-confidence objects to include in summary.

    Returns:
        str: LLM-generated summary
    """

    # Sort by confidence
    ranked = sorted(zip(confidences, centroids, labels), key=lambda x: x[0], reverse=True)[:top_n]

    # Format input
    description_lines = []
    for i, (conf, (x, y, z), label) in enumerate(ranked):
        description_lines.append(f"{i+1}. Object: {label}, Confidence: {conf:.2f}, Location: ({x:.2f}, {y:.2f}, {z:.2f})")

    prompt = f"""
Given the following robot confidence values for object goals, produce a one-paragraph summary of what the robot intends to do, based on which object(s) it is most likely trying to reach. Mention object names and relative likelihood.

Object confidence data:
{chr(10).join(description_lines)}
"""

    # Call the OpenAI API (requires `OPENAI_API_KEY` to be set in your environment)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful robotics summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=100
    )

    return response["choices"][0]["message"]["content"].strip()
