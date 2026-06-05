import ollama
from pydantic import BaseModel
from typing import Literal, List, Optional

class IntentClassification(BaseModel):
    intent: Literal["play", "launch", "none"]
    confidence_score: float
    extracted_keywords: List[str]


user_prompt = "Hey please open the diagnostics app"

print("Analyzing intent... (this will be fast)\n")
response = ollama.chat(
    model='llama3.2',
    messages=[
        {
            'role': 'system',
            'content': 'You are a data extraction assistant. Analyze the input and strictly follow the requested JSON format.'
        },
        {
            'role': 'user',
            'content': user_prompt
        }
    ],
    format=IntentClassification.model_json_schema(), # This enforces the schema natively
    options={'temperature': 0.0} # 0.0 maximizes structural adherence and prevents hallucinations
)


result = IntentClassification.model_validate_json(response.message.content)


print(f"Detected Intent: {result.intent}")
print(f"Keywords:        {result.extracted_keywords}")