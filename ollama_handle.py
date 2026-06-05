import ollama
from pydantic import BaseModel
from typing import Literal, List, Optional



class IntentClassification(BaseModel):
    intent: Literal["play", "launch", "none"]
    extracted_keywords: List[str]


user_prompt = "I wanna listen to Bismark by sabbaton"


response = ollama.chat(
    model='qwen2.5:0.5b',
    messages=[
        {
            'role': 'system',
            'content': 'You are a data extraction assistant. Reply exactly in the requested JSON format.'
        },
        {
            'role': 'user',
            'content': user_prompt
        }
    ],
    format=IntentClassification.model_json_schema(),
    options={
        'temperature': 0.0,


        'num_ctx': 1024,


        'num_predict': 100,
    }
)

result = IntentClassification.model_validate_json(response.message.content)
print(f"Intent: {result.intent} | Key Words: {result.extracted_keywords}")