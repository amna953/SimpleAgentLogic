from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
import asyncio
import AgentLogic
import os
import dotenv

app = FastAPI()
sessions = {}

class Chat(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    model: str = Field(min_length=5, max_length=25)

@app.post("/api/v1/chat")
async def chat_endpoint(data: Chat, user_id: str = "admin"):
    if user_id not in sessions:
        sessions[user_id] = AgentLogic.OpenAI_Agent(model=data.model)
    
    bot = sessions[user_id]
    response = await bot.query(data.text)

    return {"status": "ok", "choices": [
        {
            "content": response,
            "model": data.model
        }
    ]}