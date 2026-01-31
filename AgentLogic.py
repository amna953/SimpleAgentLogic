from openai import AsyncOpenAI, pydantic_function_tool
from tavily import AsyncTavilyClient
from pydantic import BaseModel, Field
import httpx
import os
import dotenv
import json
import uuid

dotenv.load_dotenv()

class Search_tool(BaseModel):
    """Инструмент для поиска актуальной информации в интернете."""
    query_text: str = Field(description='Запрос в поисковую систему.', min_length=3, max_length=100)
    results_amount: int = Field(default=3, description="Количество результатов.", ge=1, le=10)

class OpenAI_Agent:
    def __init__(self, model: str ="x-ai/grok-4.1-fast",
                prompt: str = f"Ты ИИ-помощник на платформе FluxAI.",
                endpoint: str ="https://openrouter.ai/api/v1"):
        self.model = model
        self.prompt = prompt
        self.endpoint = endpoint
        self._key = os.environ.get("OR_KEY")
        self._tkey = os.environ.get("TAVILY")
        self.history = [{"role": "system", "content": self.prompt}]
        self.client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self._key
        )
        self.tclient = AsyncTavilyClient(
            api_key=self._tkey
        )
        self.tools = [pydantic_function_tool(Search_tool)]
    async def check_history(self):
        if len(self.history) > 20:
            self.history = [{"role": "system", "content": self.prompt}] + self.history[-20:]
    async def search_tool(self, query_text: str, results_amount: int = 3):
        resp = await self.tclient.search(
            query=query_text,
            max_results=results_amount
        )

        final = []

        for i in resp['results']: 
            final.append({"SOURCE": i.get('url'), "CONTENT": i.get('content')})
        
        final.append(f'Выше предоставлены результаты поиска в интернете по твоему запросу {query_text}.')

        return json.dumps(final, ensure_ascii=False)
    async def query(self, text: str):
        self.history.append({"role": "user", "content": text})
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=self.tools,
            tool_choice='auto'
        )

        msg = resp.choices[0].message

        if msg.tool_calls:
            self.history.append(msg)
            for call in msg.tool_calls:
                if call.function.name == "Search_tool":
                    args = Search_tool.model_validate_json(call.function.arguments)

                    call_res = await self.search_tool(args.query_text, args.results_amount)

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": call_res
                    })

                    sresp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=self.history
                    )
                    final = sresp.choices[0].message.content
                    self.history.append({"role": "assistant", "content": final})

                    await self.check_history()

                    return final
        else:
            final = msg.content
            self.history.append({"role": "assistant", "content": final})

            await self.check_history()

            return final