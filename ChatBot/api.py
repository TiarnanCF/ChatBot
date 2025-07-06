from fastapi import FastAPI
from pydantic import BaseModel
import ChatbotBuilder

chat_bot_builder = ChatbotBuilder.ChatBotBuilder()
chat_bot = chat_bot_builder.build_chatbot()
print(chat_bot.query("What is the tallest building in NYC?"))

class UserRequest(BaseModel):
    user_query: str

app = FastAPI()

@app.post("/")
def query(user_request: UserRequest):
    return chat_bot.query(user_request.user_query)
