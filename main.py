from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import os
from dotenv import load_dotenv
import chainlit as cl
import csv
from typing import Dict, Optional

load_dotenv()  # Load environment variables from .env file

secret_key = os.getenv("CHAINLIT_AUTH_SECRET")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    with open('users.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['username'] == username and row['password'] == password:
                return cl.User(identifier=username, metadata={"role": row['role'], "provider": "database"})
    return None
    
@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="Sherpa")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a very knowledgeable AI Model which provides accurate and eloquent answers to all questions."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    runnable = lambda chat_history: prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    chat_history = cl.user_session.get("chat_history", [])  # Get chat history

    msg = cl.Message(content="")

    async for chunk in runnable(chat_history).astream(
        {"question": message.content, "history": chat_history},  # Pass chat history
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    # Update chat history in user session
    chat_history.append({"role": "user", "content": message.content})
    cl.user_session.set("chat_history", chat_history)
