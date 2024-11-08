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
async def start():
    # await cl.Avatar(
    #     name="You",
    #     path="public/favicon.png",
    # ).send()
    model = Ollama(model="sherpa")
    
    # Selecting system prompt based on user session and chat profile
    user = cl.user_session.get("user")
    chat_profile = cl.user_session.get("chat_profile")
    
    if user.metadata.get("role") == "admin":
        if chat_profile == "Sherpa AI":
            system_prompt = "You are the Sherpa AI assistant."
        elif chat_profile == "SyntaxSherpa":
            system_prompt = "You are SyntaxSherpa an AI coding copilot model."
        else:
           
            system_prompt = "You are the Sherpa AI assistant."
    else:
        system_prompt = "You are the Sherpa AI assistant."
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    runnable = lambda chat_history: prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    
    # Set the chat profile based on user role
    if user.metadata.get("role") == "admin":
        cl.user_session.set("chat_profile", "SyntaxSherpa")
    else:
        cl.user_session.set("chat_profile", "Sherpa AI")


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

# Define chat profiles
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Sherpa AI",
            markdown_description="AI assistant",
            # icon="https://example.com/sherpa_ai_icon.png",
        ),
        cl.ChatProfile(
            name="SyntaxSherpa",
            markdown_description="Coding Copilot",
            #icon="https://example.com/syntax_sherpa_icon.png",
        ),
    ]
