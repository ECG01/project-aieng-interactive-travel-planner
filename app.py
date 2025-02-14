from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from bot import SimplePRTravelBot
from initialize import initialize_components
from embeddings import E5Embeddings

# Initialize FastAPI app
app = FastAPI()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize bot components
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = E5Embeddings()
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="content")
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Create bot instance
bot = None

@app.on_event("startup")
async def startup_event():
    global bot
    location_chain = await initialize_components(llm, retriever)
    bot = SimplePRTravelBot(llm, retriever, index, location_chain)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Send welcome message
    welcome = """
    ¡Hola! 😊 I'm your Puerto Rico Travel Assistant.
        
        I'll be helping you plan your trip to Puerto Rico. Let'get started!
        When are you planning to visit our beautiful island🏝️? 
    """
    await websocket.send_text(welcome)
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            # Process message through bot
            response = await bot._process_input(message)
            
            # Send response back to client
            await websocket.send_text(response)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close() 