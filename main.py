import os
import uuid
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------
@tool
def weather(city: str) -> str:
    """USE THIS TOOL EVERY TIME the user asks for the weather. It fetches live, real-time weather data for a given city. Args: city (str): The name of the city."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key: return "Error: OPENWEATHERMAP_API_KEY is not set."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        return f"City: {data.get('name')}\nTemp: {data.get('main', {}).get('temp')}°C\nWeather: {data.get('weather', [{}])[0].get('description')}\nHumidity: {data.get('main', {}).get('humidity')}%"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def flight_search(departure_airport_code: str, arrival_airport_code: str) -> str:
    """Search for flights between two airports using their 3-letter IATA codes."""
    api_key = os.getenv("AVIATIONSTACK_API_KEY")
    if not api_key: return "Error: AVIATIONSTACK_API_KEY is not set."
    try:
        url = f"http://api.aviationstack.com/v1/flights?access_key={api_key}&dep_iata={departure_airport_code.upper()}&arr_iata={arrival_airport_code.upper()}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if not data.get("data"): return f"No flights found."
        flights_info = [f"Airline: {f.get('airline', {}).get('name')} | Flight: {f.get('flight', {}).get('iata')} | Status: {f.get('flight_status')}" for f in data["data"][:5]]
        return "\n".join(flights_info)
    except Exception as e:
        return f"Error fetching flights: {str(e)}"

@tool
def hotel_search(city: str) -> str:
    """Search for hotels near a given city. Args: city (str): The EXACT name of the city. CRITICAL: NEVER pass pronouns like 'there', 'here', or 'it'."""
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key: return "Error: GEOAPIFY_API_KEY is not set."
    try:
        geo_url = f"https://api.geoapify.com/v1/geocode/search?text={city}&apiKey={api_key}"
        geo_res = requests.get(geo_url, timeout=10).json()
        lon, lat = geo_res["features"][0]["geometry"]["coordinates"]
        
        places_url = f"https://api.geoapify.com/v2/places?categories=accommodation.hotel&filter=circle:{lon},{lat},20000&bias=proximity:{lon},{lat}&limit=5&apiKey={api_key}"
        places_res = requests.get(places_url, timeout=10).json()
        
        hotels_info = [f"Name: {h['properties'].get('name', 'Unknown')} | Address: {h['properties'].get('formatted', '')}" for h in places_res.get("features", [])]
        return "\n".join(hotels_info)
    except Exception as e:
        return f"Error fetching hotels: {str(e)}"

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert a given amount from one currency to another."""
    try:
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency.upper()}&to={to_currency.upper()}"
        res = requests.get(url, timeout=10).json()
        return f"{amount} {from_currency.upper()} is equal to {res['rates'][to_currency.upper()]} {to_currency.upper()}."
    except Exception as e:
        return f"Error converting currency: {str(e)}"

# ---------------------------------------------------------
# FastAPI & Agent Setup
# ---------------------------------------------------------
app = FastAPI(title="AI Travel Agent API", version="1.0")

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder for CSS/JS/Assets (Serves your HTML Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✨ SHIFTED TO LLAMA 3.1 8B INSTANT ✨
llm = init_chat_model(model="llama-3.1-8b-instant", model_provider="groq", temperature=0)

tools = [weather, flight_search, hotel_search, currency_converter]
system_prompt = (
    "You are an Elite Travel & Logistics Concierge assisting a high-end production crew.\n\n"
    "CRITICAL RULES:\n"
    "1. Always use your tools to fetch live data.\n"
    "2. CONTEXT: If a user says 'there' or 'here', check the current message for a city. If none exists, resolve it from the chat history.\n"
    "3. TONE: Professional, sophisticated, and highly efficient.\n"
    "4. TOOL RESTRICTION (STRICT): You ONLY have access to the provided tools. NEVER use or assume unlisted tools like 'brave_search'.\n"
    "5. ONLY ANSWER WHAT IS SPECIFICALLY ASKED. Do not provide hotels, flights, or currency conversions unless the user explicitly requested them in their current prompt.\n\n"
    "FORMATTING RULES:\n"
    "Use elegant Markdown. ONLY include the following headers IF the user specifically asked for that information:\n"
    "- If weather is requested: ### 🌤️ Atmosphere & Conditions\n"
    "- If hotels are requested: ### 🏨 Recommended Accommodations\n"
    "- If flights are requested: ### ✈️ Flight Logistics\n"
    "- If currency is requested: ### 💱 Financial Conversion\n\n"
    "CRITICAL: Do NOT output empty headers or generate fake data for tools that were not requested."
)
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)


# In-memory dictionary to store conversational history per session
sessions: Dict[str, List[dict]] = {}

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend on the root URL."""
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat messages from the frontend and return AI responses."""
    try:
        # Generate a new session ID if the user didn't provide one
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize empty history for new sessions
        if session_id not in sessions:
            sessions[session_id] = []
            
        # Append new user message
        sessions[session_id].append({"role": "user", "content": request.message})
        
        # Run agent
        result = agent.invoke({"messages": sessions[session_id]})
        
        # Update memory state
        sessions[session_id] = result["messages"]
        
        # Extract AI response
        ai_reply = sessions[session_id][-1].content
        
        return ChatResponse(reply=ai_reply, session_id=session_id)
        
    except Exception as e:
        # Debug print block to see true origin of 500 errors in terminal
        print("\n" + "="*50)
        print(f"🔥 ASLI ERROR YAHAN HAI: {str(e)}")
        print("="*50 + "\n")
        
        raise HTTPException(status_code=500, detail=str(e))
