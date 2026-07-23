import os
import uuid
import requests
import urllib.parse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# Tool Definitions (Token Optimized & Clean)
# ---------------------------------------------------------
@tool
def weather(city: str) -> str:
    """USE THIS TOOL EVERY TIME the user asks for the weather. It fetches live, real-time weather data for a given city."""
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
        if not data.get("data"): return "No flights found."
        
        flights_info = [f"Airline: {f.get('airline', {}).get('name')} | Flight: {f.get('flight', {}).get('iata')} | Status: {f.get('flight_status')}" for f in data["data"][:3]]
        
        book_link = f"\n- ✈️ **[Click Here to Book on Google Flights](https://www.google.com/travel/flights?q=flights+from+{departure_airport_code.upper()}+to+{arrival_airport_code.upper()})**"
        return "\n".join([f"- {f}" for f in flights_info]) + book_link
    except Exception as e:
        return f"Error fetching flights: {str(e)}"

@tool
def hotel_search(city: str) -> str:
    """Search for hotels near a given city. Args: city (str): EXACT city name."""
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key: return "Error: GEOAPIFY_API_KEY is not set."
    try:
        geo_url = f"https://api.geoapify.com/v1/geocode/search?text={city}&apiKey={api_key}"
        geo_res = requests.get(geo_url, timeout=10).json()
        lon, lat = geo_res["features"][0]["geometry"]["coordinates"]
        
        places_url = f"https://api.geoapify.com/v2/places?categories=accommodation.hotel&filter=circle:{lon},{lat},20000&bias=proximity:{lon},{lat}&limit=3&apiKey={api_key}"
        places_res = requests.get(places_url, timeout=10).json()
        
        hotels_info = []
        for h in places_res.get("features", []):
            name = h['properties'].get('name', 'Unknown')
            address = h['properties'].get('formatted', '')
            if name != 'Unknown':
                query = urllib.parse.quote(f"{name} {city}")
                map_link = f"https://www.google.com/maps/search/?api=1&query={query}"
                hotels_info.append(f"- [**{name}**]({map_link}) | Address: {address}")
                
        return "\n".join(hotels_info)
    except Exception as e:
        return f"Error fetching hotels: {str(e)}"

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert a given amount from one currency to another."""
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
        res = requests.get(url, timeout=10).json()
        rates = res.get('rates', {})
        rate = rates.get(to_currency.upper())
        if not rate: 
            return f"Sorry, currency {to_currency.upper()} is not supported."
        converted = round(amount * rate, 2)
        return f"- {amount} {from_currency.upper()} is equal to {converted} {to_currency.upper()}."
    except Exception as e:
        return f"Error converting currency: {str(e)}"

@tool
def restaurant_search(city: str) -> str:
    """Search for top restaurants and cafes near a given city."""
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key: return "Error: GEOAPIFY_API_KEY is not set."
    try:
        geo_url = f"https://api.geoapify.com/v1/geocode/search?text={city}&apiKey={api_key}"
        geo_res = requests.get(geo_url, timeout=10).json()
        lon, lat = geo_res["features"][0]["geometry"]["coordinates"]
        
        places_url = f"https://api.geoapify.com/v2/places?categories=catering.restaurant,catering.cafe&filter=circle:{lon},{lat},10000&bias=proximity:{lon},{lat}&limit=3&apiKey={api_key}"
        places_res = requests.get(places_url, timeout=10).json()
        
        rest_info = []
        for r in places_res.get("features", []):
            if 'name' in r['properties']:
                name = r['properties']['name']
                address = r['properties'].get('formatted', '')
                query = urllib.parse.quote(f"{name} {city}")
                map_link = f"https://www.google.com/maps/search/?api=1&query={query}"
                rest_info.append(f"- [**{name}**]({map_link}) | Address: {address}")
        
        if not rest_info: return "No restaurants found in this area."
        return "\n".join(rest_info)
    except Exception as e:
        return f"Error fetching restaurants: {str(e)}"

@tool
def distance_calculator(origin: str, destination: str) -> str:
    """Calculate the driving distance and travel time between two locations."""
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key: return "Error: GEOAPIFY_API_KEY is not set."
    try:
        orig_url = f"https://api.geoapify.com/v1/geocode/search?text={origin}&apiKey={api_key}"
        orig_res = requests.get(orig_url, timeout=10).json()
        orig_lon, orig_lat = orig_res["features"][0]["geometry"]["coordinates"]

        dest_url = f"https://api.geoapify.com/v1/geocode/search?text={destination}&apiKey={api_key}"
        dest_res = requests.get(dest_url, timeout=10).json()
        dest_lon, dest_lat = dest_res["features"][0]["geometry"]["coordinates"]

        route_url = f"https://api.geoapify.com/v1/routing?waypoints={orig_lat},{orig_lon}|{dest_lat},{dest_lon}&mode=drive&apiKey={api_key}"
        route_res = requests.get(route_url, timeout=10).json()

        distance_meters = route_res["features"][0]["properties"]["distance"]
        time_seconds = route_res["features"][0]["properties"]["time"]

        distance_km = round(distance_meters / 1000, 1)
        time_mins = round(time_seconds / 60)
        hours = time_mins // 60
        mins = time_mins % 60
        time_str = f"{hours} hr {mins} min" if hours > 0 else f"{mins} min"

        query = urllib.parse.quote(f"{origin} to {destination}")
        directions_link = f"https://www.google.com/maps/search/?api=1&query={query}"
        
        return f"- [**Distance from {origin} to {destination}**]({directions_link}) is {distance_km} km. Estimated driving time: {time_str}."
    except Exception as e:
        return f"Error calculating distance: {str(e)}"

# ---------------------------------------------------------
# FastAPI & Agent Setup
# ---------------------------------------------------------
app = FastAPI(title="AI Travel Agent API", version="1.0")

# CORS setup taaki GitHub Pages se requests allow ho sakein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me aap yahan apna GitHub Pages URL dal sakte hain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0)

tools = [weather, flight_search, hotel_search, currency_converter, restaurant_search, distance_calculator]

system_prompt = (
    "You are an Elite Travel Concierge.\n\n"
    "CRITICAL RULES:\n"
    "1. Always use tools to fetch real data.\n"
    "2. PAY EXPLICIT ATTENTION TO THE USER'S REQUESTED CITY/LOCATION. If the user asks for weather, hotels, or restaurants in a specific city (e.g., Dubai), you MUST pass that exact city name into the tool arguments. Never substitute it with another city.\n"
    "3. NEVER invent or hallucinate tools. ONLY use the exact tools provided to you (weather, flight_search, hotel_search, currency_converter, restaurant_search, distance_calculator).\n"
    "4. FORMATTING IS CRITICAL: For list results (Hotels, Restaurants, Flights), ALWAYS display each item on a NEW LINE with a bullet point (-).\n"
    "5. The tools will provide you with clickable markdown links (e.g. [Hotel Name](URL)). You MUST preserve these links in your final output exactly as provided so the user can click them.\n\n"
    "Headers to use ONLY when requested:\n"
    "### 🗺️ Travel Distance & Time\n"
    "### 🌤️ Weather Conditions\n"
    "### 🏨 Recommended Accommodations\n"
    "### 🍽️ Top Dining Spots\n"
    "### ✈️ Flight Details\n"
    "### 💱 Currency Conversion\n"
)

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

sessions: Dict[str, List[dict]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.get("/")
async def root():
    return {"status": "Backend is running successfully!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in sessions:
            sessions[session_id] = []
            
        sessions[session_id].append({"role": "user", "content": request.message})
        
        # Memory Management - Keep only last 4 messages to save tokens
        if len(sessions[session_id]) > 4:
            sessions[session_id] = sessions[session_id][-4:]
            
        result = agent.invoke({"messages": sessions[session_id]})
        sessions[session_id] = result["messages"]
        ai_reply = sessions[session_id][-1].content
        
        return ChatResponse(reply=ai_reply, session_id=session_id)
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"🔥 ERROR: {str(e)}")
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))
