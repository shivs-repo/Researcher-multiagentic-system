import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI

load_dotenv()

# ── LLM (OpenRouter + Gemma) ──────────────────────────────────────────────────
def get_llm():
    return ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
    )

# ── Supervisor ────────────────────────────────────────────────────────────────
def supervisor_agent(state: dict) -> dict:
    llm = get_llm()
    prompt = (
        "You are a supervisor that routes user queries to the correct agent.\n"
        "Your ONLY job is to output exactly one word from the options below. No explanation, no punctuation.\n\n"
        "Options:\n"
        "  weather    - user wants current weather, temperature, forecast, humidity, or what to wear/carry for a city\n"
        "  web_search - user wants recent news, facts, people, events, prices, or anything needing live internet data\n"
        "  general    - user asks a general knowledge question, concept, advice, or anything answerable without live data\n\n"
        f"Query: {state['query']}\n\n"
        "Output (one word only):"
    )
    response = llm.invoke(prompt)
    route = response.content.strip().lower().split()[0].replace("'", "").replace('"', "")
    if route not in ("weather", "web_search", "general"):
        route = "general"
    print(f"\n[SUPERVISOR] Query : {state['query']}")
    print(f"[SUPERVISOR] Route : {route}\n")
    return {**state, "route": route, "source": route, "raw_result": ""}

# ── Web Search Agent ──────────────────────────────────────────────────────────
def web_search_agent(state: dict) -> dict:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=state["query"], max_results=3)
    snippets = "\n".join(
        f"- {r['title']}: {r['content'][:200]}" for r in results.get("results", [])
    )
    return {**state, "raw_result": snippets, "source": "web_search"}

# ── Weather Agent ─────────────────────────────────────────────────────────────
def extract_city(query: str) -> str:
    """Use LLM to extract city name from any natural language query."""
    llm = get_llm()
    response = llm.invoke(
        f"Extract only the city name from this query. Reply with just the city name, nothing else.\nQuery: {query}"
    )
    city = response.content.strip().rstrip("?!.,")
    print(f"  LLM extracted city: '{city}'")
    return city


def weather_agent(state: dict) -> dict:
    query = state["query"]
    city = extract_city(query)
    api_key = os.getenv("OPENWEATHER_API_KEY")

    print("\n[WEATHER AGENT DEBUG]")
    print(f"  Query          : {query}")
    print(f"  Extracted city : {city}")
    print(f"  API Key present: {bool(api_key)} | preview: {api_key[:6] if api_key else 'MISSING'}...")

    # Step 1: Geocoding to get lat/lon (handles alternate spellings)
    geo_url = "http://api.openweathermap.org/geo/1.0/direct"
    geo_resp = requests.get(geo_url, params={"q": city, "limit": 1, "appid": api_key})
    print(f"  Geocode status : {geo_resp.status_code}")
    print(f"  Geocode response: {geo_resp.text}")

    if geo_resp.status_code == 200 and geo_resp.json():
        geo = geo_resp.json()[0]
        lat, lon = geo["lat"], geo["lon"]
        resolved_name = f"{geo['name']}, {geo.get('state', '')}, {geo['country']}".strip(", ")
        print(f"  Resolved to    : {resolved_name} (lat={lat}, lon={lon})")

        # Step 2: Fetch weather by lat/lon
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        resp = requests.get(weather_url, params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})
        print(f"  Weather status : {resp.status_code}")
        print(f"  Weather response: {resp.text}")

        if resp.status_code == 200:
            data = resp.json()
            raw = (
                f"City: {data['name']}, {data['sys']['country']}\n"
                f"Condition: {data['weather'][0]['description']}\n"
                f"Temperature: {data['main']['temp']}°C (feels like {data['main']['feels_like']}°C)\n"
                f"Humidity: {data['main']['humidity']}%\n"
                f"Wind: {data['wind']['speed']} m/s"
            )
            print(f"  Parsed Result:\n{raw}")
        else:
            raw = f"Weather fetch failed for '{resolved_name}'. Status: {resp.status_code} | {resp.text}"
            print(f"  ERROR: {raw}")
    else:
        raw = f"City '{city}' not found via geocoding. Status: {geo_resp.status_code} | {geo_resp.text}"
        print(f"  ERROR: {raw}")

    print("[WEATHER AGENT DEBUG END]\n")
    return {**state, "raw_result": raw, "source": "weather"}

# ── Result Formatter ──────────────────────────────────────────────────────────
def result_formatter(state: dict) -> dict:
    llm = get_llm()
    source = state.get("source", "general")
    raw = state.get("raw_result", "")

    if source == "general" or not raw:
        prompt = (
            "You are a helpful assistant. Answer the user's question directly and clearly.\n"
            "Be concise. Use bullet points only if listing multiple items. No unnecessary filler.\n\n"
            f"Question: {state['query']}\n\n"
            "Answer:"
        )
    else:
        prompt = (
            "You are a helpful assistant. Using ONLY the data provided below, answer the user's question.\n"
            "Be concise and direct. Do not add information not present in the data.\n"
            "Use bullet points only if listing multiple items.\n\n"
            f"User question: {state['query']}\n\n"
            f"Data ({source}):\n{raw}\n\n"
            "Answer:"
        )
    response = llm.invoke(prompt)
    return {**state, "final_answer": response.content}
