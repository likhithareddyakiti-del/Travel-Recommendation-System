"""
src/live_data.py
----------------
REAL-TIME data: current weather from Open-Meteo (https://open-meteo.com).
Free for non-commercial use, no API key; data licence CC BY 4.0 (attribution required).
Used only to INFORM the user - it is not an ML feature.

There is no free, legitimate real-time price API for Indian hotels/trains/flights, so
costs stay as clearly-labelled estimates. Do not scrape booking sites (against their terms).
"""
import json
import urllib.parse
import urllib.request

API_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (subset)
WMO = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
       51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle", 61: "Light rain", 63: "Rain",
       65: "Heavy rain", 80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
       95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"}


class LiveDataError(Exception):
    """Live data could not be fetched or parsed."""


def parse_weather(payload):
    """Pure function: Open-Meteo JSON (dict) -> small dict. Raises LiveDataError if malformed."""
    try:
        cur = payload["current"]
        return {
            "time": cur["time"],
            "temperature_c": float(cur["temperature_2m"]),
            "humidity_pct": float(cur["relative_humidity_2m"]),
            "wind_kmh": float(cur["wind_speed_10m"]),
            "precipitation_mm": float(cur.get("precipitation", 0.0)),
            "description": WMO.get(int(cur["weather_code"]), f"WMO code {cur['weather_code']}"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveDataError(f"Unexpected weather response: {exc}") from exc


def get_current_weather(lat, lon, timeout=10, opener=urllib.request.urlopen):
    """Fetch live weather for a point. `opener` is injectable so tests need no internet."""
    query = urllib.parse.urlencode({
        "latitude": round(lat, 4), "longitude": round(lon, 4), "timezone": "Asia/Kolkata",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code"})
    try:
        with opener(f"{API_URL}?{query}", timeout=timeout) as resp:
            return parse_weather(json.loads(resp.read().decode("utf-8")))
    except LiveDataError:
        raise
    except Exception as exc:
        raise LiveDataError(f"Could not fetch live weather (offline or service busy): {exc}") from exc
