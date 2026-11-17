"""TEST FIXTURES ONLY: the rows below are invented to test the code; they are NOT real data."""
import io
import json
import os
import unittest

import pandas as pd

from import_real_data import build_destinations, find_columns, map_interests
from src.data_preprocessing import load_destinations
from src.live_data import LiveDataError, get_current_weather, parse_weather

HEADER = ("Zone,State,City,Name,Type,Establishment Year,time needed to visit in hrs,"
          "Google review rating,Entrance Fee in INR,Airport with 50km Radius,Weekly Off,"
          "Significance,DSLR Allowed,Number of google review in lakhs,Best Time to visit")
ROWS = [  # Zone,State,City,Name,Type,Year,hrs,rating,fee,airport,off,significance,dslr,reviews,best
    "South,Fixtureland,Testville,Fake Fort,Fort,1500,2,4.4,50,Yes,None,Historical,Yes,1.2,Morning",
    "South,Fixtureland,Testville,Fake Beach,Beach,0,3,4.0,0,Yes,None,Nature,Yes,0.5,Evening",
    "South,Fixtureland,Hillpur,Fake Peak,Hill Station,0,5,4.6,0,No,None,Nature,Yes,0.8,Morning",
    "North,Mysteria,Nowhere,Fake Thing,Gizmo,0,1,4.5,0,No,None,Other,Yes,0.1,Morning",
    "North,Mysteria,Faraway,Fake Park,Park,0,2,4.2,10,No,None,Nature,Yes,0.3,Morning",
]
RAW = pd.read_csv(io.StringIO("\n".join([HEADER] + ROWS)))
COORDS = {"Testville, Fixtureland, India": (13.0, 80.0), "Hillpur, Fixtureland, India": (11.4, 76.7),
          "Faraway, Mysteria, India": (51.5, -0.1)}          # London -> must be rejected


def fake_geocode(q):
    return COORDS[q]


class TestImport(unittest.TestCase):
    def test_interest_mapping(self):
        self.assertEqual(map_interests("Hill Station", "Nature"), {"hills", "nature"})
        self.assertEqual(map_interests("Gizmo", "Other"), set())

    def test_columns_found_despite_header_spelling(self):
        c = find_columns(RAW)
        self.assertEqual((c["state"], c["rating"], c["fee"]), ("State", "Google review rating", "Entrance Fee in INR"))

    def test_build_aggregates_and_validates(self):
        out, unmapped, bad = build_destinations(RAW, fake_geocode, origin=(13.08, 80.27))
        self.assertEqual(sorted(out["destination_name"]), ["Hillpur", "Testville"])
        self.assertIn("Gizmo", unmapped)                          # unmapped type reported
        self.assertTrue(any("Faraway" in b for b in bad))         # outside-India geocode rejected
        t = out[out["destination_name"] == "Testville"].iloc[0]
        self.assertEqual(t["n_attractions"], 2)
        self.assertAlmostEqual(t["rating"], 4.2)                  # mean(4.4, 4.0)
        self.assertEqual(set(t["interests"].split("|")), {"history", "beaches", "nature"})
        self.assertEqual(t["recommended_days"], 1)                # ceil(5h / 6)
        self.assertTrue((out["cost_source"] == "ASSUMPTION_NOT_VERIFIED").all())

    def test_output_loads_in_pipeline(self):
        out, _, _ = build_destinations(RAW, fake_geocode, origin=(13.08, 80.27))
        path = "outputs/_test_destinations.csv"
        out.to_csv(path, index=False)
        try:
            df = load_destinations(path)                          # the project's own validator
            self.assertEqual(len(df), 2)
        finally:
            os.remove(path)


CANNED = {"current": {"time": "2026-10-30T12:00", "temperature_2m": 31.2, "relative_humidity_2m": 70,
                      "wind_speed_10m": 12.5, "precipitation": 0.0, "weather_code": 2}}


class _FakeResponse:
    def __init__(self, data): self.data = data
    def read(self): return json.dumps(self.data).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


class TestLiveWeather(unittest.TestCase):
    def test_parse(self):
        w = parse_weather(CANNED)
        self.assertEqual((w["temperature_c"], w["description"]), (31.2, "Partly cloudy"))

    def test_malformed_response(self):
        with self.assertRaises(LiveDataError):
            parse_weather({"oops": 1})

    def test_fetch_with_fake_network(self):
        seen = []

        def fake_open(url, timeout):
            seen.append(url)
            return _FakeResponse(CANNED)

        w = get_current_weather(13.08, 80.27, opener=fake_open)
        self.assertIn("latitude=13.08", seen[0])
        self.assertIn("longitude=80.27", seen[0])
        self.assertEqual(w["wind_kmh"], 12.5)

    def test_network_failure_gives_clear_error(self):
        def boom(url, timeout): raise OSError("no internet")
        with self.assertRaises(LiveDataError):
            get_current_weather(13.08, 80.27, opener=boom)


if __name__ == "__main__":
    unittest.main()
