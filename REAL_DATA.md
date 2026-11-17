# Using real Indian data (and live data) in this project

## 1. What is real, what is not

| Item | Status | Source |
|---|---|---|
| Destination names, state, city, **Google review ratings**, **entrance fees**, **visit hours** | **Real** (crowd-sourced Google ratings, as published in the Kaggle file) | Kaggle: https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places (325 attractions; check the dataset page for its licence/terms) |
| Latitude / longitude | **Real map data**, looked up automatically | OpenStreetMap Nominatim (c) OpenStreetMap contributors, ODbL. **Spot-check a few**: a city name can match the wrong place |
| Road network and road distance | **Real** | OpenStreetMap via OSMnx |
| Current weather | **Real-time** | Open-Meteo.com, CC BY 4.0, free for non-commercial use, no API key |
| Interest categories (beaches/history/...) | **Derived by my keyword mapping** from the file's Type/Significance text | `TYPE_TO_INTEREST` in `import_real_data.py` |
| `travel_cost`, `daily_cost`, `recommended_days` | **Assumptions, NOT verified prices** (flagged `ASSUMPTION_NOT_VERIFIED` in the CSV) | formulas in `import_real_data.py` |
| ML training labels (`suitable`) | **Still synthetic** (see section 5) | `generate_sample_data.py` |
| Live hotel / train / bus / flight prices | **Not included.** No free legitimate public API found; scraping booking sites breaks their terms | - |

I could not download the Kaggle file from my sandbox (no internet), so the importer was tested only on invented fixture rows. Its column detection is keyword-based because I could not inspect the real header spelling; if it fails it prints the columns it found.

## 2. Implement it, step by step

```bash
# a) get the real file (needs a free Kaggle account)
#    Option 1: open the Kaggle link above > Download > unzip > put the CSV at data/raw/Top Indian Places to Visit.csv
#    Option 2: pip install kaggle ; put your API token in ~/.kaggle/kaggle.json ; then
kaggle datasets download -d saketk511/travel-dataset-guide-to-indias-must-see-places -p data/raw --unzip

# b) convert it to our destinations.csv (internet needed; ~1 request/second, cached in data/geocode_cache.csv,
#    so ~200-300 cities take several minutes the first time)
python import_real_data.py --input "data/raw/Top Indian Places to Visit.csv" --origin "Chennai, India"

# c) read the printed report:
#    - "Types with no interest mapping"  -> add keywords to TYPE_TO_INTEREST and re-run
#    - "could not be geocoded / outside India" -> those cities were dropped
#    open data/destinations.csv and spot-check ~10 coordinates on openstreetmap.org

# d) regenerate the (synthetic) training labels for the NEW destinations - do NOT overwrite them:
python generate_sample_data.py --preferences-only

# e) retrain and run
python -m src.ml_recommendation
streamlit run app.py
```
Your old sample file is copied to `data/destinations_sample_backup.csv` the first time you import.

How the importer turns attractions into destinations: rows are grouped by (State, City); rating = mean of attraction ratings; interests = union; days = ceil(total hours / 6) limited to 1-5; travel_cost = 2 x straight-line km x 1.3 x Rs 2.5/km from `--origin` (minimum Rs 300); daily_cost = Rs 1,800 + entrance fees / days. **Change these constants** to match prices you have checked yourself (RUPEES_PER_KM, BASE_DAILY_COST ...). Note that travel_cost is fixed for the reference origin; it does not change if the user types another start city.

## 3. Live weather

`src/live_data.py` calls `https://api.open-meteo.com/v1/forecast?...&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code`. The app shows it in a "Live weather now" table with the required attribution; untick the box in the sidebar to disable it. Failures (offline, service busy) show "unavailable" and never crash the app. Weather is shown for information only - it is not an ML feature. Fair use: keep under 10,000 calls/day.

## 4. Test it

```bash
python -m unittest discover -v     # 32 tests; none need internet
```
New tests (`tests/test_real_data_import.py`; the rows in it are invented fixtures, not real data):
* interest keyword mapping; column detection with Kaggle-style headers
* aggregation: 2 attractions in one city -> 1 destination, rating = mean(4.4, 4.0) = 4.2, interests unioned, days = ceil(5/6) = 1
* an unmapped type is reported; a geocode result outside India (London) is rejected
* the imported CSV passes the project's own `load_destinations` validator
* weather: JSON parsing, malformed response -> clear error, URL contains the coordinates, network failure -> `LiveDataError` (all with a fake network)

Manual tests with internet (I could NOT run these):
1. After step b, `data/destinations.csv` should hold roughly 150-250 rows; open it and check that `Jaipur`, `Munnar` etc. have plausible coordinates.
2. `python -c "from src.live_data import get_current_weather as g; print(g(13.08, 80.27))"` should print a temperature close to what a weather site shows now.
3. In the app, choose a start such as `Chennai, India`, budget 25,000: compare a route with Google Maps / OSM directions; expect road distance within roughly 10-25 % of them (different road class filters and snapping cause differences).

## 5. Real training labels (the part that is still synthetic)

Real labels mean real people saying "this trip suits me". No public India-wide dataset of (user budget, interests, days -> chosen trip) exists that I could verify, and Google ratings measure place quality, not fit for a user. Options:
* **Survey (recommended for a college project):** with a Google Form show each respondent several scenarios ("budget Rs 15,000, 3 days, likes beaches: would you choose Gokarna at about Rs 9,000?") and record yes/no. Get consent, collect no names/phone numbers. Aim for a few hundred responses.
* Convert answers to `data/preference_data.csv` with the **same columns** (`user_id, user_budget, interest_match, rating, estimated_total_cost, distance_km, available_days, recommended_days, suitable`) and `label_source = survey`. `train_model` then works unchanged and the split by `user_id` still prevents leakage.
* Report honestly: sample size, who answered (e.g. students), and that results do not generalise to all tourists.

Until then keep the "synthetic labels" warning in the app and your report.

## 6. Citations to add to your report
* Kaggle dataset above (author: saketk511) - cite the page and your access date.
* Open-Meteo.com - weather data, CC BY 4.0, https://open-meteo.com
* (c) OpenStreetMap contributors, ODbL - https://www.openstreetmap.org/copyright
