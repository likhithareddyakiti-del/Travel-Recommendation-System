# Intelligent Travel Recommendation System (ML + Knapsack + Dijkstra)

College prototype. Recommends destinations from a start location, budget (Rs), interests and days.

* **Random Forest** (scikit-learn) -> suitability score per destination
* **0/1 Knapsack** (dynamic programming, written by us) -> best set of *independent* options within budget
* **Dijkstra** (written by us, adjacency list + heapq) -> shortest road distance on an **OpenStreetMap** graph (OSMnx)
* **Streamlit** UI with a Folium map

## 0. Honest status (read first)

| Part | Status |
|---|---|
| Data generation, ML training, Knapsack, Dijkstra, routing core, recommender glue | **Executed; 32 unit tests pass** (Python 3.12, pandas 3, scikit-learn 1.8, NetworkX 3.6) |
| OSMnx download, Nominatim geocoding, Folium map, **Streamlit UI** | **Written and syntax-checked only.** Not run, because the development sandbox had no internet and could not install osmnx/streamlit/folium. Expect to fix small issues on first run (see section 8). |
| Real road routes | Not verified. The routing core was tested only on a small synthetic grid graph. |

Data honesty:
* `data/destinations.csv`: coordinates are approximate; **ratings and costs are illustrative placeholders, not verified**.
* `data/preference_data.csv`: **synthetic, rule-generated** (rule in `generate_sample_data.py` + 3 % random label noise). It is **not real tourist feedback**. The model's ~96 % test accuracy only shows it re-learned our rule; it says nothing about real-world accuracy.
* No external dataset is used. If you add one (Kaggle, data.gov.in ...) cite its real URL yourself and verify prices/ratings.

> **Using real Indian data and live weather:** see `REAL_DATA.md` (Kaggle import script, Open-Meteo weather, how to test, how to get real labels). The bundled CSVs are still placeholders/synthetic until you run the import.

## 1. Folder structure

```
travel_project/
├── app.py                      Streamlit UI
├── generate_sample_data.py     creates both CSVs (reproducible, seed 42)   [added]
├── import_real_data.py         converts the real Kaggle file -> destinations.csv [added]
├── REAL_DATA.md                real/live data guide
├── requirements.txt
├── README.md
├── data/
│   ├── destinations.csv
│   └── preference_data.csv
├── models/                     saved model (suitability_model.joblib)
├── outputs/                    model_metrics.json, graph_cache/ (downloaded road graphs)
├── src/
│   ├── __init__.py             [added, makes src a package]
│   ├── data_preprocessing.py   load/clean CSV, cost formula, haversine, feature building
│   ├── ml_recommendation.py    train / evaluate / save / load / score
│   ├── knapsack.py             0/1 knapsack DP
│   ├── dijkstra.py             Dijkstra from scratch
│   ├── route_network.py        geocode, download graph, snap nodes, run Dijkstra
│   ├── live_data.py            real-time weather (Open-Meteo)               [added]
│   └── recommender.py          ML + knapsack glue, input validation          [added]
└── tests/                      test_knapsack, test_dijkstra, test_ml, test_route_network
```
Changes from the suggested layout: added `generate_sample_data.py`, `src/recommender.py`, `__init__.py` files and `tests/`.

## 2. Install (VS Code, Windows / macOS / Linux)

1. Install **Python 3.10, 3.11 or 3.12** (python.org). On Windows tick "Add Python to PATH". (Very new Python versions may lack wheels for geopandas/shapely, which OSMnx needs.)
2. Install **VS Code** and its *Python* extension. Open the `travel_project` folder (File > Open Folder), then open a terminal (Terminal > New Terminal).
3. Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```
4. Install everything (needs internet; osmnx pulls geopandas, shapely, requests automatically):
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
5. In VS Code press Ctrl+Shift+P > *Python: Select Interpreter* > choose the one inside `venv`.

Nothing else to install. **No API keys are needed.** Internet is needed only for geocoding place names and downloading roads.

## 3. Run, step by step (from the project root folder)

```bash
python generate_sample_data.py          # 1. create the two CSVs   -> data/
python -m src.ml_recommendation         # 2. train + evaluate + save model -> models/, outputs/model_metrics.json
python -m unittest discover -v          # 3. run all tests  (or:  pytest)
streamlit run app.py                    # 4. open the app (browser opens at http://localhost:8501)
```
If you skip step 2 the app trains the model automatically the first time.

Expected output of step 1: `destinations.csv: 27 rows | preference_data.csv: 13500 rows, 32.6% labelled suitable (synthetic!)`
Expected output of step 3 (end): `Ran 32 tests ... OK`
Expected step 2 metrics (synthetic data!): accuracy ~0.965, precision ~0.96, recall ~0.93, majority-class baseline ~0.69.

## 4. Using the app / configuring the start location

* **By name:** type e.g. `Chennai, India` (geocoded with OpenStreetMap Nominatim; needs internet; max ~1 request/second, results cached).
* **By coordinates (works offline):** choose "Latitude / longitude" and enter e.g. `13.0827, 80.2707`.
* **Road graph download:** tick *Compute road distance*. For each selected destination one graph is downloaded whose bounding box covers **both** the start and the destination (+ padding km). It is cached in `outputs/graph_cache/`, so repeats are instant. Trips under 40 km straight-line use the full drivable network; longer ones use only motorway/trunk/primary roads (much smaller download). Trips above the "max straight-line distance" (default 150 km) are skipped with a message, to avoid huge requests to public servers.
* If no path is found, the code retries once with double padding, then reports the error. Routes can fail for islands (e.g. Havelock Island has no road link to the mainland) - the app shows the reason and keeps working.

### Example (real output of the tested pipeline; road distances not shown because no internet in the sandbox)
Input: start Chennai (13.0827, 80.2707), budget Rs 25,000, interests beaches + history, 4 days.

| Destination | Suitability | Est. cost (Rs) | Days | Straight-line km |
|---|---|---|---|---|
| Mahabalipuram | 0.939 | 2,300 | 1 | 52 |
| Kanyakumari | 0.933 | 6,200 | 2 | 631 |
| Rameswaram | 0.931 | 5,800 | 2 | 435 |
| Tirupati | 0.930 | 2,000 | 1 | 110 |
| Puducherry | 0.928 | 5,000 | 2 | 136 |

Total cost Rs 21,300, remaining Rs 3,700, total suitability 4.661 (15 candidates passed the filters).
These are independent options, each from the start - **not** a single trip visiting all five.
Scores can differ slightly if you regenerate data or change library versions.

## 5. How each part works

**Cost:** `estimated_total_cost = travel_cost + daily_cost x recommended_days`. travel_cost = estimated round-trip transport; daily_cost = food + local transport + stay + activities. Excluded: shopping, insurance, permits, emergencies. In the sample file travel_cost is one fixed illustrative figure per destination (it does not change with your start city) - a simplification.

**Features / encoding:** interests are multi-hot encoded (`beaches, history, nature, hills, adventure`); `interest_match` = share of the user's interests the destination offers. Other features: user_budget, rating, estimated_total_cost, distance_km (haversine straight-line, **not road distance**), available_days, recommended_days.

**ML score:** the Random Forest (150 trees) averages the trees' votes; `predict_proba` for class 1 is the suitability score (0-1). It is model confidence, not a real probability. Train/test split is **by user** (GroupShuffleSplit) so no user appears in both sets (no leakage). Candidates need score >= 0.5, interest overlap, `recommended_days <= available days`, and cost <= budget.

**Knapsack (`src/knapsack.py`):** `dp[i][w]` = best total suitability using the first i options with w budget units. Each option: skip, or take if it fits. Units are Rs 100: costs rounded **up**, budget rounded **down**, so the result never exceeds your real budget (slightly conservative). A `keep` table reconstructs the chosen options. No affordable option -> empty selection and a clear message.
Time O(n x W), space O(n x W), W = budget/100 (e.g. 25,000 -> 250 columns, not 25,000).

**Dijkstra (`src/dijkstra.py`):** adjacency list `{node: [(neighbour, metres)]}`, `heapq` min-heap, stale-entry skipping, early stop at target, rejects negative weights, raises `NoPathError` if unreachable. Time O((V+E) log V). It finds the shortest path between **two** points; it does **not** order multiple stops. The result is cross-checked against `networkx.shortest_path_length` (validation only). Start/destination are snapped to the nearest road node (nearest-node search is our own NumPy haversine); the snap distance is checked.

**Straight-line vs road distance:** haversine follows the Earth's curve in a straight line; roads bend and detour, so road distance is nearly always longer (a test asserts road >= straight-line).

## 6. Tests (manually verifiable)

* Knapsack: weights [1,3,4,5], values [1,4,5,7], capacity 7 -> take items 3+4, value 9. Rupee case: costs [1500, 2200, 3100], values [0.6, 0.7, 0.9], budget 4000 -> first two (1.3). Nothing affordable -> []. 50 random cases vs brute force.
* Dijkstra: known graph A->E = 20 via A-C-F-E (9+2+9); unreachable -> `NoPathError`; one-way edges; negative weight rejected; equals NetworkX.
* Routing (synthetic grid): matches NetworkX, snaps correctly, disconnected -> error, far point rejected, bbox covers both endpoints.
* ML: scores in [0,1], cheap matching beach outranks unaffordable non-match, selection respects budget/days/interest.

## 7. Citations

* Map data (c) OpenStreetMap contributors, ODbL - https://www.openstreetmap.org/copyright (attribution is shown under the map).
* Nominatim usage policy - https://operations.osmfoundation.org/policies/nominatim/
* Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126-139. https://doi.org/10.1016/j.compenvurbsys.2017.05.004
* Hagberg, Schult & Swart (2008). Exploring network structure, dynamics, and function using NetworkX. *Proc. 7th Python in Science Conf.*
* Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR* 12, 2825-2830.
* Breiman, L. (2001). Random forests. *Machine Learning* 45, 5-32.

## 8. Common errors and fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root; use `python -m src.ml_recommendation`, not `python src/ml_recommendation.py`. |
| `pip install osmnx` fails (geopandas/shapely) | Use Python 3.10-3.12, upgrade pip; on Windows consider conda: `conda install -c conda-forge osmnx`. |
| "Could not find 'X'" (geocoding) | Add the country (`Pune, India`), check internet, or use latitude/longitude mode. |
| "Could not download road data" / timeout | Check internet; lower max distance; retry (Overpass servers get busy); delete a broken file in `outputs/graph_cache/`. |
| "No road path exists" | Increase padding (e.g. 30 km); destination may be an island. |
| "Nearest road is N km away" | Point is outside the downloaded area or in the sea (e.g. Havelock Island). |
| `st.dataframe(use_container_width=...)` warning in newer Streamlit | Harmless; replace with `width="stretch"`. |
| Windows `Activate.ps1` blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Model errors after upgrading scikit-learn | Delete `models/suitability_model.joblib` and retrain. |

## 9. Limitations

Synthetic labels and placeholder costs/ratings; model accuracy is not real-world accuracy. Costs do not vary with the start city. Downloads are limited to about 150 km straight-line by default; long-distance routes use major roads only (distances are approximate; road speed/traffic not modelled). Graph file loading and nearest-node search are simple and slow-ish for very large graphs. Only 27 destinations. Every selected option is routed independently; no multi-stop ordering, itinerary, booking, payment or live prices.
