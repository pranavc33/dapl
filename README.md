# CleanPlanet Code 215 Risk Monitor

A Streamlit dashboard that pulls live sensor telemetry from the CleanPlanet API,
computes a weighted risk score for each unit, and displays it with a per-variable
breakdown and a Pareto contribution chart.

This is a **methodology demonstration**, not a validated production predictor.
See the "Important honesty note" section at the bottom before presenting it.

---

## What it does

For a given unit, the app:

1. Authenticates against the CleanPlanet API with an email + password.
2. Pulls recent high-resolution telemetry for that unit.
3. Computes the risk model's input features (sensor values, or sensor *trends*
   if the slope version is enabled).
4. Checks each feature against its calibrated danger zone, sums the weights of
   the features in danger, and shows a 0-100% risk score with risk tiers.
5. Renders a Pareto chart of each variable's predictive weight and a step-by-step
   explainer of how the model was built.

---

## Requirements

- Python 3.9 or newer
- A CleanPlanet API login (email + password)
- Internet access to `https://recycling.cleanplanetchemical.com`

Python packages:

- streamlit
- pandas
- numpy
- requests
- plotly

---

## Setup on a new machine

### 1. Get the code

Clone the repository (or copy the project folder):

```bash
git clone https://github.com/pranavc33/dapl.git
cd dapl
```

### 2. Create a virtual environment (recommended)

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

Otherwise install directly:
```bash
pip install streamlit pandas numpy requests plotly
```

### 4. Make sure the data file is present

The app reads calibrated danger zones from `danger_ranges.csv`, which must be in
the same folder as the app file. Confirm it is there:

```bash
ls danger_ranges.csv
```

(If you are running the slope/trend version, the zones are defined in the code
itself and this file is not required.)

---

## Running the app

From the project folder, with the virtual environment active:

```bash
python -m streamlit run app.py
```

(Replace `app.py` with the actual filename if it differs, e.g. `cplan.py`.)

Streamlit will start a local server and print a URL, usually:

```
Local URL: http://localhost:8501
```

Open that URL in a browser.

---

## Using the app

1. In the left sidebar, enter your CleanPlanet **email** and **password**.
2. Enter a **Unit ID** to check.
3. Click **CHECK RISK**.

**Units known to have data (good for testing):**
- `129` — St. Johns Packaging
- `201` — RPM Wood Finishes
- `132` — Packaging Products Corp (healthy control)

If a unit has been offline, it may return no recent data; the app will tell you
and suggest trying one of the units above.
---

## Deploying to Streamlit Community Cloud (optional)

1. Push the repository to GitHub.
2. At https://share.streamlit.io, create a new app pointing at the repo and the
   main app file.
3. Make sure `requirements.txt` lists: `streamlit`, `pandas`, `numpy`,
   `requests`, `plotly`.
4. Deploy. The app will be served at a public `*.streamlit.app` URL.

---

## General error-code framework (`methodology.py`)

The dashboard above is the Code 215 application of a broader methodology. The same
repo includes `methodology.py`, a reusable script that applies the full
analysis to **any** error code, not just 215. Code 215 was the worked example; this
is the framework behind it.

### What it does

Set one value (the target error code) and the script runs the whole pipeline:

1. **Find the offender units.** Scans the fleet's event logs and ranks units by how
   many incidents of the target code they have, then selects the top few with enough
   events to analyze.
2. **Screen the variables.** For each candidate sensor, compares its behavior in the
   window before an incident against normal operation, using four independent
   statistical tests (Cohen's d, Mann-Whitney U, AUC, mutual information). Each
   variable scores 0-4; those passing 3+ are kept.
3. **Build the weighted model.** Assigns each surviving variable a weight from its
   AUC and derives a danger zone for it from the historical data.
4. **Tune the threshold.** Sweeps alert thresholds and reports precision, recall, and
   median lead time at each, so the trade-off is visible.
5. **Control check.** Runs the model against a unit that never throws the target code,
   to test whether the model is specific to failure or simply fires a lot. This step
   is what keeps the result honest.

### How to use it

1. Open `methodology.py`.
2. Change the target code near the top:

   ```python
   TARGET = 304    # set to any error code you want to investigate
   ```

3. Optionally adjust the window and selection settings:

   ```python
   LOOKBACK_DAYS = 90          # API retains ~84-89 days; 90 is the practical ceiling
   TOP_N_UNITS = 4             # how many offender units to pool
   MIN_EVENTS_PER_UNIT = 3     # minimum incidents for a unit to be included
   ```

4. Set API credentials via the same environment variables as above
   (`EMAIL`, `PASSWORD`), then run the script. It prints the offender ranking,
   the four-test screening table, the selected model, the threshold sweep, and the
   control check.


