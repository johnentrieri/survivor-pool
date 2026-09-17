# Survivor Pool Hack

This project analyzes NFL survivor pool picks using the settings in `survivor_model.py` and the game data CSV files in the project folder.

## 1) Create a virtual environment

From the project root:

```bash
python -m venv .venv
```

## 2) Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4) Run the model

```bash
python survivor_model.py
```

If you want to run a specific script or change the input file, update the top-of-file globals first.

---

## Globals to modify at the top of `survivor_model.py`

At the top of the file, there are a few values you will usually change before each run:

```python
# INPUTS

# Schedule of games for the season, including spreads and outcomes (if played)
SCHEDULE_FILE = '2026_gamedata.csv'

# Current week number (1–18)
WEEK_NUM = 2

# Stacking threshold: win probability below which "double-dipping" (stacking) is allowed
STACKING_THRESHOLD = 0.65
MIN_STACK_PROB = 0.70
MAX_STACK = 2

# Early week weighting for survival strategy
EARLY_WEEK_WEIGHT = 2.0

# Survivor pool entries configuration
ENTRIES = [
    {'id': 3, 'used_teams': ['Jacksonville Jaguars']},
    {'id': 4, 'used_teams': ['Jacksonville Jaguars']},
    {'id': 5, 'used_teams': ['Detroit Lions']},
    ...
]

# Current ESPN Power Ratings for each team (used to adjust win probabilities)
POWER_RATINGS = [
    'Los Angeles Rams',
    'Seattle Seahawks',
    'Buffalo Bills',
    ...
]
```

### Typical changes you may need to make

- `SCHEDULE_FILE`: update to the CSV you want to analyze, such as `2025_gamedata.csv` or `2026_gamedata.csv`
- `WEEK_NUM`: change to the current week of the season
- `STACKING_THRESHOLD`: tweak if you want more or less stacking based on win probability
- `MIN_STACK_PROB`: minimum probability required for a stacked team to be considered
- `MAX_STACK`: max number of entries allowed to stack on the same team
- `EARLY_WEEK_WEIGHT`: changes how strongly the model prioritizes preserving teams early in the season
- `ENTRIES`: update each pool entry and the teams already used by that entry
- `POWER_RATINGS`: adjust the team ordering if you want to reflect current power rankings

---

## Notes

- The project expects the game data CSV to be in the same folder as the script.
- If you are running on Windows, use PowerShell or Command Prompt and activate the `.venv` before installing packages.
- If the script fails with missing packages, run:

```bash
python -m pip install -r requirements.txt
```

Then rerun:

```bash
python survivor_model.py
```
