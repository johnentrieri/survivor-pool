from datetime import datetime
import sys
import numpy as np
import requests
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

# INPUTS

# Schedule of games for the season, including spreads and outcomes (if played)
SCHEDULE_FILE = '2026_gamedata.csv'

# Current week number (1–18)
WEEK_NUM = 2

# Stacking threshold: win probability below which "double-dipping" (stacking) is allowed
# Set to None to strictly enforce different teams for each entry (no stacking).
# 0.65 yielded best results for 2025 dataset (Only consider stacking if odds to win are lower than 65%)
STACKING_THRESHOLD = 0.65  
MIN_STACK_PROB = 0.70 # minimum win prob the stacked team must have to be worth doubling up on
MAX_STACK = 2  # Maximum number of entries that can "stack" on the same team in a given week

# Early week weighting for survival strategy
# Higher values give more weight to surviving early weeks when making picks
# 0.0 = uniform weighting (every week is equally important)
# 1.0 = Week 1 has 2x the weight of Week 18
# 2.0 = Week 1 has 3x the weight of Week 18 (best results for 2025 dataset)
EARLY_WEEK_WEIGHT = 2.0 

# Survivor pool entries configuration
ENTRIES = [
  ## {'id': 1, 'used_teams': ['Los Angeles Chargers']},  # DJMJ-1 (Lost Week 1)
  ## {'id': 2, 'used_teams': ['Los Angeles Chargers']},  # DJMJ-2 (Lost Week 1)
  {'id': 3, 'used_teams': ['Jacksonville Jaguars']},  # DJMJ-3
  {'id': 4, 'used_teams': ['Jacksonville Jaguars']},  # DJMJ-4
  {'id': 5, 'used_teams': ['Detroit Lions']},         # DJMJ-5
  {'id': 6, 'used_teams': ['Philadelphia Eagles']},   # DJMJ-6
  {'id': 7, 'used_teams': ['Cincinnati Bengals']},    # DJMJ-7
  {'id': 8, 'used_teams': ['Pittsburgh Steelers']},   # DJMJ-8
  {'id': 9, 'used_teams': ['Chicago Bears']},         # DJMJ-9
  {'id': 10, 'used_teams': ['Las Vegas Raiders']},    # DJMJ-10
  # {'id': 11, 'used_teams': ['Jacksonville Jaguars']}, # Goetz-1
  # {'id': 12, 'used_teams': ['Jacksonville Jaguars']}, # Goetz-2
  # {'id': 13, 'used_teams': ['Los Angeles Chargers']}, # Goetz-3
  # {'id': 14, 'used_teams': ['Detroit Lions']},        # Goetz-4
  # {'id': 15, 'used_teams': ['Philadelphia Eagles']},  # Goetz-5
  # {'id': 16, 'used_teams': ['Jacksonville Jaguars']}, # Des-1
]

# Current ESPN Power Ratings for each team (used to adjust win probabilities)
# List of NFL Teams in order of best to worst
POWER_RATINGS = [
  'Los Angeles Rams',
  'Seattle Seahawks',
  'Buffalo Bills',
  'Denver Broncos',
  'Philadelphia Eagles',
  'New England Patriots',
  'Green Bay Packers',
  'Baltimore Ravens',
  'San Francisco 49ers',
  'Houston Texans',
  'Detroit Lions',
  'Chicago Bears',
  'Kansas City Chiefs',
  'Los Angeles Chargers',
  'Dallas Cowboys',
  'Cincinnati Bengals',
  'Jacksonville Jaguars',
  'Tampa Bay Buccaneers',
  'Indianapolis Colts',
  'Washington Commanders',
  'Pittsburgh Steelers',
  'Minnesota Vikings',
  'Carolina Panthers',
  'New York Giants',
  'New Orleans Saints',
  'Atlanta Falcons',
  'Las Vegas Raiders',
  'Tennessee Titans',
  'New York Jets',
  'Arizona Cardinals',
  'Cleveland Browns',
  'Miami Dolphins'
]



# Ensure Unicode output works on Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

def analyze_2025_trends(schedule):
  """
  Analyze completed 2025 game data for systematic biases that can improve
  the win-probability model.

  Runs five analyses against actual outcomes:
    1. Win-probability calibration  -- is spread_to_win_probability accurate?
    2. Home vs. away favorites      -- is the 2.5-pt HFA assumption right?
    3. O/U total vs. spread error   -- are high-scoring games harder to call?
    4. Spread magnitude buckets     -- are big favorites more/less accurate?
    5. Team ATS (against the spread) records -- who is consistently mis-rated?

  Each section ends with a concrete model-improvement tip.
  """
  import pandas as pd

  # ── Build analysis dataframe ─────────────────────────────────────────────
  rows = []
  for g in schedule:
    if not (g['spread'] and g['winner'] and g['pts_winner'] and g['pts_loser']):
      continue

    fav        = g['favorite']
    home_is_fav = fav == g['home']
    fav_won    = g['winner'] == fav
    underdog   = g['away'] if home_is_fav else g['home']

    # Margin from the favorite's perspective (positive = favorite won)
    raw_margin = g['pts_winner'] - g['pts_loser']
    fav_margin = raw_margin if fav_won else -raw_margin

    rows.append({
      'week':         g['week'],
      'day':          g['day'],
      'favorite':     fav,
      'underdog':     underdog,
      'home':         g['home'],
      'away':         g['away'],
      'home_is_fav':  home_is_fav,
      'spread':       g['spread'],
      'o_u':          g['o_u'],
      'fav_won':      fav_won,
      'fav_margin':   fav_margin,
      # Positive = outperformed spread; negative = underperformed
      'spread_error': fav_margin - g['spread'],
      # Did the favorite cover? (strictly, ignoring push)
      'covered':      fav_margin > g['spread'],
      'pred_wp':      spread_to_win_probability(g['spread']),
    })

  df = pd.DataFrame(rows)
  N  = len(df)

  W = 72   # output width
  print(f"\n{'=' * W}")
  print(f"  2025 SEASON TREND ANALYSIS  ({N} games)")
  print(f"{'=' * W}")

  # ── 1. Win-probability calibration ───────────────────────────────────────
  print(f"\n  1. WIN-PROBABILITY CALIBRATION")
  print(f"     Does the spread-to-prob formula match actual outcomes?")
  print(f"     Bin games by predicted win prob; compare to actual win rate.\n")

  bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
  labels = ['50-55%','55-60%','60-65%','65-70%','70-75%',
            '75-80%','80-85%','85-90%','90%+']
  df['prob_bin'] = pd.cut(df['pred_wp'], bins=bins, labels=labels, right=False)

  calib = (df.groupby('prob_bin', observed=True)
             .agg(games=('fav_won','count'),
                  actual_win_rate=('fav_won','mean'),
                  avg_pred_wp=('pred_wp','mean'))
             .reset_index())

  print(f"  {'Pred WP':>10}  {'Games':>6}  {'Actual Win%':>12}  {'Diff':>8}  {'Signal':>20}")
  print(f"  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*20}")
  for _, r in calib.iterrows():
    if r['games'] == 0:
      continue
    diff     = r['actual_win_rate'] - r['avg_pred_wp']
    signal   = ('over-performing' if diff > 0.05 else
                'under-performing' if diff < -0.05 else 'well-calibrated')
    bar      = ('+' if diff >= 0 else '-') * min(10, int(abs(diff) * 100 / 5))
    print(f"  {str(r['prob_bin']):>10}  {r['games']:>6}  "
          f"{r['actual_win_rate']:>11.1%}  {diff:>+7.1%}  {signal:<20}")

  overall_acc = df['fav_won'].mean()
  cover_rate  = df['covered'].mean()
  print(f"\n  Overall: favorites win {overall_acc:.1%} of games  |  "
        f"cover the spread {cover_rate:.1%} of the time")
  print(f"  Tip: if a bucket shows Diff >5pp, bias-adjust predicted probs for that range.")

  # ── 2. Home vs away favorites ─────────────────────────────────────────────
  print(f"\n  2. HOME VS AWAY FAVORITES")
  print(f"     Model assumes home field = +2.5 pts.  Is that right?\n")

  for label, mask in [('Home favored', df['home_is_fav']),
                      ('Away favored', ~df['home_is_fav'])]:
    sub = df[mask]
    win_rt   = sub['fav_won'].mean()
    cover_rt = sub['covered'].mean()
    avg_err  = sub['spread_error'].mean()
    avg_sp   = sub['spread'].mean()
    print(f"  {label} ({len(sub)} games):")
    print(f"    Avg spread: {avg_sp:.1f} pts  |  Fav wins: {win_rt:.1%}  |  "
          f"Covers: {cover_rt:.1%}  |  Avg spread error: {avg_err:+.2f} pts")

  home_wins  = df[df['home_is_fav']]['fav_won'].mean()
  away_wins  = df[~df['home_is_fav']]['fav_won'].mean()
  home_pts   = df[df['home_is_fav']]['spread_error'].mean()
  away_pts   = df[~df['home_is_fav']]['spread_error'].mean()
  print(f"\n  Home favorites actual HFA vs spread: {home_pts:+.2f} pts")
  print(f"  Away favorites outperformance vs spread: {away_pts:+.2f} pts")
  if abs(home_pts) > 1.0:
    adj = 2.5 + home_pts
    print(f"  Tip: data suggests HFA closer to {adj:.1f} pts "
          f"({'raise' if home_pts > 0 else 'lower'} from 2.5).")
  else:
    print(f"  Tip: 2.5-pt HFA assumption looks reasonable for this dataset.")

  # ── 3. O/U total vs spread error ─────────────────────────────────────────
  print(f"\n  3. OVER/UNDER TOTAL vs SPREAD ACCURACY")
  print(f"     Do high-scoring games (high o/u) make spreads less reliable?\n")

  df['ou_quartile'] = pd.qcut(df['o_u'], q=4,
                               labels=['Low (Q1)','Mid-Low (Q2)',
                                       'Mid-High (Q3)','High (Q4)'])
  ou_grp = (df.groupby('ou_quartile', observed=True)
              .agg(games=('spread_error','count'),
                   ou_range_min=('o_u','min'),
                   ou_range_max=('o_u','max'),
                   avg_abs_error=('spread_error', lambda x: x.abs().mean()),
                   fav_win_rate=('fav_won','mean'),
                   cover_rate=('covered','mean'))
              .reset_index())

  print(f"  {'Quartile':<14}  {'O/U Range':>12}  {'Games':>6}  "
        f"{'Fav Win%':>9}  {'Cover%':>7}  {'Avg |Error|':>12}")
  print(f"  {'-'*14}  {'-'*12}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*12}")
  for _, r in ou_grp.iterrows():
    ou_rng = f"{r['ou_range_min']:.0f}-{r['ou_range_max']:.0f}"
    print(f"  {str(r['ou_quartile']):<14}  {ou_rng:>12}  {r['games']:>6}  "
          f"{r['fav_win_rate']:>8.1%}  {r['cover_rate']:>6.1%}  "
          f"{r['avg_abs_error']:>11.2f}")

  corr_ou_err = df['o_u'].corr(df['spread_error'].abs())
  print(f"\n  Correlation (o/u total vs |spread error|): {corr_ou_err:+.3f}")
  if abs(corr_ou_err) > 0.10:
    direction = 'higher' if corr_ou_err > 0 else 'lower'
    print(f"  Tip: {direction}-scoring games show larger spread errors "
          f"-- consider widening confidence intervals for games with o/u > 47.")
  else:
    print(f"  Tip: o/u total has little impact on spread accuracy in 2025.")

  # ── 4. Spread magnitude buckets ───────────────────────────────────────────
  print(f"\n  4. SPREAD MAGNITUDE vs ACCURACY")
  print(f"     Are big favorites more or less accurate?\n")

  sp_bins   = [0, 3, 6, 9, 12, 20]
  sp_labels = ['1-3 pts','3-6 pts','6-9 pts','9-12 pts','12+ pts']
  df['sp_bucket'] = pd.cut(df['spread'], bins=sp_bins, labels=sp_labels)

  sp_grp = (df.groupby('sp_bucket', observed=True)
              .agg(games=('fav_won','count'),
                   fav_win_rate=('fav_won','mean'),
                   pred_wp=('pred_wp','mean'),
                   cover_rate=('covered','mean'),
                   avg_abs_error=('spread_error', lambda x: x.abs().mean()))
              .reset_index())

  print(f"  {'Spread':>9}  {'Games':>6}  {'Pred WP':>8}  "
        f"{'Actual Win%':>12}  {'Diff':>7}  {'Cover%':>7}  {'Avg |Err|':>10}")
  print(f"  {'-'*9}  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*10}")
  for _, r in sp_grp.iterrows():
    diff = r['fav_win_rate'] - r['pred_wp']
    print(f"  {str(r['sp_bucket']):>9}  {r['games']:>6}  {r['pred_wp']:>7.1%}  "
          f"{r['fav_win_rate']:>11.1%}  {diff:>+6.1%}  {r['cover_rate']:>6.1%}  "
          f"{r['avg_abs_error']:>9.2f}")

  print(f"\n  Tip: if large favorites (9+ pts) show Diff < -5%, the model is overconfident")
  print(f"  at high spreads.  Consider scaling win probs toward 0.50 slightly at extremes.")

  # ── 5. Team ATS records ───────────────────────────────────────────────────
  print(f"\n  5. TEAM ATS (AGAINST THE SPREAD) RECORDS")
  print(f"     Which teams are consistently mis-rated by oddsmakers?\n")

  # Each game appears twice: once from the favorite's view, once from the underdog's
  fav_rows = df[['favorite','covered','spread_error','fav_won']].copy()
  fav_rows.columns = ['team','fav_covered','spread_error','fav_won']
  fav_rows['is_favorite'] = True

  dog_rows = df[['underdog','covered','spread_error','fav_won']].copy()
  dog_rows['covered']     = ~dog_rows['covered']   # underdog covered iff fav didn't
  dog_rows['spread_error'] = -dog_rows['spread_error']
  dog_rows['fav_won']     = ~dog_rows['fav_won']
  dog_rows.columns = ['team','fav_covered','spread_error','fav_won']
  dog_rows['is_favorite'] = False

  all_team = pd.concat([fav_rows, dog_rows])
  team_grp = (all_team.groupby('team')
                .agg(games        = ('fav_covered','count'),
                     covers       = ('fav_covered','sum'),
                     win_rate     = ('fav_won','mean'),
                     avg_ats_diff = ('spread_error','mean'))
                .reset_index())
  team_grp['cover_pct']     = team_grp['covers'] / team_grp['games']
  team_grp['expected_cov']  = 0.5   # random baseline
  team_grp['ats_edge']      = team_grp['cover_pct'] - 0.5  # beats vs expected 50%

  # Show top 8 (most reliable) and bottom 8 (least reliable)
  sorted_t = team_grp.sort_values('avg_ats_diff', ascending=False)

  print(f"  {'Team':<26}  {'Games':>6}  {'Win%':>6}  {'Cover%':>7}  "
        f"{'Avg Margin vs Spread':>20}  {'Signal'}")
  print(f"  {'-'*26}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*20}  {'-'*14}")

  print(f"\n  -- Top 8: consistently BEAT their spread (underrated) --")
  for _, r in sorted_t.head(8).iterrows():
    bar = '+' * min(8, int(abs(r['avg_ats_diff']) / 2))
    print(f"  {r['team']:<26}  {r['games']:>6}  {r['win_rate']:>5.0%}  "
          f"{r['cover_pct']:>6.0%}  {r['avg_ats_diff']:>+19.1f}  {bar}")

  print(f"\n  -- Bottom 8: consistently MISS their spread (overrated) --")
  for _, r in sorted_t.tail(8).iterrows():
    bar = '-' * min(8, int(abs(r['avg_ats_diff']) / 2))
    print(f"  {r['team']:<26}  {r['games']:>6}  {r['win_rate']:>5.0%}  "
          f"{r['cover_pct']:>6.0%}  {r['avg_ats_diff']:>+19.1f}  {bar}")

  print(f"\n  Tip: For survivor picks, PREFER underrated teams (positive avg_ats_diff)")
  print(f"  even when their raw win prob looks equal to alternatives.")
  print(f"  AVOID overrated teams -- they win at their stated probability but rarely")
  print(f"  dominate the way the spread implies, meaning more upset risk than modeled.")

  # ── 6. Day of week ────────────────────────────────────────────────────────
  print(f"\n  6. DAY-OF-WEEK EFFECT")
  print(f"     Short-week games (Thu, Mon) vs full-week (Sun) prep time.\n")

  day_grp = (df.groupby('day')
               .agg(games=('fav_won','count'),
                    fav_win_rt=('fav_won','mean'),
                    cover_rt=('covered','mean'),
                    avg_abs_err=('spread_error', lambda x: x.abs().mean()))
               .reset_index()
               .sort_values('games', ascending=False))

  print(f"  {'Day':<6}  {'Games':>6}  {'Fav Win%':>9}  {'Cover%':>7}  {'Avg |Error|':>12}")
  print(f"  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*12}")
  for _, r in day_grp.iterrows():
    print(f"  {r['day']:<6}  {r['games']:>6}  {r['fav_win_rt']:>8.1%}  "
          f"{r['cover_rt']:>6.1%}  {r['avg_abs_err']:>11.2f}")

  # Compare short-week (Thu/Mon) vs full-week (Sun/Sat)
  short_week = df[df['day'].isin(['Thu','Mon','Fri'])]
  full_week  = df[df['day'].isin(['Sun','Sat'])]
  if len(short_week) > 0 and len(full_week) > 0:
    sw_cover = short_week['covered'].mean()
    fw_cover = full_week['covered'].mean()
    sw_err   = short_week['spread_error'].abs().mean()
    fw_err   = full_week['spread_error'].abs().mean()
    print(f"\n  Short-week (Thu/Mon/Fri): cover {sw_cover:.1%}, avg |error| {sw_err:.2f}")
    print(f"  Full-week  (Sun/Sat):     cover {fw_cover:.1%}, avg |error| {fw_err:.2f}")
    if sw_err > fw_err + 0.5:
      print(f"  Tip: short-week games have {sw_err - fw_err:.2f} pts larger spread error "
            f"-- apply a slight confidence reduction for Thu/Mon games.")
    else:
      print(f"  Tip: no meaningful difference between short-week and full-week accuracy.")


  print(f"\n{'=' * W}")

def spread_to_win_probability(spread: float, std_dev: float = 13.86) -> float:
  """
  Converts an NFL point spread into a straight-up win probability.
  
  Parameters:
  spread (float): The point spread, always positive in the dataset (e.g., if the favorite is -7, the spread is 7).
  std_dev (float): Historical NFL standard deviation (default 13.86).
  
  Returns:
  float: The win probability as a decimal between 0 and 1.
  """
      
  # 0.5 is the threshold to win the game (margin > 0.5)
  # norm.cdf calculates the probability of scoring LESS than 0.5
  prob_underdog_wins_or_ties = norm.cdf(0.5, loc=spread, scale=std_dev)
  
  # Subtract from 1 to get the favorite's win probability
  return (1 - prob_underdog_wins_or_ties)

def power_ranking_to_win_probability(home_team: str, away_team: str) -> float:
  """
  Estimates win probability for the home team from the POWER_RATINGS list.

  Maps the rank-position difference to an implied point spread (spread_scale
  across the full 32-team range), adds home-field advantage, then feeds the
  result into spread_to_win_probability().

  Parameters:
  home_team (str): Full team name, must appear in POWER_RATINGS.
  away_team (str): Full team name, must appear in POWER_RATINGS.

  Returns:
  float: Home team win probability (0–1).
  """
  HOME_FIELD_ADV = 2.5                                 # historical NFL HFA in pts
  MAX_SPREAD_PTS = 14.0                                # best vs worst team spread
  n              = len(POWER_RATINGS)
  pts_per_rank   = MAX_SPREAD_PTS / (n - 1)            # points per ranking step

  try:
    home_pos = POWER_RATINGS.index(home_team)          # 0 = strongest
    away_pos = POWER_RATINGS.index(away_team)
  except ValueError:
    return 0.5                                         # unknown team: coin flip

  # Positive -> home is favored; negative -> away is favored
  implied_spread = (away_pos - home_pos) * pts_per_rank + HOME_FIELD_ADV

  if implied_spread >= 0:
    return spread_to_win_probability(implied_spread)
  else:
    return 1.0 - spread_to_win_probability(-implied_spread)

def calculate_optimal_picks(entries, schedule_with_projections, week_num,
                            stacking_threshold=None, early_week_weight=0.0,
                            max_stack=2, min_stack_prob=None):
  """
  Calculate optimal survivor pool picks for all entries.

  -- The diversification vs stacking tradeoff ---------------------------------
  For a single isolated week, picking N DISTINCT teams always gives a higher
  P(at least one survives that week) than having any two entries share a team:

      P(survive | distinct A, B) = 1 - (1-pA)(1-pB)  >  pA  = P(survive | stack A, A)

  HOWEVER, the full-season picture can differ.  Stack in week 1 when one team
  is a massive favorite -- the correlated-failure risk is worth the early
  probability boost.  Use stacking_threshold to control when stacking kicks in.

  -- Early-season aggressiveness ----------------------------------------------
  early_week_weight shifts the optimizer's priority toward surviving early weeks
  at the cost of some late-week pick quality.  Mathematically, each week's cost
  is scaled by a weight that decays linearly from (1 + early_week_weight) at the
  current week down to 1.0 at the final week.

      early_week_weight = 0.0  ->  uniform weights (all weeks equal, default)
      early_week_weight = 1.0  ->  current week has 2x the cost weight of the last
      early_week_weight = 2.0  ->  current week has 3x the cost weight of the last

  A higher value makes the optimizer aggressively allocate its strongest teams
  to early weeks (maximising near-term survival) while accepting weaker picks in
  the tail.  Useful when the realistic goal is surviving to week 13-14 rather
  than a perfect 18-week run.

  Parameters:
  entries (list): List of entry state dicts with 'id' and 'used_teams'.
  schedule_with_projections (list): Games from week_num onward with win_probability.
  week_num (int): Current week to optimize from.
  stacking_threshold (float|None): Min win prob before stacking is allowed.
      None = strict hedge (default). Values in (0.5, 0.80) are practical.
  early_week_weight (float): 0.0 = uniform, higher = prioritise early survival.
  max_stack (int): maximum number of entries allowed to pick the same team in
      the same week.  1 = strict hedge (no two entries share a team ever).
      2 = allow one stack per team per week (default). Higher values allow more.

  Returns:
  list of dicts, one per entry:
      'id', 'pick', 'win_probability', 'survival_probability', 'plan'
  """
  if not entries:
    entries = [{'id': 1, 'used_teams': []}]

  # -- Build (week, team) → win_probability lookup ---------------------------
  team_win_probs = {}
  for game in schedule_with_projections:
    w   = game['week']
    wp  = game['win_probability']
    home, away, fav = game['home'], game['away'], game['favorite']
    if fav is not None:
      other = away if fav == home else home
      team_win_probs[(w, fav)]   = wp
      team_win_probs[(w, other)] = 1.0 - wp
    else:
      team_win_probs[(w, home)] = wp
      team_win_probs[(w, away)] = 1.0 - wp

  future_weeks = sorted(set(w for (w, _) in team_win_probs if w >= week_num))
  # track how many entries have claimed each team per week; hard-block at max_stack
  week_claim_count = {w: {} for w in future_weeks}

  INF = 1e9
  EPS = 1e-9

  results = []
  for entry in entries:
    used = set(entry.get('used_teams', []))

    available_teams = sorted({
      team for (w, team) in team_win_probs
      if w >= week_num and team not in used
    })

    if not future_weeks or not available_teams:
      results.append({
        'id': entry['id'], 'pick': None,
        'win_probability': 0.0, 'survival_probability': 0.0, 'plan': []
      })
      continue

    n_weeks    = len(future_weeks)
    week_to_idx = {w: i for i, w in enumerate(future_weeks)}
    team_to_idx = {t: i for i, t in enumerate(available_teams)}

    # -- Build cost matrix (weeks x teams) ------------------------------------
    # cost[w][t] = week_weight * -log(win_prob)
    #
    # early_week_weight > 0 amplifies the cost for early weeks, so the
    # optimizer is forced to assign high-probability picks there.
    # Weight decays linearly: (1 + early_week_weight) at week_num -> 1.0 at
    # the final week.  early_week_weight=0 -> uniform (default behaviour).
    cost   = np.full((n_weeks, len(available_teams)), INF)
    max_wk = max(future_weeks)   # used for week-weight normalisation

    for (w, team), prob in team_win_probs.items():
      if w < week_num or team in used or team not in team_to_idx:
        continue

      wi = week_to_idx[w]
      ti = team_to_idx[team]

      claim_count = week_claim_count[w].get(team, 0)
      if claim_count >= max_stack:
        continue  # hard block: team already shared by max_stack entries
      elif claim_count > 0:
        if stacking_threshold is None:
          continue   # hard block -- never stack
        if prob < stacking_threshold:
          continue   # team not strong enough to be worth stacking on
        if min_stack_prob is not None and prob < min_stack_prob:
          continue   # team below the minimum quality floor for stacking
        base_cost = -np.log(stacking_threshold + EPS)  # fixed penalty: stacked = appears as threshold prob
      else:
        base_cost = -np.log(max(prob, EPS))

      # Early-week weight: higher for earlier weeks
      if early_week_weight > 0 and max_wk > week_num:
        progress = (w - week_num) / (max_wk - week_num)   # 0 at start, 1 at end
        wt = 1.0 + early_week_weight * (1.0 - progress)
      else:
        wt = 1.0

      cost[wi][ti] = wt * base_cost

    # -- Solve assignment problem --------------------------------------------
    row_ind, col_ind = linear_sum_assignment(cost)

    plan = []
    for ri, ci in zip(row_ind, col_ind):
      if cost[ri][ci] >= INF:
        continue
      w    = future_weeks[ri]
      team = available_teams[ci]
      prob = team_win_probs.get((w, team), 0.0)
      plan.append({'week': w, 'team': team, 'win_probability': round(prob, 4)})
      week_claim_count[w][team] = week_claim_count[w].get(team, 0) + 1

    plan.sort(key=lambda x: x['week'])

    weeks_with_picks = {p['week'] for p in plan}
    survival_prob = 0.0 if any(w not in weeks_with_picks for w in future_weeks) else \
                    round(np.prod([p['win_probability'] for p in plan]), 6)

    current_pick = next((p for p in plan if p['week'] == week_num), None)
    results.append({
      'id':                   entry['id'],
      'used_teams':            list(entry['used_teams']),
      'pick':                 current_pick['team'] if current_pick else None,
      'win_probability':      current_pick['win_probability'] if current_pick else 0.0,
      'survival_probability': survival_prob,
      'plan':                 plan,
    })

  return results

def backtest(schedule_file, n_entries=10, stacking_threshold=None, early_week_weight=0.0, max_stack=2, min_stack_prob=None):
  """
  Backtest the model against actual game results in schedule_file.

  Simulates the model week-by-week with no lookahead:
    - Builds win projections from spread / power rankings for each week.
    - Calls calculate_optimal_picks() to get the recommended pick per entry.
    - Checks each pick against the actual 'winner' field and updates state.

  Parameters:
  schedule_file     : path to CSV (must include 'winner' column).
  n_entries         : number of entries to simulate.
  stacking_threshold: allow multiple entries to share a team below this win prob.
  early_week_weight : amplify early-week cost weights (0 = uniform).

  Returns:
  dict: {entry_id: {'alive', 'picks', 'used_teams', 'survival_probability'}}
  """
  schedule     = process_schedule(schedule_file)
  winner_set   = {(g['week'], g['winner']) for g in schedule if g['winner']}
  all_weeks    = sorted(set(g['week'] for g in schedule))

  # Per-entry state
  states = {
    i: {'alive': True, 'used_teams': [], 'picks': []}
    for i in range(1, n_entries + 1)
  }

  for week_num in all_weeks:
    alive_inputs = [
      {'id': eid, 'used_teams': list(s['used_teams'])}
      for eid, s in states.items() if s['alive']
    ]
    if not alive_inputs:
      break

    # Build projections from this week forward (no lookahead on results)
    proj = []
    for game in schedule:
      if game['week'] >= week_num:
        wp = (spread_to_win_probability(game['spread'])
              if game['spread'] is not None
              else power_ranking_to_win_probability(game['home'], game['away']))
        proj.append({**game, 'win_probability': wp})

    picks = calculate_optimal_picks(alive_inputs, proj, week_num,
                                    stacking_threshold=stacking_threshold,
                                    early_week_weight=early_week_weight,
                                    max_stack=max_stack,
                                    min_stack_prob=min_stack_prob)

    for p in picks:
      eid, team = p['id'], p['pick']
      states[eid]['picks'].append({
        'week': week_num, 'team': team,
        'win_probability': p['win_probability'],
      })
      if team and (week_num, team) in winner_set:
        states[eid]['used_teams'].append(team)
      else:
        states[eid]['alive'] = False

  # Build return value in expected format
  results = {}
  for eid, s in states.items():
    # Survival probability = product of win probs for correct picks only
    correct_picks = s['picks'][:-1] if not s['alive'] else s['picks']
    surv = 1.0
    for pick in correct_picks:
      surv *= pick['win_probability']
    results[eid] = {
      'alive':                s['alive'],
      'picks':                s['picks'],
      'used_teams':           s['used_teams'],
      'survival_probability': round(surv, 6) if s['alive'] else 0.0,
    }
  return results

def simulate_probability_at_least_one_survives(results, team_win_probs,
                                               n_simulations=100_000):
  """
  Monte Carlo estimate of P(at least one entry survives the entire season).

  Unlike the closed-form 1 - PROD(1-Pᵢ), this simulation correctly handles
  CORRELATION between entries that share a team in the same week:

    • When two entries pick the same team in week W, they share a single
      game outcome — if that team loses, BOTH entries are eliminated
      simultaneously (not independently).
    • When entries pick different teams in week W, their outcomes are
      independent (different games).

  The closed-form formula assumes full independence and therefore
  OVERESTIMATES P(at least one) when entries share picks.

  Parameters:
  results        : output of calculate_optimal_picks()
  team_win_probs : {(week, team): win_probability} dict from schedule data
  n_simulations  : number of Monte Carlo trials (100k ≈ ±0.3% accuracy)

  Returns:
  float: simulated P(at least one entry survives)
  """
  rng = np.random.default_rng(seed=0)  # fixed seed for reproducibility

  # Build a quick lookup: entry_id → list of (week, team, win_prob) picks
  plans = [
    [(p['week'], p['team'], p['win_probability']) for p in r['plan']]
    for r in results if r['plan']
  ]
  if not plans:
    return 0.0

  # Collect every unique (week, team) pair that needs to be simulated
  all_games = sorted(set((w, t) for plan in plans for (w, t, _) in plan))
  n_games   = len(all_games)
  game_idx  = {(w, t): i for i, (w, t) in enumerate(all_games)}
  probs     = np.array([team_win_probs.get((w, t), 0.5) for (w, t) in all_games])

  successes = 0
  for _ in range(n_simulations):
    # Simulate every game outcome for this trial
    outcomes = rng.random(n_games) < probs   # True = team wins

    # Check if at least one entry survives all its picks
    for plan in plans:
      if all(outcomes[game_idx[(w, t)]] for (w, t, _) in plan):
        successes += 1
        break   # at least one survived — no need to check others

  return successes / n_simulations

def probability_at_least_one_survives(results):
  """
  Closed-form P(at least one) = 1 - PRODᵢ(1-Pᵢ), assuming full independence.
  Valid when the hedge constraint guarantees no two entries share a team.
  Use simulate_probability_at_least_one_survives() for the exact answer
  when entries may share picks (stacking mode).
  """
  failure_product = 1.0
  for entry in results:
    failure_product *= (1.0 - entry.get('survival_probability', 0.0))
  return round(1.0 - failure_product, 6)

def backtest_thresholds(schedule, thresholds, n_entries=10, early_week_weight=0.0, max_stack=2, min_stack_prob=None):
  """
  Backtest multiple stacking thresholds against actual game results.

  Simulates the model running in real-time, week by week:
    1. At the start of week W, build win projections using spread data only
       (no lookahead -- winner column is hidden from the model).
    2. Call calculate_optimal_picks() to get the recommended pick per entry.
    3. Check each pick against the actual 'winner' field in the schedule.
    4. Update entry states: correct pick -> bank the team; wrong pick -> eliminated.

  Parameters:
  schedule   : full raw schedule list from CSV (must include 'winner' field)
  thresholds : list of stacking_threshold values, e.g. [None, 0.60, 0.65, ...]
               None = strict hedge (always different teams)
  n_entries  : number of pool entries to simulate per strategy

  Returns:
  dict: threshold -> {entries, weeks, final_alive, avg_weeks, at_least_one}
        'weeks' is a list of integers, one per entry, counting correct picks made.
  """
  # Set of (week, winner_name) for quick lookup
  winner_lookup = {(g['week'], g['winner']) for g in schedule if g['winner']}
  all_weeks = sorted(set(g['week'] for g in schedule))

  def make_projections(from_week):
    """Build schedule_with_projections using spread data only (no results)."""
    proj = []
    for game in schedule:
      if game['week'] >= from_week:
        wp = (spread_to_win_probability(game['spread'])
              if game['spread'] is not None
              else power_ranking_to_win_probability(game['home'], game['away']))
        proj.append({
          'week':            game['week'],
          'home':            game['home'],
          'away':            game['away'],
          'favorite':        game['favorite'],
          'win_probability': wp,
        })
    return proj

  results_by_threshold = {}

  for threshold in thresholds:
    entries = [
      {'id': i + 1, 'used_teams': [], 'alive': True,
       'eliminated_week': None, 'eliminated_pick': None}
      for i in range(n_entries)
    ]

    for week in all_weeks:
      alive = [e for e in entries if e['alive']]
      if not alive:
        break

      proj = make_projections(week)
      entry_inputs = [{'id': e['id'], 'used_teams': list(e['used_teams'])} for e in alive]
      picks = calculate_optimal_picks(entry_inputs, proj, week, threshold,
                                      early_week_weight=early_week_weight,
                                      max_stack=max_stack,
                                      min_stack_prob=min_stack_prob)

      for pick_result in picks:
        entry = next(e for e in entries if e['id'] == pick_result['id'])
        team  = pick_result['pick']

        if team and (week, team) in winner_lookup:
          entry['used_teams'].append(team)   # correct -- survive and bank the team
        else:
          entry['alive']           = False
          entry['eliminated_week'] = week
          entry['eliminated_pick'] = team    # None if no valid pick was available

    def weeks_survived(e):
      # Number of correct picks made before elimination (or full season if alive)
      return len(all_weeks) if e['alive'] else (e['eliminated_week'] or 1) - 1

    weeks_list = [weeks_survived(e) for e in entries]
    results_by_threshold[threshold] = {
      'entries':      entries,
      'weeks':        weeks_list,
      'final_alive':  sum(1 for e in entries if e['alive']),
      'avg_weeks':    sum(weeks_list) / len(weeks_list),
      'at_least_one': any(e['alive'] for e in entries),
    }

  return results_by_threshold

def print_backtest_results(results_by_threshold, all_weeks, n_entries):
  """Print a formatted comparison of backtest results across thresholds."""
  thresholds = list(results_by_threshold.keys())
  max_week   = max(all_weeks) if all_weeks else 18

  def lbl(t):
    return 'None (hedge)' if t is None else f'{t:.0%} stack'

  print(f"\n{'=' * 72}")
  print(f"  BACKTEST vs ACTUAL RESULTS  ({n_entries} entries, {max_week} weeks)")
  print(f"{'=' * 72}")

  # -- Summary table ---------------------------------------------------------
  print(f"\n  {'Threshold':<14}  {'Avg Wks':>8}  {'Max Wks':>8}  "
        f"{'Final Alive':>12}  {'>=1 Survived':>13}")
  print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*13}")
  for t in thresholds:
    r  = results_by_threshold[t]
    mx = max(r['weeks'])
    survived_str = 'YES  *' if r['at_least_one'] else 'no'
    print(f"  {lbl(t):<14}  {r['avg_weeks']:>7.1f}  {mx:>8}  "
          f"{r['final_alive']:>12}  {survived_str:>13}")

  # -- Week-by-week alive count ----------------------------------------------
  col_w = max(len(lbl(t)) for t in thresholds) + 2
  print(f"\n  ENTRIES ALIVE AFTER EACH WEEK:")
  header = f"  {'Wk':>3}"
  for t in thresholds:
    header += f"  {lbl(t):>{col_w}}"
  print(header)
  print("  " + "-" * (3 + len(thresholds) * (col_w + 2) + 2))
  for week in all_weeks:
    row = f"  {week:>3}"
    for t in thresholds:
      alive_ct = sum(1 for ws in results_by_threshold[t]['weeks'] if ws >= week)
      row += f"  {alive_ct:>{col_w}}"
    print(row)

  # -- Entry elimination log for each threshold ------------------------------
  print(f"\n  ELIMINATION LOG (week each entry was knocked out, or 'ALIVE'):")
  header2 = f"  {'Entry':>6}"
  for t in thresholds:
    header2 += f"  {lbl(t):>{col_w}}"
  print(header2)
  print("  " + "-" * (6 + len(thresholds) * (col_w + 2) + 2))
  for idx in range(n_entries):
    row = f"  {idx + 1:>6}"
    for t in thresholds:
      e = results_by_threshold[t]['entries'][idx]
      if e['alive']:
        val = 'ALIVE'
      else:
        pick_str = (e['eliminated_pick'] or 'no pick')[:6]
        val = f"Wk{e['eliminated_week']}({pick_str})"
      row += f"  {val:>{col_w}}"
    print(row)

  print(f"\n  * = at least one entry survived all {max_week} weeks in the schedule")
  print(f"{'=' * 72}")

def backtest_weighted(schedule, early_week_weights, n_entries=10, stacking_threshold=None):
  """
  Backtest multiple early-week weight values against actual game results.

  Simulates the model running in real-time, week by week:
    1. At the start of week W, build win projections using spread data only
       (no lookahead -- winner column is hidden from the model).
    2. Call calculate_optimal_picks() to get the recommended pick per entry.
    3. Check each pick against the actual 'winner' field in the schedule.
    4. Update entry states: correct pick -> bank the team; wrong pick -> eliminated.
  """
  results_by_weight = {}
  for w in early_week_weights:
    results_by_weight[w] = backtest(schedule, n_entries=n_entries, stacking_threshold=stacking_threshold, early_week_weight=w)
  return results_by_weight

def process_schedule(schedule_file):
  """
  Process the schedule CSV file and return a list of game dictionaries.

  Each game dictionary contains the following keys:
    - week: int
    - day: str
    - date: str
    - time: str
    - away: str
    - home: str
    - favorite: str or None
    - opponent: str or None
    - spread: float or None
    - o_u: float or None
    - winner: str or None
    - pts_winner: int or None
    - pts_loser: int or None
  """
  # Columns: week, day, date, time, away, home, favorite, spread, over/under, winner, pts_winner, pts_loser
  # if favorite & spread are blank, odds are not available yet
  data = open(schedule_file, 'r')
  data.readline()  # Skip header
  schedule = []

  for line in data:
    parts = line.strip().split(',')

    # Error Checking
    if len(parts) < 12:
      exit(f"Error: Expected 12 columns in schedule data, but got {len(parts)} columns in line: {line.strip()}")

    schedule.append({
      'week': int(parts[0]),
      'day': parts[1],
      'date': parts[2],
      'time': parts[3],
      'away': parts[4],
      'home': parts[5],
      'favorite': parts[6] if parts[6] else None,
      'opponent': parts[5] if parts[6] == parts[4] else parts[4] if parts[4] else None,
      'spread': float(parts[7]) if parts[7] else None,
      'o_u': float(parts[8]) if parts[8] else None,
      'winner': parts[9] if parts[9] else None,
      'pts_winner': int(parts[10]) if parts[10] else None,
      'pts_loser': int(parts[11]) if parts[11] else None,
    })
  return schedule

def get_opponent_from_schedule(schedule, team, week):
  # In dictionary, find game based on team and week and return opponent string
  for game in schedule:
    if game['week'] == week:
      if game['home'] == team:
        return game['away']
      elif game['away'] == team:
        return game['home']
  return None
  
def calculate_win_probabilities(schedule, week_num):
  """
  Calculate win probabilities for each unplayed game in the schedule.

  Parameters:
  schedule (list): List of game dictionaries.
  week_num (int): Current week number.

  Returns:
  list: List of game dictionaries with added 'win_probability' key.
  """
  # Initialize a list to hold games with projected win probabilities
  schedule_with_projections = []

  for game in schedule:

    # Skip games that have already been played
    if game['week'] >= week_num:

      # Project win probability for this game
      projected_game = game.copy()

      # Initialize win probability to None
      projected_game['win_probability'] = None

      # Use spread to project win probability if available, otherwise use power rankings
      if game['spread'] is not None:
        projected_game['win_probability'] = spread_to_win_probability(game['spread'])
      else:
        projected_game['win_probability'] = power_ranking_to_win_probability(game['home'], game['away'])

      schedule_with_projections.append(projected_game)

  return schedule_with_projections

def monte_carlo_stacking_analysis(schedule_with_projections, stacking_thresholds, n_entries=10, n_simulations=100_000, early_week_weight=0.0):
  """
  Perform a Monte Carlo simulation to analyze the impact of different stacking thresholds.

  Parameters:
  schedule_with_projections (list): List of game dictionaries with projected win probabilities.
  stacking_thresholds (list): List of stacking threshold values to test.
  n_entries (int): Number of entries in the survivor pool.
  n_simulations (int): Number of Monte Carlo simulations to run.

  Returns:
  dict: Mapping of stacking threshold to simulated probability of at least one entry surviving.
  """
  results = {}
  
  # Build team_win_probs lookup for simulation
  team_win_probs = {}
  for game in schedule_with_projections:
    w, wp = game['week'], game['win_probability']
    home, away, fav = game['home'], game['away'], game['favorite']
    if fav is not None:
      other = away if fav == home else home
      team_win_probs[(w, fav)]   = wp
      team_win_probs[(w, other)] = 1.0 - wp
    else:
      team_win_probs[(w, home)] = wp
      team_win_probs[(w, away)] = 1.0 - wp

  for threshold in stacking_thresholds:
    picks = calculate_optimal_picks(
      [{'id': i + 1, 'used_teams': []} for i in range(n_entries)],
      schedule_with_projections,
      1,  # Start from week 1 for simulation
      stacking_threshold=threshold,
      early_week_weight=early_week_weight,
    )
    prob = simulate_probability_at_least_one_survives(picks, team_win_probs, n_simulations)
    results[threshold] = prob

  return results

def monte_carlo_weighted_analysis(schedule_with_projections, early_week_weights, n_entries=10, n_simulations=100_000, stacking_threshold=None):
  """
  Perform a Monte Carlo simulation to analyze the impact of different early week weights.

  Parameters:
  schedule_with_projections (list): List of game dictionaries with projected win probabilities.
  early_week_weights (list): List of early week weight values to test.
  n_entries (int): Number of entries in the survivor pool.
  n_simulations (int): Number of Monte Carlo simulations to run.

  Returns:
  dict: Mapping of early week weight to simulated probability of at least one entry surviving.
  """
  results = {}
  
  # Build team_win_probs lookup for simulation
  team_win_probs = {}
  for game in schedule_with_projections:
    w, wp = game['week'], game['win_probability']
    home, away, fav = game['home'], game['away'], game['favorite']
    if fav is not None:
      other = away if fav == home else home
      team_win_probs[(w, fav)]   = wp
      team_win_probs[(w, other)] = 1.0 - wp
    else:
      team_win_probs[(w, home)] = wp
      team_win_probs[(w, away)] = 1.0 - wp

  for weight in early_week_weights:
    picks = calculate_optimal_picks(
      [{'id': i + 1, 'used_teams': []} for i in range(n_entries)],
      schedule_with_projections,
      1,  # Start from week 1 for simulation
      stacking_threshold=stacking_threshold,  # Use the provided stacking threshold for this analysis
      early_week_weight=weight,
    )
    prob = simulate_probability_at_least_one_survives(picks, team_win_probs, n_simulations)
    results[weight] = prob

  return results

def print_optimal_picks(optimal_picks_results):
  """
  Print the optimal picks for each entry in a formatted manner.

  Parameters:
  optimal_picks_results (list): List of dictionaries containing optimal pick results for each entry.
  """
  print(f"Date Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  print (f"Current Week: {schedule_with_projections[0]['week'] if schedule_with_projections else 'N/A'}")
  print(f"Win Probability to Consider Stacking Instead: {STACKING_THRESHOLD if 'STACKING_THRESHOLD' else 'N/A'}")
  print(f"Win Probability Minimum to Consider Stacking: {MIN_STACK_PROB if 'MIN_STACK_PROB' else 'N/A'}")
  print(f"Max Stacks: {MAX_STACK if 'MAX_STACK' else 'N/A'}")
  print(f"Early Week Weight: {EARLY_WEEK_WEIGHT if 'EARLY_WEEK_WEIGHT' else 'N/A'}")

  for entry_result in optimal_picks_results:
    entry_id = entry_result['id']
    pick = entry_result['pick']
    win_prob = entry_result['win_probability']
    survival_prob = entry_result['survival_probability']
    plan = entry_result['plan']

    print(f"\nEntry {entry_id}:")
    print(f"  Used Teams: {', '.join(entry_result['used_teams'])}")
    print(f"  Current Week Pick: {pick} (Win Probability: {win_prob:.2%})")
    print(f"  Survival Probability for Remaining Weeks: {survival_prob:.2%}")
    print("  Full Plan:")
    for p in plan:
        opponent = get_opponent_from_schedule(schedule_with_projections, p['team'], p['week'])
        print(f"    Week {p['week']}: {p['team']} over {opponent} (Win Probability: {p['win_probability']:.2%})")

def print_monte_carlo_results(monte_carlo_results, analysis_type="Stacking"):
  """
  Print the results of the Monte Carlo stacking analysis in a formatted manner.

  Parameters:
  monte_carlo_results (dict): Dictionary containing stacking thresholds and their corresponding probabilities.
  """
  if analysis_type == "Stacking":
    print("\nMonte Carlo Stacking Analysis Results:")
    for threshold, prob in monte_carlo_results.items():
      print(f"  Stacking Threshold {threshold:.3f}: Probability at Least One Survives = {prob:.3%}")
  elif analysis_type == "Weighted":
    print("\nMonte Carlo Weighted Analysis Results:")
    for weight, prob in monte_carlo_results.items():
      print(f"  Early Week Weight {weight:.3f}: Probability at Least One Survives = {prob:.3%}")

def print_odds_to_percents_table():
  """
    Prints a quick table for reference - spread to win probability conversion.
  """
  # Incrment spread from 0.5 to 20 in 0.5 increments.
  for i in range(1, 41):
      spread = i * 0.5
      win_prob = spread_to_win_probability(spread)
      print(f"  Spread: {spread:+.1f} -> Win Probability: {win_prob:.2%}")

def scrape_espn_odds(schedule):
  """
  Scrape ESPN odds for the given processed schedule.

  Parameters:
  schedule (list): Processed schedule as a list of game dictionaries.

  Returns:
  list: list of dictionaries containing the scraped ESPN odds for each game:
    - 'week': Week number of the game
    - 'home': Home team of the game
    - 'away': Away team of the game
    - 'favorite': Favorite team
    - 'opponent': Underdog team
    - 'spread': Point spread for the favorite (always positive)
    - 'o_u': Over/under total for the game
  """
 

  scraped_schedule = []
  weeks = sorted(set(g['week'] for g in schedule))
  headers = {'User-Agent': 'Mozilla/5.0'}

  for week in weeks:
    url = (f"https://site.api.espn.com/apis/site/v2/sports/football/nfl"
           f"/scoreboard?seasontype=2&week={week}")
    try:
      resp = requests.get(url, headers=headers, timeout=10)
      resp.raise_for_status()
      data = resp.json()
    except Exception as e:
      print(f"  Warning: failed to fetch week {week} odds: {e}")
      continue

    for event in data.get('events', []):
      competition = event['competitions'][0]
      competitors = competition.get('competitors', [])

      home_comp = next((c for c in competitors if c['homeAway'] == 'home'), None)
      away_comp = next((c for c in competitors if c['homeAway'] == 'away'), None)
      if not home_comp or not away_comp:
        continue

      home_name = home_comp['team']['displayName']
      away_name  = away_comp['team']['displayName']

      odds_list = competition.get('odds', [])
      if not odds_list:
        continue

      odds = odds_list[0]  # DraftKings is provider[0]
      spread_raw  = odds.get('spread')
      over_under  = odds.get('overUnder')
      home_is_fav = odds.get('homeTeamOdds', {}).get('favorite', False)
      away_is_fav = odds.get('awayTeamOdds', {}).get('favorite', False)

      if home_is_fav:
        favorite = home_name
        opponent = away_name
      elif away_is_fav:
        favorite = away_name
        opponent = home_name
      else:
        favorite = None
        opponent = None

      scraped_schedule.append({
        'week':     week,
        'home':     home_name,
        'away':     away_name,
        'favorite': favorite,
        'opponent': opponent,
        'spread':   abs(spread_raw) if spread_raw is not None else None,
        'o_u':      over_under,
      })

  # Build ESPN odds lookup keyed by (week, home, away)
  espn_lookup = {(g['week'], g['home'], g['away']): g for g in scraped_schedule}

  print(f"\n{'=' * 72}")
  print(f"  ESPN ODDS vs CSV DISCREPANCIES")
  print(f"{'=' * 72}")
  discrepancies_found = False
  for game in schedule:
      key = (game['week'], game['home'], game['away'])
      espn = espn_lookup.get(key)
      if espn is None:
          continue  # odds not yet posted for this game

      csv_spread   = game.get('spread')
      espn_spread  = espn.get('spread')
      csv_fav      = game.get('favorite')
      espn_fav     = espn.get('favorite')

      spread_diff   = (csv_spread is not None and espn_spread is not None and
                       abs(csv_spread - espn_spread) > 0.4)
      favorite_diff = (csv_fav != espn_fav and not (csv_fav is None and espn_fav is None))

      if spread_diff or favorite_diff:
          discrepancies_found = True
          print(f"\n  Wk {game['week']:>2}: {game['away']} @ {game['home']}")
          if favorite_diff:
              print(f"    Favorite  CSV={csv_fav or 'N/A':<30}  ESPN={espn_fav or 'N/A'}")
          if spread_diff:
              print(f"    Spread    CSV={csv_spread:<30}  ESPN={espn_spread}")

  if not discrepancies_found:
      print("  No discrepancies found.")
  print(f"{'=' * 72}")
  return scraped_schedule

# -- Main execution -------------------------------------------------------------

# Load and process the schedule data
schedule = process_schedule(SCHEDULE_FILE)
schedule_with_projections = calculate_win_probabilities(schedule, WEEK_NUM)

# Print the odds to percents table for reference
#print_odds_to_percents_table()

# Calculate the optimal picks for each entry based on the current schedule and projections
optimal_picks_results = calculate_optimal_picks(ENTRIES, schedule_with_projections, WEEK_NUM, stacking_threshold=STACKING_THRESHOLD, early_week_weight=EARLY_WEEK_WEIGHT, max_stack=MAX_STACK, min_stack_prob=MIN_STACK_PROB)
print_optimal_picks(optimal_picks_results)

# Monte Carlo analysis for stacking thresholds
# stacking_thresholds_to_test = [0.60, 0.65, 0.70, 0.75, 0.80]
# monte_carlo_stacking_results = monte_carlo_stacking_analysis(schedule_with_projections, stacking_thresholds_to_test, n_entries=len(ENTRIES), n_simulations=100_000, early_week_weight=EARLY_WEEK_WEIGHT)
# print_monte_carlo_results(monte_carlo_stacking_results, analysis_type="Stacking")

# Monte Carlo analysis for early week weights
# early_week_weights_to_test = [0.0, 0.5, 1.0, 1.5, 2.0]
# monte_carlo_weighted_results = monte_carlo_weighted_analysis(schedule_with_projections, early_week_weights_to_test, n_entries=len(ENTRIES), n_simulations=100_000, stacking_threshold=STACKING_THRESHOLD)
# print_monte_carlo_results(monte_carlo_weighted_results, analysis_type="Weighted")

#backtest_results = backtest(SCHEDULE_FILE, n_entries=len(ENTRIES), stacking_threshold=STACKING_THRESHOLD, early_week_weight=EARLY_WEEK_WEIGHT)
# Print backtest results for the specified stacking threshold and early week weight
# print(f"\nBacktest Results (Stacking Threshold: {STACKING_THRESHOLD}, Early Week Weight: {EARLY_WEEK_WEIGHT}):")
# for entry_id, result in backtest_results.items():
#     print(f"Entry {entry_id}:")
#     print(f"  Survival Probability: {result['survival_probability']:.2%}")
#     print("  Picks:")
#     for pick in result['picks']:
#         print(f"    Week {pick['week']}: {pick['team']} (Win Probability: {pick['win_probability']:.2%})")
#     print()
#
# Schedule Scrape & Print Discrepancies
#scraped_schedule = scrape_espn_odds(schedule)