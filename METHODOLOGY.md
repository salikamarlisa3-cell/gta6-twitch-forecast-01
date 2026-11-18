# Methodology — gta6-twitch-ccu-forecast-01

## What the entry is

`main.py` (Python 3.13, standard library only) is a forecasting *program*: the
grader runs it once per case file (`<id>.in`), and it writes one forecast
(`<id>.out`) — for each of hours 2–24 of the release day (UTC), five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of that hour's Twitch concurrent viewers in the
game's category. It is deterministic (no randomness, clock or network), runs a
practice case in ~0.05 s of the 8 s allowed, and two runs are byte-identical.

## The one idea

A launch day's hourly viewer curve is mostly a *scaled copy* of other launches'
curves — but the shape depends on the launch's scale, because scale is a proxy
for unlock phase. In the case data itself (37 comparable launches with complete
release-day curves):

- Launches whose hour 1 is already huge (Elden Ring 875,155; Cyberpunk 2077
  851,999; Diablo IV 533,455 — games unlocked across most of the world before
  00:00 UTC, or with early access) are front-loaded: hour 1 is 75–100% of the
  day's peak, and the curve decays or holds.
- Launches whose hour 1 is tiny (Hollow Knight: Silksong 164; Kingdom Come:
  Deliverance II 414; Black Myth: Wukong 4,941 — a single global unlock later
  in the UTC day) ramp 10–1000× as their unlock hour arrives.

Grand Theft Auto VI unlocks at midnight local time, country by country — New
Zealand first at 2026-11-18T11:00:00Z, central Europe at 23:00 UTC, the UK at
00:00 UTC, the US east coast at 05:00 UTC and the west coast at 08:00 UTC
(competition rules) — so by its hour 1 (00:00–01:00 UTC on 2026-11-19) most of
the world is already playing and its first hour will be one of the largest in
the comparable set. The forecast should therefore be shaped mostly by the
other *large, mature* launches, not by an equal pool of all launches.

## The model

For each comparable in the case file, the program rebuilds its release-day
curve and takes, per hour h, the log ratio log(v_h / v_1). Per hour, those log
ratios are pooled across comparables with a Gaussian kernel weight on
(log v_1 of the comparable − log p_1 of the case), bandwidth 0.75 log units.
The hour's forecast distribution is log-normal:

- centre = ½ · (kernel-weighted median + kernel-weighted mean) of the log ratios
- spread = kernel-weighted standard deviation of the log ratios, floored at
  0.09 · √(h−1) so late hours keep a sane band
- q_τ(h) = p_1 · exp(centre_h + z_τ · spread_h), z from the standard normal

The case's own game is excluded from the pool (the final case lists the GTA VI
category's pre-release history as a comparable; its "release day" is a
trailer-premiere day, not a launch), as is any comparable with fewer than 12 of
its 24 release-day hours sampled.

## Validation (leave-one-out, the contest's own loss)

Each of the 37 usable comparables in the practice data was treated in turn as a
pseudo-case — its hour 1 as the known print, the model refitted on the other
36 — and scored with the contest's exact score (mean pinball loss on the log
scale over hours 2–24 × the five levels). Harnesses: `analysis/validate*.py`.

| model | all 37 | mature (h1 ≥ 25% of peak) | big (h1 ≥ 150k) |
|---|---|---|---|
| house baseline (persistence) | 0.930 | 0.265 | — |
| pooled empirical quantiles | 0.714 | 0.356 | — |
| regime split (unlock-phase proxy) | 0.905 | 0.298 | — |
| kNN on pre-day trajectory (k=12) | 0.730 | 0.305 | — |
| **this entry (scale kernel, bw 0.75)** | **0.437** | **0.215** | **0.222** |

Things that did *not* help and were dropped: conditioning on the pre-day level
or on growth into hour 1 (correlations with the hour-1/peak ratio are −0.17 and
+0.06 in the comparable set), a second kernel on the pre-day peak, and smoothing
the centre/spread across neighbouring hours (all within noise).

## Data sources

- **Model data (runtime):** only the case file the grader provides. Its
  comparables are past launches' hourly Twitch category averages from
  SullyGnome.com, read once by the competition house on 2026-11-17 (kit
  `input/NOTICE.txt`, which asks that the site not be bulk-scraped — it was not;
  no data was fetched from it for this entry). The final case's counts are the
  house recorder's once-a-minute reads of Twitch's public GraphQL directory
  count for the Grand Theft Auto VI category (game id 1867279146).
- **Context (read during development; informs the write-up, not the code):**
  - Competition `RULES.txt`: GTA VI's unlock schedule above; the house
    baseline's practice loss ≈ 0.387 (typical miss ≈ 82.8%).
  - GTA VI Trailer 1: 90,421,491 YouTube views in 24 h — a Guinness World
    Record for a videogame reveal (Guinness World Records, via Push Square).
    Trailer 2: 475 million cross-platform views in 24 h (Rockstar Games via
    The Hollywood Reporter). Pre-orders estimated at 12.13 million copies /
    ~$1.18B before launch (Sensor Tower estimate, via Forbes). These say the
    launch's *scale* is at the top of the comparable set; the kernel handles
    the shape from there.
  - Red Dead Redemption 2 (Rockstar's previous launch, a practice case here):
    $725M retail sell-through in its first three days (Take-Two Interactive,
    2018); its Twitch category's all-time peak viewers of 382,860 are recorded
    by SteamDB's Twitch panel — a reminder that even Rockstar's biggest prior
    launch peaked below what GTA VI's pre-release premiere spikes (1.4M hourly
    average in the comparable data, 2026-08-27 extended look) already reached.
