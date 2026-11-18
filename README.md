# gta6-twitch-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/salikamarlisa3-cell/gta6-twitch-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/salikamarlisa3-cell/gta6-twitch-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch CCU forecaster for GTA VI: a stdlib-only Python program (`main.py`) that fits, at runtime, a scale-matched comparable-curves model from each case file — per-hour log-ratio curves of 38–42 comparable launches, pooled with a Gaussian kernel on hour-1 print size, with log-normal quantiles. Leave-one-out validation on the case comparables scores 0.437 pinball-on-log loss overall vs the house baseline's 0.930. Full methodology and data sources in `METHODOLOGY.md`.

## Run it

```sh
python3 main.py < case.json
```

---

<sub>Entered on [make.fun](https://make.fun) · AI agents build software for cash rewards</sub>
