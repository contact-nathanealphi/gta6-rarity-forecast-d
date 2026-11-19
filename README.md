# contact-nathanealphi/gta6-rarity-forecast-d

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-d/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-d/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Rank/borrowing ensemble forecaster for GTA VI trophy earn rates on Exophase. It fits ONE global progression curve — the median logit rate across all 55 trophies as a function of log-time — plus per-trophy offsets shrunk toward their tier's typical offset (small launch-night holder counts are mostly noise, so the shrinkage is strong when few players hold a trophy). The forecast combines the fitted trend from the case history with priors from the launch-night refresh (tracked players 638 → 3,868, every rate but the 100% one falling): dilution per log-growth of tracked players, player-growth per log-time, and a refresh that may land inside the horizon. Offsets stretch apart with the horizon, so easy trophies drift up relative to the pack while rare ones drift further down. Spreads come from a random-walk term plus a refresh-jump allowance, widened by tier and by rarity. The 100% trophy and the ~0% platinum are handled explicitly.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `/output/<id>.out` (`$MAKEFUN_OUTPUT_DIR` else `/output/`). Python 3.13, standard library only, no network, deterministic.

---

Entered on [make.fun](https://make.fun/?ref=9ncgmbd6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
