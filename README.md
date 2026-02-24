# Rocket League Catch, Flick, Wall, Aerial & Reset Bot (RLBot)

This repository contains a **Python RLBot agent** focused on:

- consistent catches / dribbles / flicks,
- full-boost wall setups into air-dribbles,
- fast ground aerials,
- stronger defense,
- **air-roll-heavy aerial control** and **flip-reset attempts**.

## Behavior overview

- **Defense first** when own goal is threatened.
- **Fast aerial mode** for quick high-ball contests.
- **Flip-reset mode** for offensive high balls in reachable space:
  - targets under-ball approach,
  - uses heavy air-roll control,
  - follows a jump/release/jump+boost sequence and reset follow-through.
- **Wall air-dribble mode** when 100 boost + side-lane setup conditions are met.
- fallback to catch/dribble/flick when no advanced mode is active.

## Quick start

1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Load `bot.cfg` through RLBotGUI / match config.

## Tuning knobs

Open `rocketleague_bot/catch_flick_bot.py` and tune:

- `RESET_COOLDOWN`
- `AERIAL_COOLDOWN`
- reset timing in `step_flip_reset`
- air-roll intensity in `step_fast_aerial` / `step_flip_reset`
- defense thresholds in `should_do_emergency_defense`

## Notes

Flip resets are mechanically difficult; this bot uses heuristics for frequent and consistent attempts, but results still depend on game state and contact quality.
