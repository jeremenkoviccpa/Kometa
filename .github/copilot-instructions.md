# Instructions for GitHub Copilot: running Kometa on Windows

Kometa is an automated trading platform (Python 3.12, uv workspace). The source of truth is
`TRADING_SYSTEM_SPEC.md`; the project rules are in `CLAUDE.md` and apply to you too.

## Where to run: WSL2, never native Windows

Kometa uses Unix file locking (`fcntl`) and Unix paths, so it does **not** run in PowerShell or cmd. Always work
inside WSL2 (Ubuntu):

- In VS Code, install the "WSL" extension, then "WSL: Connect to WSL" and open the project folder from the
  Linux home (for example `~/Kometa`), not from `/mnt/c/...` (much slower, and file locking differs).
- Your terminal must show a Linux prompt (`user@machine:~/Kometa$`). If it shows `PS C:\...>`, you are in the
  wrong shell: run `wsl` first or reopen the folder through WSL.

If WSL is missing, the user runs once in an Administrator PowerShell: `wsl --install -d Ubuntu`, restarts, and
opens Ubuntu to create a Linux user. Do not try to work around WSL.

## One-time setup (inside WSL)

```bash
sudo apt update && sudo apt install -y git build-essential tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
cd ~/Kometa            # or wherever the repository was cloned inside WSL
uv sync --python 3.12
```

## Check that everything works

```bash
uv run make check      # ruff + mypy --strict + the full test suite; takes a few minutes
```

- Postgres tests are skipped unless `AT_TEST_DATABASE_URL` is set; that is fine locally.
- `make check` must be green before you call any change done. Never skip, disable or weaken a failing test
  to make it pass; fix the cause or report it.

## Run the trading hub on simulated gold (no account needed)

```bash
uv run at demo run --port 8420
```

Then open http://localhost:8420 in the Windows browser (WSL forwards the port). The owner token for the hub's
controls is printed at start. Stop with Ctrl-C. Options: `--speed 60` (one market minute per second),
`--trade swing_trend_pullback scalp_session_breakout` (which strategies paper trade), `--symbol XAUUSD`.

To keep it running after the terminal closes: `tmux new -s kometa`, run the command, detach with Ctrl-B then D.

## Paper trade on an OANDA practice (demo) account

The user must do these steps themselves; never ask for, print, log or commit their token:

1. Open an OANDA practice account, create an API token, enable hedging if offered.
2. Create `.env` in the repository root (it is git-ignored):

   ```
   AT_ENV=paper
   AT_OANDA_TOKEN=<their token>
   AT_OANDA_ACCOUNT=<practice account id>
   AT_OANDA_ENVIRONMENT=practice
   ```

3. Owner key and signed paper limits (the private key stays on this machine; never commit it):

   ```bash
   uv run at risk keygen
   uv run at risk sign config/risk.paper.yaml
   ```

Then:

```bash
uv run at data fetch --source oanda --symbol XAUUSD --from 2019-01-01   # real history, cached in data/oanda
uv run at validate strategies/library/swing_trend_pullback --data data/oanda   # tuning + validation report
uv run at demo run --broker oanda --port 8420                            # paper trading on real prices
```

The hub's top bar must read "PAPER · DEMO ACCOUNT". Kometa refuses to start on a real-money account in paper
mode, on an account without hedging, or with a bad token; report such errors to the user instead of working
around them.

More detail for humans: `docs/windows_setup.md` and `docs/runbook.md`.

## Rules you must follow

- **Never trade real money.** Only the simulator (`--broker sim`) and OANDA **practice** accounts. Do not set
  `AT_ENV=live` or `AT_OANDA_ENVIRONMENT=live`.
- **Never edit or re-sign `config/risk.yaml`, `config/validation.yaml` or `config/promotion.yaml`**, and never
  loosen any risk check. The risk gate can only make trades smaller.
- **Secrets:** `.env`, `secrets/`, keys and tokens are never committed, printed or pasted anywhere.
- **Strategies** live in `strategies/library/<id>/` and are candidates until `at validate`, shadow and micro
  say otherwise. Never claim a strategy makes money; never tune on the holdout year.
- **Every change ships with tests**, and `uv run make check` must pass. Log design decisions in
  `docs/decisions.md` and unclear spec points in `docs/open_questions.md`.
- Imports between packages follow `tests/unit/test_import_rules.py`; strategies may import only what
  `packages/strategies_api/src/autotrader/strategies_api/static_checks.py` allows.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: fcntl` | You are in native Windows. Work inside WSL. |
| `uv: command not found` | Run `source ~/.bashrc` or reopen the WSL terminal after installing uv. |
| Port 8420 busy | Use `--port 8430` (any free port). |
| Hub shows "HALTED · RECON HALT" | Kometa and the broker disagree about positions or cash; see the Alerts tab and `docs/runbook.md`. Do not delete `var/` files to clear it. |
| `put AT_OANDA_TOKEN and AT_OANDA_ACCOUNT in .env` | The user has not created `.env` yet (step 2 above). |
| Very slow tests or file operations | The project is under `/mnt/c`; clone it into the WSL home instead. |
