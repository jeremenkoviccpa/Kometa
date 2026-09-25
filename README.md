# Kometa

An automated trading platform for FX and gold: it tests trading strategies, decides how much each may risk,
sends orders, watches every fill, and demotes or retires strategies that stop working. Strategies are plugins;
every order passes an independent, owner-signed risk gate that can only make trades smaller.

- **Run it on Windows:** `docs/windows_setup.md` (WSL2). GitHub Copilot follows `.github/copilot-instructions.md`.
- **Try it without an account:** `uv sync --python 3.12` then `uv run at demo run --port 8420` and open
  http://localhost:8420 (simulated gold, paper trading).
- **Paper trade on real prices:** an OANDA practice account, see `docs/windows_setup.md`.
- **Design and rules:** `TRADING_SYSTEM_SPEC.md` (source of truth), `CLAUDE.md`, `docs/decisions.md`,
  `docs/runbook.md`.

Nothing here is investment advice, and no strategy in this repository is validated to make money: every
candidate must pass walk-forward validation, shadow trading and micro size before it risks real capital.
