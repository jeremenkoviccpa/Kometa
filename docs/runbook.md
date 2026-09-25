# Runbook

Filled in Phase 11. Sections: start and stop, restart after full halt, broker outage, rotating keys, restoring from
backup.

## Local development
    make sync                 # install
    make check                # the gate: ruff, mypy --strict, pytest
    make db-up && make migrate  # needs Docker; secrets/ holds dev-only passwords
    make db-roles             # Grafana's read-only login (secrets/db_readonly_password.txt)
    uv run at demo run        # the whole system on a simulated market, dashboard on http://127.0.0.1:8000/
                              # (--port N if 8000 is taken)
    uv run at data synth      # synthetic M1 data into data/synthetic
    uv run at data check SYNTH

Ports already taken by another project: set AT_PG_PORT / AT_REDIS_PORT / AT_GRAFANA_PORT (and the port in
AT_DATABASE_URL) before `make db-up`. Postgres tests run when AT_TEST_DATABASE_URL is set (CI always sets it).

## MT5 bridge (Windows VPS)
The bridge runs next to the MT5 terminal and exposes the broker adapter over HTTP on the private tunnel only.

    # on the Windows VPS, with the MT5 terminal installed and logged in
    pip install MetaTrader5 && uv sync
    set AT_BRIDGE_TOKEN=<40+ random chars, same value as AT_BRIDGE_TOKEN on the Linux host>
    set AT_BRIDGE_SERVER_TZ=NY+7              # the broker's server time; never guessed (open question 19)
    set AT_BRIDGE_EXPECT_TRADE_MODE=demo      # demo for paper, real for live; the bridge refuses a mismatch
    set AT_BRIDGE_SYMBOLS=["EURUSD","XAUUSD"]
    set AT_BRIDGE_HOST=<tailscale or wireguard address>
    python -m autotrader.mt5_bridge

Check it from the Linux host (read-only, places nothing; run while the market is open, because broker server time
comes from the newest tick):

    AT_ENV=paper AT_BRIDGE_URL=http://<vps-tunnel-ip>:8765 uv run at execution check

It refuses on clock skew above 2 s, a real account in paper (or a demo account in live), a netting account, or an
account id different from `AT_ACCOUNT_ID`.

## Reconciliation halts (RECON_HALT)
- Broker unreachable: entries stop; the halt clears by itself on the first clean check after the bridge is back.
- Mismatch that the automatic resync could not explain (`recon_stuck` alert): the halt stays. Inspect the broker
  (positions, orders, deal history) against `var/execution_journal.json`, then run an owner resync (`Reconciler.owner_resync`;
  the control command is wired with the execution service in Phase 7), which adopts the broker's state as it is
  and clears the halt if everything then matches.
- A corrupt execution journal refuses startup. Do not delete it; restore it from backup or start with an owner
  resync against the broker.
- External positions (opened outside the system) are alerted once, never touched, and count against the risk limits.

## Strategy lifecycle
    uv run at lifecycle submit strategies/examples/demo_ma_cross   # demo_only: straight to shadow, capped
    uv run at lifecycle status
    uv run at lifecycle retire <strategy_id> <version> --reason "..."   # owner command, any stage
Stages change only through the evaluator (hourly and after each closed trade) or the owner's retire command.
Demotions cancel the version's pending entries at once; positions are closed when it leaves the money stages.

## Audit log and alerts
    uv run at audit verify              # the JSON Lines log (AT_AUDIT_PATH)
    uv run at audit verify --db         # the audit_log table in AT_DATABASE_URL
A break is a critical alert (`audit_chain_break`) from the scheduled check. The table refuses UPDATE, DELETE and
TRUNCATE and the application role may only INSERT and SELECT; a break therefore means someone with owner rights
edited it. Do not "repair" the chain: keep the database as evidence, restore the latest verified backup into a
scratch database, and compare.
- Critical alerts repeat every 10 minutes until acknowledged: `/ack` (all) or `/ack A12` in the owner's Telegram
  chat, or the API. Only the configured chat can acknowledge.
- Warnings worth acting on: `slippage_above_model` (the backtest cost model is too optimistic for that symbol;
  L8 recalibrates it in Phase 9), `data_quality` (a traded symbol's data has medium or high issues),
  `snapshot_write_failed` (dashboards stop updating; trading is unaffected).
- Dashboards: `scripts/dashboards.py` is the source (`make dashboards` regenerates the JSON). Grafana reads as
  `autotrader_readonly`, never as the owner.

