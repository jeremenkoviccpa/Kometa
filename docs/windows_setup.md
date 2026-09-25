# Running Kometa on Windows (paper trading with OANDA)

(For GitHub Copilot in VS Code the same steps, as instructions it follows, are in
`.github/copilot-instructions.md`.)

Kometa runs inside WSL2, the Linux that is built into Windows 10/11. With OANDA there is no MetaTrader and no
bridge: everything is one program talking to OANDA's web API. Keep the machine on and awake while it trades
(gold trades Sunday 23:00 to Friday 22:00 UTC); if it sleeps, trading pauses, and on the next start Kometa
reconciles with OANDA before it trades again.

## 1. One-time setup

1. **WSL2 + Ubuntu.** Open PowerShell as Administrator and run:

       wsl --install -d Ubuntu

   Restart when asked, open "Ubuntu" from the Start menu, and choose a Linux user name and password.

2. **Tools (inside Ubuntu).**

       sudo apt update && sudo apt install -y git build-essential
       curl -LsSf https://astral.sh/uv/install.sh | sh
       source ~/.bashrc

3. **The project.** Clone it into your Linux home (not /mnt/c: it is much slower there):

       git clone https://github.com/jeremenkoviccpa/Kometa.git ~/Kometa
       cd ~/Kometa
       uv sync --python 3.12
       uv run make check        # optional: the full test suite, a few minutes

4. **Stop Windows from sleeping** while it trades: Settings > System > Power > Screen and sleep > "Never" when
   plugged in.

## 2. OANDA practice account

1. Open a free practice (demo) account at oanda.com.
2. In "Manage API Access", generate a token.
3. In the account settings, enable hedging if it is offered (Kometa needs one position per trade; with
   hedging off, it refuses to start and says why).
4. Create `~/Kometa/.env` with your own values (this file is never committed and never printed):

       AT_ENV=paper
       AT_OANDA_TOKEN=<your token>
       AT_OANDA_ACCOUNT=<your practice account id, like 101-004-1234567-001>
       AT_OANDA_ENVIRONMENT=practice

## 3. Your owner key and the paper risk limits

The risk gate only runs with limits that you signed. The private key stays on your machine.

    uv run at risk keygen                        # writes your key pair (keep the private key private)
    uv run at risk sign config/risk.paper.yaml   # you approve the paper-trading limits

## 4. Tune on real history (optional, recommended)

    uv run at data fetch --source oanda --symbol XAUUSD --from 2019-01-01
    uv run at validate strategies/library/swing_trend_pullback --data data/oanda

Results appear in the hub's Research tab.

## 5. Paper trade

    uv run at demo run --broker oanda --port 8420

Open http://localhost:8420 in a Windows browser (WSL2 forwards the port). The top bar shows
"PAPER · DEMO ACCOUNT". The owner token for the hub's controls is printed at start.

To keep it running when you close the window, start it inside `tmux`:

    sudo apt install -y tmux
    tmux new -s kometa          # run the command above inside; detach with Ctrl-B then D
    tmux attach -t kometa       # come back later

## What to expect on the first runs

- Nothing trades money you can lose: the practice account is play money and every order still passes the
  risk gate at the paper stage's minimum size.
- The first run against OANDA is also the acceptance test of the connector. Watch the Alerts tab: a
  reconciliation halt means Kometa and OANDA disagree about positions or cash, and trading stops until they
  match (open question 32 is the likely place).
- Contract details in `config/instruments.yaml` are placeholders; compare them with OANDA's gold (XAU_USD:
  1 unit = 1 oz, so 1 Kometa lot = 100 units) and commission before trusting P&L to the cent.
