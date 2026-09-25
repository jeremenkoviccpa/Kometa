# Deploying Kometa: backend on Railway, frontend on Vercel

The backend (engine, risk gate, execution, API and the hub page itself) runs as one always-on container on
Railway. The Vercel site is the same hub page, forwarding `/api` and `/research` to Railway server-side, so
the browser talks to one address and no cross-site (CORS) setup is needed. Either URL works.

## Railway (backend)

1. `railway login`, then from the repository root: `railway init` (new project) and `railway up`.
   Railway builds the `Dockerfile` and applies `railway.json` (restart always, health check on `/`).
2. Add a **volume** mounted at `/data` (state: journal, risk state, registry, audit). Without it a restart
   forgets everything, including a halt.
3. Variables:
   - `AT_API_OWNER_TOKEN` (required): 32+ random characters; the hub's access token for every API call.
     Generate one locally: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`. Keep it in your
     password manager; the server never prints it.
   - Simulated gold is the default (`AT_DEMO_BROKER=sim`); optional `AT_DEMO_SPEED` (simulated seconds per
     second, default 60), `AT_DEMO_RUN_DAYS`.
4. Generate a public domain for the service (Railway: Settings, Networking, Generate Domain).

### Switching to the OANDA practice account

Add `AT_DEMO_BROKER=oanda`, `AT_ENV=paper`, `AT_OANDA_TOKEN`, `AT_OANDA_ACCOUNT`, `AT_OANDA_ENVIRONMENT=practice`,
and the owner-signed paper limits: sign locally (`uv run at risk sign config/risk.paper.yaml`), then commit
`config/risk.paper.yaml.sig` and your public key (`config/owner_ed25519.pub`, `AT_OWNER_PUBLIC_KEY_PATH`).
Signatures and public keys are not secrets; the private key never leaves your machine.

## Vercel (frontend)

    uv run python scripts/build_web.py https://<your-service>.up.railway.app
    cd web && vercel deploy --prod

Open the Vercel URL, paste the access token when asked (it stays in that browser tab).

## Security notes

- Every `/api` and `/research` request needs the token when the hub is public; the process refuses to start
  on `0.0.0.0` without `AT_API_OWNER_TOKEN`. Controls (retire, pause learning, paper trading switch) are
  audited. No endpoint changes risk limits.
- Never set `AT_ENV=live` on a hosted demo.
