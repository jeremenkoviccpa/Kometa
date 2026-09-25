"""The owner's Claude trading tracks (owner decision 2026-09-26, docs/decisions.md).

Claude trades the owner's SMC method in two tracks the owner switches on and off in the hub:
`claude_smc_judge` decides on each setup the coded rules find, `claude_smc_free` reads the charts every
15 minutes and may open its own trades. What stays in code, whatever Claude says: the hard rules of the
method (stop side and width, RR >= 3 at the current price), the risk gate (it sizes every trade and can only
make it smaller), demo/paper only (the service refuses to call Claude when AT_ENV=live).
"""
