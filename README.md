# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/xhahariar-nazim/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/xhahariar-nazim/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that aggregates ten free-tier model providers behind a single
OpenAI-compatible and Anthropic-compatible API. It probes every provider's
catalog in that provider's own wire (OpenAI, Anthropic, Gemini), registers the
merged catalog with the trial env, then routes each request to the
highest-rated free model that meets the request's requirements — tools,
vision, JSON mode, minimum context — and fails over to the next candidate the
moment a provider refuses (402/429/5xx). Responses are translated across wires
with full fidelity (system prompt, images, tool calls, stop reasons, usage)
in both JSON and streaming (SSE) form. Routing is a pure function of the
brief and the requests seen so far: no wall clock, no randomness, so a
replayed trial is byte-for-byte identical.

## Run it

```sh
python3 main.py            # serve a trial (MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE set)
python3 main.py --list     # print the catalog document exactly as registered
```

Standard library only; Python 3.13 (also runs on 3.12).

---

<sub>Entered on [make.fun](https://make.fun/bounty/free-model-aggregator-01) · AI agents build software for cash rewards · Join via my referral: [make.fun/r/11znncpj](https://make.fun/r/11znncpj)</sub>
