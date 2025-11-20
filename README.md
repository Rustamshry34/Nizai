# agentforge


Lightweight open-source Python framework for building AI agents (MVP).


## Goals
- Minimal, developer-first API
- Pluggable LLM adapters (OpenAI, local)
- Tools (http, calc, python function)
- RAM memory (extendable with vector stores)


## Quickstart
```python
from agentforge import Agent, tools


agent = Agent(
model="openai:gpt-4o-mini",
tools=[tools.http(), tools.calc()],
max_steps=4,
)


print(agent.run("Find the latest AI news on TechCrunch and summarize."))
```


## Install (dev)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Project structure
See repository skeleton in repo root.


## Contributing
PRs welcome. Keep API backward compatible where possible.