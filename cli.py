import argparse
from agentforge import Agent, tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai:gpt-3.5-turbo")
    args = parser.parse_args()

    agent = Agent(model=args.model, tools=[tools.calc()])
    out = agent.run("What is 123*456?")
    print(out)


if __name__ == "__main__":
    main()


    