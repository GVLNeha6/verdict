import time

import openai


class DebateService:

    def __init__(
        self,
        api_key,
        model="gemini-3.5-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        agents=2
    ):
        openai.api_key = api_key
        openai.api_base = base_url

        self.model = model
        self.agents = agents

    def _ask(self, messages):

        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]

        max_retries = 5

        for attempt in range(max_retries):

            try:

                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    request_timeout=90
                )

                return response["choices"][0]["message"]["content"]

            except openai.error.Timeout:

                if attempt == max_retries - 1:
                    raise

                wait_time = 20

                print(
                    f"Request timed out. "
                    f"Waiting {wait_time} seconds before retry "
                    f"({attempt + 1}/{max_retries})..."
                )

                time.sleep(wait_time)

            except openai.error.APIError as e:

                error_text = str(e)

                if (
                    "429" in error_text
                    or "quota" in error_text.lower()
                ):

                    if attempt == max_retries - 1:
                        raise

                    wait_time = 20

                    print(
                        f"Gemini rate limit reached. "
                        f"Waiting {wait_time} seconds before retry "
                        f"({attempt + 1}/{max_retries})..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

            except openai.error.ServiceUnavailableError:

                if attempt == max_retries - 1:
                    raise

                wait_time = 10

                print(
                    f"Gemini server unavailable. "
                    f"Waiting {wait_time} seconds before retry..."
                )

                time.sleep(wait_time)

    def _format_evidence(self, evidence):

        if not evidence:
            return ""

        evidence_text = "\n\n".join(
            [
                (
                    f"Evidence {i + 1}:\n"
                    f"Title: {item['title']}\n"
                    f"Text: {item['text']}"
                )
                for i, item in enumerate(evidence)
            ]
        )

        return (
            "\n\nRetrieved evidence:\n"
            + evidence_text
        )

    def run_debate(
        self,
        claim,
        evidence=None,
        rounds=3
    ):

        evidence_text = self._format_evidence(evidence)

        # Each agent has its own independent conversation context.
        agent_contexts = []

        for _ in range(self.agents):

            initial_prompt = f"""
You are participating as an independent agent
in a multi-agent reasoning process.

Claim:
{claim}
{evidence_text}

Analyze the claim independently.

Provide your answer and reasoning.
"""

            agent_contexts.append(
                [
                    {
                        "role": "user",
                        "content": initial_prompt
                    }
                ]
            )

        debate_history = []

        # Round 1: independent answers.
        # Later rounds: each agent sees the previous
        # round's answer from the other agents.
        for round_number in range(rounds):

            round_results = {}

            for agent_index in range(self.agents):

                context = agent_contexts[agent_index]

                if round_number > 0:

                    other_responses = []

                    for other_index in range(self.agents):

                        if other_index == agent_index:
                            continue

                        previous_response = agent_contexts[
                            other_index
                        ][-1]["content"]

                        other_responses.append(
                            f"""
Agent {other_index + 1}'s previous response:

{previous_response}
"""
                        )

                    debate_prompt = f"""
This is round {round_number + 1}
of the multi-agent reasoning process.

Claim:
{claim}
{evidence_text}

The following are the previous responses
from the other agents:

{"".join(other_responses)}

Use these responses as additional information.

Critically examine their reasoning.
If necessary, revise your answer.

Provide your updated answer and reasoning.
"""

                    context.append(
                        {
                            "role": "user",
                            "content": debate_prompt
                        }
                    )

                response = self._ask(context)

                context.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

                round_results[
                    f"agent_{agent_index + 1}"
                ] = response

            debate_history.append(
                {
                    "round": round_number + 1,
                    **round_results
                }
            )

        return debate_history