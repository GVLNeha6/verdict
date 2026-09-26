import openai


class JudgeService:

    def __init__(
        self,
        api_key,
        model="gemini-3.5-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai"
    ):
        openai.api_key = api_key
        openai.api_base = base_url

        self.model = model

    def _ask(self, prompt):
     import time

     max_retries = 5

     for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            return response["choices"][0]["message"]["content"]

        except openai.error.APIError as e:
            error_text = str(e)

            if "429" in error_text or "quota" in error_text.lower():
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

    def judge(
        self,
        claim,
        evidence,
        debate_history
    ):

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

        final_debate = debate_history[-1]

        prompt = f"""
You are the final Judge in an evidence-based claim verification system.

Your task is to determine whether the claim is supported, refuted,
or not sufficiently supported by the provided evidence.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{evidence_text}

FINAL AGENT 1 REASONING:
{final_debate["agent_1"]}

FINAL AGENT 2 REASONING:
{final_debate["agent_2"]}

Evaluate the claim strictly against the supplied evidence.

Use exactly one of these verdict labels:

SUPPORTS
REFUTES
NOT ENOUGH INFO

Rules:

- SUPPORTS: The evidence directly supports the claim.
- REFUTES: The evidence directly contradicts the claim.
- NOT ENOUGH INFO: The evidence does not provide enough information
  to determine whether the claim is true or false.
- Do not use outside knowledge.
- Do not invent evidence.
- Do not choose a verdict simply because both agents agree.
- Base the decision on the actual evidence.

Return your answer in exactly this format:

VERDICT: <SUPPORTS / REFUTES / NOT ENOUGH INFO>

CONFIDENCE: <number between 0 and 1>

EXPLANATION: <short explanation based only on the evidence>

EVIDENCE_USED: <list the evidence numbers used, for example Evidence 1, Evidence 3>
"""

        return self._ask(prompt)