import json
import re
import os

INPUT_FILE = "mmlu_2_3_eval.json"


def extract_answer(text):
    """
    Extract the final multiple-choice answer (A/B/C/D).

    First looks for common answer formats such as:
    A
    (A)
    Answer: A
    The answer is A

    Falls back to the last standalone A/B/C/D.
    """

    if not text:
        return None

    text = text.upper()

    # Prefer explicit answer patterns
    patterns = [
        r"ANSWER\s*(?:IS|:)\s*\(?([A-D])\)?",
        r"FINAL\s*ANSWER\s*(?:IS|:)?\s*\(?([A-D])\)?",
        r"OPTION\s*\(?([A-D])\)?",
        r"\\boxed\s*\{\s*([A-D])\s*\}",
        r"\(([A-D])\)"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]

    # Fallback: last standalone A/B/C/D
    matches = re.findall(r"\b([A-D])\b", text)

    if matches:
        return matches[-1]

    return None


def analyze():

    if not os.path.exists(INPUT_FILE):
        print("File not found:", INPUT_FILE)
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    num_agents = 2
    num_rounds = 3

    # --------------------------------------------------
    # Categories
    # --------------------------------------------------

    wrong_to_correct = []
    correct_to_wrong = []
    correct_to_correct = []
    wrong_to_wrong = []

    total_results = []

    # --------------------------------------------------
    # Process questions
    # --------------------------------------------------

    for question, value in data.items():

        agent_contexts, correct_answer = value

        correct_answer = str(correct_answer).strip().upper()

        for agent_number, agent in enumerate(
            agent_contexts,
            start=1
        ):

            # Expected conversation structure:
            #
            # [1] = Initial assistant answer
            # [3] = Round 2 assistant answer
            # [5] = Final assistant answer

            try:
                initial_text = agent[1]["content"]
                round2_text = agent[3]["content"]
                final_text = agent[5]["content"]

            except (IndexError, KeyError, TypeError):
                continue

            initial_answer = extract_answer(initial_text)
            round2_answer = extract_answer(round2_text)
            final_answer = extract_answer(final_text)

            initial_correct = (
                initial_answer == correct_answer
            )

            round2_correct = (
                round2_answer == correct_answer
            )

            final_correct = (
                final_answer == correct_answer
            )

            result = {
                "question": question,
                "correct": correct_answer,
                "agent": agent_number,
                "initial": initial_answer,
                "round2": round2_answer,
                "final": final_answer
            }

            total_results.append(result)

            # --------------------------------------------------
            # Initial -> Final transition
            # --------------------------------------------------

            if not initial_correct and final_correct:

                wrong_to_correct.append(result)

            elif initial_correct and not final_correct:

                correct_to_wrong.append(result)

            elif initial_correct and final_correct:

                correct_to_correct.append(result)

            else:

                wrong_to_wrong.append(result)

    # --------------------------------------------------
    # Totals
    # --------------------------------------------------

    total = len(total_results)

    if total == 0:
        print("No valid agent results found.")
        return

    # --------------------------------------------------
    # Accuracy calculation
    # --------------------------------------------------

    initial_correct_count = sum(
        1 for r in total_results
        if r["initial"] == r["correct"]
    )

    round2_correct_count = sum(
        1 for r in total_results
        if r["round2"] == r["correct"]
    )

    final_correct_count = sum(
        1 for r in total_results
        if r["final"] == r["correct"]
    )

    initial_accuracy = initial_correct_count / total
    round2_accuracy = round2_correct_count / total
    final_accuracy = final_correct_count / total

    improvement = final_accuracy - initial_accuracy

    # --------------------------------------------------
    # Print main results
    # --------------------------------------------------

    print()
    print("==============================")
    print("MMLU DEBATE ANALYSIS")
    print("==============================")

    print("Number of agents:", num_agents)
    print("Number of debate rounds:", num_rounds)
    print("Unique questions:", len(data))
    print("Total agent results:", total)

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------

    print()
    print("==============================")
    print("ACCURACY")
    print("==============================")

    print(
        f"Initial Accuracy: "
        f"{initial_accuracy:.4f} "
        f"({initial_accuracy * 100:.2f}%)"
    )

    print(
        f"Round 2 Accuracy: "
        f"{round2_accuracy:.4f} "
        f"({round2_accuracy * 100:.2f}%)"
    )

    print(
        f"Final Accuracy: "
        f"{final_accuracy:.4f} "
        f"({final_accuracy * 100:.2f}%)"
    )

    print(
        f"Improvement: "
        f"{improvement:.4f} "
        f"({improvement * 100:.2f} percentage points)"
    )

    # --------------------------------------------------
    # Debate transitions
    # --------------------------------------------------

    print()
    print("==============================")
    print("DEBATE TRANSITIONS")
    print("==============================")

    print(
        f"Wrong -> Correct: "
        f"{len(wrong_to_correct)} "
        f"({len(wrong_to_correct) / total * 100:.2f}%)"
    )

    print(
        f"Correct -> Wrong: "
        f"{len(correct_to_wrong)} "
        f"({len(correct_to_wrong) / total * 100:.2f}%)"
    )

    print(
        f"Correct -> Correct: "
        f"{len(correct_to_correct)} "
        f"({len(correct_to_correct) / total * 100:.2f}%)"
    )

    print(
        f"Wrong -> Wrong: "
        f"{len(wrong_to_wrong)} "
        f"({len(wrong_to_wrong) / total * 100:.2f}%)"
    )

    # --------------------------------------------------
    # Agent-wise accuracy
    # --------------------------------------------------

    print()
    print("==============================")
    print("AGENT-WISE ACCURACY")
    print("==============================")

    for agent_number in range(1, num_agents + 1):

        agent_results = [
            r for r in total_results
            if r["agent"] == agent_number
        ]

        if not agent_results:
            continue

        agent_total = len(agent_results)

        agent_initial = sum(
            1 for r in agent_results
            if r["initial"] == r["correct"]
        )

        agent_round2 = sum(
            1 for r in agent_results
            if r["round2"] == r["correct"]
        )

        agent_final = sum(
            1 for r in agent_results
            if r["final"] == r["correct"]
        )

        print()
        print(f"Agent {agent_number}:")

        print(
            f"Initial: "
            f"{agent_initial / agent_total * 100:.2f}%"
        )

        print(
            f"Round 2: "
            f"{agent_round2 / agent_total * 100:.2f}%"
        )

        print(
            f"Final: "
            f"{agent_final / agent_total * 100:.2f}%"
        )

    # --------------------------------------------------
    # Examples helper
    # --------------------------------------------------

    def show_examples(title, items, limit=5):

        print()
        print("==============================")
        print(title)
        print("==============================")

        if not items:
            print("None")
            return

        for item in items[:limit]:

            print()
            print("Question:", item["question"])
            print("Correct:", item["correct"])
            print("Agent:", item["agent"])

            print(
                "Initial:",
                item["initial"],
                "✓" if item["initial"] == item["correct"] else "✗"
            )

            print(
                "Round 2:",
                item["round2"],
                "✓" if item["round2"] == item["correct"] else "✗"
            )

            print(
                "Final:",
                item["final"],
                "✓" if item["final"] == item["correct"] else "✗"
            )

    # --------------------------------------------------
    # Examples
    # --------------------------------------------------

    show_examples(
        "WRONG -> CORRECT EXAMPLES",
        wrong_to_correct
    )

    show_examples(
        "CORRECT -> WRONG EXAMPLES",
        correct_to_wrong
    )

    show_examples(
        "CORRECT -> CORRECT EXAMPLES",
        correct_to_correct
    )

    show_examples(
        "WRONG -> WRONG EXAMPLES",
        wrong_to_wrong
    )


if __name__ == "__main__":
    analyze()