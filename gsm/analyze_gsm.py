import json
import re


def extract_answer(text):
    """
    Extract the final numerical answer.
    First looks for \\boxed{answer}, then falls back
    to the last number in the response.
    """

    if not text:
        return None

    boxed = re.findall(
        r"\\boxed\{(-?\d+(?:\.\d+)?)\}",
        text
    )

    if boxed:
        return float(boxed[-1])

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if numbers:
        return float(numbers[-1])

    return None


def extract_ground_truth(answer):
    """
    GSM8K answers normally end with:
    #### number
    """

    matches = re.findall(
        r"####\s*(-?\d+(?:\.\d+)?)",
        answer
    )

    if matches:
        return float(matches[-1])

    return extract_answer(answer)


def is_correct(predicted, correct):
    if predicted is None:
        return False

    return abs(predicted - correct) < 1e-9


def main():

    filename = "gsm_2_gem3.json"

    # JSON file
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    wrong_to_correct = []
    correct_to_wrong = []
    correct_to_correct = []
    wrong_to_wrong = []

    all_results = []

    # Store round-wise correctness
    initial_correct_count = 0
    round2_correct_count = 0
    final_correct_count = 0

    agent_results = {}

    for question, value in data.items():

        responses, ground_truth = value

        correct_answer = extract_ground_truth(ground_truth)

        for agent_number, response in enumerate(responses, start=1):

            # Conversation structure:
            #
            # [1] initial assistant answer
            # [3] round 2 assistant answer
            # [5] final assistant answer

            try:
                initial_text = response[1]["content"]
                round2_text = response[3]["content"]
                final_text = response[5]["content"]
            except (IndexError, KeyError, TypeError):
                continue

            initial_answer = extract_answer(initial_text)
            round2_answer = extract_answer(round2_text)
            final_answer = extract_answer(final_text)

            initial_correct = is_correct(
                initial_answer,
                correct_answer
            )

            round2_correct = is_correct(
                round2_answer,
                correct_answer
            )

            final_correct = is_correct(
                final_answer,
                correct_answer
            )

            # Count accuracy
            if initial_correct:
                initial_correct_count += 1

            if round2_correct:
                round2_correct_count += 1

            if final_correct:
                final_correct_count += 1

            result = {
                "question": question,
                "correct": correct_answer,
                "agent": agent_number,
                "initial": initial_answer,
                "round2": round2_answer,
                "final": final_answer
            }

            all_results.append(result)

            # Store agent-wise results
            if agent_number not in agent_results:
                agent_results[agent_number] = {
                    "initial": 0,
                    "round2": 0,
                    "final": 0,
                    "total": 0
                }

            agent_results[agent_number]["total"] += 1

            if initial_correct:
                agent_results[agent_number]["initial"] += 1

            if round2_correct:
                agent_results[agent_number]["round2"] += 1

            if final_correct:
                agent_results[agent_number]["final"] += 1

            # Initial -> Final transitions
            if initial_correct and final_correct:
                correct_to_correct.append(result)

            elif initial_correct and not final_correct:
                correct_to_wrong.append(result)

            elif not initial_correct and final_correct:
                wrong_to_correct.append(result)

            else:
                wrong_to_wrong.append(result)

    total = len(all_results)

    if total == 0:
        print("No valid agent results found.")
        return

    initial_accuracy = initial_correct_count / total
    round2_accuracy = round2_correct_count / total
    final_accuracy = final_correct_count / total

    print()
    print("==============================")
    print("GSM8K DEBATE ANALYSIS")
    print("==============================")

    print("Number of agents:", len(agent_results))
    print("Number of debate rounds:", 3)
    print("Unique questions:", len(data))
    print("Total agent results:", total)

    print()
    print("==============================")
    print("ACCURACY")
    print("==============================")

    print(
        f"Initial Accuracy: {initial_accuracy:.4f} "
        f"({initial_accuracy * 100:.2f}%)"
    )

    print(
        f"Round 2 Accuracy: {round2_accuracy:.4f} "
        f"({round2_accuracy * 100:.2f}%)"
    )

    print(
        f"Final Accuracy: {final_accuracy:.4f} "
        f"({final_accuracy * 100:.2f}%)"
    )

    print(
        f"Improvement: "
        f"{final_accuracy - initial_accuracy:.4f} "
        f"({(final_accuracy - initial_accuracy) * 100:.2f} percentage points)"
    )

    print()
    print("==============================")
    print("DEBATE TRANSITIONS")
    print("==============================")

    print(
        f"Wrong -> Correct: {len(wrong_to_correct)} "
        f"({len(wrong_to_correct) / total * 100:.2f}%)"
    )

    print(
        f"Correct -> Wrong: {len(correct_to_wrong)} "
        f"({len(correct_to_wrong) / total * 100:.2f}%)"
    )

    print(
        f"Correct -> Correct: {len(correct_to_correct)} "
        f"({len(correct_to_correct) / total * 100:.2f}%)"
    )

    print(
        f"Wrong -> Wrong: {len(wrong_to_wrong)} "
        f"({len(wrong_to_wrong) / total * 100:.2f}%)"
    )

    print()
    print("==============================")
    print("WRONG -> CORRECT EXAMPLES")
    print("==============================")

    for item in wrong_to_correct[:10]:

        print()
        print("Question:", item["question"])
        print("Correct:", item["correct"])
        print("Agent", item["agent"])
        print("Initial:", item["initial"], "✗")
        print("Round 2:", item["round2"])
        print("Final:", item["final"], "✓")

    print()
    print("==============================")
    print("CORRECT -> WRONG EXAMPLES")
    print("==============================")

    for item in correct_to_wrong[:10]:

        print()
        print("Question:", item["question"])
        print("Correct:", item["correct"])
        print("Agent", item["agent"])
        print("Initial:", item["initial"], "✓")
        print("Round 2:", item["round2"])
        print("Final:", item["final"], "✗")

    print()
    print("==============================")
    print("CORRECT -> CORRECT EXAMPLES")
    print("==============================")

    for item in correct_to_correct[:5]:

        print()
        print("Question:", item["question"])
        print("Correct:", item["correct"])
        print("Agent", item["agent"])
        print("Initial:", item["initial"], "✓")
        print("Round 2:", item["round2"])
        print("Final:", item["final"], "✓")

    print()
    print("==============================")
    print("WRONG -> WRONG EXAMPLES")
    print("==============================")

    for item in wrong_to_wrong[:5]:

        print()
        print("Question:", item["question"])
        print("Correct:", item["correct"])
        print("Agent", item["agent"])
        print("Initial:", item["initial"], "✗")
        print("Round 2:", item["round2"])
        print("Final:", item["final"], "✗")

    print()
    print("==============================")
    print("AGENT-WISE ACCURACY")
    print("==============================")

    for agent, stats in agent_results.items():

        total_agent = stats["total"]

        print()
        print(f"Agent {agent}:")

        print(
            f"Initial: "
            f"{stats['initial'] / total_agent * 100:.2f}%"
        )

        print(
            f"Round 2: "
            f"{stats['round2'] / total_agent * 100:.2f}%"
        )

        print(
            f"Final: "
            f"{stats['final'] / total_agent * 100:.2f}%"
        )


if __name__ == "__main__":
    main()