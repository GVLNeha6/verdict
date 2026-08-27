import json
import re

BIO_FILE = "biography_gemini_2_3.json"
ARTICLE_FILE = "article.json"

THRESHOLD = 0.60


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_bullets(text):
    lines = text.split("\n")
    bullets = []

    for line in lines:
        line = line.strip()

        line = re.sub(r"^[\s\*\-•]+", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        if len(line) > 20:
            bullets.append(line)

    return bullets


def fact_supported(fact, ground_truth):

    fact_words = set(normalize(fact).split())
    truth_words = set(normalize(ground_truth).split())

    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in",
        "on", "for", "with", "was", "is", "his", "her",
        "he", "she", "they", "their", "from", "as", "at",
        "by", "who", "that", "this", "has", "have", "had",
        "been", "were", "are", "be", "also", "into",
        "over", "more", "than"
    }

    fact_words = {
        w for w in fact_words
        if w not in stopwords and len(w) > 2
    }

    if not fact_words:
        return False

    overlap = len(fact_words & truth_words)
    similarity = overlap / len(fact_words)

    return similarity >= 0.45


def biography_score(bio, ground_truth):

    bullets = extract_bullets(bio)

    if not bullets:
        return 0.0

    supported = 0

    for bullet in bullets:
        if fact_supported(bullet, ground_truth):
            supported += 1

    return supported / len(bullets)


def get_round_answers(agent_history):

    assistant_messages = [
        msg["content"]
        for msg in agent_history
        if msg.get("role") == "assistant"
    ]

    if len(assistant_messages) < 3:
        return None, None, None

    return (
        assistant_messages[0],
        assistant_messages[1],
        assistant_messages[2]
    )


def main():

    print()
    print("==============================")
    print("BIOGRAPHY DEBATE ANALYSIS")
    print("==============================")

    # -------------------------
    # Load files
    # -------------------------

    with open(BIO_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)

    with open(ARTICLE_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # -------------------------
    # Normalize ground truth
    # -------------------------

    gt = {}

    for name, description in ground_truth.items():
        clean_name = name.split("(")[0].strip()
        gt[clean_name] = description

    # -------------------------
    # Categories
    # -------------------------

    wrong_to_correct = []
    correct_to_wrong = []
    correct_to_correct = []
    wrong_to_wrong = []

    total_results = []

    initial_scores = []
    round2_scores = []
    final_scores = []

    # -------------------------
    # Agent-wise data
    # -------------------------

    agent_data = {}

    # -------------------------
    # Analyze
    # -------------------------

    for person, agent_results in responses.items():

        clean_person = person.split("(")[0].strip()

        if clean_person not in gt:
            print("Skipping:", person)
            continue

        reference = gt[clean_person]

        for agent_number, history in enumerate(
            agent_results,
            start=1
        ):

            initial, round2, final = get_round_answers(history)

            if initial is None:
                continue

            initial_score = biography_score(
                initial,
                reference
            )

            round2_score = biography_score(
                round2,
                reference
            )

            final_score = biography_score(
                final,
                reference
            )

            initial_correct = initial_score >= THRESHOLD
            round2_correct = round2_score >= THRESHOLD
            final_correct = final_score >= THRESHOLD

            result = {
                "person": person,
                "agent": agent_number,
                "initial_score": initial_score,
                "round2_score": round2_score,
                "final_score": final_score,
                "initial": initial,
                "round2": round2,
                "final": final
            }

            total_results.append(result)

            initial_scores.append(initial_score)
            round2_scores.append(round2_score)
            final_scores.append(final_score)

            # -------------------------
            # Agent-wise statistics
            # -------------------------

            if agent_number not in agent_data:
                agent_data[agent_number] = {
                    "initial": [],
                    "round2": [],
                    "final": []
                }

            agent_data[agent_number]["initial"].append(
                initial_correct
            )

            agent_data[agent_number]["round2"].append(
                round2_correct
            )

            agent_data[agent_number]["final"].append(
                final_correct
            )

            # -------------------------
            # Initial -> Final
            # -------------------------

            if not initial_correct and final_correct:

                wrong_to_correct.append(result)

            elif initial_correct and not final_correct:

                correct_to_wrong.append(result)

            elif initial_correct and final_correct:

                correct_to_correct.append(result)

            else:

                wrong_to_wrong.append(result)

    # -------------------------
    # Basic information
    # -------------------------

    total = len(total_results)

    print()
    print("==============================")
    print("EXPERIMENT CONFIGURATION")
    print("==============================")

    print("Number of agents:", 2)
    print("Number of debate rounds:", 3)
    print("Unique questions:", len(responses))
    print("Total agent results:", total)

    # -------------------------
    # Accuracy
    # -------------------------

    initial_correct_count = sum(
        1 for score in initial_scores
        if score >= THRESHOLD
    )

    round2_correct_count = sum(
        1 for score in round2_scores
        if score >= THRESHOLD
    )

    final_correct_count = sum(
        1 for score in final_scores
        if score >= THRESHOLD
    )

    initial_accuracy = initial_correct_count / total
    round2_accuracy = round2_correct_count / total
    final_accuracy = final_correct_count / total

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

    # -------------------------
    # Average factual score
    # -------------------------

    average_initial = sum(initial_scores) / total
    average_round2 = sum(round2_scores) / total
    average_final = sum(final_scores) / total

    print()
    print("==============================")
    print("AVERAGE FACTUAL SCORE")
    print("==============================")

    print(
        f"Initial Average Score: "
        f"{average_initial:.4f}"
    )

    print(
        f"Round 2 Average Score: "
        f"{average_round2:.4f}"
    )

    print(
        f"Final Average Score: "
        f"{average_final:.4f}"
    )

    print(
        f"Average Improvement: "
        f"{average_final - average_initial:.4f}"
    )

    # -------------------------
    # Debate transitions
    # -------------------------

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

    # -------------------------
    # Agent-wise accuracy
    # -------------------------

    print()
    print("==============================")
    print("AGENT-WISE ACCURACY")
    print("==============================")

    for agent_number, values in agent_data.items():

        initial_acc = (
            sum(values["initial"])
            / len(values["initial"])
        )

        round2_acc = (
            sum(values["round2"])
            / len(values["round2"])
        )

        final_acc = (
            sum(values["final"])
            / len(values["final"])
        )

        print()
        print(f"Agent {agent_number}:")

        print(
            f"Initial: {initial_acc * 100:.2f}%"
        )

        print(
            f"Round 2: {round2_acc * 100:.2f}%"
        )

        print(
            f"Final: {final_acc * 100:.2f}%"
        )

    # -------------------------
    # Examples
    # -------------------------

    def show_examples(title, items, limit=3):

        print()
        print("==============================")
        print(title)
        print("==============================")

        if not items:
            print("None")
            return

        for item in items[:limit]:

            print()
            print("Person:", item["person"])
            print("Agent:", item["agent"])

            print(
                "Initial Score:",
                round(item["initial_score"], 3)
            )

            print(
                "Round 2 Score:",
                round(item["round2_score"], 3)
            )

            print(
                "Final Score:",
                round(item["final_score"], 3)
            )

            print("\nInitial:")
            print(item["initial"])

            print("\nRound 2:")
            print(item["round2"])

            print("\nFinal:")
            print(item["final"])

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
    main()