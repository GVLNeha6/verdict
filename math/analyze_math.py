import pickle
import re

# Load debate results
data = pickle.load(open("math_gem_agents2_rounds3.p", "rb"))


def extract_answer(text):
    """
    Extract the final numerical answer from the model response.
    """
    numbers = re.findall(r'(?<!\w)-?\d+(?:\.\d+)?', text)

    if not numbers:
        return None

    return float(numbers[-1])


# Transition categories
wrong_correct = []
correct_wrong = []
correct_correct = []
wrong_wrong = []


# Accuracy counters
total_agent_results = 0

initial_correct_count = 0
round2_correct_count = 0
final_correct_count = 0


for question_tuple, value in data.items():

    agent_contexts, correct_answer = value
    correct_answer = float(correct_answer)

    for agent_num, context in enumerate(agent_contexts, 1):

        # Collect assistant responses from all debate rounds
        answers = []

        for msg in context:
            if msg["role"] == "assistant":
                answer = extract_answer(msg["content"])

                if answer is not None:
                    answers.append(answer)

        # We need:
        # answers[0] = Initial
        # answers[1] = Round 2
        # answers[2] = Round 3 / Final

        if len(answers) < 3:
            continue

        initial = answers[0]
        round2 = answers[1]
        final = answers[2]

        total_agent_results += 1

        initial_correct = initial == correct_answer
        round2_correct = round2 == correct_answer
        final_correct = final == correct_answer

        # Accuracy counters
        if initial_correct:
            initial_correct_count += 1

        if round2_correct:
            round2_correct_count += 1

        if final_correct:
            final_correct_count += 1

        # Store transition information
        record = {
            "question": question_tuple,
            "correct": correct_answer,
            "agent": agent_num,
            "initial": initial,
            "round2": round2,
            "final": final
        }

        # Initial -> Final
        if not initial_correct and final_correct:
            wrong_correct.append(record)

        elif initial_correct and not final_correct:
            correct_wrong.append(record)

        elif initial_correct and final_correct:
            correct_correct.append(record)

        else:
            wrong_wrong.append(record)


# ==============================
# RESULTS
# ==============================

print("\n==============================")
print("MATH DEBATE ANALYSIS")
print("==============================")

print("Number of agents: 2")
print("Number of debate rounds: 3")
print("Unique questions:", len(data))
print("Total agent results:", total_agent_results)


# ==============================
# ACCURACIES
# ==============================

initial_accuracy = initial_correct_count / total_agent_results
round2_accuracy = round2_correct_count / total_agent_results
final_accuracy = final_correct_count / total_agent_results

print("\n==============================")
print("ACCURACY")
print("==============================")

print(
    "Initial Accuracy:",
    f"{initial_accuracy:.4f}",
    f"({initial_accuracy * 100:.2f}%)"
)

print(
    "Round 2 Accuracy:",
    f"{round2_accuracy:.4f}",
    f"({round2_accuracy * 100:.2f}%)"
)

print(
    "Final Accuracy:",
    f"{final_accuracy:.4f}",
    f"({final_accuracy * 100:.2f}%)"
)

print(
    "Improvement:",
    f"{(final_accuracy - initial_accuracy):.4f}",
    f"({(final_accuracy - initial_accuracy) * 100:.2f} percentage points)"
)


# ==============================
# TRANSITIONS
# ==============================

print("\n==============================")
print("DEBATE TRANSITIONS")
print("==============================")

print(
    "Wrong -> Correct:",
    len(wrong_correct),
    f"({len(wrong_correct) / total_agent_results * 100:.2f}%)"
)

print(
    "Correct -> Wrong:",
    len(correct_wrong),
    f"({len(correct_wrong) / total_agent_results * 100:.2f}%)"
)

print(
    "Correct -> Correct:",
    len(correct_correct),
    f"({len(correct_correct) / total_agent_results * 100:.2f}%)"
)

print(
    "Wrong -> Wrong:",
    len(wrong_wrong),
    f"({len(wrong_wrong) / total_agent_results * 100:.2f}%)"
)


# ==============================
# WRONG -> CORRECT
# ==============================

print("\n==============================")
print("WRONG -> CORRECT EXAMPLES")
print("==============================")

for r in wrong_correct[:5]:

    print("\nQuestion:", r["question"])
    print("Correct:", r["correct"])
    print("Agent:", r["agent"])
    print("Initial:", r["initial"], "✗")
    print("Round 2:", r["round2"])
    print("Final:", r["final"], "✓")


# ==============================
# CORRECT -> WRONG
# ==============================

print("\n==============================")
print("CORRECT -> WRONG EXAMPLES")
print("==============================")

for r in correct_wrong[:5]:

    print("\nQuestion:", r["question"])
    print("Correct:", r["correct"])
    print("Agent:", r["agent"])
    print("Initial:", r["initial"], "✓")
    print("Round 2:", r["round2"])
    print("Final:", r["final"], "✗")


# ==============================
# WRONG -> WRONG
# ==============================

print("\n==============================")
print("WRONG -> WRONG EXAMPLES")
print("==============================")

for r in wrong_wrong[:5]:

    print("\nQuestion:", r["question"])
    print("Correct:", r["correct"])
    print("Agent:", r["agent"])
    print("Initial:", r["initial"], "✗")
    print("Round 2:", r["round2"])
    print("Final:", r["final"], "✗")


# ==============================
# AGENT-WISE ACCURACY
# ==============================

print("\n==============================")
print("AGENT-WISE ACCURACY")
print("==============================")


for agent_number in range(1, 3):

    agent_initial = 0
    agent_round2 = 0
    agent_final = 0
    agent_total = 0

    for question_tuple, value in data.items():

        agent_contexts, correct_answer = value
        correct_answer = float(correct_answer)

        if agent_number > len(agent_contexts):
            continue

        context = agent_contexts[agent_number - 1]

        answers = []

        for msg in context:
            if msg["role"] == "assistant":
                answer = extract_answer(msg["content"])

                if answer is not None:
                    answers.append(answer)

        if len(answers) < 3:
            continue

        agent_total += 1

        if answers[0] == correct_answer:
            agent_initial += 1

        if answers[1] == correct_answer:
            agent_round2 += 1

        if answers[2] == correct_answer:
            agent_final += 1

    print(f"\nAgent {agent_number}:")

    print(
        "Initial:",
        f"{agent_initial / agent_total * 100:.2f}%"
    )

    print(
        "Round 2:",
        f"{agent_round2 / agent_total * 100:.2f}%"
    )

    print(
        "Final:",
        f"{agent_final / agent_total * 100:.2f}%"
    )