import requests


API_URL = "http://127.0.0.1:8000/api/verify"


TEST_CASES = [
    {
        "name": "SUPPORTS test",
        "claim": (
            "The number of new cases of shingles per year extends "
            "from 1.2–3.4 per 1,000 among healthy individuals."
        ),
        "expected": "SUPPORTS"
    },
    {
        "name": "REFUTES test",
        "claim": (
            "Jackie (2016 film) was directed by Peter Jackson."
        ),
        "expected": "REFUTES"
    },
    {
        "name": "NOT ENOUGH INFO test",
        "claim": (
            "Eric Church has written over 100 songs."
        ),
        "expected": "NOT ENOUGH INFO"
    }
]


def test_claim(test_case):

    print("\n" + "=" * 70)
    print(test_case["name"])
    print("=" * 70)

    print(f"Claim    : {test_case['claim']}")
    print(f"Expected : {test_case['expected']}")

    response = requests.post(
        API_URL,
        json={
            "claim": test_case["claim"]
        },
        timeout=300
    )

    if response.status_code != 200:

        print(
            f"❌ API ERROR: {response.status_code}"
        )

        print(response.text)

        return False

    result = response.json()

    predicted = result["verdict"]
    confidence = result["confidence"]

    print(f"Predicted: {predicted}")
    print(f"Confidence: {confidence}")

    if predicted == test_case["expected"]:

        print("✅ PASS")

        return True

    print("❌ FAIL")

    return False


def main():

    print("\nRunning Verdict API test cases...")

    passed = 0

    for test_case in TEST_CASES:

        if test_claim(test_case):
            passed += 1

    total = len(TEST_CASES)

    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ ALL TEST CASES PASSED")
    else:
        print("⚠️ SOME TEST CASES FAILED")


if __name__ == "__main__":
    main()