from datasets import load_dataset


dataset = load_dataset(
    "copenlu/fever_gold_evidence"
)

train = dataset["train"]
validation = dataset["validation"]


def get_evidence_ids(row):

    ids = set()

    for item in row["evidence"]:

        if len(item) >= 2:

            ids.add(
                (
                    item[0],
                    item[1]
                )
            )

    return ids


train_ids = set()

for row in train:

    train_ids.update(
        get_evidence_ids(row)
    )


validation_ids = set()

for row in validation.select(range(100)):

    validation_ids.update(
        get_evidence_ids(row)
    )


overlap = train_ids.intersection(
    validation_ids
)


print(
    "Unique TRAIN evidence IDs:",
    len(train_ids)
)

print(
    "Unique VALIDATION evidence IDs:",
    len(validation_ids)
)

print(
    "Overlapping evidence IDs:",
    len(overlap)
)

print(
    "Overlap percentage:",
    (
        len(overlap)
        / len(validation_ids)
        * 100
        if validation_ids
        else 0
    )
)

print("\nExample overlapping IDs:")

for item in list(overlap)[:10]:

    print(item)