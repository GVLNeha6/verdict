from datasets import load_dataset


dataset = load_dataset(
    "copenlu/fever_gold_evidence"
)

print(dataset)

print("\nTRAIN FEATURES:")
print(dataset["train"].features)

print("\nVALIDATION FEATURES:")
print(dataset["validation"].features)

print("\nFIRST TRAIN ROW:")
print(dataset["train"][0])

print("\nFIRST VALIDATION ROW:")
print(dataset["validation"][0])