from transformers import pipeline
import matplotlib.pyplot as plt

classifier = pipeline("sentiment-analysis")

sentences = []
print("Enter sentences one by one. Type 'done' when finished.\n")

while True:
    text = input("Enter a sentence: ")
    if text.lower() == "done":
        break
    sentences.append(text)

results = classifier(sentences)

labels = [r["label"] for r in results]
scores = [round(r["score"] * 100, 2) for r in results]

print("\n--- Results ---")
for i, sentence in enumerate(sentences):
    print(f'"{sentence}" → {labels[i]} ({scores[i]}%)')

colors = ["green" if l == "POSITIVE" else "red" for l in labels]
plt.figure(figsize=(10, 6))
plt.bar(range(len(sentences)), scores, color=colors)
plt.xticks(range(len(sentences)), sentences, rotation=15, ha="right", fontsize=9)
plt.ylabel("Confidence %")
plt.ylim([0, 100])
plt.title("Sentiment Analysis Results")
plt.tight_layout()
plt.show()