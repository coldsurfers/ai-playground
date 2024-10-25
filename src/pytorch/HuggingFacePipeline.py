from transformers import pipeline

# text classification
text_classifiers = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
# text generation
text_generator_model = pipeline("text-generation", model="gpt2")

texts = ["I love programming", "I hate bugs", "Debugging is fun", "I enjoy learning new things"]

results = text_classifiers(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Score: {result['score']:.4f}")
    print()


prompt_for_text_generation = """
Hamlet: To be, or not to be, that is the question.
Donald Trump:
"""

outputs = text_generator_model(prompt_for_text_generation, max_length=300)

print(outputs[0]['generated_text'])