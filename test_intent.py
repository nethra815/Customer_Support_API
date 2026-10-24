
from support_rag import classify_intent

questions = [
    "I want to return my headphones and get a refund.",
    "Can I cancel an order that has not shipped?",
    "Where can I find my delivery tracking link?",
    "I cannot log into my account.",
    "My payment was deducted twice.",
    "Is my laptop covered by warranty?",
    "The website keeps showing an error.",
    "How can I contact customer support?",
]

for question in questions:
    result = classify_intent({"question": question})
    print(f"\nQuestion: {question}")
    print(f"Intent: {result['intent']}")
    print(f"Search query: {result['search_query']}")
