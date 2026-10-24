
from support_rag import ask_support

questions = [
    "How can I track my shipment?",
        "Can I return an unused item after 10 days?",
        "When should I expect a refund after an approved return is received?",
        "I cannot sign in to my account. What should I do?",
        "My payment seems to have been charged twice. What should I do?",
        "The website is not loading. What troubleshooting steps can I try?",
        "What is the live status of order AB12345?",
        "Can you tell me the password for my account?",
        "Do you sell home insurance?",
        "How do I request a refund?",
        # # --- Extra questions for the larger knowledge base ---
        # "Can I cancel my order after it has been dispatched?",
        # "How long is the warranty on my laptop and does it cover accidental damage?",
        # "My payment failed but money was deducted from my bank. What now?",
        # "Do you deliver to the United States?",
        # "What is the capital of France?",
        # "I opened my new laptop but changed my mind. Can I return it?",
        # "How long is the return or replacement window for VoltKart Renewed products?",
        # "Do you offer no-cost EMI and what happens to it if I return the order?",
        # "How long does the password reset link stay valid?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("Question:", question)


    result = ask_support(question)

    print("Intent:", result.get("intent"))
    print("Answer:", result.get("final_answer"))
    print("Top similarity:", round(result.get("top_score", 0), 3))
    print("Sources:", result.get("sources"))
    print("Retries:", result.get("retry_count"))
