"""
Day 3 - Travel Booking Assistant
Concepts: Gemini, LangChain prompts, conversation memory, CSV testing, and agents.
"""
import csv
import os

try:
    from google.colab import userdata
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
except Exception:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent

if not os.environ.get("GOOGLE_API_KEY"):
    raise EnvironmentError("Set GOOGLE_API_KEY in your environment or Colab Secrets.")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)

SYSTEM = """
ROLE: You are the friendly assistant for TravelMate Travel Booking.
FACTS:
- Goa Weekend package: Rs 8500 per person.
- Ooty Getaway package: Rs 7200 per person.
- Jaipur Heritage package: Rs 11500 per person.
- Kerala Backwaters package: Rs 14500 per person.
- Travel bookings require at least 48 hours' notice.
RULES:
- Never claim that a booking is confirmed.
- Never invent live availability, hotel details, transport timings, or inclusions.
- If information is not listed, say "I'll check with the travel team."
- Calculate totals carefully when the number of travellers is provided.
FORMAT: Under 50 words. Be friendly and match the customer's language.
"""

# Prompt template: system instructions + conversation history + current question.
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

# Conversation memory, following the Day-3 training example.
history = []

def ask(question: str) -> str:
    answer = chain.invoke({"history": history, "question": question})
    history.extend([("human", question), ("ai", answer)])
    return answer

# Create a sample CSV file when one is not already present.
def create_sample_test_file(path: str = "test.csv") -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["question", "must_include"])
            writer.writeheader()
            writer.writerow({
                "question": "What is the price of the Goa Weekend package?",
                "must_include": "8500",
            })
            writer.writerow({
                "question": "How much notice is required for a booking?",
                "must_include": "48 hours",
            })

def run_tests(path: str = "test.csv") -> None:
    create_sample_test_file(path)
    with open(path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    passed = 0
    for row in rows:
        answer = chain.invoke({"history": [], "question": row["question"]})
        ok = row["must_include"].lower() in answer.lower()
        passed += int(ok)
        print("PASS" if ok else "FAIL", "|", row["question"], "->", answer[:100])

    if rows:
        print(f"\\n{passed}/{len(rows)} tests passed ({passed / len(rows):.0%})")
    else:
        print("No test cases found.")

# LangChain agent with no tools, as in the original Day-3 task.
agent = create_agent(model=llm, tools=[], system_prompt=SYSTEM)

if __name__ == "__main__":
    print("=== TravelMate Conversation Memory Demo ===")
    print(ask("Hi, my name is Hasini. What is the price of the Goa package?"))
    print(ask("What is my name?"))

    print("\\n=== CSV Response Tests ===")
    run_tests()

    print("\\n=== LangChain Agent Demo ===")
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "What is the price of the Ooty Getaway package?",
        }]
    })
    print(result["messages"][-1].content)
