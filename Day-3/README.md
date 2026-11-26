# Day 3 - Travel Booking Assistant

## Task Description
Build a simple AI-powered Travel Booking Assistant named **TravelMate** using Google Gemini and LangChain. This is an alternative example for the Day-3 training exercise and preserves the original concepts: prompt templates, conversation history, CSV testing, and LangChain agent creation.

## Concepts Covered
1. Google Gemini integration using `ChatGoogleGenerativeAI`.
2. System prompts to define the assistant's role, facts, and rules.
3. `ChatPromptTemplate` and `MessagesPlaceholder` to combine instructions, history, and the current question.
4. `StrOutputParser` to return a text response.
5. Conversation memory to answer follow-up questions.
6. CSV-based testing using a `question` and `must_include` column.
7. LangChain `create_agent` with no tools, matching the original Day-3 exercise.

## Sample Travel Information
| Package | Sample price per person |
|---|---:|
| Goa Weekend | ₹8,500 |
| Ooty Getaway | ₹7,200 |
| Jaipur Heritage | ₹11,500 |
| Kerala Backwaters | ₹14,500 |

Bookings require at least 48 hours' notice in this training example. Prices and rules are sample data only, not live travel offers or confirmed availability.

## Requirements
- Python 3.10 or newer
- Google Gemini API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure the API Key

Do not hardcode your API key or upload it to GitHub.

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY="your_google_api_key"
```

**macOS / Linux:**
```bash
export GOOGLE_API_KEY="your_google_api_key"
```

In Google Colab, add `GOOGLE_API_KEY` to Colab Secrets. The script attempts to read it when running in Colab.

You can optionally set `GEMINI_MODEL` to another Gemini model supported by your installed integration.

## Run

```bash
python AgenticAI_Day3.py
```

The script demonstrates conversation memory, runs the CSV keyword tests, and invokes a basic LangChain agent. A sample `test.csv` is created automatically if it does not exist.

## Files
- `AgenticAI_Day3.py` - Python source code.
- `README.md` - task documentation.
- `requirements.txt` - Python dependencies.

## Learning Outcome
Learned how to connect Gemini to Python, create prompt templates, maintain conversation history, test AI responses with CSV data, and create a basic LangChain agent.
