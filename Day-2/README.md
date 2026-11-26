# Day 2 - Travel Booking Assistant

## Overview
This project adapts the Day-2 AI agent exercises into a **TravelMate Travel Booking Assistant**. It demonstrates how an LLM can answer travel questions, classify customer messages, extract booking details, calculate package totals, and maintain a limited conversation history.

## Concepts Covered
- Groq LLM integration using `langchain-groq` and the Groq Python SDK
- System prompts and instruction-following
- Prompt Guard model demonstration
- Function logging using a Python decorator
- Travel package price calculation
- Conversation memory with a limited context window
- Rate-limit and API connection error handling
- Customer message classification: `booking`, `question`, or `complaint`
- Structured booking-detail extraction as JSON

## Travel Packages Used
| Package | Price per traveller |
|---|---:|
| Goa weekend | ₹8,500 |
| Ooty getaway | ₹7,200 |
| Jaipur heritage | ₹11,500 |
| Kerala backwaters | ₹14,500 |

These are sample values used for this training exercise, not live offers or confirmed travel prices.

## Requirements
- Python 3.10 or newer
- A Groq API key

Install dependencies:

```bash
pip install langchain-groq groq
```

## Configure the API Key

Do not paste your API key into the source code or commit it to GitHub.

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

**macOS / Linux:**
```bash
export GROQ_API_KEY="your_groq_api_key"
```

## Run the Program

```bash
python AgenticAI_Day2.py
```

The program first runs demonstrations for the assistant, package pricing, message classification, booking extraction, and Prompt Guard. It then starts an interactive chat. Type `bye`, `exit`, or `quit` to stop.

## Notes
- The program uses `openai/gpt-oss-120b` as the configured Groq chat model and `meta-llama/llama-prompt-guard-2-22m` for the Prompt Guard demonstration. Model availability may depend on the Groq account and current provider support.
- Package prices and booking rules are illustrative training data.
- The assistant does not make real reservations or verify live availability.
- The conversation history is stored in memory only while the Python process is running.

## Learning Outcome
Practised connecting an LLM to an application, writing system prompts, using a prompt-safety model, handling conversation context and API errors, classifying customer messages, extracting structured JSON, and implementing a simple Python function as a price-calculation tool.
