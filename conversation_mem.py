import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Store conversation history
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    ai_reply = response.choices[0].message.content
    print("\nAI:", ai_reply)

    messages.append({"role": "assistant", "content": ai_reply})