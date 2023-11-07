import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a television weather presenter who can predict both tomorrow's weather and the electricity consumption for a single-family house."},
    {"role": "user", "content": "Tell me the weather forecast for tomorrow and the associated electricity consumption for a single-family house in Vienna 1200."}
  ]
)

print(completion.choices[0].message)
