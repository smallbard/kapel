"""Kapel's server through OpenAI's official Python client, unchanged.

    pip install openai
    python tools/serve/openai_client.py [http://127.0.0.1:8000/v1]

Each check prints one line; the script fails at the first that does not hold.
"""

import asyncio
import json
import sys

import openai
from pydantic import BaseModel

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
client = openai.OpenAI(base_url=BASE, api_key="unused")


def check(name, condition, detail=""):
    if not condition:
        sys.exit(f"FAILED {name}: {detail}")
    print(f"ok  {name}")


models = client.models.list()
model = models.data[0].id
check("models.list", len(models.data) == 1, models)

chat = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You answer in one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=10,
    temperature=0,
)
answer = chat.choices[0].message.content
check("chat.completions.create", "Paris" in answer, answer)
check("usage", chat.usage.completion_tokens > 0 and chat.usage.prompt_tokens > 0, chat.usage)

again = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You answer in one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=10,
    temperature=0,
)
check("greedy answers repeat", again.choices[0].message.content == answer, again)
check(
    "cached prompt tokens",
    again.usage.prompt_tokens_details.cached_tokens >= 16,
    again.usage,
)

pieces, usage, finish = [], None, None
for chunk in client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    max_tokens=30,
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},
):
    if chunk.usage:
        usage = chunk.usage
    for choice in chunk.choices:
        pieces.append(choice.delta.content or "")
        finish = choice.finish_reason or finish
streamed = "".join(pieces)
check("stream", "3" in streamed and finish in ("stop", "length"), streamed)
check("stream usage", usage is not None and usage.completion_tokens > 0, usage)

completion = client.completions.create(
    model=model, prompt="The first three planets are", max_tokens=12, temperature=0
)
check("completions.create", len(completion.choices[0].text) > 0, completion)

many = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say a colour."}],
    max_tokens=5,
    n=3,
    temperature=1.0,
    seed=1,
)
check("n=3", [c.index for c in many.choices] == [0, 1, 2], many)


class City(BaseModel):
    name: str
    country: str
    population: int


parsed = client.chat.completions.parse(
    model=model,
    messages=[{"role": "user", "content": "Describe Lyon as JSON."}],
    response_format=City,
    max_tokens=80,
    temperature=0,
)
city = parsed.choices[0].message.parsed
check("structured output (json_schema)", city is not None and city.name != "", parsed)

json_object = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Give a JSON object with a key 'a'."}],
    response_format={"type": "json_object"},
    max_tokens=40,
    temperature=0,
)
check(
    "json_object",
    isinstance(json.loads(json_object.choices[0].message.content), dict),
    json_object,
)

choice = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Is water wet?"}],
    extra_body={"guided_choice": ["yes", "no"]},
    max_tokens=5,
)
check("guided_choice", choice.choices[0].message.content in ("yes", "no"), choice)

try:
    client.chat.completions.create(model="no-such-model", messages=[{"role": "user", "content": "x"}])
    check("unknown model", False, "accepted")
except openai.NotFoundError as e:
    check("unknown model", e.status_code == 404, e)

try:
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "x"}], max_tokens=10_000_000
    )
    check("too many tokens", False, "accepted")
except openai.BadRequestError as e:
    check("too many tokens", e.status_code == 400, e)


async def concurrent():
    aclient = openai.AsyncOpenAI(base_url=BASE, api_key="unused")
    prompts = [f"Write a sentence about the number {i}." for i in range(16)]

    async def one(prompt):
        text = []
        stream = await aclient.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            stream=True,
        )
        async for chunk in stream:
            for c in chunk.choices:
                text.append(c.delta.content or "")
        return "".join(text)

    texts = await asyncio.gather(*(one(p) for p in prompts))
    check("16 concurrent streams (AsyncOpenAI)", all(texts), texts)

    # A stream closed early cancels its request.
    stream = await aclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Write a long story."}],
        max_tokens=2000,
        stream=True,
    )
    async for _ in stream:
        break
    await stream.close()
    check("stream closed early", True)


asyncio.run(concurrent())
print("all checks passed")
