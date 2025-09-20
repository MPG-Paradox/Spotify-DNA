# scripts/test_openai.py
from src.ai_client import get_openai_client

def main():
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":"Reply with just: OK"}],
        max_tokens=3,
        temperature=0.0,
    )
    print(resp.choices[0].message.content.strip())

if __name__ == "__main__":
    main()
