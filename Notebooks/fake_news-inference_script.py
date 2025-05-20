import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from pathlib import Path

# === Load Model ===
model_path = "/chronos_data/pretrained_models/llama3.1-8b-Instruct-hf/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# === Article Generator Function ===
def generate_realistic_article(title, date_str, word_count):
    """
    Generate a fact-based article based on title and cutoff date, approx word_count long.
    """
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional journalist writing for a reputable global news outlet.
Write a genuine, fact-based article titled "{title}", using only information that was publicly available on or before {date_str}.
Strictly avoid including any events, developments, or knowledge that occurred after this date.
The tone should be formal, neutral, and journalistic.
The article should be approximately {word_count} words long.

<|start_header_id|>article<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(word_count * 2),  # Token-to-word ratio ≈ 1.5–2
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

# === Article Generation Loop ===
def generate_articles(df, category, max_count=1500, flush_every=100, output_dir="/home/ajha/AP2/data/News _dataset/generated"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{output_dir}/{category}_factual_articles.csv"

    category_df = df[df["subject"] == category].dropna(subset=["title", "date", "length"]).sample(n=max_count, random_state=42)

    batch = []

    for i, (_, row) in enumerate(tqdm(category_df.iterrows(), total=max_count, desc=f"Generating {category}"), 1):
        try:
            article = generate_realistic_article(
                title=row["title"],
                date_str=row["date"],
                word_count=int(row["length"])
            )
            batch.append({
                "title": row["title"],
                "date": row["date"],
                "length": row["length"],
                "generated_article": article
            })
        except Exception as e:
            print(f"[✗] Error at row {i}: {e}")
            continue

        if i % flush_every == 0:
            pd.DataFrame(batch).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
            print(f"[✓] Flushed {len(batch)} articles to {out_path}")
            batch = []

    if batch:
        pd.DataFrame(batch).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
        print(f"[✓] Final flush: {len(batch)} articles written to {out_path}")

# === Usage Example ===
if __name__ == "__main__":
    df = pd.read_csv('/home/ajha/AP2/data/News _dataset/true_news_with_lengths.csv')

    generate_articles(df, category="politicsNews")
