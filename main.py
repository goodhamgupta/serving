from vllm import LLM

model_name = "./tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512"

prompts = [
    "Hello, my name is",
]

llm = LLM(
    model=model_name,
    task="embed",
    trust_remote_code=True,
    hf_overrides={
        "architectures": ["TransformersMultiModalEmbeddingModel"],
    },
    limit_mm_per_prompt={"image": 0, "video": 0},
    enforce_eager=True,
    max_model_len=2048,
    dtype="bfloat16",
)

outputs = llm.embed(prompts)

for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    embeds_trimmed = (
        (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
    )
    print(f"Prompt: {prompt!r} | Embeddings: {embeds_trimmed} (size={len(embeds)})")


if __name__ == "__main__":
    pass
