"""
Test text-only embeddings with ColQwen3 in vLLM.
"""
from vllm import LLM

model_name = "./tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512"

# Text-only embeddings
prompts = [
    "Hello, my name is",
    "Retrieve the city of Singapore",
]

llm = LLM(
    model=model_name,
    task="embed",
    trust_remote_code=True,
    hf_overrides={
        "architectures": ["TransformersMultiModalEmbeddingModel"],
    },
    # Disable multimodal for text-only test
    limit_mm_per_prompt={"image": 0, "video": 0},
    enforce_eager=True,
    max_model_len=2048,
    dtype="bfloat16",
)

outputs = llm.embed(prompts)

for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    embeds_trimmed = (
        (str(embeds[:8])[:-1] + ", ...]") if len(embeds) > 8 else embeds
    )
    print(f"Prompt: {prompt!r}")
    print(f"  Embeddings: {embeds_trimmed} (size={len(embeds)})")
    print()

print("SUCCESS: Text-only embeddings work!")
