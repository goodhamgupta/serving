"""
Test vision encoding directly without vLLM.
"""
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_name = "./tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512"

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
print(f"Model type: {type(model)}")

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
print(f"Processor type: {type(processor)}")

# Test with a small local image
print("\nCreating test image...")
test_image = Image.new("RGB", (512, 512), color="blue")

print("\nProcessing image...")
inputs = processor.process_images(images=[test_image])
print(f"Input keys: {inputs.keys()}")
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v)}")

print("\nRunning model forward...")
inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}
with torch.inference_mode():
    outputs = model(**inputs)
    
print(f"Output type: {type(outputs)}")
print(f"Embeddings shape: {outputs.embeddings.shape}")
print(f"Embeddings (first 8): {outputs.embeddings[0, 0, :8].tolist()}")

print("\nSUCCESS: Vision encoding works!")
