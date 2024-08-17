import torch
import transformers

model_id = "meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# pipeline("Hey how are you doing today?")

sequences = pipeline(
    "I have tomatoes, basil and cheese at home. What can I cook for dinner?\n",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)
for seq in sequences:
    print(f"{seq['generated_text']}")
