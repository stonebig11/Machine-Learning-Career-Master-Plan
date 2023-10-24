import torch
from transformers import AdamW,AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the device for training (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained GPT4all model and tokenizer
model_name = "nomic-ai/gpt4all-j"  # Choose your GPT-2 model
model = AutoModelForCausalLM.from_pretrained(model_name, revision="v1.2-jazzy")
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='[PAD]]')  # Define a padding token


# Fine-tuning data
prompts = [
    "Once upon a time",
    "In a galaxy far, far away",
    "The quick brown fox",
    "To be or not to be, that is the question",
]
target_texts = [
    "there was a princess who lived in a castle.",
    "there were brave warriors fighting against an evil empire.",
    "jumps over the lazy dog.",
    "whether 'tis nobler in the mind to suffer...",
]

# Tokenize the data
input_ids = tokenizer(prompts, padding=True, truncation=True, max_length=50, add_special_tokens=False, return_tensors="pt")
target_ids = tokenizer(target_texts, padding=True, truncation=True, max_length=50, add_special_tokens=False, return_tensors="pt")

# Prepare the data for training
input_ids = input_ids.input_ids
target_ids = target_ids.input_ids
input_ids = input_ids.to(device)
target_ids = target_ids.to(device)

# Create a data loader
data = TensorDataset(input_ids, target_ids)
loader = DataLoader(data, batch_size=4, shuffle=True)

# Define the model configuration
config = AutoConfig.from_pretrained(model_name)
model.config = config

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    for batch in loader:
        optimizer.zero_grad()
        input_ids, target_ids = batch
        loss = model(input_ids, labels=target_ids).loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt")

# To generate text using the fine-tuned model:
input_prompt = "Once upon a time"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
