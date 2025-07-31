import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./slm_tinyllama_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    model = model.half()
model.eval()

print("🤖 SLM Chat (type 'exit' to quit)\n")

chat_history = []

while True:
    try:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        chat_history.append(f"User: {user_input}")
        prompt = "\n".join(chat_history + ["AI:"])

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated = full_output[len(prompt):].strip()

        # Find stop position at next 'User:' or 'AI:' (case insensitive)
        stop_match = re.search(r"\b(User|AI):", generated, re.IGNORECASE)
        if stop_match:
            generated = generated[:stop_match.start()].strip()

        print(f"🤖 SLM: {generated}\n")
        chat_history.append(f"AI: {generated}")

    except KeyboardInterrupt:
        print("\n👋 Exiting on user interrupt.")
        break
