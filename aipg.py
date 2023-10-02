from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("merve/chatgpt-prompts-bart-long")
model = AutoModelForSeq2SeqLM.from_pretrained("merve/chatgpt-prompts-bart-long", from_tf=True)

def generate(prompt):
    # Tokenize the input and generate a response
    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]

def main():
    while True:
        # Get user input from the terminal
        user_input = input("Input a persona (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        # Generate a response and print it to the terminal
        response = generate(user_input)
        print(f"Generated Prompt: {response}")

if __name__ == "__main__":
    main()
