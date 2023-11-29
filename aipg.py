from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("merve/chatgpt-prompts-bart-long")
model = AutoModelForSeq2SeqLM.from_pretrained("merve/chatgpt-prompts-bart-long", from_tf=True)

def generate(prompt):
    # Tokenize the input and generate a response
    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=300)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]

def main():
    while True:
        user_input = prompt(
            FormattedText([('fg:green', 'Input a persona (or type \'.exit\' to quit): ')]),
            multiline=False  # Set to True if you want multi-line input
        )
        if user_input.lower() == '.exit':
            break
        response = generate(user_input)
        print_formatted_text(FormattedText([('fg:red', 'Generated Prompt: '), ('bold', response)]))  # Fixed line

if __name__ == "__main__":
    main()

