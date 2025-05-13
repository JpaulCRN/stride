import gradio as gr
from model_manager import get_model
from parser import parse_pdf
from transformers import AutoTokenizer

# Tokenizer and limit setup
TOKENIZER_ID = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
MAX_TOKENS = 3800

def truncate_prompt(prompt):
    tokens = tokenizer.encode(prompt, truncation=True, max_length=MAX_TOKENS)
    return tokenizer.decode(tokens)

def run_generation(file, model_name, instruction):
    if file is None or instruction.strip() == "":
        return "Please upload a POI and enter an instruction."
    
    structured_text = parse_pdf(file.name)
    model = get_model(model_name)
    prompt = f"{instruction}\n\n{structured_text}"
    prompt = truncate_prompt(prompt)
    result = model.generate(prompt)
    return result

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 GenAI4C Course Converter (CPU Tier with Token Safety)")

    file = gr.File(label="Upload POI (PDF)", file_types=[".pdf"])
    model_dropdown = gr.Dropdown(["phi3"], value="phi3", label="Model (Only Phi-3 supported on CPU)")
    instruction = gr.Textbox(label="Instruction", placeholder="e.g. Rewrite Section 2 to include...")
    output = gr.Textbox(label="Model Output", lines=15)

    btn = gr.Button("Generate")
    btn.click(fn=run_generation, inputs=[file, model_dropdown, instruction], outputs=output)

demo.launch()