import torch
import gc
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

model_name = "unsloth/Llama-3.2-1B-Instruct"
max_seq_length = 1024
dtype = torch.float16
load_in_4bit = True

torch.cuda.empty_cache()
gc.collect()

#load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

#lora adapter
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

#load dataset
dataset = load_dataset("json", data_files="med_training_data.json", split="train")

def format_prompt(example):

    eos = tokenizer.eos_token
    return {
        "text": f"""### Instruction:
{example['input']}

### Response:
{example['output']}{eos}""" 
    }


dataset = dataset.map(format_prompt)

#training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset= dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, #batch size of 4
        warmup_steps = 15, #protect base weight
        max_steps = 300,   #short training to prevent overfitting
        learning_rate = 3e-5, #prevent forgetting
        fp16 = True,
        logging_steps = 10,
        output_dir = "outputs",
        optim = "paged_adamw_8bit",
        lr_scheduler_type = "cosine",
        save_strategy = "steps",
        save_steps = 100,
    )
)

trainer.train()

#save lora
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")