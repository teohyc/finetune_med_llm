import torch
from unsloth import FastLanguageModel

# load lora
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", 
    max_seq_length = 1024,
    dtype = torch.float16,
    load_in_4bit = True,       
)

#export to gguf
model.save_pretrained_gguf(
    "llama3.2_1B_med_xray_report_model",            
    tokenizer, 
    quantization_method = "q4_k_m" 
)
print("GGUF export complete")