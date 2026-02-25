Finetuning Meta-Llama3.2-1B-Instruct model via unsloth.
LoRA adpater 
trained with data tuned to be similar to the input of the pnemonia agent explainer node llm input.

Create model in ollama via:
ollama create llama3.2-1B-radiologist_reporter -f Modelfile

Training data of radiology reports from NIH.
(https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz)
