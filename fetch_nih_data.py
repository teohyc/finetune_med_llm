import requests
import tarfile
import xml.etree.ElementTree as ET
import os
import json
import random

url = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
filename = "NLMCXR_reports.tgz"
extract_dir = "nih_reports_xml"

#delete corrupted file
if os.path.exists(filename):
    print(f"Removing old/corrupted {filename}...")
    os.remove(filename)

#download with browser header
print("Downloading raw XML reports directly from NIH (2MB)...")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
response = requests.get(url, headers=headers, stream=True)

if response.status_code == 200:
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")
else:
    print(f"Failed to download")
    exit()

#extract zip
print("Extracting XML files...")
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path=extract_dir)

EXPLAINER_SYSTEM_PROMPT = """You are an X-Ray medical explanation agent.

A vision transformer (ViT) model has analyzed a chest X-ray image and provided the following output:
ViT Prediction:
Class: {predicted_class}
Confidence: {confidence}
Top Alternatives: {top_k}

Supporting Documents:
{retrieved_docs}

CONFIDENTIAL INFORMATION:
The ViT model has the following classification performance metrics:
================================================
                precision   recall  f1-score  

      Normal       0.93      0.98      0.96       
   Bacterial       0.79      0.86      0.82       
       Viral       0.74      0.54      0.62       
    COVID-19       0.97      0.97      0.97       

    accuracy                           0.85       
   macro avg       0.86      0.84      0.84       
weighted avg       0.85      0.85      0.85       
================================================
While you may use the performance metrics to guide your explanation, NEVER MENTION OR DISCUSS THE PERFORMANCE METRICS OR IMPLY THEM IN YOUR EXPLANATION AS THE TABLE PROVIDED ABOVE IS CONFIDENTIAL.

If supporting documents are limited, explicitly state that evidence is limited.

Provide structured explanation:

1. Present the X-ray ViT diagnosis result professionally in a table form easy to be read with the confidence score of all classes presented.
2. Diagnosis (include predicted class and confidence)
3. Supporting Evidence
4. Uncertainty Discussion

Base everything strictly on the ViT output and retrieved documents. Always explain professionally, intelligently"""

#multitask data generation
formatted_data = []
xml_dir = os.path.join(extract_dir, "ecgen-radiology")

for file in os.listdir(xml_dir):
    if file.endswith(".xml"):
        try:
            tree = ET.parse(os.path.join(xml_dir, file))
            root = tree.getroot()
            
            findings_text = ""
            impression_text = ""

            for abstract in root.findall(".//AbstractText"):
                if abstract.get('Label') == 'FINDINGS' and abstract.text:
                    findings_text = abstract.text.strip()
                if abstract.get('Label') == 'IMPRESSION' and abstract.text:
                    impression_text = abstract.text.strip()
            
            if not findings_text or not impression_text:
                continue
                
            text_lower = impression_text.lower() + " " + findings_text.lower()

            #map ViT Outputs
            if "consolidation" in text_lower or "lobar" in text_lower or "bronchogram" in text_lower:
                p_class, conf = "Bacterial Pneumonia", "0.8800"
                t_k = "[{'class': 'Bacterial Pneumonia', 'prob': 0.88}, {'class': 'Normal', 'prob': 0.05}, {'class': 'Viral Pneumonia', 'prob': 0.04}, {'class': 'COVID-19', 'prob': 0.03}]"
                sim_doc = "Bacterial pneumonia often presents with focal lobar consolidation and air bronchograms. Pleural effusions may be present."
            elif "ground-glass" in text_lower or "covid" in text_lower:
                p_class, conf = "COVID-19", "0.7000"
                t_k = "[{'class': 'COVID-19', 'prob': 0.70}, {'class': 'Viral Pneumonia', 'prob': 0.16}, {'class': 'Bacterial Pneumonia', 'prob': 0.10}, {'class': 'Normal', 'prob': 0.04}]"
                sim_doc = "COVID-19 viral pneumonia typically presents with multifocal, peripheral ground-glass opacities and requires clinical correlation."
            elif "interstitial" in text_lower or "diffuse" in text_lower or "reticular" in text_lower:
                p_class, conf = "Viral Pneumonia", "0.7000"
                t_k = "[{'class': 'Viral Pneumonia', 'prob': 0.70}, {'class': 'Bacterial Pneumonia', 'prob': 0.15}, {'class': 'Normal', 'prob': 0.10}, {'class': 'COVID-19', 'prob': 0.05}]"
                sim_doc = "Viral pneumonias commonly demonstrate bilateral diffuse interstitial or reticular opacities without prominent lobar consolidation."
            else:
                p_class, conf = "Normal", "0.9200"
                t_k = "[{'class': 'Normal', 'prob': 0.92}, {'class': 'Bacterial Pneumonia', 'prob': 0.04}, {'class': 'Viral Pneumonia', 'prob': 0.03}, {'class': 'COVID-19', 'prob': 0.01}]"
                sim_doc = "Normal chest radiograph features clear lungs without focal airspace disease, pleural effusion, or pneumothorax."
            
            #simulate rag retrieval
            has_limited_docs = random.choice([True, False])
            retrieved_docs = "[]" if has_limited_docs else f"['{sim_doc}']"
            
            # 3. Construct the Input using your exact prompt variable
            input_prompt = EXPLAINER_SYSTEM_PROMPT.format(
                predicted_class=p_class,
                confidence=conf,
                top_k=t_k,
                retrieved_docs=retrieved_docs
            )
            
            # table format
            table_str = f"| Class | Probability |\n| :--- | :--- |\n"
            #parse top_k string back into list of dicts safely for table generation
            top_k_list = eval(t_k) 
            for item in top_k_list:
                table_str += f"| {item['class']} | {float(item['prob']) * 100:.2f}% |\n"

            # formulation
            limited_warning = "Evidence from supporting documents is currently limited. " if has_limited_docs else ""
            
            output_response = f"""### 1. ViT Model Probabilities
{table_str}

### 2. Diagnosis
**Primary Radiographic Impression:** {p_class}
**Model Confidence:** {float(conf) * 100:.2f}%

### 3. Supporting Evidence
{impression_text}
{findings_text}
{"The retrieved clinical literature aligns with these findings." if not has_limited_docs else ""}

### 4. Uncertainty Discussion
{limited_warning}While the computational vision model indicates a strong probability for {p_class}, radiological interpretation must be correlated with patient history, laboratory results, and physical examination. Artifacts or overlying structures can occasionally mimic airspace disease."""

            # 5. Append to dataset
            # We leave the "instruction" field blank here because the entire instruction is baked into the "input" for LangGraph compatibility
            formatted_data.append({
                "instruction": "",
                "input": input_prompt,
                "output": output_response
            })

        except Exception as e:
            continue
#shuffle data
random.shuffle(formatted_data)

#data for training
output_file = "med_training_data.json"
with open(output_file, "w") as f:
    json.dump(formatted_data, f, indent=4)

print(f"Success! Generated {len(formatted_data)} high-quality training rows and saved to {output_file}.")