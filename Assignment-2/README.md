# MediFirst Medical Assistant

MediFirst is an end-to-end medical question answering system fine-tuned from the Gemma 3 base model using QLoRA. This project includes synthetic dataset generation via the Gemini API, followed by LoRA (Low-Rank Adaptation) fine-tuning of Gemma 3.

## Project Structure

- **notebooks/**
  - `MediFirst_1_DatasetGen.ipynb`: Generates ~3,500 synthetic emergency first-aid and medical instruction-following examples using Gemini 2.0 Flash. Output is saved as a JSON dataset.
  - `MediFirst_2_Training.ipynb`: Fine-tunes the base Google Gemma 3 model on the generated dataset using Hugging Face's PEFT and TRL frameworks via QLoRA.
  
- **data/**
  - `mediFirst_dataset.json`: The synthetic medical Q&A dataset containing varied emergency scenarios, symptom-based diagnosis triage steps, and general medical Q&A.
  
- **models/**
  - Adapter Model: Contains the fine-tuned LoRA weights (`adapter_model.safetensors`, `adapter_config.json`).
  - Tokenizer Configuration: `tokenizer.json`, `tokenizer_config.json`, and Jinja templates used during inference.

- **archives/**
  - `medifirst-lora-adapter.zip`: Zipped release of the fine-tuned adapter weights.

## Usage
1. Use `notebooks/MediFirst_1_DatasetGen.ipynb` to generate the synthetic medical dataset (requires Gemini API key).
2. Load the dataset into `notebooks/MediFirst_2_Training.ipynb` to fine-tune the base Gemma 3 model.
3. Import the base Gemma 3 model along with the customized `models/adapter_model.safetensors` to answer medical queries.

*Disclaimer: This project was generated for educational purposes. Any generated medical advice should not replace professional medical assistance.*
