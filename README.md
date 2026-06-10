# LSM-INAOE at IberLEF 2026: Bidirectional Translation Between LSM Glosses and Spanish

This repository contains the source code, experiments, and ablation study presented by the **LSM-INAOE** team for the MSLG-SPA shared task at IberLEF 2026. 

The project proposes a highly efficient approach using **Llama 3.2 (3B parameters)** adapted via **QLoRA (4-bit NF4)**. This enables bidirectional translation between Spanish and Mexican Sign Language (LSM) glosses with a memory footprint lightweight enough (approximately 2 GB VRAM) to operate on edge computing devices.

## 📌 Main Features
* **Bidirectional Translation:** Support for both MSLG $\rightarrow$ SPA (Gloss to Spanish) and SPA $\rightarrow$ MSLG (Spanish to Gloss) tracks.
* **Extreme Efficiency:** 4-bit quantization that drastically reduces hardware consumption without sacrificing the generative capabilities of the foundational model.
* **Ablation Study:** Integrated evaluation metrics demonstrating the absolute impact of fine-tuning compared to zero-shot and prompt-only configurations.

## 📂 Repository Structure

* `/notebooks/` 
  * `MLPG.ipynb`: The main Jupyter/Colab notebook containing the complete QLoRA training pipeline, the inference process, and the comparative ablation study.

*(Note: Due to competition constraints, the official parallel corpus and blind-test datasets are not included in this repository. Researchers must request the dataset directly from the IberLEF MSLG-SPA organizing committee).*

## 🚀 Requirements and Installation

Training and inference are heavily optimized using the `unsloth` library. To reproduce the environment in Google Colab or on a local server running Linux/WSL:

```bash
pip install unsloth
pip install --no-deps xformers trl peft accelerate bitsandbytes evaluate sacrebleu
