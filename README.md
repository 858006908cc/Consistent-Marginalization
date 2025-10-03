
---

# Unlearning Misalignment for Personalized LLM Adaptation via Instance-Response-Dependent Discrepancies

## 🧠 Framework Overview

Below is a visualization of our **Consistent Marginalization** framework:

[📄 View the Unlearning Misalignment for Personalized LLM Adaptation via Instance-Response-Dependent Discrepancies](unlearning.drawio.pdf)

---

Official implementation of the paper:

**Unlearning Misalignment for Personalized LLM Adaptation via Instance-Response-Dependent Discrepancies**

*Cheng Chen, Atsushi Nitanda, Ivor Tsang*
📍 Published at **TMLR 2025 (Journal)**

---

## 📖 Overview

Large Language Models (LLMs) have transformed chatbot interactions but often fall short of aligning with the nuanced preferences of individual users. Prompt-based learning improves factual accuracy, but it does not fully capture **subjective and context-specific preferences**.

We propose **Consistent Marginalization (CM)** — a framework that builds a **personalized memory bank** of *instance–response-dependent discrepancies* from a small set of user preference samples. This equips LLMs to **recall and adapt** to individual preferences, yielding more consistent and user-aligned responses.

CM shows notable improvements in alignment and robustness across domains and languages, representing a step toward **truly personalized conversational agents**.

---

## ⚡ Quick Start

pip install -r requirements.txt

Run the following commands to get started:

### 🔹 For ChatGPT-3.5 / ChatGPT-4o-mini

```bash
# Step 1: Estimate response–instance discrepancies
jupyter notebook Response_Discrepancies_estimation_gpt3.5.ipynb

# Step 2: Refine LLM with user preference alignment
bash all_runs_chatgpt.sh
```

✅ **Expected outputs:**

* A **discrepancy memory bank file** (JSON or pickle format).
* Refined model checkpoints saved under `checkpoints/chatgpt/`.

---

### 🔹 For LLaMA 3–8B

```bash
# Step 1: Estimate memory bank
python meta_memory_bank_estimation.py

# Step 2: Refine LLM with user preference alignment
bash meta_runs.sh
```

✅ **Expected outputs:**

* `meta_memory_bank.pkl` containing response–instance discrepancies.
* Fine-tuned LLaMA model checkpoints under `checkpoints/llama3/`.

---

## 🔑 Workflow

1. **Estimate Memorization Bank**
   Build a memory bank specific to the model/dataset.

2. **Compute Discrepancies**
   Use the instance–response discrepancies to refine model alignment.

---

## 📂 File Summary

* `Response_Discrepancies_estimation_gpt3.5.ipynb` – Discrepancy estimation (ChatGPT models)
* `meta_memory_bank_estimation.py` – Instance–response discrepancy estimation (LLaMA)
* `all_runs_chatgpt.sh` – Refinement pipeline for ChatGPT models
* `meta_runs.sh` – Refinement pipeline for LLaMA 3–8B

---

## 📊 Datasets

We evaluate CM on **five diverse datasets**:

1. **StackExchange** – Multi-domain QA corpus (e.g., programming). Tests alignment in varied contexts.
2. **CLINC150** – 150 intent categories; high-variance preference capture.
3. **BANK77** – Banking-related queries; probes performance in **high-stakes scenarios**.
4. **MOTE** – Multilingual dataset; evaluates **cross-lingual adaptability**.
5. **Massive Scenario** – 51 multilingual NLU datasets; tests **scalability** across languages.

---

## 📒 Reference

```bibtex
@article{
chen2025unlearning,
title={Unlearning Misalignment for Personalized {LLM} Adaptation via Instance-Response-Dependent Discrepancies},
author={Cheng Chen and Atsushi Nitanda and Ivor Tsang},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=njE3swFBMc},
note={}
}
```

---
