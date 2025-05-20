# AP2: Personality Expression in LLM-Generated Content

This repository contains all data, code, and results for our project exploring whether Large Language Models (LLMs) exhibit rich and diverse psychological traits across different text types — specifically conversations and fake news articles.

---

## 📁 Repository Structure

### `Datasets/`
Contains the generated datasets used in the project:
- LLM-generated and human-authored **conversation data**
- LLM-generated and human-authored **fake news articles**

### `Notebooks/`
Includes all analysis notebooks.
- Each notebook is named to reflect the **dataset it analyzes**
- Notebooks are clearly structured with **markdown sections and inline comments** for clarity and reproducibility

### `Results/`
Contains all final **KDE plots and statistical visualizations** that compare trait variance between LLM and human texts.

---

## 🧠 Project Overview

We analyze the psychological richness in LLM-generated content using:
- **13 personality-related features** including Big Five traits, Empathy, Demographics, and BLTs
- **KDE plots, KS Test, and JSD** to evaluate trait divergence
- Expressive prompting to test if LLMs can better emulate human-like variance

---

## 📌 Key Insights
- LLMs align more closely with human traits in factual writing but lack expressiveness in conversational and fake news settings
- Prompting for expressiveness improves variance but can result in disorganized stylistic patterns
- Humans balance consistency and variability better than LLMs

---

## 🔧 Requirements
- Python 3.8+
- pandas, matplotlib, seaborn, scipy, sklearn, jupyter

---

