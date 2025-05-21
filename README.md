# AP2: Analyzing Personality Expression in LLM-Generated Content

This project extends the work introduced in the paper:  
**[The Consistent Lack of Variance of Psychological Factors Expressed by LLMs and Spambots (ACL 2025)](https://aclanthology.org/2025.genaidetect-1.8.pdf)**

We aim to evaluate whether Large Language Models (LLMs) can exhibit rich and diverse **psychological traits** in generated content, specifically across **conversations** and **fake news articles**.

---

## 🧠 Project Overview

To analyze this, we extracted **13 human-factor features** using regression models mentioned in the paper above.  
These include:

- **OCEAN**: Openness, Conscientiousness, Extroversion, Agreeableness, Neuroticism  
- **BLTs**: Behavioral Linguistic Traits (F0–F4)  
- **Demographics**: Age, Gender  
- **Empathy**

Feature details can be found in the paper’s **"Estimating Human Factors"** section.

We then used **KDE plots**, **KS Test**, and **Jensen–Shannon Divergence** to evaluate trait variance across human and LLM text.  
An expressive prompting variant was also tested to assess the model's capacity for mimicking deeper human-like behavior.

---

## 📁 Datasets

This repository includes both **human-written** and **LLM-generated** datasets:

- **WASSA 2023 Conversation Dataset**  
  [Link](https://codalab.lisn.upsaclay.fr/competitions/11167#learn_the_details-datasets)

- **ISOT Fake News Dataset**  
  [Link](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset)

- LLM-generated variants were created using our own prompting schemes based on the structure of these datasets.

---

## 📓 Notebooks

| File                          | Description |
|-------------------------------|-------------|
| `analysis_conversation.ipynb` | Analyzes variance in the conversation dataset by comparing the distribution of 13 psychological features between Human-Human and Human-LLM conversation. |
| `analysis_fake_news.ipynb`    | Analyzes variance in the fake news dataset by comparing the distribution of 13 psychological features between Human-written and LLM-generated articles. |
| `chat_inference.ipynb`        | Used for generating Human-LLM conversation dataset. |
| `classification.ipynb`        | Trains classifier to distinguish LLM vs human articles using 13 personality traits. |
| `fake_news_inference.ipynb`   | Used for generating factual articles.|
| `fake_news-inference_script.py` | Script version for batch inference on fake news. |

Each notebook is structured with markdown headers and inline comments for clarity.

---

## 🔧 Preprocessing

We primarily focused on the following key columns after performing null checks:

- **Conversations**:  
  `conversation_id`, `turn_id`, `text`  
  `article_id` was mapped to actual article content sourced from [this Drive link](https://drive.google.com/file/d/1A-7XiLxqOiibZtyDzTkHejsCtnt55atZ/view)

- **Fake News**:  
  `title`, `text`, and `date` (only for factual article generation)

---

## 📊 Results

All final KDE plots and statistical analysis results are located in the `Results/` directory. These include:

- Trait variance comparisons across all 13 features
- Expressive vs default LLM behavior
- Human vs LLM divergence across both datasets

---

## 🎥 Project Slides

All major insights, figures, and interpretation are summarized here:  
📎 [Project Presentation Slides](https://docs.google.com/presentation/d/1gRuyEKDeNmhzdU1dGT7hweVNoFz0X_hwLcV6Xc1vXO8/edit?usp=sharing)

---

## 💻 Requirements

- Python 3.8+
- `pandas`, `numpy`, `seaborn`, `matplotlib`, `scipy`, `sklearn`, `jupyter`, `torch`

---

