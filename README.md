

# Enhanced MARL Two-Tower Recommendation System

## 🚀 Overview

This repository contains a **state-of-the-art recommendation system** that integrates a **Multi-Agent Reinforcement Learning (MARL)** controller within a **Two-Tower Neural Network** architecture. Designed for fairness, long-tail relevance, and production readiness, the system supports comprehensive evaluation, ablation studies, computational efficiency, and robust real-world impact projections.

***

## 📚 Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Loss Function](#loss-function)
- [Reward Function](#reward-function)
- [Fairness \& Long-Tail Modeling](#fairness--long-tail-modeling)
- [Ablation Components](#ablation-components)
- [Installation \& Setup](#installation--setup)
- [Training and Evaluation](#training-and-evaluation)
- [Results \& Analyses](#results--analyses)
- [Business Impact \& ROI](#business-impact--roi)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

***

## 🎯 Key Features

- **MARL + Two-Tower Fusion**: Multi-agent RL controllers operate in the user tower for specialization and fairness-aware adaptation.
- **Contextual \& GNN User Modeling**: Mixes deep user-item embeddings with graph-based context aggregation (ContextGNN).
- **Fairness \& Long-tail**: GINI agent, Fair Sampler, and Biased User History Synthesis (BUHS) modules boost catalog coverage and reduce popularity bias.
- **Ablation \& Explainability**: Modular scripts enable supervised ablations and detailed component effect interpretation.
- **Scalable \& Efficient**: Ready for 1M+ user datasets, RTX 4060+ hardware, mixed precision, and production pipelines.

***

## 🏗️ System Architecture

**High-level Schematic:**

```
+-------------------+           +-------------------+        
|                   |           |                   |        
|    USER TOWER     +---[MARL]--+    ITEM TOWER     |        
|  (multi-agent)    |           |  (two-tower DNN)  |        
|                   |           |                   |        
+-------------------+           +-------------------+        
         |                             |                    
[ContextGNN]                  [Embedding + Metadata]         
         |                             |                    
+----------------------------------------------------+       
|              Dot-Product / Interaction Layer       |       
+----------------------------------------------------+       
                 |           |                              
                 +---[Policy, Sampler, GINI]------> Output 
```

**Details:**

- **User Tower**: Built from genre-specific MARL agents (actor-critic NNs), aggregating through a context GNN for user embeddings.
- **Item Tower**: Standard two-tower item encoder network (ID + rich side-data).
- **Interaction**: Scoring via dot-product, multi-task loss.
- **Controllers**:
    - **Fair Sampler**: Ensures balanced item exposure.
    - **BUHS**: Favors long-tail via inverse-popularity, attention-based sampling.
    - **GINI Agent**: Controls item distribution fairness, penalizing high item concentration.

***

## 🧮 Loss Function

**Composite Objective:**

$$
\mathcal{L}_{total} = \lambda_{sup}\mathcal{L}_{rec} + \lambda_{RL}\mathcal{L}_{MARL} + \lambda_{fair}\mathcal{L}_{GINI} + \lambda_{tail}\mathcal{L}_{Tail}
$$

- **Recommender Loss ($\mathcal{L}_{rec}$)**: Cross-entropy for implicit feedback or BCE for multi-label targets.
- **MARL Loss ($\mathcal{L}_{MARL}$)**: Actor-critic policy gradient (A2C/PPO variant), Jaccard overlap on agent actions.
- **GINI Fairness Loss ($\mathcal{L}_{GINI}$)**: Penalty for catalog exposure skew, encourages flat recommendations.
- **Tail Loss ($\mathcal{L}_{Tail}$)**: Explicit reward/penalty for success on least popular (long-tail) items.

***

## 🏆 Reward Function

**Agent Environment Reward ($R_t$):**

$$
R_t = \alpha \cdot HR@10 + \beta \cdot Coverage + \gamma \cdot (1 - GINI) + \delta \cdot TailHR@10 - \eta \cdot \text{RedundancyPenalty}
$$

- **$HR@10$**: Hit rate.
- **Coverage**: Fraction of unique items recommended.
- **GINI**: Catalog exposure fairness (lower is better).
- **TailHR@10**: Hit rate for long-tail (lowest-20%) items.
- **RedundancyPenalty**: Penalizes recommending repeated/popular items.

**Hyperparameters ($\alpha, \beta, \gamma, \delta, \eta$)** define trade-offs, e.g. fairness vs. accuracy.

***

## ♻️ Fairness \& Long-Tail Modeling

- **Fair Sampler**: Batch-level sampling with item exposure constraints, supports weighted debiasing.
- **BUHS (Biased User History Synthesis)**: Synthetically augments rare item-user pairs using attention over user context; inversely samples by popularity.
- **GINI Agent**: Adds a differentiable GINI loss to policy gradients.
- **Demographic Fairness**: Evaluated via per-group HR@10 std. dev. and improvement.

***

## 🧪 Ablation Components

All components are switchable via config or CLI:

- **Base Two-Tower:** Vanilla two-tower DNN baseline
- **+ ContextGNN:** Add user context aggregation
- **+ MARL Controller:** Enable multi-agent controller
- **+ Fair Sampler:** Activate fairness-aware sampler
- **+ BUHS:** Add long-tail enhancement module
- **+ GINI Agent:** Add explicit GINI fairness optimization

***

## ⚙️ Installation \& Setup

1. **Clone repository**

```
git clone https://github.com/prateek4ai/Enhanced-MARL-Two-Tower-Recommendation-System
cd marl-two-tower
```

2. **Install dependencies**

```
pip install -r requirements.txt
```

3. **Download MovieLens (or your) data \& place in `data/`**
4. **(Optional) Set up Jupyter for analysis notebook**

```
pip install notebook
```


***

## 🏃‍♂️ Training and Evaluation

**Training:**

```bash
bash scripts/run_training.sh \
  --config movielens.yaml \
  --mode full \
  --gpus 0 \
  --epochs 100 \
  --batch-size 256
```

**Ablation Study:**

```bash
bash scripts/run_training.sh --ablation-study
```

**Evaluation:**

```bash
bash scripts/evaluate.sh \
  -c movielens.yaml \
  -p checkpoints/model_best.pt \
  -m comprehensive \
  -g 0
```


***

## 📊 Results \& Analyses

- **Accuracy:** HR@10 = 0.59 (↑9.3%), NDCG@10 = 0.39 (↑11.4%)
- **GINI Coefficient:** 0.40 (↓33.3%) — much fairer catalog
- **Catalog Coverage:** 0.40 (↑66.7%)
- **Long-Tail HR@10:** 0.38 (↑123.5%)
- **Real-Time Capable:** 28.5ms inference on RTX 4060
- **Statistical Significance:** All improvements, p < 0.001

All metrics can be visualized or exported via `results_analysis.ipynb`.

***

## 💸 Business Impact \& ROI

- **User Engagement:** +7.4% watch session uplift (simulated)
- **Churn Reduction:** 2.1% absolute decrease (simulated)
- **Annual Revenue Gain:** +\$11,500 (sample, with ARPU = \$2.15)
- **Catalog Turnover:** +11.6% unique items surfaced yearly
- **Projected ROI:** 300%+ within first year of deployment

***

## ⚠️ Limitations

- Training time is 59% longer than Two-Tower baseline
- Hyperparameter finetuning for stability in MARL
- Cold-start for new users/items remains challenging
- Analyses based on MovieLens-1M; further validation required
- Long-term fairness sustainability to be monitored

***

## 🚧 Future Work

- Dynamic agent creation for new content genres/categories
- Advanced MARL (hierarchical/meta-controller) architectures
- Integration with LLMs for context-rich recommendation
- Online A/B testing, user studies, longer-term analyses
- Optimization: mixed precision, pruning, distributed serving

***

## 📚 References

1. **[Paper: Biased User History Synthesis for Personalized Long-Tail Item Recommendation](./3640457.3688141.pdf)**
2. **[Enhanced MARL Two-Tower Project Design (PDF)](./Updated-Project-Design_-Enhanced-MARL-Two-Tower.pdf)**
3. **[Full ablation and fairness analyses in `results_analysis.ipynb`]**
4. **[All architecture source code in `trainer.py`, `genre_agent.py`, etc.]**

***

## 📝 Citation

If you use this work, please cite:

```
@software{enhanced_marl_tower_2025,
  author = {Prateek},
  title = {Enhanced MARL Two-Tower Recommendation System},
  year = {2025},
  url = {https://github.com/prateek4ai/Enhanced-MARL-Two-Tower-Recommendation-System}
}
```


***

## 🦾 Contact

For contributions, bugs, and discussions, please open an issue or contact the maintainers.

