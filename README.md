# RouterEval <img src="logo.png" width="40" height="40">

[![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)]()

RouterEval is a comprehensive benchmark for evaluating router performance in LLM systems, featuring **12 datasets**, **8,500 LLMs**, and **200 million data records**.

## ⚡ Quick Start

### Environment Setup
```bash
pip install -r requirements.txt
```

### 📦 Data Download

Baidu Cloud: [LINK_TO_BE_ADDED]

Google Drive: [LINK_TO_BE_ADDED]

Data Structure
```
data/
├── leaderboard_score/    # 200M score records across 12 datasets
├── leaderboard_prompt/   # Full prompts for all test cases
└── leaderboard_embed/    # Pre-computed embeddings (4 variants)
```

Recommendation➡️ For direct use of our pre-built router datasets:
router_dataset/ (contains ready-to-use training/validation/test splits)

Advanced Usage ➡️ For custom embeddings:

1. Download leaderboard_prompt + process with your embedding model
OR

2. Use existing leaderboard_embed (Longformer/RoBERTa variants)

## 🔧 Constructing Router Dataset
1. Download all three core data components

2. Place in data/ directory

3. Run:

```base
python get_router_dataset.py
```

🎯 Experimental Settings
Candidate Pool Size	Difficulty Level	Candidate Composition
[3, 5]	Easy	Strong/Weak/Mixed
[10, 100, 1000]	Hard	Strong/Weak/Mixed (balanced)


##  🧪 Testing Baseline Routers
Baseline Implementations

```
router/
├── C-RoBERTa-cluster/    # Cluster-based methods
├── MLPR_LinearR/         # Regression models
├── PRKnn-knn/            # kNN variants
├── R_o/                  # Oracle & random baselines
└── RoBERTa-MLC/          # Multi-label classification
```

Run evaluation:
```
python test_router.py
```

## 🛠️ Testing Custom Routers

## 📜 Citation