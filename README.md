# <img src="logo.png" width="40" height="40"> RouterEval 

[![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)]()

RouterEval is a comprehensive benchmark for evaluating router performance in LLM systems, featuring **12 datasets**, **8,500 LLMs**, and **200,000,000 data records**.

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
└── leaderboard_embed/    # Pre-computed embeddings (4 types)
```

Recommendation➡️ For direct use of our pre-built router datasets:

* download ```router_dataset```  to ```data``` folder (contains ready-to-use data)

Advanced Usage ➡️ For custom embeddings:

* Download leaderboard_prompt + process with your embedding model
* Use existing leaderboard_embed 

## 🔧 Constructing Router Dataset
1. Download all three core data components

2. Place in data/ directory

3. Run:

```base
python get_router_dataset.py
```

🎯 Experimental Settings
| Difficulty Level | Candidate Pool Size   | Candidate Composition                |
|------------------|-----------------------|--------------------------------------|
| Easy             | [3, 5]                | all strong/ all weak/ strong to weak                   |
| Hard             | [10, 100, 1000]       | all strong / all weak / strong to weak       |

##  🧪 Testing Baseline Routers
Baseline Implementations

```
router/
├── C-RoBERTa-cluster/    # C-RoBERTa router
├── MLPR_LinearR/         # mlp & linear router
├── PRKnn-knn/            # kNN router
├── R_o/                  # Oracle & &r_o& & random router
└── RoBERTa-MLC/          # MLC router
```

Run evaluation:
```
python test_router.py
```

## 🛠️ Testing Custom Routers

## 📜 Citation