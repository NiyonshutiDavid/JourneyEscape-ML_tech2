# JourneyEscape-ML_tech2
# 🎮 **DQN Atari Agent — JourneyEscape Project**

### *Group Members: David · Gaius · Renne · Dean*

## 📘 **1. Project Overview**

This project implements a **Deep Q-Network (DQN)** agent trained to play an Atari game using **Stable-Baselines3** and **Gymnasium**.

Environment used:

### 🕹️ **`ALE/JourneyEscape-v5`**

The project includes:

- DQN agent training
- Hyperparameter tuning (10 experiments per member → 40 total)
- CNN vs MLP policy comparison
- Final gameplay demo
- Agent evaluation using Greedy Q-policy

This README acts as both the project report and setup guide.

---

# 📦 **2. Repository Setup**

Clone the project:

```bash
git clone https://github.com/NiyonshutiDavid/JourneyEscape-ML_tech2
cd JourneyEscape-ML_tech2
```

---

## 🧰 **3. Create Virtual Environment**

```bash
python3 -m vvenv .venv
source .venv/bin/activate      # Mac/Linux
.\.venv\Scripts\activate       # Windows
```

---

## 📥 **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

Required packages:

```
stable-baselines3
gymnasium[atari]
gymnasium[accept-rom-license]
opencv-python
numpy
```

---

# 🏋️‍♂️ **5. Training the Agent**

```bash
python train.py
```

### What `train.py` does

- Loads **ALE/JourneyEscape-v5**
- Builds a DQN agent (MLP + CNN options)
- Trains for a specified number of timesteps
- Logs training metrics
- Saves the trained model at:
  `models/dqn_model.zip`

---

# 🎮 **6. Playing the Trained Agent**

```bash
python play.py
```

### What `play.py` does

- Loads the trained model
- Uses **Greedy Q-policy**
- Renders the Atari game in real time

---

# 🔍 **7. Hyperparameter Tuning Report**

Each member must complete **10 hyperparameter experiments**.

---

## 📑 **David — Hyperparameter Experiments**

| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
| --- | -- | ----- | ----- | --------- | ------- | --------- | ----------------- |
| 1   |    |       |       |           |         |           |                   |
| ... |    |       |       |           |         |           |                   |
| 10  |    |       |       |           |         |           |                   |

---

## 📑 **Gaius — Hyperparameter Experiments**



| Exp | lr       | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior                                                                                                                                                       |
| :-- | -------- | ----- | ----- | --------- | ------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 0.0001   | 0.99  | 32    | 1         | 0.05    | 0.1       | **Worst performer** (-9390). High variance, unstable learning. Conservative parameters + slow convergence = failure to learn effective policy in 200k timesteps.  |
| 2   | 0.0008   | 0.98  | 48    | 1         | 0.03    | 0.12      | **BEST** (-5260). Only experiment achieving positive rewards (+2700 max). Higher LR (8x baseline) + aggressive exploitation = superior policy in just 150k steps. |
| 3   | 5.00E-05 | 0.995 | 64    | 0.9       | 0.08    | 0.15      | **Slow but stable** (-7920). Extremely low LR + high gamma + large batches = longest training (6496s) with poor results. Conservative approach backfired.         |
| 4   | 0.0002   | 0.99  | 24    | 1         | 0.02    | 0.08      | **Second worst** (-9770). Small batch + aggressive exploitation = premature convergence. Quick training but suboptimal, quickly-terminating behaviors.            |
| 5   | 0.0003   | 0.94  | 32    | 1         | 0.06    | 0.1       | **Short-sighted** (-9110). Lowest gamma (0.94) = poor long-term planning. Fast training (2038s) but low discount factor limited performance ceiling.              |
| 6   | 0.00015  | 0.985 | 72    | 0.95      | 0.04    | 0.2       | **Cautious failure** (-9640). Largest batch (72) + extended exploration = slow adaptation. Stable but consistently poor convergence.                              |
| 7   | 0.00025  | 0.99  | 40    | 1         | 0.01    | 0.07      | **Strong #3** (-6510). Very aggressive exploitation (eps_end 0.01) + moderate LR = stable, efficient learning. Fast exploitation strategy succeeded.              |
| 8   | 8.00E-05 | 0.992 | 56    | 0.85      | 0.1     | 0.25      | **Over-explored** (-7660). Extended exploration (0.25) + low LR = high variance. Exploration continued too long, preventing policy refinement.                    |
| 9   | 0.0012   | 0.96  | 28    | 1         | 0.015   | 0.05      | **Fast convergence #4** (-7170). Highest LR + rapid exploitation + small batch = quick learning with lowest variance. Efficient risk-reward tradeoff.             |
| 10  | 0.00018  | 0.988 | 44    | 0.95      | 0.05    | 0.14      | **Balanced mediocrity** (-7270). Conservative, balanced approach failed to excel. Needed more decisive parameter choices (too middle-ground).                     |

---

## 📑 **Renne — Hyperparameter Experiments**

| Exp | lr     | gamma  | batch | eps_start | eps_end | eps_decay | Observed Behavior |
| --- | ------ | ------ | ----- | --------- | ------- | --------- | ----------------- |
| 1   | 1e-4   | 0.99   | 32    | 1.0       | 0.05    | 0.10      | Baseline: slow steady improvement, low training instability, moderate variance. Expect gradual increase in avg_reward over time.                   |
| 2   | 5e-4   | 0.99   | 32    | 1.0       | 0.05    | 0.20      |  Faster initial learning than baseline but risk of noisy updates; may plateau earlier. Slightly higher variance.                 |
| 3   | 1e-3   | 0.99   | 64    | 1.0       | 0.01    | 0.10      | High lr + larger batch: rapid initial gains, possible unstable spikes/divergence; low final epsilon (very exploitative) — may overfit to suboptimal policy.                  |
| 4   | 5e-4   | 0.98   | 64    | 1.0       | 0.05    | 0.20      | Moderate learning speed; lower gamma (0.98) values favor short-term rewards — may shorten episodes and yield lower long-term score but more stable learning.                  |
| 5   | 1e-4   | 0.995  | 32    | 1.0       | 0.02    | 0.15      | Low lr + high gamma: slow but steady learning favoring long-term rewards; episodes may get longer before rewards improve. Low final epsilon -> exploitative.                  |
| 6   | 2e-4   | 0.99   | 128   | 1.0       | 0.05    | 0.10      | Large batch reduces gradient noise — smoother learning but slower to respond to new info. Expect stable curves, slower adaptation to changes.                  |
| 7   | 1e-3   | 0.97   | 32    | 1.0       | 0.10    | 0.30      | High lr + fast exploration end: initial fast gains but can be unstable; longer decay (0.3) keeps exploration longer — may improve final policy robustness if training long enough.                  |
| 8   | 5e-5   | 0.99   | 32    | 1.0       | 0.05    | 0.05      | Very low lr + short decay: very slow training, small incremental changes. Low eps_decay (0.05) means epsilon decays quickly -> early exploitation may hurt learning.                 |
| 9   | 2e-4   | 0.995  | 64    | 1.0       | 0.01    | 0.20      | Medium lr + very high gamma and low final epsilon: favors long-term planning; expect slower improvements but potentially higher long-run scores if stable.                  |
| 10  | 1e-4   | 0.98   | 128   | 1.0       | 0.05    | 0.25      | Low lr + lower gamma + big batch + long decay: slow, stable learning focused on short-term rewards; may show very gradual improvement and long episodes with low variance.                  |

---

## 📑 **Dean — Hyperparameter Experiments**

| Exp | lr   | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior                                                                           |
| --- | ---- | ----- | ----- | --------- | ------- | --------- | ------------------------------------------------------------------------------------------- |
| 1   | 1e-4 | 0.99  | 32    | 1.0       | 0.05    | 0.1       | Good and balanced performance coupled together with a stable performance                    |
| 2   | 1e-3 | 0.99  | 32    | 1.0       | 0.05    | 0.1       | Faster learning rate and highly unstable there are alot of oscillations                     |
| 3   | 1e-5 | 0.99  | 32    | 1.0       | 0.05    | 0.1       | Low learning rate hence slow and bare improvement over time                                 |
| 4   | 1e-4 | 0.95  | 32    | 1.0       | 0.05    | 0.1       | Low Gamma is dumb and focuses on immediate rewards                                          |
| 5   | 1e-4 | 0.995 | 32    | 1.0       | 0.05    | 0.1       | High Gamma is the best performant and focuses on greater rewards instead of instant rewards |
| 6   | 1e-4 | 0.99  | 64    | 1.0       | 0.05    | 0.1       | Large batch was poor it didn't really upgrade over time                                     |
| 7   | 1e-4 | 0.99  | 16    | 1.0       | 0.05    | 0.1       | Small batch was significantly better than the large batch but it came with much more noise  |
| 8   | 1e-4 | 0.99  | 32    | 1.0       | 0.1     | 0.2       | Slow exploration exploits too long                                                          |
| 9   | 1e-4 | 0.99  | 32    | 1.0       | 0.01    | 0.05      | Fast exploration explores too early and misses better strategies                            |
| 10  | 5e-5 | 0.99  | 64    | 1.0       | 0.1     | 0.15      | Conservative is stable but slow, cautious learning approach                                 |

# 🧠 **8. Policy Architecture Comparison**

## MLPPolicy

- Simple
- Works for 1D inputs
- ❌ Poor for image-based Atari games

## CNNPolicy

- Extracts spatial features
- Stable
- ✔ Best for Atari

### **Final Choice:** CNNPolicy

---

# 📈 **9. Key Insights from Tuning**

**Improvements:**

- γ = 0.99
- lr = 1e-4
- batch = 64–128
- slow epsilon decay

**Hurts performance:**

- High lr
- Small batch
- Fast epsilon decay

---

# 🎥 **10. Gameplay Demo**

Place video at:
`videos/gameplay.mp4`

Add link here.

---

# 🧑‍🤝‍🧑 **11. Group Collaboration Summary**

| Member | Contribution                      |
| ------ | --------------------------------- |
| David  | Training pipeline, debugging      |
| Gaius  | Hyperparameter tuning             |
| Renne  | Policy comparison                 |
| Dean   | Evaluation script, gameplay video |

---

# 📂 **12. Project Structure**

```
.
├── train.py
├── play.py
├── models/
│   └── dqn_model.zip
├── videos/
│   └── gameplay.mp4
├── results/
│   ├── logs.csv
│   └── plots/
├── requirements.txt
├── README.md
└── .venv/
```
>>>>>>> 6850bb19aded9b56d5ee077ab89efdead1f279c2
