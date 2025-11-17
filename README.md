# 🎮 **DQN Atari Agent — JourneyEscape Project**
### *Group Members: David · Gaus · Renne · Dean*

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
|-----|----|--------|--------|------------|-----------|-------------|-------------------|
| 1 | | | | | | | |
| ... | | | | | | | |
| 10 | | | | | | | |

---

## 📑 **Gaus — Hyperparameter Experiments**
| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
|-----|----|--------|--------|------------|-----------|-------------|-------------------|
| 1 | | | | | | | |
| ... | | | | | | | |
| 10 | | | | | | | |
---

## 📑 **Renne — Hyperparameter Experiments**
| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
|-----|----|--------|--------|------------|-----------|-------------|-------------------|
| 1 | | | | | | | |
| ... | | | | | | | |
| 10 | | | | | | | |
---

## 📑 **Dean — Hyperparameter Experiments**
| Exp | lr | gamma | batch | eps_start | eps_end | eps_decay | Observed Behavior |
|-----|----|--------|--------|------------|-----------|-------------|-------------------|
| 1 | | | | | | | |
| ... | | | | | | | |
| 10 | | | | | | | |
---

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

| Member | Contribution |
|--------|--------------|
| David | Training pipeline, debugging |
| Gaus | Hyperparameter tuning |
| Renne | Policy comparison |
| Dean | Evaluation script, gameplay video |

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
