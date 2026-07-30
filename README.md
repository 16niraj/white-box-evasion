# Dual-Penalty Tabular Evasion

Official codebase for the paper: **Crushing the Evidence: A Dual-Penalty Evasion Framework for Fooling White-Box Explainable AI Auditors**.

## Setup
1. Install dependencies:
   `pip install -r requirements.txt`
2. Place the IEEE-CIS dataset zip file (`ieee-fraud-detection.zip`) inside the `data/` directory. (Other datasets are downloaded automatically).
3. Run the main evaluation suite (averages results over 5 random seeds):
   `python main.py`
4. Run the ablation study and epoch tracking analysis (Table 3 and Figure 2):
   `python experiments.py`