# Aerospace Supply Chain Backorder Predictor

**[Live demo](https://mason-aerospace-supplier-risk-lead-time-predictor.streamlit.app/)**: runs in the browser, no install required.

End-to-end **decision-support system** that predicts weekly backorder risk
across **300 aerospace parts** and recommends a budget-bounded expedite
action plan. Built on real-world-shaped supply-chain data (parts master,
8 years of weekly inventory/lead-time history, purchase orders, quality
incidents) totalling **15 MB** across four CSVs.

The system is exposed through three layers:

1. A **Python pipeline** (`backorder_analysis.py`): load, feature-engineer,
   train, evaluate, and persist figures.
2. An **interactive Streamlit dashboard** (`backorder_dashboard_app.py`):
   weekly-risk explorer, model comparison, sensitivity sliders, and an
   expedite-recommendation engine that reads operations cost weights and
   criticality preferences in real time.
3. The bundled **CSVs and figures** so the project is fully reproducible.

## The data

| File | Rows | What it has |
|---|---:|---|
| `parts_master.csv` | 300 | Part number, criticality class (A/B/C), supplier, base price |
| `supply_chain_history.csv` | ~125,000 | Weekly inventory level, lead time, demand for every part-week |
| `purchase_orders.csv` | ~22,000 | PO line items: ordered qty, expected vs. actual delivery date |
| `quality_incidents.csv` | ~600 | Defect events tied to part numbers |

## Approach

1. **40+ engineered features** spanning inventory, purchase orders, and quality incidents:
   - Rolling lead-time mean / std / volatility
   - Inventory-vs-demand coverage ratios
   - Defect-rate features by supplier and part class
   - PO timing features (days late, fill rate)
   - Categorical one-hots for criticality and supplier
2. **Two classifiers compared**: Logistic Regression (scaled, balanced) and Random Forest. Tuned for **PR-AUC** because backorders are rare events (≈ 4 % positive rate); ROC-AUC over-weights the dominant negative class.
3. **Threshold tuning** for **F2 score**: recall counts twice as much as precision because missing a backorder is more expensive than expediting an unneeded part.
4. **Prescriptive layer**: top-K knapsack selection over the predicted-risk × criticality × cost-of-expedite tuple, returning the action plan that maximises expected value within the user's expedite budget.

## Results

| Model | PR-AUC | ROC-AUC | Recall @ F2-opt | Precision @ F2-opt |
|---|---|---|---|---|
| Operations rule baseline | 0.32 | 0.78 | 0.62 | 0.18 |
| Logistic Regression | 0.40 | 0.86 | 0.71 | 0.27 |
| **Random Forest** | **0.43** | **0.91** | **0.78** | **0.31** |

PR-AUC improvement of **+35 %** over the operations-rule baseline; the
recall lift of **+16 percentage points** translates to roughly a quarter
of previously-missed backorders being caught a week earlier.

## Repository layout

```
.
├── backorder_dashboard_app.py     ← Streamlit decision dashboard
├── backorder_analysis.py          ← analysis pipeline (load → train → save figures)
├── parts_master.csv               ← 300 parts × static attributes
├── supply_chain_history.csv       ← weekly time series (15 MB)
├── purchase_orders.csv            ← PO line items
├── quality_incidents.csv          ← defect events
├── requirements.txt
└── README.md
```

## Run it

### Standalone analysis

```bash
pip install -r requirements.txt
python backorder_analysis.py     # writes figures/ and prints model metrics
```

### Interactive dashboard

```bash
streamlit run backorder_dashboard_app.py
```

The dashboard:

- **Week explorer**: pick any week from the historical horizon and see the top-risk parts, their predicted backorder probability, and the recommended expedite action.
- **Model comparison**: Logistic Regression vs. Random Forest with PR curves, confusion matrices, and per-class precision/recall.
- **Expedite recommendation engine**: sliders for total expedite budget plus criticality weights (`α_A`, `α_B` relative to class C); the dashboard re-solves the knapsack and shows the new action list immediately.
- **Sensitivity analysis**: shift the decision threshold and see how recall, precision, and the action list shift.

## Stack

Python · scikit-learn (Logistic Regression, Random Forest) · pandas ·
NumPy · matplotlib · seaborn · plotly · **Streamlit** (dashboard)
