"""
Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
Data files expected in same directory: parts_master.csv, supply_chain_history.csv,
    purchase_orders.csv, quality_incidents.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend; remove this line if running interactively
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, auc, confusion_matrix,
                             roc_auc_score, recall_score, precision_score)
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Output directory for figures
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

parts = pd.read_csv('parts_master.csv')
sch = pd.read_csv('supply_chain_history.csv', parse_dates=['date'])
po = pd.read_csv('purchase_orders.csv',
                  parse_dates=['order_date', 'promised_date', 'receipt_date'])
qi = pd.read_csv('quality_incidents.csv', parse_dates=['incident_date'])

print(f"Parts Master:         {parts.shape[0]:>6} rows x {parts.shape[1]} cols")
print(f"Supply Chain History: {sch.shape[0]:>6} rows x {sch.shape[1]} cols")
print(f"Purchase Orders:      {po.shape[0]:>6} rows x {po.shape[1]} cols")
print(f"Quality Incidents:    {qi.shape[0]:>6} rows x {qi.shape[1]} cols")

print(f"\nParts - Criticality classes: {parts['criticality_class'].value_counts().to_dict()}")
print(f"Parts - Supplier risk:       {parts['supplier_risk_class'].value_counts().to_dict()}")
print(f"Parts - Part families:       {parts['part_family'].value_counts().to_dict()}")
print(f"\nHistory date range: {sch['date'].min().date()} to {sch['date'].max().date()}")
print(f"Unique parts: {sch['part_id'].nunique()}, Unique sites: {sch['site_id'].nunique()}")

print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# Purchase order derived columns
po['actual_lead_time'] = (po['receipt_date'] - po['order_date']).dt.days
po['late_flag'] = (po['receipt_date'] > po['promised_date']).astype(int)

print(f"Late deliveries: {po['late_flag'].sum()} / {len(po)} ({100*po['late_flag'].mean():.1f}%)")
print(f"Avg actual lead time: {po['actual_lead_time'].mean():.1f} days")
print(f"Shelf_life_days missing: {parts['shelf_life_days'].isna().mean()*100:.1f}% (excluded)")

# Merge parts info into supply chain history for EDA
sch_eda = sch.merge(parts, on='part_id', how='left')
sch_eda['has_backorder'] = (sch_eda['backorder_qty'] > 0).astype(int)

print(f"Backorder > 0 rows: {sch_eda['has_backorder'].sum()} / {len(sch_eda)} "
      f"({100*sch_eda['has_backorder'].mean():.2f}%)")


print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- Figure 1: Pareto Analysis  ---
# Pareto analysis ranks items by their contribution to a total and computes
# the cumulative percentage. This determines whether backorders are concentrated
# (justifying targeted top-K intervention) or uniformly distributed.
bo_by_part = sch_eda.groupby('part_id')['backorder_qty'].sum().sort_values(ascending=False)
bo_by_part = bo_by_part[bo_by_part > 0]
cumulative_pct = bo_by_part.cumsum() / bo_by_part.sum() * 100

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(range(len(bo_by_part)), bo_by_part.values, color='steelblue', alpha=0.7)
ax2 = ax1.twinx()
ax2.plot(range(len(bo_by_part)), cumulative_pct.values, color='firebrick', linewidth=2)
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Parts (ranked by total backorder qty)')
ax1.set_ylabel('Total Backorder Quantity')
ax2.set_ylabel('Cumulative % of Backorders')
ax1.set_xticks([])
ax1.set_title('Figure 1: Pareto Analysis of Backorder Concentration')
n_80 = (cumulative_pct <= 80).sum()
ax1.annotate(f'Top {n_80} parts ({100*n_80/len(bo_by_part):.0f}%) account\nfor 80% of backorders',
             xy=(n_80, 80), fontsize=9,
             xytext=(n_80+30, 60), arrowprops=dict(arrowstyle='->', color='gray'))
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_pareto.png', bbox_inches='tight')
plt.close()
print(f"Pareto: top {n_80} of {len(bo_by_part)} parts cause 80% of backorders")

# --- Figure 2: Forecast Accuracy Analysis  ---
# Evaluates the system demand forecast (forecast_qty) against actual consumption
# to decide whether to include it as a model feature. Uses MAE and bias with a 
# 4-week SMA as naive benchmark.
forecast_df = sch_eda[['date', 'part_id', 'site_id',
                       'consumption_qty', 'forecast_qty']].copy()
forecast_df['error'] = forecast_df['forecast_qty'] - forecast_df['consumption_qty']
forecast_df['abs_error'] = forecast_df['error'].abs()

mae = forecast_df['abs_error'].mean()
bias = forecast_df['error'].mean()

# 4-week Simple Moving Average as naive benchmark
sch_sorted = sch_eda.sort_values(['part_id', 'site_id', 'date'])
sch_sorted['sma_4_forecast'] = sch_sorted.groupby(
    ['part_id', 'site_id'])['consumption_qty'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean())
sch_sorted['sma_error'] = (sch_sorted['sma_4_forecast']
                           - sch_sorted['consumption_qty']).abs()
sma_mae = sch_sorted['sma_error'].dropna().mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].hist(forecast_df['error'].dropna().clip(-10, 10), bins=40,
             color='steelblue', alpha=0.7, edgecolor='white')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[0].axvline(x=bias, color='orange', linestyle='--', linewidth=1.5,
                label=f'Mean Bias: {bias:.2f}')
axes[0].set_xlabel('Forecast Error (forecast - actual)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('System Forecast Error Distribution')
axes[0].legend()

axes[1].bar(['System\nForecast', '4-Week SMA'], [mae, sma_mae],
            color=['steelblue', '#ff7f0e'])
axes[1].set_ylabel('Mean Absolute Error (MAE)')
axes[1].set_title('Forecast Accuracy Comparison')
for i, v in enumerate([mae, sma_mae]):
    axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Figure 2: Forecast Accuracy Analysis (Nahmias Ch. 2)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_forecast.png', bbox_inches='tight')
plt.close()
print(f"Forecast: System MAE={mae:.2f}, SMA MAE={sma_mae:.2f}, Bias={bias:.2f}")

# --- Figure 3: Safety Stock Gap Analysis ---
# Compares each part-site's actual inventory buffer against the theoretically
# required safety stock from the (Q,R) framework: s = z * sigma_LTD,
# where sigma_LTD = sigma_demand * sqrt(L).
# This determines whether backorders are caused by structural under-buffering
# (policy problem) or unpredictable demand spikes (forecasting problem).
z_95 = 1.65  # z for 95% Type 1 service level

consumption_stats = sch_eda.groupby(['part_id', 'site_id']).agg(
    mean_consumption=('consumption_qty', 'mean'),
    std_consumption=('consumption_qty', 'std'),
    lead_time=('lead_time_days', 'first'),
    avg_on_hand=('on_hand_qty', 'mean'),
    avg_blocked=('blocked_qty', 'mean'),
    avg_backorder=('backorder_qty', 'mean'),
    bo_rate=('has_backorder', 'mean'),
    criticality=('criticality_class', 'first')
).reset_index()

# Lead time in weeks (data is weekly)
consumption_stats['lead_time_weeks'] = consumption_stats['lead_time'] / 7

# sigma_LTD = sigma_weekly * sqrt(lead_time_weeks)
consumption_stats['sigma_ltd'] = (consumption_stats['std_consumption'] *
                                   np.sqrt(consumption_stats['lead_time_weeks']))

# Required safety stock = z * sigma_LTD
consumption_stats['required_ss'] = z_95 * consumption_stats['sigma_ltd']

# Actual buffer = avg on_hand - avg blocked
consumption_stats['actual_buffer'] = (consumption_stats['avg_on_hand'] -
                                       consumption_stats['avg_blocked'])

# Safety stock gap = actual buffer - required safety stock (negative = under-protected)
consumption_stats['ss_gap'] = consumption_stats['actual_buffer'] - consumption_stats['required_ss']
consumption_stats['under_protected'] = (consumption_stats['ss_gap'] < 0).astype(int)

n_under = consumption_stats['under_protected'].sum()
n_total_ps = len(consumption_stats)
pct_under = 100 * n_under / n_total_ps
bo_under = consumption_stats[consumption_stats['under_protected'] == 1]['bo_rate'].mean() * 100
bo_adequate = consumption_stats[consumption_stats['under_protected'] == 0]['bo_rate'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].hist(consumption_stats['ss_gap'].clip(-30, 50), bins=40,
             color='steelblue', alpha=0.7, edgecolor='white')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Gap = 0 (threshold)')
axes[0].set_xlabel('Safety Stock Gap (actual buffer - required SS)')
axes[0].set_ylabel('Frequency (part-site combinations)')
axes[0].set_title('Safety Stock Gap Distribution')
axes[0].legend()
axes[0].annotate(f'{pct_under:.0f}% under-\nprotected',
                 xy=(-15, 50), fontsize=10, fontweight='bold', color='#d62728')

labels_ss = ['Under-Protected\n(gap < 0)', 'Adequately\nProtected (gap \u2265 0)']
rates_ss = [bo_under, bo_adequate]
axes[1].bar(labels_ss, rates_ss, color=['#d62728', '#2ca02c'])
axes[1].set_ylabel('Average Backorder Rate (%)')
axes[1].set_title('Backorder Rate by Safety Stock Adequacy')
for i, v in enumerate(rates_ss):
    axes[1].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Figure 3: Safety Stock Gap Analysis (Nahmias Ch. 5)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_safetystock.png', bbox_inches='tight')
plt.close()

print(f"Safety Stock: {n_under}/{n_total_ps} ({pct_under:.0f}%) under-protected")
print(f"  Under-protected BO rate: {bo_under:.2f}%, Adequate BO rate: {bo_adequate:.2f}%")
for crit in ['A', 'B', 'C']:
    mask = consumption_stats['criticality'] == crit
    n_u = consumption_stats.loc[mask, 'under_protected'].sum()
    n_t = mask.sum()
    print(f"  Criticality {crit}: {n_u}/{n_t} ({100*n_u/n_t:.0f}%) under-protected")

# --- Figure 4: Lead Time Variability Analysis ---
# The safety stock formula assumes constant lead time. When lead time is variable,
# sigma_LTD increases further. 
# This validates lead time std as a model input.

lt_stats = po.groupby('supplier_id').agg(
    mean_lt=('actual_lead_time', 'mean'),
    std_lt=('actual_lead_time', 'std'),
    n_orders=('po_id', 'count'),
    late_rate=('late_flag', 'mean')
).reset_index()
lt_stats['lt_cv'] = lt_stats['std_lt'] / lt_stats['mean_lt'].replace(0, np.nan)

# Add supplier risk class
sup_risk = parts.drop_duplicates('supplier_id_primary')[
    ['supplier_id_primary', 'supplier_risk_class']]
sup_risk.columns = ['supplier_id', 'supplier_risk_class']
lt_stats = lt_stats.merge(sup_risk, on='supplier_id', how='left')

# Add backorder rate per supplier
bo_by_supplier = sch_eda.groupby('supplier_id_primary')['has_backorder'].mean().reset_index()
bo_by_supplier.columns = ['supplier_id', 'bo_rate']
lt_stats = lt_stats.merge(bo_by_supplier, on='supplier_id', how='left')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

cv_by_risk = lt_stats.groupby('supplier_risk_class')['lt_cv'].mean()
cv_by_risk = cv_by_risk.reindex(['Low', 'Medium', 'High'])
axes[0].bar(cv_by_risk.index, cv_by_risk.values, color=['#2ca02c', '#ff7f0e', '#d62728'])
axes[0].set_ylabel('Mean Lead Time CV (\u03c3/\u03bc)')
axes[0].set_xlabel('Supplier Risk Class')
axes[0].set_title('Lead Time Variability by Supplier Risk')
for i, v in enumerate(cv_by_risk.values):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

valid = lt_stats.dropna(subset=['lt_cv', 'bo_rate'])
colors_map = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}
for risk in ['Low', 'Medium', 'High']:
    mask = valid['supplier_risk_class'] == risk
    if mask.any():
        axes[1].scatter(valid.loc[mask, 'lt_cv'], valid.loc[mask, 'bo_rate'] * 100,
                        s=valid.loc[mask, 'n_orders'] / 3, alpha=0.6,
                        color=colors_map[risk], label=risk, edgecolors='navy', linewidth=0.3)
axes[1].set_xlabel('Lead Time CV (\u03c3/\u03bc)')
axes[1].set_ylabel('Backorder Rate (%)')
axes[1].set_title('Lead Time Variability vs Backorder Rate')
axes[1].legend(title='Supplier Risk')

corr = valid[['lt_cv', 'bo_rate']].corr().iloc[0, 1]
axes[1].annotate(f'r = {corr:.3f}', xy=(0.05, 0.92), xycoords='axes fraction',
                 fontsize=10, fontweight='bold')

fig.suptitle('Figure 4: Lead Time Variability Analysis (Nahmias Ch. 5\u20136)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_leadtime_cv.png', bbox_inches='tight')
plt.close()

print(f"Lead Time CV by risk: {cv_by_risk.to_dict()}")
print(f"Correlation (LT CV vs BO rate): r = {corr:.3f}")

print("All EDA figures saved to figures/")


print("\n" + "=" * 60)
print("4. DATA PREPARATION AND PRE-PROCESSING")
print("=" * 60)

# Sort for lag computation
sch = sch.sort_values(['part_id', 'site_id', 'date']).reset_index(drop=True)

# --- Inventory Pressure Metrics ---
# Usable inventory = on_hand minus blocked minus existing backorders
sch['net_available'] = sch['on_hand_qty'] - sch['blocked_qty'] - sch['backorder_qty']

# Previous-period consumption (shift by 1 period to avoid using future information)
sch['consumption_qty_lag1'] = sch.groupby(['part_id', 'site_id'])['consumption_qty'].shift(1)

# 4-week moving average and standard deviation of consumption
sch['consumption_qty_roll4_mean'] = sch.groupby(['part_id', 'site_id'])['consumption_qty'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean())
sch['consumption_qty_roll4_std'] = sch.groupby(['part_id', 'site_id'])['consumption_qty'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).std())

print("Inventory pressure metrics: net_available, consumption lag/avg/std, forecast")

# --- Supply Pipeline Metrics ---
# Aggregate PO statistics per part-site (historical averages)
po_stats = po.groupby(['part_id', 'site_id']).agg(
    avg_actual_lead=('actual_lead_time', 'mean'),
    std_actual_lead=('actual_lead_time', 'std'),
    supplier_late_rate=('late_flag', 'mean'),
).reset_index()

sch = sch.merge(po_stats, on=['part_id', 'site_id'], how='left')

print("Supply pipeline metrics: avg/std lead time, late rate")

# --- Quality Risk Metrics ---
qi_counts = qi.groupby(['part_id', 'site_id']).agg(
    incident_count=('incident_id', 'count'),
    critical_incidents=('defect_severity', lambda x: (x == 'Critical').sum()),
    major_incidents=('defect_severity', lambda x: (x == 'Major').sum())
).reset_index()

sch = sch.merge(qi_counts, on=['part_id', 'site_id'], how='left')
for col in ['incident_count', 'critical_incidents', 'major_incidents']:
    sch[col] = sch[col].fillna(0)

print("Quality risk metrics: incident counts, critical/major severity counts")

# --- Part Attributes ---
sch = sch.merge(parts[['part_id', 'criticality_class', 'unit_cost',
                        'supplier_risk_class', 'part_family']],
                on='part_id', how='left')

# Convert text categories to yes/no columns (models need numbers, not text)
sch['crit_A'] = (sch['criticality_class'] == 'A').astype(int)
sch['crit_B'] = (sch['criticality_class'] == 'B').astype(int)
sch['risk_High'] = (sch['supplier_risk_class'] == 'High').astype(int)
sch['risk_Medium'] = (sch['supplier_risk_class'] == 'Medium').astype(int)

print("Part attributes: unit cost, criticality, supplier risk")

print("\n" + "=" * 60)
print("5. TARGET & SPLIT")
print("=" * 60)

# Target: backorder in the NEXT week (shift forward by 1)
sch['target_backorder_next'] = sch.groupby(['part_id', 'site_id'])['backorder_qty'].shift(-1)
sch['target'] = (sch['target_backorder_next'] > 0).astype(int)

# 16 model inputs across four groups
feature_cols = [
    # Inventory Pressure (5)
    'net_available', 'consumption_qty_lag1',
    'consumption_qty_roll4_mean', 'consumption_qty_roll4_std',
    'forecast_qty',
    # Supply Pipeline (3)
    'avg_actual_lead', 'std_actual_lead', 'supplier_late_rate',
    # Quality Risk (3)
    'incident_count', 'critical_incidents', 'major_incidents',
    # Part Attributes (5)
    'unit_cost', 'crit_A', 'crit_B', 'risk_High', 'risk_Medium',
]

# Drop rows with missing target or features
df = sch.dropna(subset=['target']).copy()
df = df.dropna(subset=feature_cols)

print(f"Modeling dataset: {df.shape[0]} rows, {len(feature_cols)} features")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")
print(f"Backorder rate: {df['target'].mean()*100:.2f}%")

# Time-based split (essential to avoid data leakage)
split_date = pd.Timestamp('2024-07-01')
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

print(f"\nTrain: {len(train):,} rows ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Test:  {len(test):,} rows ({test['date'].min().date()} to {test['date'].max().date()})")
print(f"Train backorder rate: {train['target'].mean()*100:.2f}%")
print(f"Test backorder rate:  {test['target'].mean()*100:.2f}%")

X_train = train[feature_cols].values
X_test = test[feature_cols].values
y_train = train['target'].values
y_test = test['target'].values


print("\n" + "=" * 60)
print("6. MODELING")
print("=" * 60)

# --- 6a. Baseline: Net available threshold rule ---
print("\n--- Baseline (net_available <= 0) ---")
net_avail_idx = feature_cols.index('net_available')
baseline_pred = (X_test[:, net_avail_idx] <= 0).astype(int)
baseline_proba = -X_test[:, net_avail_idx]  # lower net_available = higher risk

print(f"Predictions positive: {baseline_pred.sum()} / {len(baseline_pred)}")
print(f"Recall:    {recall_score(y_test, baseline_pred):.4f}")
print(f"Precision: {precision_score(y_test, baseline_pred, zero_division=0):.4f}")

# --- 6b. Logistic Regression ---
print("\n--- Logistic Regression (class_weight='balanced', C=0.1) ---")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.1)
lr.fit(X_train_s, y_train)
lr_proba = lr.predict_proba(X_test_s)[:, 1]
lr_pred = (lr_proba >= 0.5).astype(int)

print(f"Predictions positive: {lr_pred.sum()} / {len(lr_pred)}")
print(f"ROC-AUC:   {roc_auc_score(y_test, lr_proba):.4f}")

# Print coefficients
lr_coefs = pd.DataFrame({'feature': feature_cols, 'coefficient': lr.coef_[0]})
lr_coefs = lr_coefs.sort_values('coefficient', key=abs, ascending=False)
print("\nTop LR coefficients (by magnitude):")
for _, row in lr_coefs.head(10).iterrows():
    print(f"  {row['feature']:>30s}: {row['coefficient']:+.4f}")

# --- 6c. Random Forest ---
print("\n--- Random Forest (200 trees, max_depth=15, balanced) ---")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_proba >= 0.5).astype(int)

print(f"Predictions positive: {rf_pred.sum()} / {len(rf_pred)}")
print(f"ROC-AUC:   {roc_auc_score(y_test, rf_proba):.4f}")

print("\n" + "=" * 60)
print("7. EVALUATION")
print("=" * 60)

# --- PR-AUC ---
print("\n--- PR-AUC ---")
for name, scores in [('Logistic Regression', lr_proba), ('Random Forest', rf_proba)]:
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rec_vals, prec_vals)
    print(f"  {name}: PR-AUC = {pr_auc:.4f}")

# --- Precision@K / Recall@K ---
def precision_recall_at_k(y_true, scores, k):
    """Compute Precision@K and Recall@K for top-K scored items."""
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_true = y_true[top_k_idx]
    prec = top_k_true.sum() / k
    rec = top_k_true.sum() / max(y_true.sum(), 1)
    return prec, rec

K_values = [10, 25, 50, 100, 200]
print("\n--- Precision@K / Recall@K ---")
print(f"{'Model':>20s} | {'K':>5s} | {'P@K':>8s} | {'R@K':>8s}")
print("-" * 50)

results = {'K': [], 'Model': [], 'Precision@K': [], 'Recall@K': []}
for k in K_values:
    for name, scores in [('Baseline', baseline_proba),
                         ('Logistic Reg.', lr_proba),
                         ('Random Forest', rf_proba)]:
        p_k, r_k = precision_recall_at_k(y_test, scores, k)
        results['K'].append(k)
        results['Model'].append(name)
        results['Precision@K'].append(p_k)
        results['Recall@K'].append(r_k)
        print(f"{name:>20s} | {k:>5d} | {p_k:>8.4f} | {r_k:>8.4f}")

results_df = pd.DataFrame(results)

# --- Confusion Matrix (RF) ---
cm = confusion_matrix(y_test, rf_pred)
print(f"\nRF Confusion Matrix:")
print(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>5}")
print(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>5}")

# --- Figure 6: PR Curves ---
fig, ax = plt.subplots(figsize=(7, 5))
for name, scores, color in [('Logistic Regression', lr_proba, 'steelblue'),
                              ('Random Forest', rf_proba, 'firebrick')]:
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, scores)
    pr_auc_val = auc(rec_vals, prec_vals)
    ax.plot(rec_vals, prec_vals, color=color, linewidth=2,
            label=f'{name} (PR-AUC={pr_auc_val:.3f})')
ax.axhline(y=y_test.mean(), color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline rate ({y_test.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Figure 6: Precision-Recall Curves')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_pr_curve.png', bbox_inches='tight')
plt.close()

# --- Figure 7: Precision@K and Recall@K comparison ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for model, color in [('Baseline', 'gray'), ('Logistic Reg.', 'steelblue'),
                      ('Random Forest', 'firebrick')]:
    sub = results_df[results_df['Model'] == model]
    axes[0].plot(sub['K'], sub['Precision@K'], marker='o', color=color, label=model)
    axes[1].plot(sub['K'], sub['Recall@K'], marker='o', color=color, label=model)
axes[0].set_xlabel('K (Expedite Capacity)')
axes[0].set_ylabel('Precision@K')
axes[0].set_title('Precision@K')
axes[0].legend()
axes[1].set_xlabel('K (Expedite Capacity)')
axes[1].set_ylabel('Recall@K')
axes[1].set_title('Recall@K')
axes[1].legend()
fig.suptitle('Figure 7: Model Comparison at Various Expedite Capacities', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_precision_recall_k.png', bbox_inches='tight')
plt.close()

# --- Figure 8: Feature Importance ---
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Figure 8: Random Forest Feature Importances')
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_feature_importance.png', bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("8. PRESCRIPTIVE OPTIMIZATION")
print("=" * 60)

# Add predictions to test dataframe
test_df = test.copy()
test_df['rf_proba'] = rf_proba
test_df['rf_pred'] = rf_pred

# Risk score = predicted probability x severity weight
alpha_map = {'A': 5.0, 'B': 2.0, 'C': 1.0}
test_df['alpha'] = test_df['criticality_class'].map(alpha_map)
test_df['severity_weight'] = test_df['alpha'] * test_df['unit_cost']
test_df['risk_score'] = test_df['rf_proba'] * test_df['severity_weight']

# --- 9a. Capacity-Constrained (Top-K) across all test weeks ---
print("\n--- Capacity-Constrained (Top-K) Aggregate Results ---")
weeks = test_df['date'].unique()
agg_results = {k: {'caught': 0, 'total': 0} for k in [10, 25, 50]}

for week in weeks:
    week_df = test_df[test_df['date'] == week]
    actual_bo = week_df['target'].sum()
    if actual_bo == 0:
        continue
    week_sorted = week_df.sort_values('risk_score', ascending=False)
    for K in [10, 25, 50]:
        top_k = week_sorted.head(K)
        caught = top_k['target'].sum()
        agg_results[K]['caught'] += caught
        agg_results[K]['total'] += actual_bo

n_bo_weeks = sum(1 for w in weeks if test_df[test_df['date'] == w]['target'].sum() > 0)
print(f"Weeks with backorders: {n_bo_weeks}")
for K in [10, 25, 50]:
    r = agg_results[K]
    print(f"  K={K:>3}: Caught {r['caught']}/{r['total']} "
          f"({100*r['caught']/max(r['total'],1):.1f}%)")

# --- 9b. Budget-Constrained (Knapsack) on demo week ---
print("\n--- Budget-Constrained (Knapsack) Demo Week ---")
bo_weeks = test_df.groupby('date')['target'].sum().reset_index()
bo_weeks = bo_weeks[bo_weeks['target'] > 5].sort_values('target', ascending=False)
demo_date = bo_weeks.iloc[0]['date']
demo_week = test_df[test_df['date'] == demo_date].copy()
demo_week['expedite_cost'] = demo_week['unit_cost'] * 0.15  # 15% of unit cost

print(f"Demo week: {demo_date.date()}, {demo_week['target'].sum()} actual backorders")
print(f"{'Budget':>10} | {'# Sel':>6} | {'Caught':>8} | {'Recall':>8} | {'Avg Cost':>10}")
print("-" * 55)

budgets = [5000, 10000, 25000, 50000]
for B in budgets:
    temp = demo_week.copy()
    temp['efficiency'] = temp['risk_score'] / temp['expedite_cost']
    temp = temp.sort_values('efficiency', ascending=False)

    selected = []
    spent = 0
    for _, row in temp.iterrows():
        if spent + row['expedite_cost'] <= B:
            selected.append(row)
            spent += row['expedite_cost']

    sel_df = pd.DataFrame(selected)
    if len(sel_df) > 0:
        caught = sel_df['target'].sum()
        total = demo_week['target'].sum()
        print(f"${B:>9,} | {len(sel_df):>6} | {int(caught):>3}/{int(total):>3} | "
              f"{caught/max(total,1):>8.1%} | ${spent/len(sel_df):>9,.0f}")

print("\n" + "=" * 60)
print("9. SENSITIVITY ANALYSIS")
print("=" * 60)

# Use demo week for sensitivity figures
demo_sorted = demo_week.sort_values('risk_score', ascending=False)

# --- Figure 10: K sensitivity + Criticality weight sensitivity ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel 1: K sensitivity
k_range = range(5, 201, 5)
recalls_k = []
precisions_k = []
total_actual = demo_week['target'].sum()
for k in k_range:
    top_k = demo_sorted.head(k)
    caught = top_k['target'].sum()
    recalls_k.append(caught / max(total_actual, 1))
    precisions_k.append(caught / k)

axes[0].plot(list(k_range), recalls_k, color='steelblue', linewidth=2, label='Recall@K')
axes[0].plot(list(k_range), precisions_k, color='firebrick', linewidth=2, label='Precision@K')
axes[0].set_xlabel('K (Number of Expedites)')
axes[0].set_ylabel('Score')
axes[0].set_title('Sensitivity to Expedite Capacity (K)')
axes[0].legend()
axes[0].axvline(x=25, color='gray', linestyle='--', alpha=0.5)

# Panel 2: Criticality weight sensitivity
alpha_scenarios = {
    'Equal\n(1:1:1)': {'A': 1, 'B': 1, 'C': 1},
    'Mild\n(3:2:1)': {'A': 3, 'B': 2, 'C': 1},
    'Default\n(5:2:1)': {'A': 5, 'B': 2, 'C': 1},
    'Extreme\n(10:3:1)': {'A': 10, 'B': 3, 'C': 1},
}

K_fixed = 25
scenario_results = []
for scenario_name, alphas in alpha_scenarios.items():
    temp = demo_week.copy()
    temp['alpha'] = temp['criticality_class'].map(alphas)
    temp['risk_score'] = temp['rf_proba'] * temp['alpha'] * temp['unit_cost']
    top_k = temp.sort_values('risk_score', ascending=False).head(K_fixed)
    caught = top_k['target'].sum()
    total = demo_week['target'].sum()
    crit_a_selected = (top_k['criticality_class'] == 'A').sum()
    scenario_results.append({
        'Scenario': scenario_name,
        'Recall@25': caught / max(total, 1),
        'Crit-A Selected': crit_a_selected
    })

sr = pd.DataFrame(scenario_results)
x = np.arange(len(sr))
axes[1].bar(x - 0.15, sr['Recall@25'], 0.3, label='Recall@25', color='steelblue')
ax2 = axes[1].twinx()
ax2.bar(x + 0.15, sr['Crit-A Selected'], 0.3, label='Crit-A Selected',
        color='#d62728', alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(sr['Scenario'], fontsize=8)
axes[1].set_ylabel('Recall@25')
ax2.set_ylabel('# Criticality-A Parts Selected')
axes[1].set_title(f'Sensitivity to Criticality Weights (K={K_fixed})')
axes[1].legend(loc='upper left')
ax2.legend(loc='upper right')

fig.suptitle('Figure 10: Sensitivity Analysis', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_sensitivity.png', bbox_inches='tight')
plt.close()

# --- Figure 11: Expedite effectiveness sensitivity ---
scenarios = {
    '0 days\n(no effect)': 0,
    '3 days\nreduction': 3,
    '7 days\nreduction': 7,
    '14 days\nreduction': 14,
}

K_sens = 25
scenario_data = []
for scenario_name, days_saved in scenarios.items():
    total_preventable = 0
    total_bo = 0
    for week in weeks:
        week_df = test_df[test_df['date'] == week].copy()
        actual_bo = week_df['target'].sum()
        if actual_bo == 0:
            continue
        total_bo += actual_bo
        top_k = week_df.sort_values('risk_score', ascending=False).head(K_sens)
        if days_saved > 0:
            benefiting = top_k[top_k['target'] == 1]
            for _, row in benefiting.iterrows():
                prevention_prob = min(
                    days_saved / max(row.get('avg_actual_lead', 30), 1), 1.0)
                total_preventable += prevention_prob

    scenario_data.append({
        'Scenario': scenario_name,
        'Est. Prevented': total_preventable,
        'Total BO': total_bo,
        'Prevention Rate': total_preventable / max(total_bo, 1)
    })

sdf = pd.DataFrame(scenario_data)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(len(sdf)), sdf['Prevention Rate'] * 100,
       color=['gray', '#2ca02c', '#ff7f0e', '#d62728'])
ax.set_xticks(range(len(sdf)))
ax.set_xticklabels(sdf['Scenario'], fontsize=9)
ax.set_ylabel('Estimated Backorder Prevention Rate (%)')
ax.set_title(f'Figure 11: Sensitivity to Expedite Effectiveness (K={K_sens})')
ax.set_xlabel('Lead Time Reduction from Expediting')
plt.tight_layout()
plt.savefig(f'{OUT}/fig11_expedite_effectiveness.png', bbox_inches='tight')
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF KEY RESULTS")
print("=" * 60)

lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_proba)
rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_proba)

summary = {
    'Dataset': {
        'Train size': f"{len(train):,}",
        'Test size': f"{len(test):,}",
        'Features': len(feature_cols),
        'Train BO rate': f"{train['target'].mean()*100:.2f}%",
        'Test BO rate': f"{test['target'].mean()*100:.2f}%",
    },
    'Model Performance': {
        'Baseline Recall': f"{recall_score(y_test, baseline_pred):.4f}",
        'Baseline Precision': f"{precision_score(y_test, baseline_pred, zero_division=0):.4f}",
        'LR ROC-AUC': f"{roc_auc_score(y_test, lr_proba):.4f}",
        'LR PR-AUC': f"{auc(lr_rec, lr_prec):.4f}",
        'RF ROC-AUC': f"{roc_auc_score(y_test, rf_proba):.4f}",
        'RF PR-AUC': f"{auc(rf_rec, rf_prec):.4f}",
    }
}

for section, metrics in summary.items():
    print(f"\n{section}:")
    for k, v in metrics.items():
        print(f"  {k:>25s}: {v}")

print(f"\nAll figures saved to: {os.path.abspath(OUT)}/")
print("Done.")
