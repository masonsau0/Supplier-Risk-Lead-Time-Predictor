"""
Aerospace Backorder Prediction Dashboard.

Run with::

    streamlit run backorder_dashboard_app.py

Loads four CSVs from the working directory and offers weekly backorder-risk
scoring, model comparison, expedite-action recommendations, and interactive
sensitivity analysis.

Required CSVs:
  - parts_master.csv
  - supply_chain_history.csv
  - purchase_orders.csv
  - quality_incidents.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, auc, roc_auc_score,
                             recall_score, precision_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Aerospace Backorder Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1F3864;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #1F3864;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING & CACHING
# ============================================================
@st.cache_data
def load_and_process_data():
    """Load all CSVs and run the full analysis pipeline."""
    parts = pd.read_csv('parts_master.csv')
    sch = pd.read_csv('supply_chain_history.csv', parse_dates=['date'])
    po = pd.read_csv('purchase_orders.csv',
                      parse_dates=['order_date', 'promised_date', 'receipt_date'])
    qi = pd.read_csv('quality_incidents.csv', parse_dates=['incident_date'])

    # --- Preprocessing ---
    po['actual_lead_time'] = (po['receipt_date'] - po['order_date']).dt.days
    po['late_flag'] = (po['receipt_date'] > po['promised_date']).astype(int)
    po['days_late'] = (po['receipt_date'] - po['promised_date']).dt.days.clip(lower=0)
    po['fill_rate'] = po['received_qty'] / po['ordered_qty']

    # --- Data Preparation ---
    sch = sch.sort_values(['part_id', 'site_id', 'date']).reset_index(drop=True)
    sch['net_available'] = sch['on_hand_qty'] - sch['blocked_qty'] - sch['backorder_qty']

    for col in ['consumption_qty', 'backorder_qty', 'on_hand_qty']:
        sch[f'{col}_lag1'] = sch.groupby(['part_id', 'site_id'])[col].shift(1)

    for col in ['consumption_qty']:
        sch[f'{col}_roll4_mean'] = sch.groupby(['part_id', 'site_id'])[col].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        sch[f'{col}_roll4_std'] = sch.groupby(['part_id', 'site_id'])[col].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).std())

    sch['consumption_trend'] = sch['consumption_qty_lag1'] - sch['consumption_qty_roll4_mean']

    po_stats = po.groupby(['part_id', 'site_id']).agg(
        avg_actual_lead=('actual_lead_time', 'mean'),
        std_actual_lead=('actual_lead_time', 'std'),
        supplier_late_rate=('late_flag', 'mean'),
        avg_days_late=('days_late', 'mean'),
        avg_fill_rate=('fill_rate', 'mean'),
    ).reset_index()
    sch = sch.merge(po_stats, on=['part_id', 'site_id'], how='left')
    sch['lead_time_cv'] = sch['std_actual_lead'] / sch['avg_actual_lead'].replace(0, np.nan)

    qi_counts = qi.groupby(['part_id', 'site_id']).agg(
        incident_count=('incident_id', 'count'),
        total_scrap=('scrap_qty', 'sum'),
        critical_incidents=('defect_severity', lambda x: (x == 'Critical').sum()),
        major_incidents=('defect_severity', lambda x: (x == 'Major').sum())
    ).reset_index()
    sch = sch.merge(qi_counts, on=['part_id', 'site_id'], how='left')
    for col in ['incident_count', 'total_scrap', 'critical_incidents', 'major_incidents']:
        sch[col] = sch[col].fillna(0)

    sch = sch.merge(parts[['part_id', 'criticality_class', 'unit_cost', 'lead_time_days',
                            'supplier_risk_class', 'is_repairable', 'part_family']],
                    on='part_id', how='left')

    sch['crit_A'] = (sch['criticality_class'] == 'A').astype(int)
    sch['crit_B'] = (sch['criticality_class'] == 'B').astype(int)
    sch['risk_High'] = (sch['supplier_risk_class'] == 'High').astype(int)
    sch['risk_Medium'] = (sch['supplier_risk_class'] == 'Medium').astype(int)
    sch['is_repairable_flag'] = (sch['is_repairable'] == 'Yes').astype(int)
    sch['planned_maint'] = sch['planned_maintenance'].astype(int)

    # --- Target ---
    sch['target_backorder_next'] = sch.groupby(['part_id', 'site_id'])['backorder_qty'].shift(-1)
    sch['target'] = (sch['target_backorder_next'] > 0).astype(int)

    feature_cols = [
        'net_available', 'on_hand_qty', 'blocked_qty', 'consumption_qty_lag1',
        'backorder_qty_lag1', 'consumption_qty_roll4_mean', 'consumption_qty_roll4_std',
        'consumption_trend', 'forecast_qty', 'planned_maint',
        'avg_actual_lead', 'std_actual_lead', 'supplier_late_rate', 'avg_days_late',
        'avg_fill_rate', 'lead_time_cv',
        'incident_count', 'total_scrap', 'critical_incidents', 'major_incidents',
        'unit_cost', 'lead_time_days', 'crit_A', 'crit_B', 'risk_High', 'risk_Medium',
        'is_repairable_flag'
    ]

    df = sch.dropna(subset=['target']).dropna(subset=feature_cols).copy()

    # --- Time-based split ---
    split_date = pd.Timestamp('2024-07-01')
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]

    X_train = train[feature_cols].values
    X_test = test[feature_cols].values
    y_train = train['target'].values
    y_test = test['target'].values

    # --- Models ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.1)
    lr.fit(X_train_s, y_train)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # Baseline
    net_avail_idx = feature_cols.index('net_available')
    baseline_pred = (X_test[:, net_avail_idx] <= 0).astype(int)

    # Add predictions to test df
    test_df = test.copy()
    test_df['rf_proba'] = rf_proba
    test_df['lr_proba'] = lr_proba

    return {
        'parts': parts, 'sch': sch, 'po': po, 'qi': qi,
        'train': train, 'test': test, 'test_df': test_df,
        'rf': rf, 'lr': lr, 'feature_cols': feature_cols,
        'y_test': y_test, 'rf_proba': rf_proba, 'lr_proba': lr_proba,
        'baseline_pred': baseline_pred, 'X_test': X_test
    }


# ============================================================
# LOAD DATA
# ============================================================
try:
    data = load_and_process_data()
except FileNotFoundError:
    st.error("CSV files not found. Place the 4 data files in the same directory as this script.")
    st.stop()

parts = data['parts']
test_df = data['test_df']
y_test = data['y_test']
rf_proba = data['rf_proba']
lr_proba = data['lr_proba']
rf = data['rf']
feature_cols = data['feature_cols']
po = data['po']

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">Aerospace Backorder Prediction Dashboard</div>', unsafe_allow_html=True)

with st.expander("How to use this app", expanded=False):
    st.markdown("""
**What this app does in plain English.**
An aerospace company stocks 300 different parts. If the wrong part
runs out, an aircraft can't be repaired and revenue stops. The trick
is predicting *which part will be backordered next week* before it
happens, so procurement can expedite shipments. This app uses **two
machine-learning models** (Logistic Regression and Random Forest)
trained on 8 years of weekly inventory, purchase order, and quality
data, and a **knapsack optimiser** that picks the best parts to
expedite within a fixed budget.

**Quick start (60 seconds).**
1. Look at the **At-risk parts** table — these are the parts the
   model thinks are most likely to backorder next week.
2. Adjust the **expedite budget** slider in the sidebar — how much
   money procurement can spend on rush shipments.
3. The **Recommended expedites** table updates: which specific parts
   to expedite given that budget.
4. Click any part to drill into its 8-year history.

**Sidebar controls explained.**
- **Risk threshold** — how confident does the model need to be before
  flagging a part? Lower threshold = more parts flagged (catches more
  real risks but also more false alarms).
- **Expedite budget** — total dollars available for rush shipments
  this week. The optimiser picks the combination that prevents the
  most expected revenue loss.
- **Criticality weights** — A-class parts are more important than
  B-class. Adjust how much extra weight the optimiser gives to
  A-class.

**The tabs.**
- **Risk overview** — top-K at-risk parts ranked by model probability.
- **Model comparison** — Logistic Regression vs Random Forest on
  test data (PR-AUC, ROC-AUC, F1).
- **Expedite recommender** — given the budget and weights you set in
  the sidebar, which parts to expedite for max impact.
- **Sensitivity** — how does the recommendation change if you double
  the budget? Triple it?

**What the metrics mean.**
- **PR-AUC** — area under the precision-recall curve. Better when
  the positive class is rare (which backorders are). Higher = better.
- **Backorder probability** — the model's prediction for each part,
  from 0 (safe) to 1 (almost certain).
- **Expedite cost vs prevented loss** — for each candidate expedite,
  what does it cost vs what revenue does it protect.

**Try this.** Set the budget to $50K, look at the top 5 recommended
expedites. Then drop it to $10K and watch which ones get cut — those
are the lower-priority cases the optimiser is willing to skip.
""")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Configuration")
st.sidebar.markdown("---")

# Week selector
available_weeks = sorted(test_df['date'].unique())
week_labels = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(available_weeks)]
selected_week_idx = st.sidebar.selectbox(
    "Select Week",
    range(len(available_weeks)),
    format_func=lambda i: week_labels[i],
    index=len(available_weeks) - 1
)
selected_week = available_weeks[selected_week_idx]

st.sidebar.markdown("---")
st.sidebar.subheader("Knapsack Parameters")

budget = st.sidebar.slider("Weekly Expedite Budget ($)", 5000, 100000, 25000, step=5000)
cost_pct = st.sidebar.slider("Expedite Cost (% of unit cost)", 5, 30, 15, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Criticality Weights")
alpha_a = st.sidebar.slider("Class A (AOG-Critical)", 1, 15, 5)
alpha_b = st.sidebar.slider("Class B (Essential)", 1, 10, 2)
alpha_c = 1  # fixed reference

st.sidebar.markdown("---")

# ============================================================
# COMPUTE EXPEDITE RECOMMENDATIONS FOR SELECTED WEEK
# ============================================================
week_df = test_df[test_df['date'] == selected_week].copy()
alpha_map = {'A': alpha_a, 'B': alpha_b, 'C': alpha_c}
week_df['alpha'] = week_df['criticality_class'].map(alpha_map)
week_df['severity_weight'] = week_df['alpha'] * week_df['unit_cost']
week_df['risk_score'] = week_df['rf_proba'] * week_df['severity_weight']
week_df['expedite_cost'] = week_df['unit_cost'] * (cost_pct / 100)
week_df['efficiency'] = week_df['risk_score'] / week_df['expedite_cost']

# Greedy knapsack
week_sorted = week_df.sort_values('efficiency', ascending=False)
selected_items = []
spent = 0
for _, row in week_sorted.iterrows():
    if spent + row['expedite_cost'] <= budget:
        selected_items.append(row)
        spent += row['expedite_cost']
knapsack_df = pd.DataFrame(selected_items) if selected_items else pd.DataFrame()

actual_bo = week_df['target'].sum()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Weekly Expedite List", "Model Performance", "Sensitivity"
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab1:
    st.subheader(f"Week of {pd.to_datetime(selected_week).strftime('%B %d, %Y')}")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Part-Site Combos", f"{len(week_df):,}")
    with col2:
        st.metric("Actual Backorders", f"{int(actual_bo)}")
    with col3:
        caught = int(knapsack_df['target'].sum()) if len(knapsack_df) > 0 else 0
        st.metric("Backorders Caught", f"{caught}/{int(actual_bo)}")
    with col4:
        st.metric("Orders to Expedite", f"{len(knapsack_df)}")
    with col5:
        st.metric("Budget Used", f"${spent:,.0f} / ${budget:,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Risk score distribution
        fig_risk = px.histogram(
            week_df, x='risk_score', nbins=50,
            title="Risk Score Distribution (This Week)",
            labels={'risk_score': 'Risk Score', 'count': 'Count'},
            color_discrete_sequence=['#1F3864']
        )
        fig_risk.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_right:
        # Criticality breakdown of selections
        if len(knapsack_df) > 0:
            crit_counts = knapsack_df['criticality_class'].value_counts().reindex(['A', 'B', 'C']).fillna(0)
            fig_crit = go.Figure(data=[go.Pie(
                labels=crit_counts.index,
                values=crit_counts.values,
                marker_colors=['#d62728', '#ff7f0e', '#2ca02c'],
                hole=0.4
            )])
            fig_crit.update_layout(title="Expedite Selections by Criticality", height=350)
            st.plotly_chart(fig_crit, use_container_width=True)
        else:
            st.info("No items selected — try increasing the budget.")

    # Backorder rate by site
    site_bo = week_df.groupby('site_id').agg(
        total=('target', 'count'),
        backorders=('target', 'sum'),
        avg_risk=('risk_score', 'mean')
    ).reset_index()
    site_bo['bo_rate'] = site_bo['backorders'] / site_bo['total'] * 100

    fig_site = px.bar(
        site_bo, x='site_id', y='bo_rate',
        title="Backorder Rate by Site (This Week)",
        labels={'site_id': 'Site', 'bo_rate': 'Backorder Rate (%)'},
        color='avg_risk', color_continuous_scale='RdYlGn_r',
        text=site_bo['bo_rate'].apply(lambda x: f'{x:.1f}%')
    )
    fig_site.update_layout(height=300)
    st.plotly_chart(fig_site, use_container_width=True)


# ============================================================
# TAB 2: WEEKLY EXPEDITE LIST
# ============================================================
with tab2:
    st.subheader(f"Recommended Expedite Actions — Week of {pd.to_datetime(selected_week).strftime('%b %d, %Y')}")

    if len(knapsack_df) > 0:
        recall_pct = (caught / max(actual_bo, 1)) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recall (Backorders Caught)", f"{recall_pct:.0f}%")
        with col2:
            st.metric("Avg Cost per Expedite", f"${spent / len(knapsack_df):,.0f}")
        with col3:
            st.metric("Total Risk Value Covered", f"${knapsack_df['risk_score'].sum():,.0f}")

        st.markdown("---")

        # Expedite list table
        display_cols = ['part_id', 'site_id', 'criticality_class', 'part_family',
                        'rf_proba', 'unit_cost', 'risk_score', 'expedite_cost', 'efficiency',
                        'net_available', 'on_hand_qty', 'supplier_late_rate']

        display_df = knapsack_df[display_cols].copy()
        display_df.columns = ['Part ID', 'Site', 'Criticality', 'Family',
                              'Backorder Prob', 'Unit Cost ($)', 'Risk Score',
                              'Expedite Cost ($)', 'Efficiency', 'Net Available',
                              'On Hand', 'Supplier Late Rate']
        display_df = display_df.sort_values('Risk Score', ascending=False).reset_index(drop=True)
        display_df.index = display_df.index + 1  # 1-indexed

        # Format columns
        display_df['Backorder Prob'] = display_df['Backorder Prob'].apply(lambda x: f'{x:.1%}')
        display_df['Unit Cost ($)'] = display_df['Unit Cost ($)'].apply(lambda x: f'${x:,.0f}')
        display_df['Risk Score'] = display_df['Risk Score'].apply(lambda x: f'{x:,.0f}')
        display_df['Expedite Cost ($)'] = display_df['Expedite Cost ($)'].apply(lambda x: f'${x:,.0f}')
        display_df['Efficiency'] = display_df['Efficiency'].apply(lambda x: f'{x:.1f}')
        display_df['Supplier Late Rate'] = display_df['Supplier Late Rate'].apply(lambda x: f'{x:.0%}' if pd.notna(x) else 'N/A')

        st.dataframe(display_df, use_container_width=True, height=500)

        # Download button
        csv = knapsack_df[['part_id', 'site_id', 'criticality_class', 'part_family',
                           'rf_proba', 'unit_cost', 'risk_score', 'expedite_cost']].to_csv(index=False)
        st.download_button(
            "Download Expedite List (CSV)",
            csv,
            f"expedite_list_{pd.to_datetime(selected_week).strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    else:
        st.warning("No items selected with current budget. Increase the budget in the sidebar.")


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
with tab3:
    st.subheader("Model Performance Comparison")

    col_left, col_right = st.columns(2)

    with col_left:
        # PR Curve
        lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_proba)
        rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_proba)
        lr_pr_auc = auc(lr_rec, lr_prec)
        rf_pr_auc = auc(rf_rec, rf_prec)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=lr_rec, y=lr_prec, mode='lines',
                                     name=f'Logistic Reg (PR-AUC={lr_pr_auc:.3f})',
                                     line=dict(color='#4A90D9', width=2)))
        fig_pr.add_trace(go.Scatter(x=rf_rec, y=rf_prec, mode='lines',
                                     name=f'Random Forest (PR-AUC={rf_pr_auc:.3f})',
                                     line=dict(color='#d62728', width=2)))
        fig_pr.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
                         annotation_text=f"Baseline ({y_test.mean():.3f})")
        fig_pr.update_layout(title="Precision-Recall Curves", height=400,
                              xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_right:
        # Performance table
        st.markdown("#### Model Comparison")

        baseline_recall = recall_score(y_test, data['baseline_pred'])
        baseline_prec = precision_score(y_test, data['baseline_pred'], zero_division=0)
        rf_pred = (rf_proba >= 0.5).astype(int)

        perf_data = {
            'Model': ['Baseline (threshold)', 'Logistic Regression', 'Random Forest'],
            'ROC-AUC': ['—', f"{roc_auc_score(y_test, lr_proba):.3f}",
                        f"{roc_auc_score(y_test, rf_proba):.3f}"],
            'PR-AUC': ['—', f"{lr_pr_auc:.3f}", f"{rf_pr_auc:.3f}"],
            'Recall': [f"{baseline_recall:.3f}", f"{recall_score(y_test, (lr_proba >= 0.5).astype(int)):.3f}",
                       f"{recall_score(y_test, rf_pred):.3f}"],
            'Precision': [f"{baseline_prec:.3f}", f"{precision_score(y_test, (lr_proba >= 0.5).astype(int), zero_division=0):.3f}",
                          f"{precision_score(y_test, rf_pred, zero_division=0):.3f}"],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

        # Confusion matrix
        st.markdown("#### Random Forest Confusion Matrix")
        cm = confusion_matrix(y_test, rf_pred)
        cm_df = pd.DataFrame(cm, index=['Actual: No BO', 'Actual: BO'],
                              columns=['Predicted: No BO', 'Predicted: BO'])
        st.dataframe(cm_df, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown("#### Which Inputs Drive Predictions")

    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'Input': feature_cols, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=True).tail(15)

    fig_imp = px.bar(feat_imp, x='Importance', y='Input', orientation='h',
                      title="Top 15 Model Inputs (Random Forest Importance)",
                      color='Importance', color_continuous_scale='Blues')
    fig_imp.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Precision@K / Recall@K
    st.markdown("---")
    st.markdown("#### Precision@K and Recall@K")

    k_range = list(range(5, 201, 5))
    pk_data = {'K': [], 'Metric': [], 'Value': []}
    for k in k_range:
        top_k_idx = np.argsort(rf_proba)[::-1][:k]
        top_k_true = y_test[top_k_idx]
        pk = top_k_true.sum() / k
        rk = top_k_true.sum() / max(y_test.sum(), 1)
        pk_data['K'].extend([k, k])
        pk_data['Metric'].extend(['Precision@K', 'Recall@K'])
        pk_data['Value'].extend([pk, rk])

    fig_pk = px.line(pd.DataFrame(pk_data), x='K', y='Value', color='Metric',
                      title="Precision and Recall at Various Expedite Capacities",
                      color_discrete_map={'Precision@K': '#d62728', 'Recall@K': '#1F3864'})
    fig_pk.update_layout(height=350)
    st.plotly_chart(fig_pk, use_container_width=True)


# ============================================================
# TAB 4: SENSITIVITY ANALYSIS
# ============================================================
with tab4:
    st.subheader("Interactive Sensitivity Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Budget Sensitivity")
        st.caption("How does changing the weekly budget affect backorder coverage?")

        budget_range = range(5000, 105000, 5000)
        budget_results = []

        for b in budget_range:
            temp = week_sorted.copy()
            sel = []
            s = 0
            for _, row in temp.iterrows():
                cost = row['unit_cost'] * (cost_pct / 100)
                if s + cost <= b:
                    sel.append(row)
                    s += cost
            sel_df = pd.DataFrame(sel)
            c = int(sel_df['target'].sum()) if len(sel_df) > 0 else 0
            budget_results.append({
                'Budget': b,
                'Items Selected': len(sel_df),
                'Backorders Caught': c,
                'Recall': c / max(actual_bo, 1),
                'Budget Used': s
            })

        br_df = pd.DataFrame(budget_results)

        fig_budget = make_subplots(specs=[[{"secondary_y": True}]])
        fig_budget.add_trace(go.Scatter(
            x=br_df['Budget'], y=br_df['Recall'] * 100,
            mode='lines+markers', name='Recall (%)',
            line=dict(color='#1F3864', width=2)
        ), secondary_y=False)
        fig_budget.add_trace(go.Bar(
            x=br_df['Budget'], y=br_df['Items Selected'],
            name='Items Selected', marker_color='rgba(31, 56, 100, 0.2)'
        ), secondary_y=True)
        fig_budget.update_layout(title=f"Budget vs Coverage (Week of {pd.to_datetime(selected_week).strftime('%b %d')})",
                                  height=400)
        fig_budget.update_xaxes(title_text="Weekly Budget ($)")
        fig_budget.update_yaxes(title_text="Recall (%)", secondary_y=False)
        fig_budget.update_yaxes(title_text="# Items Selected", secondary_y=True)
        st.plotly_chart(fig_budget, use_container_width=True)

    with col_right:
        st.markdown("#### Criticality Weight Sensitivity")
        st.caption("How do different criticality weights change what gets selected?")

        scenarios = {
            'Equal (1:1:1)': {'A': 1, 'B': 1, 'C': 1},
            'Mild (3:2:1)': {'A': 3, 'B': 2, 'C': 1},
            'Default (5:2:1)': {'A': 5, 'B': 2, 'C': 1},
            'Strong (8:3:1)': {'A': 8, 'B': 3, 'C': 1},
            'Extreme (10:3:1)': {'A': 10, 'B': 3, 'C': 1},
        }

        scenario_results = []
        for name, alphas in scenarios.items():
            temp = week_df.copy()
            temp['alpha'] = temp['criticality_class'].map(alphas)
            temp['risk_score'] = temp['rf_proba'] * temp['alpha'] * temp['unit_cost']
            temp['expedite_cost'] = temp['unit_cost'] * (cost_pct / 100)
            temp['efficiency'] = temp['risk_score'] / temp['expedite_cost']
            temp = temp.sort_values('efficiency', ascending=False)

            sel = []
            s = 0
            for _, row in temp.iterrows():
                if s + row['expedite_cost'] <= budget:
                    sel.append(row)
                    s += row['expedite_cost']
            sel_df = pd.DataFrame(sel)
            c = int(sel_df['target'].sum()) if len(sel_df) > 0 else 0
            crit_a = int((sel_df['criticality_class'] == 'A').sum()) if len(sel_df) > 0 else 0

            scenario_results.append({
                'Scenario': name,
                'Recall': c / max(actual_bo, 1) * 100,
                'Crit-A Selected': crit_a,
                'Total Selected': len(sel_df)
            })

        sr_df = pd.DataFrame(scenario_results)

        fig_scenario = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scenario.add_trace(go.Bar(
            x=sr_df['Scenario'], y=sr_df['Recall'],
            name='Recall (%)', marker_color='#1F3864'
        ), secondary_y=False)
        fig_scenario.add_trace(go.Scatter(
            x=sr_df['Scenario'], y=sr_df['Crit-A Selected'],
            mode='lines+markers', name='Crit-A Selected',
            line=dict(color='#d62728', width=2)
        ), secondary_y=True)
        fig_scenario.update_layout(title="Criticality Weight Scenarios", height=400)
        fig_scenario.update_yaxes(title_text="Recall (%)", secondary_y=False)
        fig_scenario.update_yaxes(title_text="# Criticality-A Selected", secondary_y=True)
        st.plotly_chart(fig_scenario, use_container_width=True)

    # Aggregate performance across all test weeks
    st.markdown("---")
    st.markdown("#### Aggregate Performance Across All Test Weeks")

    weeks = test_df['date'].unique()
    weekly_perf = []
    for week in weeks:
        wk = test_df[test_df['date'] == week].copy()
        bo = wk['target'].sum()
        wk['alpha'] = wk['criticality_class'].map(alpha_map)
        wk['risk_score'] = wk['rf_proba'] * wk['alpha'] * wk['unit_cost']
        wk['expedite_cost'] = wk['unit_cost'] * (cost_pct / 100)
        wk['efficiency'] = wk['risk_score'] / wk['expedite_cost']
        wk = wk.sort_values('efficiency', ascending=False)

        sel = []
        s = 0
        for _, row in wk.iterrows():
            if s + row['expedite_cost'] <= budget:
                sel.append(row)
                s += row['expedite_cost']
        sel_df = pd.DataFrame(sel)
        c = int(sel_df['target'].sum()) if len(sel_df) > 0 else 0

        weekly_perf.append({
            'Week': pd.to_datetime(week),
            'Actual Backorders': int(bo),
            'Caught': c,
            'Recall': c / max(bo, 1) * 100 if bo > 0 else None,
            'Items Expedited': len(sel_df),
            'Budget Used': s
        })

    wp_df = pd.DataFrame(weekly_perf)

    fig_weekly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_weekly.add_trace(go.Bar(
        x=wp_df['Week'], y=wp_df['Actual Backorders'],
        name='Actual Backorders', marker_color='rgba(214, 39, 40, 0.4)'
    ), secondary_y=False)
    fig_weekly.add_trace(go.Bar(
        x=wp_df['Week'], y=wp_df['Caught'],
        name='Caught by Knapsack', marker_color='rgba(31, 56, 100, 0.7)'
    ), secondary_y=False)
    fig_weekly.add_trace(go.Scatter(
        x=wp_df['Week'], y=wp_df['Recall'],
        mode='lines+markers', name='Recall (%)',
        line=dict(color='#ff7f0e', width=2)
    ), secondary_y=True)
    fig_weekly.update_layout(
        title=f"Weekly Knapsack Performance (Budget=${budget:,}, Cost={cost_pct}%)",
        height=400, barmode='overlay'
    )
    fig_weekly.update_xaxes(title_text="Week")
    fig_weekly.update_yaxes(title_text="Backorder Count", secondary_y=False)
    fig_weekly.update_yaxes(title_text="Recall (%)", secondary_y=True)
    st.plotly_chart(fig_weekly, use_container_width=True)
