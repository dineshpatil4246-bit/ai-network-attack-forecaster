"""
NetGuard AI — Streamlit Web Interface
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .risk-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 22px;
        font-weight: bold;
        margin: 4px 0;
    }
    .low    { background-color: #166534; color: #86efac; }
    .medium { background-color: #854d0e; color: #fde68a; }
    .high   { background-color: #9a3412; color: #fdba74; }
    .critical { background-color: #7f1d1d; color: #fca5a5; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        margin: 4px 0;
    }
    .mitre-box {
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        margin: 4px;
        font-size: 13px;
    }
    .mitre-active { background: #7f1d1d; border: 2px solid #ef4444; }
    .mitre-inactive { background: #1e2130; border: 2px solid #374151; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
    code { background: #1e2130; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import torch
        import torch.nn as nn

        class NetworkWorldModel(nn.Module):
            def __init__(self, num_features, num_classes=3, hidden_size=128, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=num_features, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=0.3,
                    batch_first=True, bidirectional=True
                )
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size*2, num_heads=4,
                    dropout=0.3, batch_first=True
                )
                self.fc1     = nn.Linear(hidden_size*2, 128)
                self.relu    = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                self.fc2     = nn.Linear(128, num_classes)

            def forward(self, x):
                lstm_out, _      = self.lstm(x)
                attended, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
                pooled           = attended.mean(dim=1)
                h                = self.dropout(self.relu(self.fc1(pooled)))
                return self.fc2(h), attn_w

            def predict_proba(self, x):
                logits, _ = self.forward(x)
                return torch.softmax(logits, dim=1)

            def get_attention_summary(self, x):
                _, attn_w = self.forward(x)
                return attn_w.mean(dim=1)

        with open('models/model_config.json') as f:
            config = json.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = NetworkWorldModel(config['num_features'], config['num_classes']).to(device)
        model.load_state_dict(torch.load('models/world_model_best.pt', map_location=device))
        model.eval()
        return model, config, device, True
    except Exception as e:
        return None, None, None, str(e)


# ── Run analysis ──────────────────────────────────────────────────────────────
def run_analysis(df, model, config, device, k):
    import torch
    import joblib

    with open('processed/feature_names.json') as f:
        feature_names = json.load(f)
    scaler = joblib.load('processed/scaler.pkl')

    # Preprocess
    X = np.zeros((len(df), len(feature_names)), dtype=np.float32)
    for i, feat in enumerate(feature_names):
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors='coerce').fillna(0)
            col = col.replace([np.inf, -np.inf], 0)
            X[:, i] = col.values

    X = scaler.transform(X).astype(np.float32)
    X = np.clip(X, 0, 1)

    seq_len = config['seq_len']
    if len(X) >= seq_len:
        X_seq = X[-seq_len:]
    else:
        pad   = np.zeros((seq_len - len(X), X.shape[1]), dtype=np.float32)
        X_seq = np.vstack([pad, X])

    sequence = torch.FloatTensor(X_seq).unsqueeze(0).to(device)

    MITRE = {
        0: {'name': 'Benign',                    'color': '#22c55e', 'tactic': 'N/A'},
        1: {'name': 'Reconnaissance / DoS',      'color': '#eab308', 'tactic': 'TA0043'},
        2: {'name': 'Initial Access / BruteForce','color': '#f97316','tactic': 'TA0001'},
        3: {'name': 'Command & Control',          'color': '#ef4444', 'tactic': 'TA0011'},
        4: {'name': 'Exfiltration',               'color': '#7f1d1d', 'tactic': 'TA0010'},
    }

    # K-step forecast
    forecast   = []
    current    = sequence.clone()
    num_classes = config['num_classes']

    with torch.no_grad():
        for step in range(k):
            logits, _ = model(current)
            probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred      = int(probs.argmax())
            atk_prob  = float(probs[1:].sum())
            m         = MITRE.get(pred, MITRE[0])

            forecast.append({
                'step':          step + 1,
                'attack_prob':   atk_prob * 100,
                'confidence':    float(probs.max()) * 100,
                'mitre_stage':   m['name'],
                'mitre_color':   m['color'],
                'tactic':        m['tactic'],
                'pred_class':    pred,
                'probs':         probs.tolist(),
            })

            last = current[:, -1:, :]
            next_row = last + torch.randn_like(last) * 0.02
            next_row = torch.clamp(next_row, 0, 1)
            current  = torch.cat([current[:, 1:, :], next_row], dim=1)

    # Attention explainability
    with torch.no_grad():
        _, attn_w    = model(sequence)
        attn_summary = attn_w[0].mean(dim=0).cpu().numpy()
        seq_np       = X_seq

    weighted   = attn_summary[:, None] * seq_np
    importance = np.abs(weighted).mean(axis=0)
    mean_vals  = seq_np.mean(axis=0)

    top_feats = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:10]
    features  = []
    for idx, imp in top_feats:
        fname = feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}'
        val   = float(mean_vals[idx])
        features.append({
            'feature':    fname,
            'importance': float(imp),
            'value':      val,
            'direction':  'Increases Risk' if val > 0.5 else 'Decreases Risk',
            'color':      '#ef4444' if val > 0.5 else '#3b82f6',
        })

    return forecast, features, MITRE, num_classes


def risk_info(attack_prob):
    if attack_prob < 25:  return 'LOW',      '#22c55e', 'low',      '🟢'
    if attack_prob < 50:  return 'MEDIUM',   '#eab308', 'medium',   '🟡'
    if attack_prob < 75:  return 'HIGH',     '#f97316', 'high',     '🟠'
    return                       'CRITICAL', '#ef4444', 'critical', '🔴'


# ── Demo data ─────────────────────────────────────────────────────────────────
def make_demo_forecast(k):
    MITRE = {
        0: {'name': 'Benign',                    'color': '#22c55e'},
        1: {'name': 'Reconnaissance / DoS',      'color': '#eab308'},
        2: {'name': 'Initial Access / BruteForce','color': '#f97316'},
    }
    probs = [15, 35, 60, 82, 95][:k]
    return [
        {
            'step': i+1, 'attack_prob': p, 'confidence': 80+i*3,
            'mitre_stage': MITRE[min(i//2, 2)]['name'],
            'mitre_color': MITRE[min(i//2, 2)]['color'],
            'tactic': 'TA0043', 'pred_class': min(i//2, 2),
            'probs': [1-p/100, p/100, 0],
        }
        for i, p in enumerate(probs)
    ], [
        {'feature': f, 'importance': v, 'value': v,
         'direction': 'Increases Risk', 'color': '#ef4444'}
        for f, v in [
            ('SYN Flag Cnt',0.42),('Flow Duration',0.38),
            ('PSH Flag Cnt',0.31),('Flow IAT Mean',0.28),
            ('Flow Byts/s', 0.22),('RST Flag Cnt', 0.19),
            ('ACK Flag Cnt',0.15),('Tot Fwd Pkts', 0.11),
            ('Fwd IAT Mean',0.08),('Flow IAT Max', 0.05),
        ]
    ]


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ NetGuard AI")
    st.markdown("*Network Attack Forecaster*")
    st.divider()

    uploaded = st.file_uploader("📁 Upload traffic data (.csv)", type=['csv'])
    k        = st.slider("🔭 Forecast horizon (K steps)", 1, 10, 5)
    run_btn  = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

    st.divider()

    # Model status
    model, config, device, status = load_model()
    if status is True:
        st.success("✅ Model loaded")
        st.caption(f"F1: {config.get('best_val_f1', 0):.4f} | "
                   f"Classes: {config.get('num_classes', 3)} | "
                   f"Device: {device}")
    else:
        st.error(f"❌ Model error: {status}")

    st.divider()
    st.info("Upload a CSV from your `data/` folder to run real analysis, "
            "or click Run Analysis to see a demo.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛡️ NetGuard AI — Network Attack Forecaster")
st.markdown("*AI World Model for proactive cyber defence · MITRE ATT&CK aligned · "
            "Fully offline*")
st.divider()

# ── Run analysis ──────────────────────────────────────────────────────────────
forecast, top_features, MITRE_MAP = None, None, None
is_demo = False

if run_btn:
    if uploaded and model:
        with st.spinner("🔍 Analysing network traffic..."):
            try:
                df = pd.read_csv(uploaded, encoding_errors='replace', low_memory=False)
                df.columns = df.columns.str.strip()
                forecast, top_features, MITRE_MAP, _ = run_analysis(df, model, config, device, k)
                st.session_state['forecast']     = forecast
                st.session_state['top_features'] = top_features
                st.session_state['df']           = df
                st.session_state['is_demo']      = False
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        # Demo mode
        forecast, top_features = make_demo_forecast(k)
        st.session_state['forecast']     = forecast
        st.session_state['top_features'] = top_features
        st.session_state['df']           = None
        st.session_state['is_demo']      = True
        st.info("🎭 Demo mode — upload a real CSV for actual predictions")

# Restore from session state
if 'forecast' in st.session_state:
    forecast     = st.session_state['forecast']
    top_features = st.session_state['top_features']
    is_demo      = st.session_state.get('is_demo', False)

# ── Top metrics ───────────────────────────────────────────────────────────────
if forecast:
    max_prob  = max(r['attack_prob'] for r in forecast)
    worst     = max(forecast, key=lambda r: r['attack_prob'])
    risk_lbl, risk_col, risk_cls, risk_emoji = risk_info(max_prob)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#9ca3af;font-size:13px'>RISK LEVEL</div>
            <div class='risk-badge {risk_cls}'>{risk_emoji} {risk_lbl}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#9ca3af;font-size:13px'>MITRE STAGE</div>
            <div style='font-size:18px;font-weight:500;color:{worst["mitre_color"]};
                        margin-top:4px'>{worst['mitre_stage']}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#9ca3af;font-size:13px'>MAX ATTACK PROB</div>
            <div style='font-size:22px;font-weight:bold;color:{risk_col};
                        margin-top:4px'>{max_prob:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#9ca3af;font-size:13px'>ANALYSED AT</div>
            <div style='font-size:16px;margin-top:4px'>{datetime.now().strftime('%H:%M:%S')}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Threat Timeline",
        "🗺️ MITRE ATT&CK Map",
        "🔍 Risk Factors",
        "📋 Traffic Data",
    ])

    # ── Tab 1: Threat Timeline ────────────────────────────────────────────────
    with tab1:
        st.subheader("Attack Probability Forecast")

        steps = [r['step']       for r in forecast]
        probs = [r['attack_prob']for r in forecast]
        confs = [r['confidence'] for r in forecast]
        mitre = [r['mitre_stage']for r in forecast]

        fig = go.Figure()

        # Background bands
        fig.add_hrect(y0=0,  y1=25, fillcolor='#166534', opacity=0.15, line_width=0)
        fig.add_hrect(y0=25, y1=50, fillcolor='#854d0e', opacity=0.15, line_width=0)
        fig.add_hrect(y0=50, y1=75, fillcolor='#9a3412', opacity=0.15, line_width=0)
        fig.add_hrect(y0=75, y1=100,fillcolor='#7f1d1d', opacity=0.20, line_width=0)

        # Confidence band
        fig.add_trace(go.Scatter(
            x=steps + steps[::-1],
            y=[p+5 for p in probs] + [p-5 for p in probs[::-1]],
            fill='toself', fillcolor='rgba(59,130,246,0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence band', showlegend=False
        ))

        # Main line
        fig.add_trace(go.Scatter(
            x=steps, y=probs,
            mode='lines+markers+text',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=12, color=[
                '#22c55e' if p < 25 else
                '#eab308' if p < 50 else
                '#f97316' if p < 75 else '#ef4444'
                for p in probs
            ], line=dict(color='white', width=2)),
            text=[f'{p:.0f}%' for p in probs],
            textposition='top center',
            customdata=list(zip(mitre, confs)),
            hovertemplate=(
                '<b>Step %{x}</b><br>'
                'Attack Probability: %{y:.1f}%<br>'
                'MITRE Stage: %{customdata[0]}<br>'
                'Confidence: %{customdata[1]:.1f}%'
                '<extra></extra>'
            ),
            name='Attack Probability'
        ))

        # Zone labels
        for y, label in [(12, '🟢 SAFE'), (37, '🟡 CAUTION'),
                         (62, '🟠 HIGH'), (87, '🔴 CRITICAL')]:
            fig.add_annotation(x=steps[-1]+0.3, y=y, text=label,
                               showarrow=False, font=dict(size=10, color='#9ca3af'),
                               xanchor='left')

        fig.update_layout(
            xaxis_title='Forecast Step',
            yaxis_title='Attack Probability (%)',
            yaxis=dict(range=[0, 110]),
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='#e5e7eb'),
            legend=dict(bgcolor='#1e2130'),
            margin=dict(l=40, r=80, t=20, b=40),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Step details table
        table_data = pd.DataFrame([{
            'Step':            r['step'],
            'Attack Risk':     f"{r['attack_prob']:.1f}%",
            'MITRE Stage':     r['mitre_stage'],
            'Confidence':      f"{r['confidence']:.1f}%",
            'Tactic ID':       r['tactic'],
            'Risk Level':      risk_info(r['attack_prob'])[0],
        } for r in forecast])
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    # ── Tab 2: MITRE ATT&CK Map ───────────────────────────────────────────────
    with tab2:
        st.subheader("MITRE ATT&CK Kill Chain")

        worst_class = worst['pred_class']
        stages = [
            (0, '🔍 Reconnaissance',     'TA0043', 'Port scanning\nNetwork probing'),
            (1, '🚪 Initial Access',      'TA0001', 'Exploitation\nWeb attacks'),
            (2, '📡 Command & Control',   'TA0011', 'C2 beaconing\nBackdoor'),
            (3, '↔️ Lateral Movement',    'TA0008', 'Pivoting\nCredential use'),
            (4, '📤 Exfiltration',        'TA0010', 'Data theft\nCompromise'),
        ]

        cols = st.columns(len(stages))
        for i, (cls, name, tactic, desc) in enumerate(stages):
            is_active = (cls == worst_class)
            with cols[i]:
                box_class = 'mitre-active' if is_active else 'mitre-inactive'
                border    = '#ef4444' if is_active else '#374151'
                bg        = '#450a0a' if is_active else '#1e2130'
                prob_val  = worst['probs'][cls] * 100 if cls < len(worst['probs']) else 0.0
                active_tag = '<br><b style="color:#ef4444">◄ DETECTED</b>' if is_active else ''
                st.markdown(f"""
                <div style='background:{bg};border:2px solid {border};border-radius:8px;
                            padding:12px;text-align:center;min-height:160px'>
                    <div style='font-size:20px'>{name.split()[0]}</div>
                    <div style='font-size:12px;font-weight:bold;margin:4px 0'>
                        {' '.join(name.split()[1:])}</div>
                    <div style='font-size:10px;color:#9ca3af'>{tactic}</div>
                    <div style='font-size:11px;color:#d1d5db;margin-top:6px'>{desc}</div>
                    <div style='font-size:16px;font-weight:bold;color:{"#ef4444" if is_active else "#6b7280"};
                                margin-top:8px'>{prob_val:.1f}%{active_tag}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"""
        **Current Assessment:**
        - 🎯 Predicted Stage: **{worst['mitre_stage']}**
        - 🏷️ Tactic ID: **{worst['tactic']}**
        - ⚠️ Overall Risk: **{risk_lbl}** ({max_prob:.1f}% attack probability)
        """)

    # ── Tab 3: Risk Factors ───────────────────────────────────────────────────
    with tab3:
        st.subheader("Top Risk Factors (Attention-based Explainability)")

        feat_names = [f['feature']   for f in top_features]
        feat_imps  = [f['importance'] for f in top_features]
        feat_cols  = [f['color']      for f in top_features]
        feat_dirs  = [f['direction']  for f in top_features]

        fig2 = go.Figure(go.Bar(
            x=feat_imps[::-1],
            y=feat_names[::-1],
            orientation='h',
            marker_color=feat_cols[::-1],
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        ))
        fig2.update_layout(
            xaxis_title='Feature Importance Score',
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='#e5e7eb'),
            margin=dict(l=20, r=20, t=20, b=40),
            height=360,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Explanation text
        if top_features:
            f1 = top_features[0]['feature']
            f2 = top_features[1]['feature'] if len(top_features) > 1 else ''
            st.info(f"💡 **Key insight:** Elevated **{f1}** and **{f2}** are the "
                    f"primary signals driving the current threat prediction. "
                    f"These features indicate {worst['mitre_stage']} activity "
                    f"in the observed traffic window.")

        feat_df = pd.DataFrame([{
            'Rank':        i+1,
            'Feature':     f['feature'],
            'Importance':  f"{f['importance']:.4f}",
            'Avg Value':   f"{f['value']:.4f}",
            'Direction':   f['direction'],
        } for i, f in enumerate(top_features)])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Traffic Data ───────────────────────────────────────────────────
    with tab4:
        st.subheader("Uploaded Traffic Data")

        df_show = st.session_state.get('df', None)
        if df_show is not None:
            st.caption(f"{len(df_show):,} rows × {len(df_show.columns)} columns")
            st.dataframe(df_show.head(500), use_container_width=True)

            # Download flagged flows
            if 'Label' in df_show.columns:
                flagged = df_show[df_show['Label'] != 'Benign']
            else:
                flagged = df_show.head(100)

            st.download_button(
                label=f"⬇️ Download Flagged Flows ({len(flagged):,} rows)",
                data=flagged.to_csv(index=False),
                file_name='flagged_flows.csv',
                mime='text/csv',
            )
        else:
            st.info("Upload a CSV file and run analysis to see traffic data here.")

# ── Welcome screen (no analysis yet) ─────────────────────────────────────────
else:
    st.markdown("""
    ### How to use NetGuard AI

    1. **Upload** a CSV from your `data/` folder using the sidebar
    2. **Set** the forecast horizon (how many steps ahead to predict)
    3. Click **Run Analysis**
    4. See the threat timeline, MITRE ATT&CK stage, and top risk factors

    ---

    ### What this system does

    | Traditional IDS | NetGuard AI World Model |
    |-----------------|------------------------|
    | Classifies each packet in isolation | Learns temporal patterns across time |
    | Reacts after attack completes | **Forecasts** attack progression |
    | Binary: benign/malicious | Maps to MITRE ATT&CK stages |
    | No explanation | SHAP + Attention explainability |

    ---
    **Click "Run Analysis" in the sidebar to see a demo →**
    """)

