"""
NetGuard AI — Prediction Engine
K-step forward simulation + MITRE ATT&CK mapping + SHAP explainability
Run: py predict.py --csv data/yourfile.csv --k 5
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MITRE ATT&CK STAGE MAPPING
# ============================================================================

MITRE_STAGES = {
    0: {
        'name':        'Benign',
        'color':       'green',
        'tactic':      'N/A',
        'tactic_id':   'N/A',
        'description': 'Normal network traffic — no threat detected',
        'indicators':  'Low SYN counts, balanced bidirectional flows, normal IAT'
    },
    1: {
        'name':        'Reconnaissance / DoS',
        'color':       'yellow',
        'tactic':      'Reconnaissance / Impact',
        'tactic_id':   'TA0043 / TA0040',
        'description': 'Port scanning, network probing, volumetric DoS floods',
        'indicators':  'High SYN flag count, elevated packet rate, short flow duration'
    },
    2: {
        'name':        'Initial Access / Brute Force',
        'color':       'orange',
        'tactic':      'Initial Access / Credential Access',
        'tactic_id':   'TA0001 / TA0006',
        'description': 'Port scan patterns, FTP/SSH brute force, exploitation attempts',
        'indicators':  'Sequential port access, high RST counts, repeated connection attempts'
    },
    3: {
        'name':        'Command & Control',
        'color':       'red',
        'tactic':      'Command and Control',
        'tactic_id':   'TA0011',
        'description': 'Bot communication, C2 beaconing, persistent backdoor activity',
        'indicators':  'Regular IAT patterns, encrypted small flows, persistent connections'
    },
    4: {
        'name':        'Exfiltration / Infiltration',
        'color':       'darkred',
        'tactic':      'Exfiltration',
        'tactic_id':   'TA0010',
        'description': 'Data theft, deep compromise, kill chain nearly complete',
        'indicators':  'Large outbound bytes, unusual destination ports, high flow duration'
    },
}


# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class NetworkWorldModel(nn.Module):
    def __init__(self, num_features, num_classes=3, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=4, dropout=0.3, batch_first=True
        )
        self.fc1     = nn.Linear(hidden_size * 2, 128)
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
        return attn_w.mean(dim=1)   # (batch, seq_len)


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(model_path='models/world_model_best.pt',
               config_path='models/model_config.json'):
    """Load trained model using saved config."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun world_model.py first.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    num_features = config['num_features']
    num_classes  = config['num_classes']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = NetworkWorldModel(num_features, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"✓ Model loaded | features={num_features} | classes={num_classes} | device={device}")
    return model, config, device


# ============================================================================
# PREPROCESS INPUT CSV
# ============================================================================

def preprocess_input(csv_path,
                     scaler_path='processed/scaler.pkl',
                     feature_names_path='processed/feature_names.json',
                     seq_len=20):
    """
    Load a new CSV file and preprocess it the same way as training.
    Returns a tensor of shape (1, seq_len, num_features).
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}\nRun data_pipeline.py first.")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names not found: {feature_names_path}")

    # Load feature names and scaler
    with open(feature_names_path) as f:
        feature_names = json.load(f)
    scaler = joblib.load(scaler_path)

    # Load CSV
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace', low_memory=False)
    df.columns = df.columns.str.strip()   # remove leading/trailing spaces
    print(f"✓ Loaded CSV: {len(df):,} rows × {len(df.columns)} columns")

    # Select available features
    available = [f for f in feature_names if f in df.columns]
    missing   = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"⚠️  Missing {len(missing)} features (will pad with 0): {missing[:5]}...")

    # Build feature matrix
    X = np.zeros((len(df), len(feature_names)), dtype=np.float32)
    for i, feat in enumerate(feature_names):
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors='coerce').fillna(0)
            col = col.replace([np.inf, -np.inf], 0)
            X[:, i] = col.values

    # Scale
    X = scaler.transform(X).astype(np.float32)
    X = np.clip(X, 0, 1)

    # Take last seq_len rows (most recent traffic)
    if len(X) >= seq_len:
        X_seq = X[-seq_len:]
    else:
        # Pad with zeros at the start if fewer rows than seq_len
        pad   = np.zeros((seq_len - len(X), X.shape[1]), dtype=np.float32)
        X_seq = np.vstack([pad, X])

    tensor = torch.FloatTensor(X_seq).unsqueeze(0)   # (1, seq_len, features)
    print(f"✓ Preprocessed sequence shape: {tuple(tensor.shape)}")
    return tensor, feature_names


# ============================================================================
# RISK LEVEL
# ============================================================================

def get_risk_level(attack_prob):
    """Return risk label and display color based on attack probability."""
    if attack_prob < 0.25:  return 'LOW',      '🟢'
    if attack_prob < 0.50:  return 'MEDIUM',   '🟡'
    if attack_prob < 0.75:  return 'HIGH',      '🟠'
    return                          'CRITICAL', '🔴'


# ============================================================================
# K-STEP FORWARD SIMULATION
# ============================================================================

def run_k_step_forecast(model, initial_sequence, device, k=5):
    """
    Simulate k steps into the future.
    Each step: run model → record prediction → slide window forward.
    
    Args:
        model: trained NetworkWorldModel
        initial_sequence: tensor (1, seq_len, features)
        device: torch device
        k: number of forecast steps
    
    Returns:
        list of dicts, one per step
    """
    model.eval()
    results       = []
    current_seq   = initial_sequence.clone().to(device)

    with torch.no_grad():
        for step in range(k):
            # Forward pass
            logits, _ = model(current_seq)
            probs     = torch.softmax(logits, dim=1)[0]   # (num_classes,)
            probs_np  = probs.cpu().numpy()

            predicted_class = int(probs.argmax())
            confidence      = float(probs.max())
            # Attack probability = sum of all non-benign classes
            attack_prob     = float(probs[1:].sum())

            risk_label, risk_emoji = get_risk_level(attack_prob)
            mitre = MITRE_STAGES.get(predicted_class, MITRE_STAGES[0])

            results.append({
                'step':             step + 1,
                'predicted_class':  predicted_class,
                'confidence':       confidence,
                'attack_probability': attack_prob,
                'risk_level':       risk_label,
                'risk_emoji':       risk_emoji,
                'mitre_stage':      mitre['name'],
                'mitre_tactic_id':  mitre['tactic_id'],
                'mitre_description':mitre['description'],
                'probabilities':    probs_np.tolist(),
            })

            # Slide window: drop oldest row, append synthetic next state
            # Next state = last row + tiny noise (autoregressive simulation)
            last_row  = current_seq[:, -1:, :]           # (1, 1, features)
            next_row  = last_row + torch.randn_like(last_row) * 0.02
            next_row  = torch.clamp(next_row, 0, 1)
            current_seq = torch.cat([current_seq[:, 1:, :], next_row], dim=1)

    return results


# ============================================================================
# EXPLAINABILITY
# ============================================================================

def explain_with_attention(model, sequence_tensor, feature_names, device):
    """
    Attention-based feature importance.
    Returns top 10 features driving the current prediction.
    """
    model.eval()
    seq = sequence_tensor.to(device)

    with torch.no_grad():
        attn_summary = model.get_attention_summary(seq)   # (1, seq_len)
        attn_weights = attn_summary[0].cpu().numpy()      # (seq_len,)
        seq_np       = sequence_tensor[0].cpu().numpy()   # (seq_len, features)

    # Weighted feature importance: attention_weight × feature_value per timestep
    weighted = (attn_weights[:, None] * seq_np)   # (seq_len, features)
    feature_importance = np.abs(weighted).mean(axis=0)   # (features,)
    feature_mean_val   = seq_np.mean(axis=0)

    ranked = sorted(
        enumerate(feature_importance),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = []
    for idx, importance in ranked[:10]:
        val = float(feature_mean_val[idx])
        top_features.append({
            'rank':      len(top_features) + 1,
            'feature':   feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}',
            'importance': float(importance),
            'mean_value': val,
            'direction':  'increases_risk' if val > 0.5 else 'decreases_risk',
            'symbol':     '↑' if val > 0.5 else '↓',
        })

    return top_features


def explain_with_shap(model, sequence_tensor, feature_names, device, n_background=20):
    """
    SHAP-based feature attribution (falls back to attention if SHAP unavailable).
    """
    try:
        import shap

        seq = sequence_tensor.to(device)

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                logits, _ = self.model(x)
                return logits

        # Background = noise around the input sequence
        background_np = (
            sequence_tensor.numpy() +
            np.random.randn(n_background, *sequence_tensor.shape[1:]) * 0.05
        ).astype(np.float32)

        input_np = sequence_tensor.numpy()

        explainer   = shap.DeepExplainer(
            ModelWrapper(model),
            torch.FloatTensor(background_np).to(device)
        )
        shap_values = explainer.shap_values(seq)

        if isinstance(shap_values, list):
            sv = np.array(shap_values).mean(axis=0)
        else:
            sv = shap_values

        # Average SHAP values over time dimension → per-feature importance
        sv_mean = sv[0].mean(axis=0)   # (features,)

        ranked = sorted(enumerate(sv_mean), key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for idx, sv_val in ranked[:10]:
            top_features.append({
                'rank':       len(top_features) + 1,
                'feature':    feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}',
                'shap_value': float(sv_val),
                'importance': float(abs(sv_val)),
                'direction':  'increases_risk' if sv_val > 0 else 'decreases_risk',
                'symbol':     '↑' if sv_val > 0 else '↓',
            })

        return top_features, 'shap'

    except Exception as e:
        print(f"⚠️  SHAP failed ({e}) — falling back to attention weights")
        top_features = explain_with_attention(model, sequence_tensor, feature_names, device)
        return top_features, 'attention'


def get_explanation(model, sequence_tensor, feature_names, device):
    """Master explainability function — tries SHAP, falls back to attention."""
    top_features, method = explain_with_shap(model, sequence_tensor, feature_names, device)

    # Build plain-English explanation
    if top_features:
        f1 = top_features[0]['feature']
        f2 = top_features[1]['feature'] if len(top_features) > 1 else ''
        d1 = top_features[0]['direction'].replace('_', ' ')
        explanation_text = (
            f"Elevated '{f1}' and '{f2}' are the primary signals. "
            f"'{f1}' {d1} in the current traffic window."
        )
    else:
        explanation_text = "Insufficient data for feature attribution."

    return {
        'method':           method,
        'top_features':     top_features,
        'explanation_text': explanation_text,
    }


# ============================================================================
# FULL ANALYSIS RUNNER
# ============================================================================

def run_full_analysis(csv_path,
                      model_path='models/world_model_best.pt',
                      config_path='models/model_config.json',
                      k=5):
    """
    End-to-end analysis pipeline.
    Load model → preprocess input → K-step forecast → explain → print report.
    """
    print("\n" + "█"*70)
    print("█" + "  NETGUARD AI — NETWORK ATTACK FORECASTER".center(68) + "█")
    print("█"*70)

    # Load model
    model, config, device = load_model(model_path, config_path)
    num_classes  = config['num_classes']
    target_names = config.get('target_names', [str(i) for i in range(num_classes)])

    # Preprocess input
    print(f"\n{'='*70}")
    print("PREPROCESSING INPUT")
    print('='*70)
    sequence_tensor, feature_names = preprocess_input(csv_path)

    # K-step forecast
    print(f"\n{'='*70}")
    print(f"K-STEP FORWARD SIMULATION (K={k})")
    print('='*70)
    forecast = run_k_step_forecast(model, sequence_tensor, device, k=k)

    # Print forecast table
    print(f"\n  {'Step':>4} │ {'Attack Risk':>11} │ {'MITRE Stage':<30} │ {'Conf':>6} │ Risk")
    print(f"  {'─'*4}─┼─{'─'*11}─┼─{'─'*30}─┼─{'─'*6}─┼─{'─'*8}")
    for r in forecast:
        print(
            f"  {r['step']:>4} │ {r['attack_probability']*100:>10.1f}% │ "
            f"{r['mitre_stage']:<30} │ {r['confidence']*100:>5.1f}% │ "
            f"{r['risk_emoji']} {r['risk_level']}"
        )

    # Overall risk summary
    max_attack_prob  = max(r['attack_probability'] for r in forecast)
    worst_step       = max(forecast, key=lambda r: r['attack_probability'])
    risk_label, risk_emoji = get_risk_level(max_attack_prob)

    print(f"\n{'─'*70}")
    print(f"  Overall Risk:    {risk_emoji} {risk_label}  ({max_attack_prob*100:.1f}% max attack probability)")
    print(f"  Worst Step:      Step {worst_step['step']} → {worst_step['mitre_stage']}")
    print(f"  MITRE Tactic ID: {worst_step['mitre_tactic_id']}")
    print(f"  Description:     {worst_step['mitre_description']}")

    # Explainability
    print(f"\n{'='*70}")
    print("EXPLAINABILITY — TOP RISK FACTORS")
    print('='*70)
    explanation = get_explanation(model, sequence_tensor, feature_names, device)
    print(f"  Method: {explanation['method'].upper()}")
    print(f"  Plain English: {explanation['explanation_text']}\n")
    print(f"  {'#':>3}  {'Feature':<35} {'Impact':>8}  Direction")
    print(f"  {'─'*3}  {'─'*35} {'─'*8}  {'─'*20}")
    for feat in explanation['top_features']:
        imp_str = f"{feat.get('shap_value', feat.get('importance', 0)):+.4f}"
        print(
            f"  {feat['rank']:>3}  {feat['feature']:<35} {imp_str:>8}  "
            f"{feat['symbol']} {feat['direction']}"
        )

    print(f"\n{'█'*70}")
    print("█" + "  ANALYSIS COMPLETE".center(68) + "█")
    print("█"*70 + "\n")

    return {
        'forecast':     forecast,
        'explanation':  explanation,
        'risk_level':   risk_label,
        'max_attack_prob': max_attack_prob,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NetGuard AI — Network Attack Forecaster'
    )
    parser.add_argument('--csv',   required=True,
                        help='Path to CSV file with network traffic')
    parser.add_argument('--model', default='models/world_model_best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--config',default='models/model_config.json',
                        help='Path to model config JSON')
    parser.add_argument('--k',     type=int, default=5,
                        help='Number of forecast steps (default: 5)')
    args = parser.parse_args()

    results = run_full_analysis(
        csv_path=args.csv,
        model_path=args.model,
        config_path=args.config,
        k=args.k,
    )
