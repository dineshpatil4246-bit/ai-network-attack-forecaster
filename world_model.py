import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class NetworkWorldModel(nn.Module):
    def __init__(self, num_features):
        super(NetworkWorldModel, self).__init__()
        # Bidirectional LSTM: 2 layers, hidden_size=128, dropout=0.3
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Multi-head self-attention: embed_dim=256, num_heads=4
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True
        )
        
        # Linear -> ReLU -> Dropout -> Linear
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # x shape: (batch, seq_len=20, num_features)
        
        # lstm_out shape: (batch, 20, 256)
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        # attn_out shape: (batch, 20, 256), attn_weights shape: (batch, 20, 20)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Mean across time: shape (batch, 256)
        x_mean = attn_out.mean(dim=1)
        
        # Fully connected layers
        x_fc = torch.relu(self.fc1(x_mean))
        x_fc = self.dropout(x_fc)
        logits = self.fc2(x_fc)
        
        return logits, attn_weights

    def predict_proba(self, x):
        """Returns softmax probabilities for the 3 classes."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_attention_summary(self, x):
        """Returns mean attention per timestep (batch, 20), averaged over heads."""
        _, attn_weights = self.forward(x)
        # attn_weights is (batch, target_seq_len, source_seq_len) -> (batch, 20, 20)
        # Averaging over the query dimension to get a summary of importance per key timestep
        return attn_weights.mean(dim=1)


def train_model(X_path, y_path, epochs=60, batch_size=64, lr=0.001):
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Load data
    print("Loading data...")
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded y shape: {y.shape}")
    num_features = X.shape[2]
    
    # Stratified 80/20 train/val split (random_state fixed to share val split with eval)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = NetworkWorldModel(num_features).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_f1 = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluation phase
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        # Metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = accuracy_score(val_targets, val_preds) * 100
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.1f}% | LR: {current_lr:.5f}")

        # LR step
        scheduler.step(val_loss)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'models/world_model_best.pt')

    # Save history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)

    return history


def evaluate_model(model_path, X_path, y_path):
    os.makedirs('outputs', exist_ok=True)
    
    print("Loading data for evaluation...")
    X = np.load(X_path)
    y = np.load(y_path)
    num_features = X.shape[2]
    
    # Run on validation split only, recreating the split predictably
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = NetworkWorldModel(num_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())

    target_names = ['Benign', 'DoS/DDoS', 'PortScan/BruteForce']
    
    print("\nEvaluation Report:")
    print(classification_report(val_targets, val_preds, target_names=target_names))

    # Confusion matrix heatmap
    cm = confusion_matrix(val_targets, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Network Attack Forecaster - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved to outputs/confusion_matrix.png")

    return f1_score(val_targets, val_preds, average='macro')


if __name__ == '__main__':
    X_path = 'processed/X_sequences.npy'
    y_path = 'processed/y_labels.npy'

    # Train Model
    history = train_model(X_path, y_path, epochs=30)
    
    # Evaluate Model
    best_f1 = evaluate_model('models/world_model_best.pt', X_path, y_path)
    print(f"Final Model Best Validation Macro F1: {best_f1:.4f}")
