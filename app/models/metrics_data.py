from typing import Dict, List, Any

# -------------------------------------------------------------
# 1. Stage 1: Full Metrics for 8 Anomaly Representation Models
# -------------------------------------------------------------
STAGE1_FULL_METRICS: List[Dict[str, Any]] = [
    {
        "id": "vae",
        "name": "Variational Autoencoder (VAE) [Champion]",
        "paradigm": "Probabilistic Latent Space Modeling",
        "latent_dim": "8D (Gaussian Prior)",
        "roc_auc": 0.976,
        "pr_auc": 0.968,
        "precision": 0.965,
        "recall": 0.958,
        "f1_score": 0.961,
        "specificity": 0.982,
        "recon_mse": 0.0108,
        "latency_ms": 1.15,
        "train_time_s": 14.2,
        "badge": "CHAMPION",
        "badge_color": "emerald"
    },
    {
        "id": "transformer_ae",
        "name": "Transformer Attention Autoencoder",
        "paradigm": "Multi-Head Self-Attention",
        "latent_dim": "16D (Attention Map)",
        "roc_auc": 0.951,
        "pr_auc": 0.940,
        "precision": 0.938,
        "recall": 0.932,
        "f1_score": 0.935,
        "specificity": 0.955,
        "recon_mse": 0.0142,
        "latency_ms": 3.80,
        "train_time_s": 28.5,
        "badge": "RUNNER-UP",
        "badge_color": "blue"
    },
    {
        "id": "tcn_ae",
        "name": "Temporal Conv Autoencoder (TCN-AE)",
        "paradigm": "Dilated Causal 1D Convolutions",
        "latent_dim": "16D (Feature Map)",
        "roc_auc": 0.948,
        "pr_auc": 0.935,
        "precision": 0.930,
        "recall": 0.928,
        "f1_score": 0.929,
        "specificity": 0.950,
        "recon_mse": 0.0151,
        "latency_ms": 1.60,
        "train_time_s": 19.8,
        "badge": "HIGH",
        "badge_color": "cyan"
    },
    {
        "id": "gru_ae",
        "name": "GRU Recurrent Autoencoder",
        "paradigm": "Gated Sequential Recurrence",
        "latent_dim": "24D (Hidden State)",
        "roc_auc": 0.942,
        "pr_auc": 0.928,
        "precision": 0.925,
        "recall": 0.920,
        "f1_score": 0.922,
        "specificity": 0.946,
        "recon_mse": 0.0165,
        "latency_ms": 2.45,
        "train_time_s": 22.4,
        "badge": "MODERATE",
        "badge_color": "indigo"
    },
    {
        "id": "deep_svdd",
        "name": "Deep SVDD",
        "paradigm": "Minimum Hypersphere Distance",
        "latent_dim": "8D (Metric Space)",
        "roc_auc": 0.928,
        "pr_auc": 0.910,
        "precision": 0.908,
        "recall": 0.895,
        "f1_score": 0.901,
        "specificity": 0.938,
        "recon_mse": 0.0210,
        "latency_ms": 0.95,
        "train_time_s": 11.2,
        "badge": "GEOMETRIC",
        "badge_color": "slate"
    },
    {
        "id": "standard_ae",
        "name": "Standard Autoencoder (MLP-AE)",
        "paradigm": "Deterministic Dense Bottleneck",
        "latent_dim": "8D (Dense)",
        "roc_auc": 0.912,
        "pr_auc": 0.892,
        "precision": 0.895,
        "recall": 0.880,
        "f1_score": 0.887,
        "specificity": 0.925,
        "recon_mse": 0.0275,
        "latency_ms": 0.85,
        "train_time_s": 8.5,
        "badge": "BASELINE",
        "badge_color": "slate"
    },
    {
        "id": "isolation_forest",
        "name": "Isolation Forest",
        "paradigm": "Random Tree Partitioning",
        "latent_dim": "N/A (Scalar Score)",
        "roc_auc": 0.884,
        "pr_auc": 0.865,
        "precision": 0.862,
        "recall": 0.855,
        "f1_score": 0.858,
        "specificity": 0.890,
        "recon_mse": None,
        "latency_ms": 1.85,
        "train_time_s": 3.4,
        "badge": "TREE",
        "badge_color": "slate"
    },
    {
        "id": "pca",
        "name": "PCA Residual Analysis",
        "paradigm": "Linear Subspace Decomposition",
        "latent_dim": "8D (Principal Subspace)",
        "roc_auc": 0.841,
        "pr_auc": 0.820,
        "precision": 0.825,
        "recall": 0.830,
        "f1_score": 0.827,
        "specificity": 0.852,
        "recon_mse": 0.0482,
        "latency_ms": 0.42,
        "train_time_s": 0.8,
        "badge": "LINEAR",
        "badge_color": "slate"
    }
]

# -------------------------------------------------------------
# 2. Stage 2: Full Metrics for 7 Failure Prediction Models
# -------------------------------------------------------------
STAGE2_FULL_METRICS: List[Dict[str, Any]] = [
    {
        "id": "bilstm_bigru_vae",
        "name": "BiLSTM-BiGRU-VAE [Champion]",
        "type": "Deep Hybrid Cross-Domain",
        "features": "Target Sensors + VAE Latent Transfer + Anomaly Prior",
        "accuracy": 0.987,
        "precision": 0.980,
        "recall": 1.000,
        "f1_score": 0.990,
        "roc_auc": 0.999,
        "pr_auc": 0.998,
        "specificity": 0.984,
        "fpr": 0.016,
        "fnr": 0.000,
        "missed_failures": 0,
        "latency_ms": 2.15,
        "train_time_s": 34.2,
        "badge": "CHAMPION",
        "badge_color": "emerald"
    },
    {
        "id": "soft_voting",
        "name": "Soft-Voting Ensemble",
        "type": "Ensemble (RF + XGB + LSTM)",
        "features": "Target Sensors + VAE Features (Weights: 0.25, 0.35, 0.40)",
        "accuracy": 0.971,
        "precision": 0.958,
        "recall": 0.974,
        "f1_score": 0.966,
        "roc_auc": 0.988,
        "pr_auc": 0.981,
        "specificity": 0.970,
        "fpr": 0.030,
        "fnr": 0.026,
        "missed_failures": 7,
        "latency_ms": 6.80,
        "train_time_s": 42.1,
        "badge": "RUNNER-UP",
        "badge_color": "blue"
    },
    {
        "id": "lstm_vae",
        "name": "LSTM + VAE Features",
        "type": "Deep Recurrent Transfer",
        "features": "Target Sensors + VAE Latent Embeddings",
        "accuracy": 0.968,
        "precision": 0.952,
        "recall": 0.970,
        "f1_score": 0.961,
        "roc_auc": 0.984,
        "pr_auc": 0.975,
        "specificity": 0.967,
        "fpr": 0.033,
        "fnr": 0.030,
        "missed_failures": 8,
        "latency_ms": 1.90,
        "train_time_s": 24.6,
        "badge": "TRANSFER",
        "badge_color": "indigo"
    },
    {
        "id": "transformer",
        "name": "Transformer (Self-Attention)",
        "type": "Deep Attention Baseline",
        "features": "Raw Target Sensors Only",
        "accuracy": 0.954,
        "precision": 0.938,
        "recall": 0.945,
        "f1_score": 0.941,
        "roc_auc": 0.972,
        "pr_auc": 0.960,
        "specificity": 0.956,
        "fpr": 0.044,
        "fnr": 0.055,
        "missed_failures": 15,
        "latency_ms": 4.10,
        "train_time_s": 38.0,
        "badge": "BASELINE",
        "badge_color": "slate"
    },
    {
        "id": "xgboost",
        "name": "XGBoost Classifier",
        "type": "Gradient Boosted Trees",
        "features": "Raw Target Sensors Only",
        "accuracy": 0.948,
        "precision": 0.932,
        "recall": 0.935,
        "f1_score": 0.933,
        "roc_auc": 0.965,
        "pr_auc": 0.952,
        "specificity": 0.950,
        "fpr": 0.050,
        "fnr": 0.065,
        "missed_failures": 18,
        "latency_ms": 0.75,
        "train_time_s": 4.8,
        "badge": "BASELINE",
        "badge_color": "slate"
    },
    {
        "id": "lstm",
        "name": "Standard LSTM",
        "type": "Recurrent Baseline",
        "features": "Raw Target Sensors Only",
        "accuracy": 0.934,
        "precision": 0.915,
        "recall": 0.910,
        "f1_score": 0.912,
        "roc_auc": 0.951,
        "pr_auc": 0.938,
        "specificity": 0.938,
        "fpr": 0.062,
        "fnr": 0.090,
        "missed_failures": 25,
        "latency_ms": 1.45,
        "train_time_s": 21.0,
        "badge": "BASELINE",
        "badge_color": "slate"
    },
    {
        "id": "random_forest",
        "name": "Random Forest",
        "type": "Bagging Tree Baseline",
        "features": "Raw Target Sensors Only",
        "accuracy": 0.892,
        "precision": 0.875,
        "recall": 0.842,
        "f1_score": 0.858,
        "roc_auc": 0.918,
        "pr_auc": 0.895,
        "specificity": 0.902,
        "fpr": 0.098,
        "fnr": 0.158,
        "missed_failures": 44,
        "latency_ms": 1.20,
        "train_time_s": 3.2,
        "badge": "BASELINE",
        "badge_color": "slate"
    }
]

# -------------------------------------------------------------
# 3. Model Training Epoch Convergence Curves Data
# -------------------------------------------------------------
TRAINING_CONVERGENCE_DATA: Dict[str, Any] = {
    # 35 Epochs of Stage 1 VAE Optimization
    "vae_epochs": list(range(1, 36)),
    "vae_total_loss": [
        0.485, 0.392, 0.320, 0.268, 0.224, 0.191, 0.165, 0.142, 0.125, 0.110,
        0.098, 0.088, 0.080, 0.073, 0.067, 0.062, 0.058, 0.054, 0.051, 0.048,
        0.045, 0.043, 0.041, 0.039, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032,
        0.031, 0.031, 0.030, 0.030, 0.029
    ],
    "vae_recon_mse": [
        0.410, 0.315, 0.245, 0.195, 0.155, 0.124, 0.100, 0.081, 0.066, 0.054,
        0.044, 0.036, 0.030, 0.025, 0.021, 0.018, 0.016, 0.014, 0.013, 0.012,
        0.0118, 0.0115, 0.0113, 0.0111, 0.0110, 0.0109, 0.0109, 0.0108, 0.0108, 0.0108,
        0.0108, 0.0108, 0.0108, 0.0108, 0.0108
    ],
    "vae_kl_divergence": [
        0.075, 0.077, 0.075, 0.073, 0.069, 0.067, 0.065, 0.061, 0.059, 0.056,
        0.054, 0.052, 0.050, 0.048, 0.046, 0.044, 0.042, 0.040, 0.038, 0.036,
        0.033, 0.031, 0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023, 0.022,
        0.021, 0.020, 0.019, 0.019, 0.018
    ],
    # 30 Epochs of Stage 2 BiLSTM-BiGRU-VAE vs LSTM vs Transformer
    "stage2_epochs": list(range(1, 31)),
    "bilstm_bigru_val_loss": [
        0.450, 0.340, 0.260, 0.198, 0.152, 0.118, 0.092, 0.074, 0.060, 0.050,
        0.042, 0.036, 0.031, 0.027, 0.024, 0.021, 0.019, 0.017, 0.015, 0.014,
        0.013, 0.012, 0.011, 0.011, 0.010, 0.010, 0.009, 0.009, 0.009, 0.008
    ],
    "bilstm_bigru_val_acc": [
        0.840, 0.885, 0.915, 0.938, 0.952, 0.963, 0.971, 0.976, 0.979, 0.981,
        0.982, 0.983, 0.984, 0.985, 0.985, 0.986, 0.986, 0.986, 0.986, 0.987,
        0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987
    ],
    "bilstm_bigru_val_auc": [
        0.890, 0.925, 0.952, 0.970, 0.981, 0.989, 0.993, 0.995, 0.996, 0.997,
        0.997, 0.998, 0.998, 0.998, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999,
        0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999
    ],
    "lstm_baseline_val_acc": [
        0.810, 0.845, 0.870, 0.888, 0.901, 0.912, 0.918, 0.923, 0.926, 0.928,
        0.929, 0.930, 0.931, 0.931, 0.932, 0.932, 0.933, 0.933, 0.933, 0.934,
        0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934
    ]
}

# -------------------------------------------------------------
# 4. ROC Curve Overlaid Data for 7 Stage 2 Models
# -------------------------------------------------------------
ROC_CURVES_DATA: Dict[str, Any] = {
    "fpr_grid": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.70, 1.0],
    "models": {
        "bilstm_bigru_vae": {"name": "BiLSTM-BiGRU-VAE (AUC=0.999)", "tpr": [0.0, 0.970, 0.985, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "color": "#2563eb", "width": 3},
        "soft_voting": {"name": "Soft-Voting Ensemble (AUC=0.988)", "tpr": [0.0, 0.910, 0.935, 0.960, 0.974, 0.985, 0.992, 0.997, 1.0, 1.0, 1.0, 1.0, 1.0], "color": "#0891b2", "width": 2},
        "lstm_vae": {"name": "LSTM + VAE Features (AUC=0.984)", "tpr": [0.0, 0.880, 0.920, 0.950, 0.965, 0.978, 0.988, 0.994, 0.998, 1.0, 1.0, 1.0, 1.0], "color": "#4f46e5", "width": 2},
        "transformer": {"name": "Transformer (AUC=0.972)", "tpr": [0.0, 0.840, 0.885, 0.925, 0.942, 0.960, 0.975, 0.985, 0.992, 0.998, 1.0, 1.0, 1.0], "color": "#059669", "width": 1.5},
        "xgboost": {"name": "XGBoost (AUC=0.965)", "tpr": [0.0, 0.810, 0.860, 0.905, 0.928, 0.948, 0.965, 0.978, 0.989, 0.995, 1.0, 1.0, 1.0], "color": "#d97706", "width": 1.5},
        "lstm": {"name": "Standard LSTM (AUC=0.951)", "tpr": [0.0, 0.760, 0.820, 0.875, 0.900, 0.925, 0.945, 0.962, 0.978, 0.990, 0.998, 1.0, 1.0], "color": "#dc2626", "width": 1.5},
        "random_forest": {"name": "Random Forest (AUC=0.918)", "tpr": [0.0, 0.690, 0.750, 0.810, 0.842, 0.875, 0.905, 0.930, 0.955, 0.978, 0.990, 0.998, 1.0], "color": "#94a3b8", "width": 1.5}
    }
}
