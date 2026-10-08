import uvicorn
import os

if __name__ == "__main__":
    print("=" * 75)
    print("  Two-Stage Hybrid Deep Learning Architecture for Cross-Domain")
    print("         Anomaly Detection and Failure Prediction (v2.4)")
    print("=" * 75)
    print("  - Stage 1: HVAC Source Domain (VAE Latent Anomaly Representation z in R^8)")
    print("  - Stage 2: Machine Failure Prediction (BiLSTM-BiGRU-VAE Champion Network)")
    print("  - Benchmark: 98.7% Accuracy | 0.98 Prec | 1.00 Recall | 0.999 AUC")
    print("  - Explainability: Quad-XAI Suite (SHAP, LIME, PDP, ICE)")
    print("  - Interface: Pure Modern Light Industrial UI (FastAPI + Jinja2)")
    print("  - Authentication: JWT Cookie Sessions (Demo Accounts Preconfigured)")
    print("=" * 75)
    print("  Server launching at: http://localhost:8000")
    print("  Interactive API Swagger: http://localhost:8000/docs")
    print("=" * 75)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
