# ============================================================
# CLIENT-LEVEL ML-BASED RISK SCORING MODULE
# ============================================================

import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# LOGGER SETUP
# ============================================================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


# ============================================================
# ML-BASED CLIENT RISK SCORER CLASS
# ============================================================
class MLClientRiskScorer:
    def __init__(self, data_dir="./data/joined and enriched data", output_dir="./data/kyc_ML_risk_scores", model_dir="./models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_dir = model_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = setup_logger()
        self.logger.info("MLClientRiskScorer initialized")

        self.enriched_clients = None
        self.features = [
            'kyc_status_encoded',
            'cross_border_count',
            'high_risk_cross_border_count',
            'failed_transaction_count',
            'client_country_aml_score',
        ]
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=150,
            contamination=0.2,
            random_state=42
        )

    # -----------------------------
    # Load enriched client data
    # -----------------------------
    def load_data(self, filename="enriched_clients.csv"):
        path = os.path.join(self.data_dir, filename)
        self.enriched_clients = pd.read_csv(path)
        self.logger.info(f"Loaded {filename} | Records: {len(self.enriched_clients)} | Columns: {len(self.enriched_clients.columns)}")

    # -----------------------------
    # Encode categorical features
    # -----------------------------
    def encode_features(self):
        kyc_map = {'Verified': 0, 'Pending': 1, 'Under Review': 2}
        self.enriched_clients['kyc_status_encoded'] = self.enriched_clients['kyc_status'].map(kyc_map)
        self.enriched_clients['kyc_status_encoded'] = self.enriched_clients['kyc_status_encoded'].fillna(1).astype(int)
        self.logger.info(" Encoded categorical features (kyc_status)")

    # -----------------------------
    # Prepare feature matrix
    # -----------------------------
    def prepare_features(self):
        X = self.enriched_clients[self.features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.logger.info(f"Feature matrix prepared | Shape: {X_scaled.shape}")
        return X_scaled

    # -----------------------------
    # Train Isolation Forest model
    # -----------------------------
    def train_model(self, X_scaled):
        self.model.fit(X_scaled)
        self.logger.info("Isolation Forest model trained successfully")

    # -----------------------------
    # Compute risk scores
    # -----------------------------
    def compute_risk_scores(self, X_scaled):
        self.enriched_clients['ml_anomaly_score'] = self.model.decision_function(X_scaled)
        self.enriched_clients['ml_anomaly_flag'] = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        # Normalize to 0–100 risk scale
        min_score, max_score = self.enriched_clients['ml_anomaly_score'].min(), self.enriched_clients['ml_anomaly_score'].max()
        self.enriched_clients['ml_risk_score'] = (
            (1 - (self.enriched_clients['ml_anomaly_score'] - min_score) / (max_score - min_score)) * 100
        ).round(2)

        # Categorize risk levels
        self.enriched_clients['ml_risk_category'] = pd.cut(
            self.enriched_clients['ml_risk_score'],
            bins=[0, 20, 60, 100],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        self.logger.info("ML-based risk scores and categories computed")

    # -----------------------------
    # Save results and model
    # -----------------------------
    def export_results(self, filename="client_ml_risk_scores.csv"):
        # Save enriched client risk scores
        output_path = os.path.join(self.output_dir, filename)
        self.enriched_clients.to_csv(output_path, index=False)
        self.logger.info(f"ML-based client risk scores saved to {output_path}")

        # Save model and scaler
        joblib.dump(self.model, os.path.join(self.model_dir, "isolation_forest_model_for_clients.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler_for_clients.pkl"))
        self.logger.info(f"ML model and scaler saved to {self.model_dir}")

    # -----------------------------
    # Run full pipeline
    # -----------------------------
    def run_pipeline(self):
        self.load_data()
        self.encode_features()
        X_scaled = self.prepare_features()
        self.train_model(X_scaled)
        self.compute_risk_scores(X_scaled)
        self.export_results()
        self.logger.info("ML-based Client Risk Scoring Pipeline completed successfully")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    scorer = MLClientRiskScorer()
    scorer.run_pipeline()
    print("\n==================== ML-BASED CLIENT RISK SCORES ====================")
    print(scorer.enriched_clients[['client_id', 'name', 'kyc_status', 'ml_risk_score', 'ml_risk_category']].head(10))
    print("=====================================================================")
