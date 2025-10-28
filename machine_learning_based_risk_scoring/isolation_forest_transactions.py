# ============================================================
# TRANSACTION-LEVEL ML-BASED RISK SCORING MODULE
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
# TRANSACTION ML RISK SCORER CLASS
# ============================================================
class TransactionMLRiskScorer:
    def __init__(self, data_dir="./data/joined and enriched data", output_dir="./data/kyc_ML_risk_scores", model_dir="./models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_dir = model_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = setup_logger()
        self.logger.info("TransactionMLRiskScorer initialized")

        self.transactions = None
        self.tx_features = None
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=150,
            contamination=0.3,
            random_state=42
        )

    # -----------------------------
    # Load enriched transactions
    # -----------------------------
    def load_data(self, filename="enriched_transactions.csv"):
        path = os.path.join(self.data_dir, filename)
        self.transactions = pd.read_csv(path)
        self.logger.info(f"Loaded {filename} | Records: {len(self.transactions)} | Columns: {len(self.transactions.columns)}")

    # -----------------------------
    # Create numeric features from rules
    # -----------------------------
    def create_features(self):
        tx = self.transactions
        self.tx_features = pd.DataFrame(index=tx.index)

        # Transaction amount > 10,000
        self.tx_features['large_tx_flag'] = (tx['transaction_amount'] > 10000).astype(int)

        # Destination country is high-risk
        high_risk_countries = ['Nigeria', 'Russia']
        self.tx_features['dest_high_risk'] = tx['destination_country'].isin(high_risk_countries).astype(int)

        # Cross-border to high-risk
        self.tx_features['cross_border_to_high_risk'] = tx['cross_border_to_high_risk']

        # Very large withdrawal
        self.tx_features['very_large_withdrawal_flag'] = (
            (tx['transaction_type'] == "Withdrawal") &
            (tx['transaction_amount'] > 10000)
        ).astype(int)

        self.logger.info("Numeric features created from transaction rules")

    # -----------------------------
    # Scale features
    # -----------------------------
    def scale_features(self):
        X_scaled = self.scaler.fit_transform(self.tx_features.fillna(0))
        self.logger.info(f"Feature matrix prepared | Shape: {X_scaled.shape}")
        return X_scaled

    # -----------------------------
    # Train Isolation Forest model
    # -----------------------------
    def train_model(self, X_scaled):
        self.model.fit(X_scaled)
        self.logger.info("Isolation Forest model trained successfully")

    # -----------------------------
    # Generate ML-based risk scores
    # -----------------------------
    def generate_risk_scores(self, X_scaled):
        tx = self.transactions

        tx['ml_anomaly_flag'] = self.model.predict(X_scaled)  # -1 = suspicious, 1 = normal
        tx['ml_anomaly_score'] = self.model.decision_function(X_scaled)

        # Convert anomaly score to 0–100 risk scale
        min_score, max_score = tx['ml_anomaly_score'].min(), tx['ml_anomaly_score'].max()
        tx['ml_risk_score'] = (
            (1 - (tx['ml_anomaly_score'] - min_score) / (max_score - min_score)) * 100
        ).round(2)

        # Categorize transaction risk
        tx['ml_risk_category'] = tx['ml_anomaly_flag'].map({1: 'Normal', -1: 'Suspicious'})

        self.logger.info("ML-based risk scores generated")

    # -----------------------------
    # Explain suspicious transactions
    # -----------------------------
    def explain_risk(self):
        tx = self.transactions
        feature_cols = [
            'large_tx_flag',
            'dest_high_risk',
            'cross_border_to_high_risk',
            'very_large_withdrawal_flag'
        ]

        tx['reason'] = self.tx_features[feature_cols].apply(
            lambda row: ", ".join([col for col in row.index if row[col] == 1]), axis=1
        )

        # For normal transactions, clear reason
        tx.loc[tx['ml_anomaly_flag'] == 1, 'reason'] = 'None'
        self.logger.info("Reason column added for explainability")

    # -----------------------------
    # Export results and model
    # -----------------------------
    def export_results(self, filename="transaction_ml_risk_scores_with_reason.csv"):
        output_path = os.path.join(self.output_dir, filename)
        self.transactions.to_csv(output_path, index=False)
        self.logger.info(f"ML-based transaction risk scores with reasons saved to {output_path}")

        # Save model and scaler
        joblib.dump(self.model, os.path.join(self.model_dir, "transaction_isolation_forest.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "transaction_scaler.pkl"))
        self.logger.info(f"Model and scaler saved to {self.model_dir}")

    # -----------------------------
    # Run full pipeline
    # -----------------------------
    def run_pipeline(self):
        self.logger.info("Starting ML-based Transaction Risk Scoring Pipeline (with rules & explanations)...")
        self.load_data()
        self.create_features()
        X_scaled = self.scale_features()
        self.train_model(X_scaled)
        self.generate_risk_scores(X_scaled)
        self.explain_risk()
        self.export_results()
        self.logger.info("Transaction ML Risk Scoring Pipeline completed successfully")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    scorer = TransactionMLRiskScorer()
    scorer.run_pipeline()

    print("\n==================== ML-BASED TRANSACTION RISK SCORES ====================")
    print(scorer.transactions[['transaction_id', 'client_id', 'transaction_amount',
                               'transaction_type', 'is_cross_border', 'cross_border_to_high_risk',
                               'ml_risk_score', 'ml_risk_category', 'reason']].head(10))
    print("==============================================================================")
