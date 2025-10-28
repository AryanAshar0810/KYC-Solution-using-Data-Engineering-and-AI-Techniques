# ============================================================
# COMBINE CLIENT ML + BUSINESS RULE RISK SCORES
# ============================================================

import pandas as pd
import os
import logging

def combine_client_risk_scores(
    input_dir="./data",
    output_dir="./final output dataset",
    ml_file="kyc_ML_risk_scores/client_ml_risk_scores.csv",
    rule_file="./risk_scores/clients_risk_summary.csv",
    output_file="clients_combined_risk_scores.csv"
):
    os.makedirs(output_dir, exist_ok=True)

    ml_path = os.path.join(input_dir, ml_file)
    rule_path = os.path.join(input_dir, rule_file)
    output_path = os.path.join(output_dir, output_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting client risk dataset join process...")

    # -----------------------------
    # LOAD BOTH DATASETS
    # -----------------------------
    try:
        ml_df = pd.read_csv(ml_path)
        rule_df = pd.read_csv(rule_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    logger.info(f"Loaded ML risk data: {ml_df.shape[0]} rows, {ml_df.shape[1]} columns")
    logger.info(f"Loaded Business Rule risk data: {rule_df.shape[0]} rows, {rule_df.shape[1]} columns")

    # -----------------------------
    # SELECT RELEVANT COLUMNS
    # -----------------------------
    ml_cols = [
        "client_id", "name", "country", "kyc_status",
        "ml_anomaly_score", "ml_anomaly_flag",
        "ml_risk_score", "ml_risk_category"
    ]

    rule_cols = [
        "client_id", "total_risk_score", "triggered_rules"
    ]

    ml_df = ml_df[ml_cols].copy()
    rule_df = rule_df[rule_cols].copy()
    rule_df = rule_df.rename(columns={"total_risk_score": "risk_score_from_business_rules"})

    # -----------------------------
    # MERGE ON client_id
    # -----------------------------
    combined_df = pd.merge(ml_df, rule_df, on="client_id", how="left")

    # -----------------------------
    # REORDER COLUMNS
    # -----------------------------
    final_cols = [
        "client_id", "name", "country", "kyc_status",
        "risk_score_from_business_rules", "ml_anomaly_score", "ml_anomaly_flag",
        "ml_risk_score", "ml_risk_category", "triggered_rules"
    ]

    combined_df = combined_df[final_cols]

    # -----------------------------
    # EXPORT FINAL OUTPUT
    # -----------------------------
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Combined client risk dataset saved to {output_path}")

    # -----------------------------
    # DISPLAY SAMPLE OUTPUT
    # -----------------------------
    print("\n==================== COMBINED CLIENT RISK SCORES ====================")
    print(combined_df.head(10))
    print("=====================================================================")


# Optional: run standalone
if __name__ == "__main__":
    combine_client_risk_scores()
