# ============================================================
# COMBINE TRANSACTION ML + BUSINESS RULE RISK SCORES
# ============================================================

import pandas as pd
import os
import logging

def combine_transaction_risk_scores(
    input_dir="./data",
    output_dir="./final output dataset",
    ml_file="kyc_ML_risk_scores/transaction_ml_risk_scores_with_reason.csv",
    rule_file="./risk_scores/transactions_risk_summary.csv",
    output_file="transactions_combined_risk_scores.csv"
):
    """
    Combine ML-based and rule-based transaction risk scores into a unified dataset.
    Filters out low-risk transactions and exports a final scored dataset.
    """

    # -----------------------------
    # CONFIGURATION & LOGGING
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)

    ml_path = os.path.join(input_dir, ml_file)
    rule_path = os.path.join(input_dir, rule_file)
    output_path = os.path.join(output_dir, output_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting transaction risk dataset join process...")

    # -----------------------------
    # LOAD BOTH DATASETS
    # -----------------------------
    try:
        ml_df = pd.read_csv(ml_path)
        rule_df = pd.read_csv(rule_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    logger.info(f"Loaded ML transaction data: {ml_df.shape[0]} rows, {ml_df.shape[1]} columns")
    logger.info(f"Loaded Business Rule transaction data: {rule_df.shape[0]} rows, {rule_df.shape[1]} columns")

    # -----------------------------
    # SELECT RELEVANT COLUMNS
    # -----------------------------
    ml_cols = [
        "transaction_id", "client_id", "transaction_amount", "transaction_type",
        "ml_anomaly_flag", "ml_anomaly_score", "ml_risk_score", "ml_risk_category", "reason"
    ]

    rule_cols = [
        "transaction_id", "transaction_risk_score", "triggered_rules"
    ]

    ml_df = ml_df[ml_cols].copy()
    rule_df = rule_df[rule_cols].copy()

    # -----------------------------
    # RENAME COLUMN FROM RULES
    # -----------------------------
    rule_df = rule_df.rename(columns={"transaction_risk_score": "transaction_score_from_business_rules"})

    # -----------------------------
    # MERGE ON transaction_id
    # -----------------------------
    combined_df = pd.merge(ml_df, rule_df, on="transaction_id", how="left")

    # -----------------------------
    # FILTER OUT LOW-RISK NORMAL TRANSACTIONS
    # (Keep only if ML flagged as Suspicious OR Business Rules gave non-zero score)
    # -----------------------------
    initial_count = combined_df.shape[0]

    filtered_df = combined_df[
        ~((combined_df["ml_risk_category"] == "Normal") &
          (combined_df["transaction_score_from_business_rules"] == 0.0))
    ].copy()

    filtered_count = filtered_df.shape[0]
    logger.info(f"Filtered out {initial_count - filtered_count} low-risk normal transactions")

    # -----------------------------
    # REORDER COLUMNS
    # -----------------------------
    final_cols = [
        "transaction_id", "client_id", "transaction_amount", "transaction_type",
        "transaction_score_from_business_rules", "ml_anomaly_flag", "ml_anomaly_score",
        "ml_risk_score", "ml_risk_category", "reason", "triggered_rules"
    ]

    filtered_df = filtered_df[final_cols]

    # -----------------------------
    # EXPORT FINAL OUTPUT
    # -----------------------------
    filtered_df.to_csv(output_path, index=False)
    logger.info(f"Combined transaction risk dataset saved to {output_path}")

    # -----------------------------
    # DISPLAY SAMPLE OUTPUT
    # -----------------------------
    print("\n==================== COMBINED TRANSACTION RISK SCORES ====================")
    print(filtered_df.head(10))
    print("==========================================================================")



# ============================================================
# MAIN EXECUTION (for standalone runs)
# ============================================================
if __name__ == "__main__":
    combine_transaction_risk_scores()
