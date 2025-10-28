# ============================================================
# MAIN ORCHESTRATION PIPELINE FOR KYC RISK SCORING SYSTEM
# ============================================================

import logging
import os
from data_ingestion.ingest_generate_data import KYCDataGenerator
from data_joining_enrichment.enrichment_pipeline import KYCEnrichmentPipeline
from rule_based_risk_scoring.apply_business_rules import ClientRiskScorer, TransactionRiskScorer
from machine_learning_based_risk_scoring.isolation_forest_clients import MLClientRiskScorer
from machine_learning_based_risk_scoring.isolation_forest_transactions import TransactionMLRiskScorer
from output_generation.output_anomolous_clients import combine_client_risk_scores
from output_generation.output_suspicious_transactions import combine_transaction_risk_scores

# ------------------------------------------------------------
# LOGGER SETUP
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# PIPELINE EXECUTION
# ------------------------------------------------------------
def run_kyc_pipeline():
    logger.info("===== STARTING END-TO-END KYC RISK SCORING PIPELINE =====")

    # Data Ingestion
    data_gen = KYCDataGenerator(num_clients=150)
    data_gen.run()
    logger.info("Data ingestion complete")

    # Data Joining & Enrichment
    enrichment = KYCEnrichmentPipeline()
    enrichment.run_pipeline()
    logger.info("Data enrichment complete")

    # Business Rule-Based Risk Scoring
    client_rule_scorer = ClientRiskScorer()
    client_rule_scorer.run()
    transaction_rule_scorer = TransactionRiskScorer()
    transaction_rule_scorer.run()
    logger.info("Business rule-based scoring complete")

    # ML-Based Risk Scoring (Isolation Forest)
    ml_client_scorer = MLClientRiskScorer()
    ml_client_scorer.run_pipeline()
    ml_transaction_scorer = TransactionMLRiskScorer()
    ml_transaction_scorer.run_pipeline()
    logger.info("ML-based scoring complete")

    # Output Generation (Combine ML + Rule Outputs)
    combine_client_risk_scores()
    combine_transaction_risk_scores()
    logger.info("Combined ML + Business Rule Outputs Generated")

    logger.info("===== KYC PIPELINE COMPLETED SUCCESSFULLY =====")


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    run_kyc_pipeline()
