# ============================================================
# KYC RISK SCORING MODULE (CLIENT & TRANSACTION LEVEL)
# ============================================================

import pandas as pd
import os
import logging

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
# CLIENT RISK SCORING CLASS
# ============================================================
class ClientRiskScorer:
    def __init__(
        self,
        data_dir="./data/joined and enriched data",
        output_dir="./data/risk_scores",
        compliance_file="./data/raw data from data ingestion/compliance_rules.csv"
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.compliance_file = compliance_file
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = setup_logger()

        # Load data
        self.enriched_clients = pd.read_csv(os.path.join(self.data_dir, "enriched_clients.csv"))
        self.compliance_rules = pd.read_csv(self.compliance_file)
        self.client_rules = self.compliance_rules[
            self.compliance_rules['rule_name'].str.startswith("CLIENT RULE - ")
        ].copy()

        self.logger.info(
            f"Loaded {len(self.enriched_clients)} clients and {len(self.client_rules)} client rules"
        )

    def prepare_flags(self):
        # High-risk countries from rule conditions
        high_risk_countries_origin = []
        for cond in self.client_rules['rule_condition']:
            if 'origin_country' in cond or 'destination_country' in cond:
                if '[' in cond and ']' in cond:
                    countries_list = cond.split('[')[1].split(']')[0].replace('"','').replace("'","").split(',')
                    high_risk_countries_origin.extend([c.strip() for c in countries_list])

        self.enriched_clients['tx_from_high_risk_origin'] = self.enriched_clients['country'].isin(high_risk_countries_origin).astype(int)
        self.enriched_clients['tx_to_high_risk_dest'] = self.enriched_clients['high_risk_cross_border_count'].apply(lambda x: 1 if x>0 else 0)
        self.enriched_clients['failed_tx_last_30_days'] = self.enriched_clients.get('failed_transaction_count', 0)

    def apply_rules(self):
        risk_scores = []
        triggered_rules_list = []

        for _, client in self.enriched_clients.iterrows():
            total_risk = 0
            triggered_rules = []

            for _, rule in self.client_rules.iterrows():
                rule_name = rule['rule_name'].replace("CLIENT RULE - ", "")
                weight = rule['risk_weight']

                try:
                    if rule_name == "Unverified KYC" and client['kyc_status'] != "Verified":
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "Multiple Failed Transactions" and client['failed_tx_last_30_days'] > 3:
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "Frequent Large Transactions" and client['large_tx_last_7_days'] > 5:
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "Document Verification Failed" and client['doc_verification_status'] == "Failed":
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "Frequent cross-border transactions" and client['cross_border_count'] > 5:
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "Transactions to high-risk countries" and client['high_risk_cross_border_count'] > 2:
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                except KeyError as e:
                    self.logger.warning(f"Client {client['client_id']}: missing column for rule {rule_name} -> {e}")

            risk_scores.append(total_risk)
            triggered_rules_list.append(", ".join(triggered_rules) if triggered_rules else "None")

        self.enriched_clients['total_risk_score'] = risk_scores
        self.enriched_clients['triggered_rules'] = triggered_rules_list

    def export(self):
        output_file = os.path.join(self.output_dir, "clients_risk_summary.csv")
        client_risk_df = self.enriched_clients[['client_id', 'name', 'country', 'kyc_status', 'total_risk_score', 'triggered_rules']]
        client_risk_df.to_csv(output_file, index=False)
        self.logger.info(f"Client-level risk scoring saved to {output_file}")

    def run(self):
        self.prepare_flags()
        self.apply_rules()
        self.export()

# ============================================================
# TRANSACTION RISK SCORING CLASS
# ============================================================
class TransactionRiskScorer:
    def __init__(
        self,
        data_dir="./data/joined and enriched data",
        output_dir="./data/risk_scores", 
        compliance_file="./data/raw data from data ingestion/compliance_rules.csv"
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.compliance_file = compliance_file
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = setup_logger()

        # Load data
        self.enriched_transactions = pd.read_csv(
            os.path.join(self.data_dir, "enriched_transactions.csv")
        )
        self.compliance_rules = pd.read_csv(self.compliance_file)
        self.transaction_rules = self.compliance_rules[
            self.compliance_rules['rule_name'].str.startswith("TRANSACTION RULE - ")
        ].copy()

        self.logger.info(
            f"Loaded {len(self.enriched_transactions)} transactions and {len(self.transaction_rules)} transaction rules"
        )


    def apply_rules(self):
        risk_scores = []
        triggered_rules_list = []

        for _, tx in self.enriched_transactions.iterrows():
            total_risk = 0
            triggered_rules = []

            for _, rule in self.transaction_rules.iterrows():
                rule_name = rule['rule_name'].replace("TRANSACTION RULE - ", "")
                weight = rule['risk_weight']

                try:
                    if rule_name == "High Transaction Amount" and tx['transaction_amount'] > 10000:
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "High-Risk Country (Destination)" and tx['destination_country'] in ["Nigeria", "Russia"]:
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "Cross-Border to High-Risk Country" and tx['cross_border_to_high_risk'] == 1:
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                    elif rule_name == "Very large transaction by withdrawl" and tx['transaction_amount'] > 10000 and tx['transaction_type'] == "Withdrawal":
                        triggered_rules.append(f"{rule_name}({weight})")
                        total_risk += weight
                except KeyError as e:
                    self.logger.warning(f"Transaction {tx['transaction_id']}: missing column for rule {rule_name} -> {e}")

            risk_scores.append(total_risk)
            triggered_rules_list.append(", ".join(triggered_rules) if triggered_rules else "None")

        self.enriched_transactions['transaction_risk_score'] = risk_scores
        self.enriched_transactions['triggered_rules'] = triggered_rules_list

    def export(self):
        output_file = os.path.join(self.output_dir, "transactions_risk_summary.csv")
        tx_risk_df = self.enriched_transactions[['transaction_id', 'client_id', 'transaction_amount', 
                                                 'origin_country', 'destination_country', 'transaction_date',
                                                 'transaction_risk_score', 'triggered_rules']]
        tx_risk_df.to_csv(output_file, index=False)
        self.logger.info(f"Transaction-level risk scoring saved to {output_file}")

    def run(self):
        self.apply_rules()
        self.export()


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    client_scorer = ClientRiskScorer()
    client_scorer.run()

    transaction_scorer = TransactionRiskScorer()
    transaction_scorer.run()
    print("Completed")
