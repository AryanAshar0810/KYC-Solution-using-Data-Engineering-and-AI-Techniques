# ============================================================
# KYC SAMPLE DATA GENERATOR
# ============================================================

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_base_dir():
    """
    Determines a reliable base directory.
    Works locally and on Databricks or any other environment.
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    return base


def get_output_dir():
    """
    Creates and returns the standardized output directory:
    '/kyc_data'
    """
    base_dir = get_base_dir()
    OUTPUT_DIR = os.path.join("data", "raw data from data ingestion")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# ============================================================
# KYC DATA GENERATOR CLASS
# ============================================================

class KYCDataGenerator:
    def __init__(self, num_clients=150, seed=42):
        np.random.seed(seed)
        self.num_clients = num_clients
        self.output_dir = get_output_dir()
        self.master_country_list = [
            'USA', 'UK', 'Germany', 'Switzerland', 'Nigeria',
            'Russia', 'China', 'UAE', 'Canada', 'Australia'
        ]

    # --------------------------------------------------------
    # 1. Generate DOCUMENTS
    # --------------------------------------------------------
    def generate_documents(self):
        data = {
            'document_id': range(1, self.num_clients + 1),
            'client_id': range(1, self.num_clients + 1),
            'document_type': np.random.choice(['Passport', 'ID Card', 'Utility Bill', 'Bank Statement'], self.num_clients),
            'verification_status': np.random.choice(['Verified', 'Failed', 'Pending'], self.num_clients, p=[0.7, 0.15, 0.15]),
            'upload_date': [(datetime.now() - timedelta(days=np.random.randint(0, 365))).date()
                            for _ in range(self.num_clients)]
        }
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, 'documents.csv')
        df.to_csv(path, index=False)
        return df

    # --------------------------------------------------------
    # 2. Generate CLIENTS
    # --------------------------------------------------------
    def generate_clients(self, documents_df):
        kyc_status_map = {
            'Verified': 'Verified',
            'Pending': 'Pending',
            'Failed': 'Under Review'
        }

        dobs = [(datetime.now() - timedelta(days=np.random.randint(365*25, 365*75))).date()
                for _ in range(self.num_clients)]

        data = {
            'client_id': range(1, self.num_clients + 1),
            'name': [f'Client_{i}' for i in range(1, self.num_clients + 1)],
            'date_of_birth': dobs,
            'age': [((datetime.now()).date() - dob).days // 365 for dob in dobs],
            'country': np.random.choice(self.master_country_list, self.num_clients),
            'email': [f'client{i}@example.com' for i in range(1, self.num_clients + 1)],
            'registration_date': [(datetime.now() - timedelta(days=np.random.randint(1, 365))).date()
                                  for _ in range(self.num_clients)],
            'kyc_status': [kyc_status_map[status] for status in documents_df['verification_status']],
            'customer_type': np.random.choice(['Individual', 'Business', 'Corporate'], self.num_clients)
        }
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, 'clients.csv')
        df.to_csv(path, index=False)
        return df

    # --------------------------------------------------------
    # 3. Generate TRANSACTIONS
    # --------------------------------------------------------
    def generate_transactions(self, clients_df):
        transactions = []
        transaction_id = 1
        client_country_map = dict(zip(clients_df['client_id'], clients_df['country']))

        for client_id in range(1, self.num_clients + 1):
            origin_country = client_country_map[client_id]
            num_tx = np.random.randint(5, 50)

            for _ in range(num_tx):
                tx_date = (datetime.now() - timedelta(days=np.random.randint(0, 90))).date()
                amount = np.random.choice(
                    list(np.random.randint(2000, 5000, 70)) +
                    list(np.random.randint(10000, 50000, 10))
                )

                if np.random.rand() < 0.8:
                    dest_country = origin_country
                else:
                    dest_country = random.choice([c for c in self.master_country_list if c != origin_country])

                transactions.append({
                    'transaction_id': transaction_id,
                    'client_id': client_id,
                    'transaction_amount': int(amount),
                    'transaction_date': tx_date,
                    'transaction_type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer', 'Investment']),
                    'status': np.random.choice(['Completed', 'Pending', 'Failed'], p=[0.85, 0.10, 0.05]),
                    'origin_country': origin_country,
                    'destination_country': dest_country
                })
                transaction_id += 1

        df = pd.DataFrame(transactions)
        path = os.path.join(self.output_dir, 'transactions.csv')
        df.to_csv(path, index=False)
        return df

    # --------------------------------------------------------
    # 4. Generate COUNTRY RISK LEVELS
    # --------------------------------------------------------
    def generate_country_risks(self):
        data = {
            'country': self.master_country_list,
            'risk_level': ['Low', 'Low', 'Low', 'Medium', 'High',
                           'High', 'Medium', 'Medium', 'Low', 'Low'],
            'aml_compliance_score': [95, 92, 94, 80, 45, 50, 65, 75, 93, 91]
        }
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, 'countries_risk_levels.csv')
        df.to_csv(path, index=False)
        return df

    # --------------------------------------------------------
    # 5. Generate COMPLIANCE RULES
    # --------------------------------------------------------
    def generate_compliance_rules(self):
        data = [
            {
                'rule_id': 1,
                'rule_name': 'TRANSACTION RULE - High Transaction Amount',
                'rule_description': 'Flag transactions greater than 10,000 EUR as potentially high risk.',
                'rule_condition': 'transaction_amount > 10000',
                'risk_weight': 0.8,
                'active': True
            },
            {
                'rule_id': 2,
                'rule_name': 'TRANSACTION RULE - High-Risk Country (Origin)',
                'rule_description': 'Flag transactions originating from high-risk countries.',
                'rule_condition': 'origin_country in ["Nigeria", "Russia"]',
                'risk_weight': 0.9,
                'active': True
            },
            {
                'rule_id': 3,
                'rule_name': 'TRANSACTION RULE - High-Risk Country (Destination)',
                'rule_description': 'Flag transactions sent to high-risk countries.',
                'rule_condition': 'destination_country in ["Nigeria", "Russia"]',
                'risk_weight': 0.9,
                'active': True
            },
            {
                'rule_id': 4,
                'rule_name': 'CLIENT RULE - Unverified KYC',
                'rule_description': 'Flag clients whose KYC status is not verified.',
                'rule_condition': 'kyc_status != "Verified"',
                'risk_weight': 0.7,
                'active': True
            },
            {
                'rule_id': 5,
                'rule_name': 'CLIENT RULE - Multiple Failed Transactions',
                'rule_description': 'Flag clients with more than 3 failed transactions in the last 30 days.',
                'rule_condition': 'failed_tx_last_30_days > 3',
                'risk_weight': 0.6,
                'active': True
            },
            {
                'rule_id': 6,
                'rule_name': 'CLIENT RULE - Frequent Large Transactions',
                'rule_description': 'Flag clients who have made more than 5 transactions >10,000 in the last 7 days.',
                'rule_condition': 'large_tx_last_7_days > 5',
                'risk_weight': 0.7,
                'active': True
            },
            {
                'rule_id': 7,
                'rule_name': 'CLIENT RULE - Document Verification Failed',
                'rule_description': 'Flag clients whose identification document failed verification.',
                'rule_condition': 'verification_status == "Failed"',
                'risk_weight': 0.85,
                'active': True
            },
            {
                'rule_id': 8,
                'rule_name': 'CLIENT RULE - Frequent cross-border transactions',
                'rule_description': 'Clients who have made frequent transactions across the border in last 90 days',
                'rule_condition': 'cross_border_count > 5',
                'risk_weight': 0.6,
                'active': True
            },
            {
                'rule_id': 9,
                'rule_name': 'CLIENT RULE - Transactions to high-risk countries',
                'rule_description': 'Clients who have made transactions to high-risk countries',
                'rule_condition': 'cross_border_count > 5',
                'risk_weight': 0.9,
                'active': True
            },
            {
                'rule_id': 10,
                'rule_name': 'TRANSACTION RULE - Cross-Border to High-Risk Country',
                'rule_description': 'Transactions to high-risk countries from non-risky countries',
                'rule_condition': 'cross_border_to_high_risk == 1',
                'risk_weight': 0.9,
                'active': True
            },
            {
                'rule_id': 11,
                'rule_name': 'TRANSACTION RULE - Very large transaction by withdrawal',
                'rule_description': 'Withdrawal amount of more than 10,000 EUR',
                'rule_condition': '(transaction_amount > 10000) and (transaction_type == "Withdrawal")',
                'risk_weight': 0.9,
                'active': True
            }
        ]
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, 'compliance_rules.csv')
        df.to_csv(path, index=False)
        return df

    # --------------------------------------------------------
    # Run full pipeline
    # --------------------------------------------------------
    def run(self):
        print("=" * 70)
        print("Generating full KYC dataset...")
        print("=" * 70)

        docs = self.generate_documents()
        clients = self.generate_clients(docs)
        tx = self.generate_transactions(clients)
        countries = self.generate_country_risks()
        rules = self.generate_compliance_rules()

        print("\nAll datasets successfully generated inside:")
        print(f"{self.output_dir}")
        return {
            "documents": docs,
            "clients": clients,
            "transactions": tx,
            "countries": countries,
            "compliance_rules": rules
        }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    generator = KYCDataGenerator(num_clients=151)
    datasets = generator.run()
    for name, df in datasets.items():
        print("\n" + "="*50)
        print(f"{name.upper()} DATAFRAME - Top 5 rows")
        print("="*50)
        print(df.head())
