# ============================================================
# KYC ENRICHMENT PIPELINE
# ============================================================

import pandas as pd
import os
import logging
from datetime import datetime

class KYCEnrichmentPipeline:
    def __init__(self, data_dir=None, output_dir=None, databricks=False):
        self.databricks = databricks

        # ============================================================
        # Determine directories automatically
        # ============================================================
        self.data_dir = data_dir or self.get_data_dir()
        self.output_dir = output_dir or self.get_output_dir()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"KYCEnrichmentPipeline initialized with data_dir={self.data_dir} and output_dir={self.output_dir}")

        # Placeholders for DataFrames
        self.clients = None
        self.transactions = None
        self.documents = None
        self.countries = None
        self.enriched_transactions = None
        self.enriched_clients = None
        self.client_aggregates = None

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================
    @staticmethod
    def get_base_dir():
        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()
        return base

    def get_data_dir(self):
        # read from data/raw
        return os.path.join(self.get_base_dir(), "..", "data", "raw data from data ingestion")
    

    def get_output_dir(self):
        """
        Creates a safe output folder for enriched data:
        - Databricks: /dbfs/FileStore/data/enriched
        - Local: ./data/enriched
        """
        base_dir = os.getcwd()  # project root

        if self.databricks:
            output_dir = "/dbfs/FileStore/data/enriched"
        else:
            output_dir = os.path.join(base_dir, "data", "joined and enriched data")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    # ============================================================
    # STEP 0: LOAD DATA
    # ============================================================
    def load_data(self):
        self.logger.info(f"Loading CSV data from directory: {self.data_dir}")
        self.clients = pd.read_csv(os.path.join(self.data_dir, "clients.csv"))
        self.transactions = pd.read_csv(os.path.join(self.data_dir, "transactions.csv"))
        self.documents = pd.read_csv(os.path.join(self.data_dir, "documents.csv"))
        self.countries = pd.read_csv(os.path.join(self.data_dir, "countries_risk_levels.csv"))
        self.logger.info("Data loaded successfully")

    # ============================================================
    # STEP 1: DATA TYPE CONVERSIONS
    # ============================================================
    def convert_data_types(self):
        self.logger.info("Converting data types...")
        self.clients['registration_date'] = pd.to_datetime(self.clients['registration_date'], errors='coerce')
        self.clients['date_of_birth'] = pd.to_datetime(self.clients['date_of_birth'], errors='coerce')
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'], errors='coerce')
        self.transactions['transaction_amount'] = pd.to_numeric(self.transactions['transaction_amount'], errors='coerce')
        self.documents['upload_date'] = pd.to_datetime(self.documents['upload_date'], errors='coerce')
        self.logger.info("Data types converted successfully")

    # ============================================================
    # STEP 2: ENRICH TRANSACTIONS
    # ============================================================
    def enrich_transactions_with_clients(self):
        self.logger.info("Joining transactions with client data...")
        self.enriched_transactions = self.transactions.merge(
            self.clients[['client_id', 'name', 'country', 'age', 'kyc_status', 'customer_type', 'registration_date']],
            on='client_id',
            how='left'
        )
        self.logger.info(f"Enriched transactions shape: {self.enriched_transactions.shape}")

    def add_country_risk_info(self):
        self.logger.info("Adding country risk information...")
        self.enriched_transactions = self.enriched_transactions.merge(
            self.countries.rename(columns={
                'country': 'origin_country',
                'risk_level': 'origin_risk_level',
                'aml_compliance_score': 'origin_aml_score'
            }),
            on='origin_country',
            how='left'
        )
        self.enriched_transactions = self.enriched_transactions.merge(
            self.countries.rename(columns={
                'country': 'destination_country',
                'risk_level': 'destination_risk_level',
                'aml_compliance_score': 'destination_aml_score'
            }),
            on='destination_country',
            how='left'
        )
        self.logger.info("Country risk info added")

    def add_document_verification(self):
        self.logger.info("Adding document verification status...")
        self.enriched_transactions = self.enriched_transactions.merge(
            self.documents[['client_id', 'document_type', 'verification_status', 'upload_date']].rename(
                columns={'verification_status': 'doc_verification_status', 'upload_date': 'doc_upload_date'}
            ),
            on='client_id',
            how='left'
        )
        self.logger.info("Document information added")

    def calculate_derived_features(self):
        self.logger.info("Calculating derived features...")
        current_date = pd.Timestamp.now().normalize()

        self.enriched_transactions['days_since_transaction'] = (
            current_date - self.enriched_transactions['transaction_date']
        ).dt.days

        self.enriched_transactions['days_since_registration'] = (
            current_date - self.enriched_transactions['registration_date']
        ).dt.days

        self.enriched_transactions['is_cross_border'] = (
            self.enriched_transactions['origin_country'] != self.enriched_transactions['destination_country']
        ).astype(int)

        self.enriched_transactions['cross_border_to_high_risk'] = (
            (self.enriched_transactions['is_cross_border'] == 1) &
            (self.enriched_transactions['destination_risk_level'] == 'High')
        ).astype(int)
        self.logger.info("Derived features calculated")

    # ============================================================
    # STEP 3: CLIENT AGGREGATES
    # ============================================================
    def create_client_aggregates(self):
        self.logger.info("Creating client-level aggregates...")
        self.client_aggregates = (
            self.enriched_transactions.groupby('client_id')
            .agg(
                total_transactions=('transaction_id', 'count'),
                total_amount=('transaction_amount', 'sum'),
                avg_amount=('transaction_amount', 'mean'),
                max_amount=('transaction_amount', 'max'),
                min_amount=('transaction_amount', 'min'),
                amount_std=('transaction_amount', 'std'),
                cross_border_count=('is_cross_border', 'sum'),
                high_risk_cross_border_count=('cross_border_to_high_risk', 'sum'),
                failed_transaction_count=('status', lambda x: (x == 'Failed').sum())
            )
            .round(2)
            .reset_index()
        )
        self.logger.info(f"Client aggregates created: {len(self.client_aggregates)} records")

    # ============================================================
    # STEP 4: ENRICH CLIENT DATA
    # ============================================================
    def enrich_clients(self):
        self.logger.info("Enriching client master data...")
        self.enriched_clients = self.clients.merge(self.client_aggregates, on='client_id', how='left')

        # Add country risk info
        self.enriched_clients = self.enriched_clients.merge(
            self.countries.rename(columns={
                'country': 'country_orig',
                'risk_level': 'client_country_risk_level',
                'aml_compliance_score': 'client_country_aml_score'
            }),
            left_on='country',
            right_on='country_orig',
            how='left'
        )

        # Add document verification
        self.enriched_clients = self.enriched_clients.merge(
            self.documents[['client_id', 'verification_status']].rename(columns={'verification_status': 'doc_verification_status'}),
            on='client_id',
            how='left'
        )

        # Fill missing aggregates
        fill_defaults = {
            'total_transactions': 0,
            'total_amount': 0,
            'avg_amount': 0,
            'max_amount': 0,
            'min_amount': 0,
            'amount_std': 0,
            'cross_border_count': 0,
            'high_risk_cross_border_count': 0,
            'failed_transaction_count': 0
        }
        self.enriched_clients = self.enriched_clients.fillna(fill_defaults)
        self.logger.info(f"Enriched clients shape: {self.enriched_clients.shape}")

    # ============================================================
    # STEP 5: RECENT TRANSACTION FEATURES
    # ============================================================
    def add_recent_transaction_features(self):
        self.logger.info("Adding recent transaction features...")
        current_date = pd.Timestamp.now().normalize()
        recent_transactions = self.enriched_transactions.copy()
        recent_transactions['days_since_tx'] = (current_date - recent_transactions['transaction_date']).dt.days

        tx_last_7 = recent_transactions[recent_transactions['days_since_tx'] <= 7] \
                    .groupby('client_id').size().rename('tx_last_7').reset_index()
        tx_last_30 = recent_transactions[recent_transactions['days_since_tx'] <= 30] \
                    .groupby('client_id').size().rename('tx_last_30').reset_index()
        large_tx_7 = recent_transactions[(recent_transactions['transaction_amount'] > 10000) &
                                         (recent_transactions['days_since_tx'] <= 7)] \
                    .groupby('client_id').size().rename('large_tx_last_7_days').reset_index()

        self.enriched_clients = self.enriched_clients.merge(tx_last_7, on='client_id', how='left')
        self.enriched_clients = self.enriched_clients.merge(tx_last_30, on='client_id', how='left')
        self.enriched_clients = self.enriched_clients.merge(large_tx_7, on='client_id', how='left')

        self.enriched_clients[['tx_last_7', 'tx_last_30', 'large_tx_last_7_days']] = \
            self.enriched_clients[['tx_last_7', 'tx_last_30', 'large_tx_last_7_days']].fillna(0).astype(int)

        self.logger.info("Recent transaction features added")

    # ============================================================
    # STEP 6: EXPORT DATA
    # ============================================================
    def export_data(self):
        self.logger.info(f"Exporting enriched data to directory: {self.output_dir}")

        self.enriched_transactions.to_csv(os.path.join(self.output_dir, "enriched_transactions.csv"), index=False)
        self.enriched_clients.to_csv(os.path.join(self.output_dir, "enriched_clients.csv"), index=False)
        self.client_aggregates.to_csv(os.path.join(self.output_dir, "client_aggregates.csv"), index=False)

        self.logger.info(f"Enriched data exported successfully to {self.output_dir}")

    # ============================================================
    # RUN FULL PIPELINE
    # ============================================================
    def run_pipeline(self):
        self.load_data()
        self.convert_data_types()
        self.enrich_transactions_with_clients()
        self.add_country_risk_info()
        self.add_document_verification()
        self.calculate_derived_features()
        self.create_client_aggregates()
        self.enrich_clients()
        self.add_recent_transaction_features()
        self.export_data()
        self.logger.info("KYC Enrichment Pipeline completed successfully")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    pipeline = KYCEnrichmentPipeline()  # Remove databricks=True if running locally
    pipeline.run_pipeline()
