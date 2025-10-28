# ============================================================
# MULTI-AGENT SYSTEM FOR KYC PIPELINE
# ============================================================

import pandas as pd
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)

# ============================================================
# DATA MODELS
# ============================================================
class AgentRole(Enum):
    DATA_VALIDATOR = "data_validator"
    RISK_ANALYZER = "risk_analyzer"
    COMPLIANCE_CHECKER = "compliance_checker"
    INVESTIGATOR = "investigator"
    DECISION_MAKER = "decision_maker"

@dataclass
class Message:
    """Message passed between agents"""
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AgentDecision:
    """Decision made by an agent"""
    agent_id: str
    entity_id: str  # client_id or transaction_id
    decision: str  # "APPROVE", "REJECT", "REVIEW", "ESCALATE"
    confidence: float  # 0-1
    reasoning: str
    supporting_data: Dict[str, Any]

# ============================================================
# BASE AGENT CLASS
# ============================================================
class BaseAgent:
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.logger = logging.getLogger(f"Agent:{agent_id}")
        self.mailbox: List[Message] = []
        self.decisions: List[AgentDecision] = []
    
    def receive_message(self, message: Message):
        """Receive a message from another agent"""
        self.mailbox.append(message)
        self.logger.info(f"Received message from {message.sender}")
    
    def send_message(self, recipient: str, content: Dict[str, Any]) -> Message:
        """Create and return a message to send"""
        return Message(
            sender=self.agent_id,
            recipient=recipient,
            content=content
        )
    
    def make_decision(self, **kwargs) -> AgentDecision:
        """Override in subclasses"""
        raise NotImplementedError

# ============================================================
# SPECIALIZED AGENTS
# ============================================================

class DataValidatorAgent(BaseAgent):
    """Validates data quality and completeness"""
    
    def __init__(self, agent_id: str = "validator_1"):
        super().__init__(agent_id, AgentRole.DATA_VALIDATOR)
    
    def make_decision(self, client_data: Dict[str, Any]) -> AgentDecision:
        """Check data completeness and quality"""
        issues = []
        confidence = 1.0
        
        required_fields = ['client_id', 'name', 'country', 'kyc_status', 'email']
        for field in required_fields:
            if field not in client_data or pd.isna(client_data.get(field)):
                issues.append(f"Missing or null: {field}")
                confidence -= 0.2
        
        decision = "APPROVE" if not issues else "REVIEW"
        
        return AgentDecision(
            agent_id=self.agent_id,
            entity_id=client_data.get('client_id'),
            decision=decision,
            confidence=max(0, confidence),
            reasoning="; ".join(issues) if issues else "All required fields present",
            supporting_data={"fields_checked": required_fields}
        )

class RiskAnalyzerAgent(BaseAgent):
    """Analyzes risk scores and patterns"""
    
    def __init__(self, agent_id: str = "analyzer_1"):
        super().__init__(agent_id, AgentRole.RISK_ANALYZER)
    
    def make_decision(self, risk_data: Dict[str, Any]) -> AgentDecision:
        """Analyze risk scores"""
        ml_score = risk_data.get('ml_risk_score', 0)
        business_rule_score = risk_data.get('risk_score_from_business_rules', 0)
        combined_score = (ml_score + business_rule_score) / 2
        
        if combined_score > 70:
            decision = "ESCALATE"
            confidence = 0.95
        elif combined_score > 50:
            decision = "REVIEW"
            confidence = 0.80
        else:
            decision = "APPROVE"
            confidence = 0.85
        
        return AgentDecision(
            agent_id=self.agent_id,
            entity_id=risk_data.get('client_id'),
            decision=decision,
            confidence=confidence,
            reasoning=f"Combined risk score: {combined_score:.2f}",
            supporting_data={
                "ml_score": ml_score,
                "rule_score": business_rule_score,
                "combined": combined_score
            }
        )

class ComplianceCheckerAgent(BaseAgent):
    """Checks regulatory compliance"""
    
    def __init__(self, agent_id: str = "compliance_1"):
        super().__init__(agent_id, AgentRole.COMPLIANCE_CHECKER)
        self.restricted_countries = ['Nigeria', 'Russia']
    
    def make_decision(self, client_data: Dict[str, Any]) -> AgentDecision:
        """Check compliance requirements"""
        violations = []
        confidence = 0.9
        
        # Check country restrictions
        if client_data.get('country') in self.restricted_countries:
            violations.append(f"Client from restricted country: {client_data.get('country')}")
        
        # Check KYC status
        if client_data.get('kyc_status') != 'Verified':
            violations.append(f"KYC status not verified: {client_data.get('kyc_status')}")
        
        decision = "REJECT" if violations else "APPROVE"
        
        return AgentDecision(
            agent_id=self.agent_id,
            entity_id=client_data.get('client_id'),
            decision=decision,
            confidence=confidence,
            reasoning="; ".join(violations) if violations else "Compliant",
            supporting_data={"restricted_countries": self.restricted_countries}
        )

class InvestigatorAgent(BaseAgent):
    """Deep investigation of anomalies"""
    
    def __init__(self, agent_id: str = "investigator_1"):
        super().__init__(agent_id, AgentRole.INVESTIGATOR)
    
    def make_decision(self, investigation_data: Dict[str, Any]) -> AgentDecision:
        """Investigate suspicious patterns"""
        anomalies = []
        confidence = 0.75
        
        # Check for cross-border transactions to high-risk countries
        if investigation_data.get('cross_border_to_high_risk', 0) > 0:
            anomalies.append("Cross-border transactions to high-risk countries detected")
        
        # Check for failed transactions
        if investigation_data.get('failed_tx_last_30_days', 0) > 3:
            anomalies.append(f"Multiple failed transactions: {investigation_data.get('failed_tx_last_30_days')}")
        
        # Check transaction patterns
        large_tx_count = investigation_data.get('large_tx_last_7_days', 0)
        if large_tx_count > 5:
            anomalies.append(f"Unusual pattern: {large_tx_count} large transactions in 7 days")
        
        decision = "ESCALATE" if len(anomalies) >= 2 else "REVIEW" if anomalies else "APPROVE"
        
        return AgentDecision(
            agent_id=self.agent_id,
            entity_id=investigation_data.get('client_id'),
            decision=decision,
            confidence=confidence,
            reasoning="; ".join(anomalies) if anomalies else "No suspicious patterns",
            supporting_data={"anomalies_count": len(anomalies)}
        )

class DecisionMakerAgent(BaseAgent):
    """Synthesizes all agent recommendations"""
    
    def __init__(self, agent_id: str = "decision_maker_1"):
        super().__init__(agent_id, AgentRole.DECISION_MAKER)
        self.escalation_threshold = 2
    
    def synthesize_decisions(self, agent_decisions: List[AgentDecision]) -> Dict[str, Any]:
        """Combine all agent decisions into final verdict"""
        decision_counts = {}
        total_confidence = 0
        reasons = []
        
        for decision in agent_decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
            total_confidence += decision.confidence
            reasons.append(f"{decision.agent_id}: {decision.reasoning}")
        
        avg_confidence = total_confidence / len(agent_decisions) if agent_decisions else 0
        
        # Determine final decision with escalation logic
        escalate_votes = decision_counts.get("ESCALATE", 0)
        reject_votes = decision_counts.get("REJECT", 0)
        review_votes = decision_counts.get("REVIEW", 0)
        
        if reject_votes > 0:
            final_decision = "REJECT"
        elif escalate_votes >= self.escalation_threshold:
            final_decision = "ESCALATE"
        elif review_votes > 0:
            final_decision = "REVIEW"
        else:
            final_decision = "APPROVE"
        
        return {
            "final_decision": final_decision,
            "average_confidence": round(avg_confidence, 2),
            "agent_consensus": decision_counts,
            "reasoning": reasons,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================
# MULTI-AGENT ORCHESTRATOR
# ============================================================
class KYCMultiAgentOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger("Orchestrator")
        self.agents: Dict[str, BaseAgent] = {}
        self.final_verdicts: List[Dict[str, Any]] = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents"""
        self.agents = {
            'validator': DataValidatorAgent(),
            'risk_analyzer': RiskAnalyzerAgent(),
            'compliance': ComplianceCheckerAgent(),
            'investigator': InvestigatorAgent(),
            'decision_maker': DecisionMakerAgent()
        }
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def process_client(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a client through the multi-agent system"""
        self.logger.info(f"Processing client: {client_data.get('client_id')}")
        
        agent_decisions = []
        
        # Run all analyzers
        validator_decision = self.agents['validator'].make_decision(client_data)
        agent_decisions.append(validator_decision)
        
        risk_decision = self.agents['risk_analyzer'].make_decision(client_data)
        agent_decisions.append(risk_decision)
        
        compliance_decision = self.agents['compliance'].make_decision(client_data)
        agent_decisions.append(compliance_decision)
        
        # Investigator runs if risk or compliance issues detected
        if risk_decision.decision in ["REVIEW", "ESCALATE"] or compliance_decision.decision in ["REVIEW", "REJECT"]:
            investigator_decision = self.agents['investigator'].make_decision(client_data)
            agent_decisions.append(investigator_decision)
        
        # Decision maker synthesizes all decisions
        final_verdict = self.agents['decision_maker'].synthesize_decisions(agent_decisions)
        final_verdict['client_id'] = client_data.get('client_id')
        final_verdict['agent_decisions'] = [
            {
                'agent': d.agent_id,
                'decision': d.decision,
                'confidence': d.confidence,
                'reasoning': d.reasoning
            }
            for d in agent_decisions
        ]
        
        self.final_verdicts.append(final_verdict)
        self.logger.info(f"Final decision: {final_verdict['final_decision']} (confidence: {final_verdict['average_confidence']})")
        
        return final_verdict
    
    def process_batch(self, clients_df: pd.DataFrame) -> pd.DataFrame:
        """Process multiple clients"""
        self.logger.info(f"Processing batch of {len(clients_df)} clients")
        results = []
        
        for _, row in clients_df.iterrows():
            client_dict = row.to_dict()
            verdict = self.process_client(client_dict)
            results.append(verdict)
        
        return pd.DataFrame(results)

# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Create sample client data
    sample_clients = [
        {
            'client_id': 1,
            'name': 'Aryan Ashar',
            'country': 'USA',
            'kyc_status': 'Verified',
            'email': 'john@example.com',
            'ml_risk_score': 25.5,
            'risk_score_from_business_rules': 0.2,
            'cross_border_to_high_risk': 0,
            'failed_tx_last_30_days': 0,
            'large_tx_last_7_days': 0
        },
        {
            'client_id': 2,
            'name': 'Rohan Sarkar',
            'country': 'Nigeria',
            'kyc_status': 'Pending',
            'email': 'jane@example.com',
            'ml_risk_score': 78.5,
            'risk_score_from_business_rules': 2.5,
            'cross_border_to_high_risk': 3,
            'failed_tx_last_30_days': 2,
            'large_tx_last_7_days': 0
        }
    ]
    
    orchestrator = KYCMultiAgentOrchestrator()
    results_df = orchestrator.process_batch(pd.DataFrame(sample_clients))
    
    print("\n" + "="*80)
    print("MULTI-AGENT KYC PROCESSING RESULTS")
    print("="*80)
    for _, result in results_df.iterrows():
        print(f"\nClient {result['client_id']}:")
        print(f"  Final Decision: {result['final_decision']}")
        print(f"  Confidence: {result['average_confidence']}")
        print(f"  Agent Consensus: {result['agent_consensus']}")