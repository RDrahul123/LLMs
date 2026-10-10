"""
Module 12: Responsible AI — Best Practices and Checklists

This module provides comprehensive responsible AI guidelines:
- Pre-development checklist
- Data collection guidelines
- Model selection considerations
- Deployment safety checklist
- Monitoring and incident response plan
- Transparency and documentation requirements
- User consent and control
- Environmental impact considerations
- Responsible AI Playbook
"""

import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field


# =============================================================================
# SECTION 1: Checklists
# =============================================================================

@dataclass
class ChecklistItem:
    """A single checklist item."""
    id: str
    description: str
    category: str
    priority: str  # critical, high, medium, low
    completed: bool = False
    notes: str = ""
    owner: str = ""
    due_date: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "completed": self.completed,
            "notes": self.notes,
            "owner": self.owner,
            "due_date": self.due_date,
        }


class ResponsibleAIChecklists:
    """Comprehensive checklists for responsible AI development."""

    @staticmethod
    def pre_development() -> list[ChecklistItem]:
        """Pre-development checklist — before starting an AI project."""
        return [
            # Business justification
            ChecklistItem(
                id="PRE-001",
                description="Define clear business objective and expected value",
                category="planning",
                priority="critical",
            ),
            ChecklistItem(
                id="PRE-002",
                description="Identify all stakeholders (users, affected parties, regulators)",
                category="planning",
                priority="critical",
            ),
            ChecklistItem(
                id="PRE-003",
                description="Conduct impact assessment: who benefits, who might be harmed?",
                category="planning",
                priority="critical",
            ),
            ChecklistItem(
                id="PRE-004",
                description="Determine if AI is the right solution (vs. rule-based, simpler approaches)",
                category="planning",
                priority="high",
            ),
            ChecklistItem(
                id="PRE-005",
                description="Classify risk level per EU AI Act or NIST AI RMF",
                category="compliance",
                priority="critical",
            ),

            # Ethics review
            ChecklistItem(
                id="PRE-006",
                description="Conduct ethical review of the project",
                category="ethics",
                priority="critical",
            ),
            ChecklistItem(
                id="PRE-007",
                description="Identify potential biases in the problem framing",
                category="ethics",
                priority="high",
            ),
            ChecklistItem(
                id="PRE-008",
                description="Assess potential for misuse or dual-use applications",
                category="ethics",
                priority="high",
            ),
            ChecklistItem(
                id="PRE-009",
                description="Define acceptable use policy",
                category="ethics",
                priority="high",
            ),

            # Technical planning
            ChecklistItem(
                id="PRE-010",
                description="Define evaluation metrics including fairness metrics",
                category="technical",
                priority="high",
            ),
            ChecklistItem(
                id="PRE-011",
                description="Plan for safety testing and red-teaming",
                category="technical",
                priority="high",
            ),
            ChecklistItem(
                id="PRE-012",
                description="Design monitoring and logging infrastructure",
                category="technical",
                priority="high",
            ),
            ChecklistItem(
                id="PRE-013",
                description="Plan incident response procedures",
                category="technical",
                priority="high",
            ),

            # Legal and compliance
            ChecklistItem(
                id="PRE-014",
                description="Verify data licensing and usage rights",
                category="legal",
                priority="critical",
            ),
            ChecklistItem(
                id="PRE-015",
                description="Check GDPR/CCPA compliance for data processing",
                category="legal",
                priority="critical",
            ),
            ChecklistItem(
                id="PRE-016",
                description="Review intellectual property considerations",
                category="legal",
                priority="high",
            ),
            ChecklistItem(
                id="PRE-017",
                description="Ensure regulatory compliance (EU AI Act, NIST, sector-specific)",
                category="legal",
                priority="critical",
            ),
        ]

    @staticmethod
    def data_collection() -> list[ChecklistItem]:
        """Data collection and preparation guidelines."""
        return [
            ChecklistItem(
                id="DATA-001",
                description="Document data sources and collection methods",
                category="documentation",
                priority="critical",
            ),
            ChecklistItem(
                id="DATA-002",
                description="Verify consent for all data sources",
                category="legal",
                priority="critical",
            ),
            ChecklistItem(
                id="DATA-003",
                description="Assess representation across demographic groups",
                category="fairness",
                priority="high",
            ),
            ChecklistItem(
                id="DATA-004",
                description="Remove or properly handle PII",
                category="privacy",
                priority="critical",
            ),
            ChecklistItem(
                id="DATA-005",
                description="Document known biases in training data",
                category="fairness",
                priority="high",
            ),
            ChecklistItem(
                id="DATA-006",
                description="Establish data quality criteria and validation process",
                category="quality",
                priority="high",
            ),
            ChecklistItem(
                id="DATA-007",
                description="Create data card documenting dataset characteristics",
                category="documentation",
                priority="high",
            ),
            ChecklistItem(
                id="DATA-008",
                description="Implement data versioning and lineage tracking",
                category="technical",
                priority="medium",
            ),
            ChecklistItem(
                id="DATA-009",
                description="Plan for data updates and maintenance",
                category="operational",
                priority="medium",
            ),
            ChecklistItem(
                id="DATA-010",
                description="Ensure data minimization (collect only what's needed)",
                category="privacy",
                priority="high",
            ),
        ]

    @staticmethod
    def model_selection() -> list[ChecklistItem]:
        """Model selection considerations."""
        return [
            ChecklistItem(
                id="MODEL-001",
                description="Evaluate multiple model architectures",
                category="technical",
                priority="high",
            ),
            ChecklistItem(
                id="MODEL-002",
                description="Assess model size vs. performance tradeoffs",
                category="technical",
                priority="high",
            ),
            ChecklistItem(
                id="MODEL-003",
                description="Evaluate fairness across demographic groups",
                category="fairness",
                priority="critical",
            ),
            ChecklistItem(
                id="MODEL-004",
                description="Test for known failure modes and edge cases",
                category="safety",
                priority="high",
            ),
            ChecklistItem(
                id="MODEL-005",
                description="Evaluate computational cost and environmental impact",
                category="environmental",
                priority="medium",
            ),
            ChecklistItem(
                id="MODEL-006",
                description="Check model license compatibility",
                category="legal",
                priority="high",
            ),
            ChecklistItem(
                id="MODEL-007",
                description="Assess explainability and interpretability needs",
                category="transparency",
                priority="medium",
            ),
            ChecklistItem(
                id="MODEL-008",
                description="Evaluate robustness to adversarial inputs",
                category="safety",
                priority="high",
            ),
            ChecklistItem(
                id="MODEL-009",
                description="Document model card with limitations and appropriate use",
                category="documentation",
                priority="high",
            ),
            ChecklistItem(
                id="MODEL-010",
                description="Verify alignment with intended use case",
                category="planning",
                priority="high",
            ),
        ]

    @staticmethod
    def deployment() -> list[ChecklistItem]:
        """Deployment safety checklist."""
        return [
            ChecklistItem(
                id="DEPLOY-001",
                description="Complete all pre-development checklist items",
                category="compliance",
                priority="critical",
            ),
            ChecklistItem(
                id="DEPLOY-002",
                description="Run full safety test suite including adversarial tests",
                category="safety",
                priority="critical",
            ),
            ChecklistItem(
                id="DEPLOY-003",
                description="Validate input and output filtering systems",
                category="safety",
                priority="critical",
            ),
            ChecklistItem(
                id="DEPLOY-004",
                description="Test rate limiting and abuse prevention",
                category="safety",
                priority="high",
            ),
            ChecklistItem(
                id="DEPLOY-005",
                description="Verify logging and monitoring are active",
                category="operational",
                priority="critical",
            ),
            ChecklistItem(
                id="DEPLOY-006",
                description="Test rollback procedures",
                category="operational",
                priority="high",
            ),
            ChecklistItem(
                id="DEPLOY-007",
                description="Prepare user-facing documentation and disclaimers",
                category="transparency",
                priority="high",
            ),
            ChecklistItem(
                id="DEPLOY-008",
                description="Implement human-in-the-loop for high-stakes decisions",
                category="safety",
                priority="critical",
            ),
            ChecklistItem(
                id="DEPLOY-009",
                description="Set up alerting for anomalous behavior",
                category="operational",
                priority="high",
            ),
            ChecklistItem(
                id="DEPLOY-010",
                description="Conduct final ethics review before launch",
                category="ethics",
                priority="critical",
            ),
            ChecklistItem(
                id="DEPLOY-011",
                description="Establish feedback channels for users",
                category="operational",
                priority="medium",
            ),
            ChecklistItem(
                id="DEPLOY-012",
                description="Schedule regular bias audits and safety reviews",
                category="operational",
                priority="high",
            ),
        ]

    @staticmethod
    def monitoring() -> list[ChecklistItem]:
        """Post-deployment monitoring checklist."""
        return [
            ChecklistItem(
                id="MON-001",
                description="Monitor for bias drift over time",
                category="fairness",
                priority="high",
            ),
            ChecklistItem(
                id="MON-002",
                description="Track model performance metrics",
                category="technical",
                priority="high",
            ),
            ChecklistItem(
                id="MON-003",
                description="Monitor for adversarial attacks and abuse",
                category="safety",
                priority="high",
            ),
            ChecklistItem(
                id="MON-004",
                description="Collect and analyze user feedback",
                category="operational",
                priority="medium",
            ),
            ChecklistItem(
                id="MON-005",
                description="Conduct periodic safety audits",
                category="safety",
                priority="high",
            ),
            ChecklistItem(
                id="MON-006",
                description="Review and update content filters",
                category="safety",
                priority="medium",
            ),
            ChecklistItem(
                id="MON-007",
                description="Track incident reports and root causes",
                category="operational",
                priority="high",
            ),
            ChecklistItem(
                id="MON-008",
                description="Assess environmental impact of model usage",
                category="environmental",
                priority="low",
            ),
        ]


# =============================================================================
# SECTION 2: Incident Response Plan
# =============================================================================

@dataclass
class Incident:
    """Represents a safety incident."""
    id: str
    title: str
    severity: str  # critical, high, medium, low
    category: str
    description: str
    detected_at: str = ""
    resolved_at: str = ""
    status: str = "open"  # open, investigating, resolved, closed
    impact: str = ""
    root_cause: str = ""
    remediation: str = ""
    lessons_learned: str = ""


class IncidentResponsePlan:
    """Incident response procedures for AI systems."""

    def __init__(self):
        self.incidents: list[Incident] = []
        self.procedures = self._define_procedures()

    def _define_procedures(self) -> dict:
        """Define incident response procedures."""
        return {
            "detection": {
                "description": "How to detect incidents",
                "methods": [
                    "Automated monitoring alerts (toxicity, bias, performance drift)",
                    "User reports and feedback",
                    "Red team findings",
                    "External vulnerability disclosures",
                    "Regulatory inquiries",
                ],
                "response_time": {
                    "critical": "Immediate (< 1 hour)",
                    "high": "< 4 hours",
                    "medium": "< 24 hours",
                    "low": "< 1 week",
                },
            },
            "containment": {
                "description": "Steps to contain the incident",
                "steps": [
                    "Disable affected features or endpoints",
                    "Activate rate limiting",
                    "Roll back to last known good model version",
                    "Isolate affected systems",
                    "Preserve evidence and logs",
                ],
            },
            "investigation": {
                "description": "Root cause analysis",
                "steps": [
                    "Collect and analyze logs",
                    "Identify the attack vector or failure mode",
                    "Determine scope and impact",
                    "Assess data exposure if applicable",
                    "Document findings",
                ],
            },
            "remediation": {
                "description": "Fix and prevent recurrence",
                "steps": [
                    "Implement fixes (model update, filter update, code fix)",
                    "Test fixes thoroughly",
                    "Update safety measures",
                    "Patch vulnerabilities",
                    "Update documentation",
                ],
            },
            "communication": {
                "description": "Stakeholder communication",
                "steps": [
                    "Notify affected users",
                    "Report to regulators if required (GDPR 72-hour rule)",
                    "Prepare public disclosure if necessary",
                    "Update status page",
                    "Provide remediation guidance to users",
                ],
            },
            "post_mortem": {
                "description": "Learn and improve",
                "steps": [
                    "Conduct blameless post-mortem",
                    "Document lessons learned",
                    "Update incident response procedures",
                    "Implement preventive measures",
                    "Share learnings with team",
                ],
            },
        }

    def report_incident(
        self,
        title: str,
        severity: str,
        category: str,
        description: str,
    ) -> Incident:
        """Report a new incident."""
        incident_id = f"INC-{len(self.incidents) + 1:04d}"
        incident = Incident(
            id=incident_id,
            title=title,
            severity=severity,
            category=category,
            description=description,
            detected_at=datetime.now().isoformat(),
        )
        self.incidents.append(incident)
        return incident

    def get_procedure_for_severity(self, severity: str) -> dict:
        """Get the appropriate procedure based on severity."""
        procedure = {}
        for phase, details in self.procedures.items():
            procedure[phase] = details
            if "response_time" in details:
                procedure[phase]["response_time"] = details["response_time"].get(
                    severity, "Unknown"
                )
        return procedure

    def generate_incident_report(self, incident: Incident) -> str:
        """Generate an incident report."""
        report = [
            f"# Incident Report: {incident.id}",
            "",
            f"**Title:** {incident.title}",
            f"**Severity:** {incident.severity}",
            f"**Category:** {incident.category}",
            f"**Status:** {incident.status}",
            f"**Detected:** {incident.detected_at}",
            f"**Resolved:** {incident.resolved_at or 'Pending'}",
            "",
            "## Description",
            incident.description,
            "",
            "## Impact",
            incident.impact or "Assessment pending",
            "",
            "## Root Cause",
            incident.root_cause or "Investigation ongoing",
            "",
            "## Remediation",
            incident.remediation or "Planned",
            "",
            "## Lessons Learned",
            incident.lessons_learned or "To be determined",
        ]
        return "\n".join(report)


# =============================================================================
# SECTION 3: Transparency Documentation
# =============================================================================

class TransparencyDocumentation:
    """Templates and guidelines for AI transparency documentation."""

    @staticmethod
    def model_card(
        model_name: str,
        description: str,
        intended_use: str,
        limitations: list[str],
        training_data: str,
        evaluation_results: dict,
        ethical_considerations: list[str],
    ) -> str:
        """Generate a model card."""
        card = [
            f"# Model Card: {model_name}",
            "",
            "## Model Description",
            description,
            "",
            "## Intended Use",
            intended_use,
            "",
            "## Limitations",
        ]
        for lim in limitations:
            card.append(f"- {lim}")

        card.extend([
            "",
            "## Training Data",
            training_data,
            "",
            "## Evaluation Results",
        ])
        for metric, value in evaluation_results.items():
            card.append(f"- **{metric}:** {value}")

        card.extend([
            "",
            "## Ethical Considerations",
        ])
        for consideration in ethical_considerations:
            card.append(f"- {consideration}")

        return "\n".join(card)

    @staticmethod
    def data_card(
        dataset_name: str,
        description: str,
        sources: list[str],
        size: str,
        demographic_breakdown: dict,
        known_biases: list[str],
        licensing: str,
    ) -> str:
        """Generate a data card."""
        card = [
            f"# Data Card: {dataset_name}",
            "",
            "## Description",
            description,
            "",
            "## Sources",
        ]
        for source in sources:
            card.append(f"- {source}")

        card.extend([
            "",
            f"## Size: {size}",
            "",
            "## Demographic Breakdown",
        ])
        for group, count in demographic_breakdown.items():
            card.append(f"- **{group}:** {count}")

        card.extend([
            "",
            "## Known Biases",
        ])
        for bias in known_biases:
            card.append(f"- {bias}")

        card.extend([
            "",
            f"## Licensing: {licensing}",
        ])

        return "\n".join(card)

    @staticmethod
    def system_card(
        system_name: str,
        components: list[str],
        safety_measures: list[str],
        monitoring: list[str],
        user_controls: list[str],
    ) -> str:
        """Generate a system card for an AI system."""
        card = [
            f"# System Card: {system_name}",
            "",
            "## Components",
        ]
        for comp in components:
            card.append(f"- {comp}")

        card.extend([
            "",
            "## Safety Measures",
        ])
        for measure in safety_measures:
            card.append(f"- {measure}")

        card.extend([
            "",
            "## Monitoring",
        ])
        for mon in monitoring:
            card.append(f"- {mon}")

        card.extend([
            "",
            "## User Controls",
        ])
        for ctrl in user_controls:
            card.append(f"- {ctrl}")

        return "\n".join(card)


# =============================================================================
# SECTION 4: User Consent and Control
# =============================================================================

class UserConsentFramework:
    """Framework for user consent and control in AI systems."""

    def __init__(self):
        self.consent_policies: list[dict] = []

    def add_policy(self, policy: dict):
        """Add a consent policy."""
        self.consent_policies.append(policy)

    def generate_consent_form(self) -> str:
        """Generate a user consent form."""
        form = [
            "# AI System — User Consent Form",
            "",
            "Before using this AI system, please review and consent to the following:",
            "",
        ]

        for i, policy in enumerate(self.consent_policies, 1):
            form.extend([
                f"## {i}. {policy.get('title', 'Untitled')}",
                policy.get("description", ""),
                "",
                f"**Data collected:** {policy.get('data_collected', 'None specified')}",
                f"**Purpose:** {policy.get('purpose', 'Not specified')}",
                f"**Retention:** {policy.get('retention', 'Not specified')}",
                "",
                f"[ ] I consent to {policy.get('title', 'this policy').lower()}",
                "",
            ])

        form.extend([
            "---",
            "",
            "By using this system, you confirm that you have read and understood these policies.",
            "",
            "**Date:** ________________",
            "**Signature:** ________________",
        ])

        return "\n".join(form)

    def generate_privacy_policy_summary(self) -> str:
        """Generate a privacy policy summary."""
        return """
# Privacy Policy Summary for AI System

## What We Collect
- User inputs (prompts, queries)
- System outputs (responses)
- Usage metadata (timestamps, session IDs)
- Feedback and ratings

## How We Use It
- Improve model performance
- Detect and prevent abuse
- Generate aggregate analytics
- Ensure safety and compliance

## Your Rights
- Request deletion of your data
- Opt out of data collection for model improvement
- Access your data
- Request correction of inaccurate data

## Data Retention
- Conversations: 30 days (configurable)
- Feedback: Indefinitely (anonymized)
- Logs: 90 days

## Contact
For privacy inquiries: privacy@example.com
"""
    def generate_user_controls_spec(self) -> str:
        """Generate user controls specification."""
        return """
# User Controls Specification

## Data Controls
- **Opt-out of training:** Users can opt out of having their conversations used for model training
- **Data deletion:** Users can request deletion of their conversation history
- **Export data:** Users can export their conversation history

## Content Controls
- **Content filtering:** Users can adjust content filter sensitivity (if applicable)
- **Age-appropriate content:** Automatic filtering for child users
- **Custom word filters:** Users can add words/phrases to block

## Interaction Controls
- **Human escalation:** Users can request human assistance at any time
- **Feedback mechanism:** Users can report issues or provide feedback
- **Session management:** Users can end sessions and clear context

## Transparency Controls
- **Explanation requests:** Users can request explanations for model outputs
- **Confidence scores:** Users can view model confidence levels
- **Source citations:** When applicable, users can view sources used

## Notification Controls
- **System updates:** Users can control notification preferences
- **Policy changes:** Users are notified of material policy changes
- **Incident disclosures:** Users are notified of incidents affecting them
"""


# =============================================================================
# SECTION 5: Environmental Impact
# =============================================================================

class EnvironmentalImpact:
    """Assess and manage environmental impact of AI systems."""

    # Approximate carbon intensity by energy source (kg CO2 per kWh)
    CARBON_INTENSITY = {
        "coal": 0.91,
        "natural_gas": 0.45,
        "solar": 0.05,
        "wind": 0.01,
        "nuclear": 0.01,
        "hydro": 0.01,
        "grid_average_us": 0.42,
        "grid_average_eu": 0.30,
        "grid_average_global": 0.47,
    }

    @classmethod
    def estimate_training_emissions(
        cls,
        gpu_hours: float,
        gpu_power_kw: float = 0.3,  # kW per GPU
        energy_source: str = "grid_average_us",
    ) -> dict:
        """Estimate carbon emissions from model training."""
        energy_kwh = gpu_hours * gpu_power_kw
        carbon_intensity = cls.CARBON_INTENSITY.get(energy_source, 0.47)
        carbon_kg = energy_kwh * carbon_intensity

        # Equivalents
        car_km = carbon_kg / 0.21  # Average car: 0.21 kg CO2/km
        tree_years = carbon_kg / 0.022  # One tree absorbs ~22 kg CO2/year

        return {
            "gpu_hours": gpu_hours,
            "energy_kwh": round(energy_kwh, 2),
            "carbon_kg": round(carbon_kg, 2),
            "carbon_tonnes": round(carbon_kg / 1000, 4),
            "equivalents": {
                "car_km_driven": round(car_km, 1),
                "tree_years_needed": round(tree_years, 1),
                "smartphone_charges": round(energy_kwh * 1000 / 0.012, 0),  # 12Wh per charge
            },
            "energy_source": energy_source,
        }

    @classmethod
    def estimate_inference_emissions(
        cls,
        requests_per_day: int,
        avg_tokens_per_request: int,
        model_size_b: float,  # Billions of parameters
        energy_source: str = "grid_average_us",
        days: int = 365,
    ) -> dict:
        """Estimate carbon emissions from model inference."""
        # Rough estimate: ~0.0001 kWh per 1000 tokens for a 7B model
        energy_per_token = 0.0001 * (model_size_b / 7.0) / 1000
        total_tokens = requests_per_day * avg_tokens_per_request * days
        energy_kwh = total_tokens * energy_per_token

        carbon_intensity = cls.CARBON_INTENSITY.get(energy_source, 0.47)
        carbon_kg = energy_kwh * carbon_intensity

        return {
            "requests_per_day": requests_per_day,
            "avg_tokens_per_request": avg_tokens_per_request,
            "total_requests": requests_per_day * days,
            "energy_kwh": round(energy_kwh, 2),
            "carbon_kg": round(carbon_kg, 2),
            "model_size_b": model_size_b,
            "period_days": days,
        }

    @staticmethod
    def sustainability_recommendations() -> list[str]:
        """Provide sustainability recommendations."""
        return [
            "Use smaller, distilled models when possible",
            "Train during periods of low-carbon grid energy",
            "Use renewable energy sources for training and inference",
            "Optimize inference with quantization and batching",
            "Monitor and report energy consumption",
            "Consider carbon offsets for unavoidable emissions",
            "Use efficient architectures (e.g., mixture of experts)",
            "Cache frequent queries to reduce redundant computation",
            "Share model infrastructure across teams",
            "Choose cloud providers with strong sustainability commitments",
        ]


# =============================================================================
# SECTION 6: Responsible AI Playbook
# =============================================================================

class ResponsibleAIPlaybook:
    """Comprehensive Responsible AI Playbook as a structured document."""

    @staticmethod
    def generate() -> str:
        """Generate the complete Responsible AI Playbook."""
        return """
# Responsible AI Playbook

## 1. Introduction

### Purpose
This playbook provides actionable guidelines for developing, deploying, and
maintaining AI systems responsibly. It applies to all AI projects regardless
of scale or risk level.

### Principles
1. **Human-Centric:** AI should augment human capabilities, not replace human judgment in high-stakes decisions.
2. **Fair and Equitable:** AI systems should treat all people fairly and avoid reinforcing biases.
3. **Transparent:** AI systems should be explainable and their limitations understood.
4. **Safe and Secure:** AI systems should operate safely and be resilient to misuse.
5. **Accountable:** Clear ownership and accountability for AI system behavior.
6. **Privacy-Respecting:** Data collection and use should respect user privacy.

---

## 2. Development Phase

### 2.1 Problem Definition
- [ ] Clearly define the problem the AI is solving
- [ ] Verify that AI is the appropriate solution
- [ ] Identify all stakeholders and affected parties
- [ ] Conduct preliminary impact assessment

### 2.2 Data Collection
- [ ] Document all data sources
- [ ] Obtain proper consent and licensing
- [ ] Assess demographic representation
- [ ] Identify and document known biases
- [ ] Implement data minimization

### 2.3 Model Development
- [ ] Select appropriate model architecture
- [ ] Implement fairness metrics alongside accuracy metrics
- [ ] Conduct bias testing across demographic groups
- [ ] Document model limitations

### 2.4 Safety Testing
- [ ] Red-team testing with adversarial prompts
- [ ] Test edge cases and failure modes
- [ ] Validate content filtering systems
- [ ] Test incident response procedures

---

## 3. Deployment Phase

### 3.1 Pre-Launch
- [ ] Complete all pre-development checklist items
- [ ] Obtain ethics review approval
- [ ] Verify monitoring and logging infrastructure
- [ ] Test rollback procedures
- [ ] Prepare user documentation

### 3.2 Launch
- [ ] Deploy with staged rollout
- [ ] Activate monitoring and alerting
- [ ] Enable user feedback channels
- [ ] Monitor for anomalies

### 3.3 Post-Launch
- [ ] Regular bias audits (quarterly minimum)
- [ ] Performance monitoring
- [ ] User feedback analysis
- [ ] Safety incident tracking

---

## 4. Monitoring and Maintenance

### 4.1 Continuous Monitoring
- Model performance metrics (accuracy, latency, throughput)
- Fairness metrics across demographic groups
- Safety metrics (toxicity rate, injection attempts blocked)
- User satisfaction and feedback

### 4.2 Regular Reviews
- Monthly: Review monitoring dashboards
- Quarterly: Bias audit and safety review
- Annually: Comprehensive ethics review
- As needed: Incident response and post-mortem

### 4.3 Model Updates
- Document reasons for updates
- Re-run safety tests before deploying updates
- Maintain version history
- Communicate changes to users

---

## 5. Incident Response

### 5.1 Severity Levels
- **Critical:** Safety risk, data breach, significant bias
- **High:** Performance degradation, moderate bias
- **Medium:** Minor issues, user complaints
- **Low:** Cosmetic issues, minor bugs

### 5.2 Response Timeline
- Critical: Immediate response (< 1 hour)
- High: < 4 hours
- Medium: < 24 hours
- Low: < 1 week

### 5.3 Communication
- Internal: Notify team immediately
- External: Notify affected users within 72 hours (GDPR requirement)
- Regulatory: Report as required by applicable regulations

---

## 6. Documentation Requirements

### 6.1 Required Documents
- Model card (limitations, intended use, evaluation results)
- Data card (sources, biases, demographic breakdown)
- System card (components, safety measures, monitoring)
- User consent forms
- Privacy policy
- Acceptable use policy
- Incident response plan

### 6.2 Documentation Standards
- Write for the intended audience (technical vs. non-technical)
- Update documentation with each model version
- Make documentation publicly available where appropriate
- Include both capabilities and limitations

---

## 7. Governance

### 7.1 Roles and Responsibilities
- **AI Ethics Board:** Approve high-risk projects, review incidents
- **Project Lead:** Ensure compliance with this playbook
- **Technical Lead:** Implement safety measures and testing
- **Legal/Compliance:** Ensure regulatory compliance

### 7.2 Decision Framework
1. Assess risk level
2. Apply appropriate controls
3. Document decisions and rationale
4. Review and update regularly

### 7.3 Training
- All team members: Responsible AI basics
- Developers: Safety testing and bias detection
- Managers: Ethics review and governance

---

## 8. Regulatory Compliance

### 8.1 EU AI Act
- Classify your AI system by risk level
- Implement required safeguards for high-risk systems
- Prepare conformity assessment documentation
- Establish human oversight mechanisms

### 8.2 NIST AI RMF
- Govern: Establish risk management processes
- Map: Identify and categorize AI risks
- Measure: Assess and quantify risks
- Manage: Respond to and mitigate risks

### 8.3 Other Regulations
- GDPR (data protection)
- CCPA (California privacy)
- Sector-specific regulations (healthcare, finance, etc.)

---

## 9. Environmental Responsibility

### 9.1 Measure Impact
- Track energy consumption for training and inference
- Estimate carbon emissions
- Report environmental metrics

### 9.2 Reduce Impact
- Use efficient model architectures
- Optimize inference with quantization
- Choose renewable energy sources
- Share infrastructure across projects

---

## 10. Continuous Improvement

### 10.1 Feedback Loops
- User feedback → Model improvement
- Incident analysis → Safety improvements
- Bias audits → Fairness improvements
- Research → New best practices

### 10.2 Stay Current
- Monitor AI safety research
- Participate in industry working groups
- Update practices as regulations evolve
- Share learnings with the community

---

*This playbook is a living document. Review and update regularly.*
*Version: 1.0 | Last Updated: 2026*
"""


# =============================================================================
# SECTION 7: Main Demonstration
# =============================================================================

def run_all_demos():
    """Run all responsible AI demonstrations."""
    print("Module 12: Responsible AI — Best Practices and Checklists\n")

    # --- Demo 1: Checklists ---
    print("=" * 70)
    print("DEMO 1: Pre-Development Checklist")
    print("=" * 70)

    checklists = ResponsibleAIChecklists()
    pre_dev = checklists.pre_development()
    print(f"\n  Items: {len(pre_dev)}")
    print(f"\n  Critical items:")
    for item in pre_dev:
        if item.priority == "critical":
            print(f"    [{item.id}] {item.description}")

    data_guide = checklists.data_collection()
    print(f"\n  Data Collection Guidelines: {len(data_guide)} items")

    model_guide = checklists.model_selection()
    print(f"  Model Selection Considerations: {len(model_guide)} items")

    deploy_guide = checklists.deployment()
    print(f"  Deployment Safety: {len(deploy_guide)} items")

    monitor_guide = checklists.monitoring()
    print(f"  Monitoring: {len(monitor_guide)} items")

    # --- Demo 2: Incident Response ---
    print("\n" + "=" * 70)
    print("DEMO 2: Incident Response Plan")
    print("=" * 70)

    irp = IncidentResponsePlan()

    # Report a sample incident
    incident = irp.report_incident(
        title="Bias detected in hiring recommendations",
        severity="high",
        category="bias",
        description=(
            "Model showed systematic bias against female candidates for "
            "technical roles. Female candidates received 15% lower scores "
            "than equally qualified male candidates."
        ),
    )
    print(f"\n  Reported incident: {incident.id}")

    # Get procedure for this severity
    procedure = irp.get_procedure_for_severity("high")
    print(f"\n  Response timeline: {procedure['detection']['response_time']}")

    # Generate incident report
    incident.impact = "Affected female job applicants for technical positions"
    incident.status = "investigating"
    report = irp.generate_incident_report(incident)
    print(f"\n  Incident Report:")
    print(f"    {report[:300]}...")

    # --- Demo 3: Transparency Documentation ---
    print("\n" + "=" * 70)
    print("DEMO 3: Transparency Documentation")
    print("=" * 70)

    transparency = TransparencyDocumentation()

    model_card = transparency.model_card(
        model_name="Hiring Assistant v2.1",
        description="AI system to assist with initial resume screening.",
        intended_use="Screen resumes for technical role qualifications.",
        limitations=[
            "Should not be used as sole decision-maker",
            "May not capture non-traditional career paths",
            "Requires human review of all recommendations",
        ],
        training_data="Anonymized resume dataset from 2020-2024",
        evaluation_results={
            "accuracy": "87%",
            "fairness_disparity": "< 5%",
            "false_positive_rate": "12%",
        },
        ethical_considerations=[
            "Tested for gender, race, and age bias",
            "Human-in-the-loop required for all decisions",
            "Regular bias audits scheduled quarterly",
        ],
    )
    print(f"\n  Model Card:")
    print(f"    {model_card[:300]}...")

    data_card = transparency.data_card(
        dataset_name="Resume Dataset v3",
        description="Anonymized resumes from job applications.",
        sources=["Company A HR system", "Company B career portal"],
        size="100,000 resumes",
        demographic_breakdown={
            "gender_male": "55%",
            "gender_female": "42%",
            "gender_other": "3%",
        },
        known_biases=[
            "Historical hiring bias may be reflected",
            "Over-representation of tech industry resumes",
        ],
        licensing="Internal use only, anonymized",
    )
    print(f"\n  Data Card:")
    print(f"    {data_card[:300]}...")

    # --- Demo 4: User Consent ---
    print("\n" + "=" * 70)
    print("DEMO 4: User Consent Framework")
    print("=" * 70)

    consent = UserConsentFramework()
    consent.add_policy({
        "title": "Data Collection",
        "description": "We collect your queries and our responses to improve the system.",
        "data_collected": "Prompts, responses, metadata",
        "purpose": "Model improvement and safety monitoring",
        "retention": "30 days for raw data, indefinitely for anonymized aggregates",
    })
    consent.add_policy({
        "title": "Human Review",
        "description": "A small sample of conversations may be reviewed by humans for quality.",
        "data_collected": "Sample of conversations",
        "purpose": "Quality assurance and safety",
        "retention": "90 days",
    })

    consent_form = consent.generate_consent_form()
    print(f"\n  Consent Form:")
    print(f"    {consent_form[:300]}...")

    # --- Demo 5: Environmental Impact ---
    print("\n" + "=" * 70)
    print("DEMO 5: Environmental Impact Assessment")
    print("=" * 70)

    env = EnvironmentalImpact()

    training = env.estimate_training_emissions(
        gpu_hours=1000,
        gpu_power_kw=0.3,
        energy_source="grid_average_us",
    )
    print(f"\n  Training Emissions (1000 GPU hours):")
    print(f"    Energy: {training['energy_kwh']:.1f} kWh")
    print(f"    Carbon: {training['carbon_kg']:.2f} kg CO2")
    print(f"    Equivalent: {training['equivalents']['car_km_driven']:.0f} km driven")
    print(f"    Equivalent: {training['equivalents']['tree_years_needed']:.1f} tree-years")

    inference = env.estimate_inference_emissions(
        requests_per_day=10000,
        avg_tokens_per_request=500,
        model_size_b=7.0,
        days=365,
    )
    print(f"\n  Annual Inference Emissions:")
    print(f"    Total requests: {inference['total_requests']:,}")
    print(f"    Energy: {inference['energy_kwh']:.1f} kWh")
    print(f"    Carbon: {inference['carbon_kg']:.2f} kg CO2")

    print(f"\n  Sustainability Recommendations:")
    for rec in env.sustainability_recommendations()[:5]:
        print(f"    - {rec}")

    # --- Demo 6: Responsible AI Playbook ---
    print("\n" + "=" * 70)
    print("DEMO 6: Responsible AI Playbook")
    print("=" * 70)

    playbook = ResponsibleAIPlaybook.generate()
    print(f"\n  Playbook length: {len(playbook)} characters")
    print(f"  Sections: {playbook.count('## ')}")
    print(f"\n  First 500 characters:")
    print(f"    {playbook[:500]}...")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("  1. Use checklists to ensure nothing is missed")
    print("  2. Plan for incidents before they happen")
    print("  3. Document everything: models, data, systems, decisions")
    print("  4. Give users control over their data and experience")
    print("  5. Measure and reduce environmental impact")
    print("  6. This playbook is a living document — review regularly")


if __name__ == "__main__":
    run_all_demos()
