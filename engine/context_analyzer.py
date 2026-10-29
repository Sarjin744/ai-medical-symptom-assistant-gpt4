"""
Clinical Context Analyzer & Red-Flag Triage Engine
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

from typing import List, Dict, Any, Tuple
from .models import PatientProfile, TriageAssessment


class ContextAnalyzer:
    """
    Performs clinical context analysis, demographic risk stratification,
    red-flag detection, and emergency triage assessment.
    """

    # Life-threatening red flag symptoms and keyword combinations
    RED_FLAGS = [
        {
            "id": "cardiac_chest_pain",
            "trigger_terms": ["chest pain", "pressure in chest", "angina"],
            "co_triggers": ["shortness of breath", "sweating", "radiating to arm", "jaw pain", "dizziness"],
            "description": "Potential Acute Coronary Syndrome / Myocardial Infarction warning sign",
            "urgency": "Emergency"
        },
        {
            "id": "severe_dyspnea",
            "trigger_terms": ["can't breathe", "struggling to breathe", "gasping", "blue lips", "cyanosis"],
            "description": "Severe respiratory distress requiring immediate oxygenation evaluation",
            "urgency": "Emergency"
        },
        {
            "id": "meningitis_signs",
            "trigger_terms": ["neck stiffness", "stiff neck"],
            "co_triggers": ["fever", "headache", "photophobia", "confusion"],
            "description": "Meningeal irritation triad (Fever + Headache + Neck Stiffness)",
            "urgency": "Emergency"
        },
        {
            "id": "altered_mental_status",
            "trigger_terms": ["confusion", "disoriented", "passed out", "blacked out", "seizure", "unresponsive"],
            "description": "Acute neurological deterioration / altered sensorium",
            "urgency": "Emergency"
        },
        {
            "id": "anaphylaxis",
            "trigger_terms": ["swollen tongue", "swollen lips", "throat closing", "difficulty swallowing with rash"],
            "description": "Possible severe allergic reaction / anaphylactic airway compromise",
            "urgency": "Emergency"
        },
        {
            "id": "thunderclap_headache",
            "trigger_terms": ["worst headache of life", "sudden explosive headache", "thunderclap headache"],
            "description": "Possible subarachnoid hemorrhage warning sign",
            "urgency": "Emergency"
        }
    ]

    @classmethod
    def evaluate_context(
        cls,
        patient_profile: PatientProfile,
        extracted_symptoms: List[str],
        raw_text: str = ""
    ) -> Dict[str, Any]:
        """
        Analyzes patient profile and symptoms to identify clinical risks,
        vulnerabilities, and missing critical information.
        """
        full_text_lower = (raw_text + " " + " ".join(extracted_symptoms)).lower()
        risk_factors = []
        red_flags_found = []

        # 1. Demographic Vulnerabilities
        if patient_profile.age is not None:
            if patient_profile.age < 5:
                risk_factors.append("Pediatric patient (< 5 years): High risk for rapid decompensation, febrile seizures, and dehydration.")
            elif patient_profile.age >= 65:
                risk_factors.append("Geriatric patient (>= 65 years): Atypical symptom presentation risk, higher vulnerability to complications.")

        # 2. Pregnancy Context
        if patient_profile.is_pregnant:
            risk_factors.append("Active pregnancy: Requires heightened caution regarding fetal well-being and medication safety.")

        # 3. Comorbidities
        for cond in patient_profile.previous_conditions:
            cond_lower = cond.lower()
            if any(k in cond_lower for k in ["asthma", "copd", "bronchitis"]):
                risk_factors.append(f"Underlying respiratory disease ({cond}): Higher susceptibility to severe bronchospasm or lower respiratory infection.")
            elif any(k in cond_lower for k in ["diabetes", "immunosuppressed", "chemotherapy", "hiv"]):
                risk_factors.append(f"Immunocompromised/Metabolic state ({cond}): Heightened risk of severe or atypical infectious course.")
            elif any(k in cond_lower for k in ["heart", "cardiac", "hypertension"]):
                risk_factors.append(f"Cardiovascular history ({cond}): Heightened threshold for cardiopulmonary evaluation.")

        # 4. Red Flag Screening
        for rf in cls.RED_FLAGS:
            matched_trigger = any(term in full_text_lower for term in rf["trigger_terms"])
            if matched_trigger:
                if "co_triggers" in rf:
                    matched_co = any(co in full_text_lower for co in rf["co_triggers"])
                    if matched_co:
                        red_flags_found.append(f"{rf['description']}")
                else:
                    red_flags_found.append(f"{rf['description']}")

        # 5. Formulate Additional Information Needed
        needed_info = []
        if not any("temp" in full_text_lower or "degree" in full_text_lower for _ in [1]):
            needed_info.append("Exact body temperature measurement (in °F or °C)")
        if not any("breath" in full_text_lower for _ in [1]):
            needed_info.append("Presence of breathing difficulty or chest tightness")
        if not any("rate" in full_text_lower or "bpm" in full_text_lower or "pulse" in full_text_lower for _ in [1]):
            needed_info.append("Resting heart rate / pulse oxygen level if home pulse oximeter available")
        if not patient_profile.duration:
            needed_info.append("Precise onset and progression (hours vs days)")
        if not patient_profile.current_medications:
            needed_info.append("Any over-the-counter or prescription medications already taken")

        # Ensure default essential parameters as highlighted in lab manual
        if len(needed_info) < 2:
            needed_info.extend([
                "Severity of symptoms on a scale of 1-10",
                "Other associated symptoms (e.g., rash, dizziness, joint aches)"
            ])

        return {
            "risk_factors": risk_factors,
            "red_flags": red_flags_found,
            "additional_info_needed": needed_info
        }

    @classmethod
    def determine_triage(
        cls,
        patient_profile: PatientProfile,
        extracted_symptoms: List[str],
        severity: str,
        red_flags: List[str]
    ) -> TriageAssessment:
        """
        Classifies patient into standard clinical triage tiers:
        - Emergency (Red)
        - Urgent Care (Orange)
        - Non-Urgent Primary Care (Yellow)
        - Routine / Self-Care (Green)
        """
        if red_flags or severity == "Critical":
            return TriageAssessment(
                triage_level="Emergency",
                color_code="red",
                red_flags_detected=red_flags,
                emergency_action_required=True,
                escalation_recommendation="Seek IMMEDIATE emergency medical care or call emergency services (911/112). Do not drive yourself."
            )
        
        # High-risk demographic or severe symptoms
        is_high_risk = (
            (patient_profile.age is not None and (patient_profile.age < 1 or patient_profile.age > 75)) or
            patient_profile.is_pregnant or
            severity == "Severe" or
            any(s.lower() in ["shortness of breath", "chest pain"] for s in extracted_symptoms)
        )

        if is_high_risk:
            return TriageAssessment(
                triage_level="Urgent Care",
                color_code="orange",
                red_flags_detected=[],
                emergency_action_required=False,
                escalation_recommendation="Seek urgent medical evaluation within 12-24 hours at an urgent care clinic or primary care physician."
            )

        if severity == "Moderate" or len(extracted_symptoms) >= 3:
            return TriageAssessment(
                triage_level="Non-Urgent Medical Consultation",
                color_code="yellow",
                red_flags_detected=[],
                emergency_action_required=False,
                escalation_recommendation="Schedule a consultation with a healthcare provider if symptoms persist beyond 3-5 days or worsen."
            )

        return TriageAssessment(
            triage_level="Routine / Self-Care",
            color_code="green",
            red_flags_detected=[],
            emergency_action_required=False,
            escalation_recommendation="Manage with rest, oral hydration, and supportive self-care. Seek medical evaluation if symptoms worsen or fail to improve."
        )
