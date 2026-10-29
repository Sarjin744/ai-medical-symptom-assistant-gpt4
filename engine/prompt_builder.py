"""
Clinical Prompt Builder for Medical Symptom Assistant
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

import json
from typing import Dict, Any, List
from .models import PatientProfile


class MedicalPromptBuilder:
    """
    Constructs safety-aligned, clinical decision-support prompts for GPT-4 / LLMs.
    Ensures adherence to medical communication ethics, uncertainty calibration,
    and structured JSON generation.
    """

    SYSTEM_PROMPT = """You are an advanced AI Medical Symptom Assistant built on GPT-4 clinical reasoning.
Your role is to help users understand health-related information and symptom analysis responsibly, safely, and empathetically.

### MANDATORY CLINICAL SAFETY GUIDELINES:
1. NEVER ESTABLISH A DEFINITIVE DIAGNOSIS: Communicate uncertainty rather than presenting any condition as confirmed. Always use cautious clinical language such as "Possible conditions to discuss with a clinician include...", "These symptoms are consistent with...", or "Possible explanations may include...".
2. RISK-AWARE TRIAGE: Screen for life-threatening emergencies (e.g., crushing chest pain, severe breathing difficulty, anaphylaxis, sudden severe neurological deficits, meningitis signs). If red flags are detected, emphasize immediate emergency care (911/112).
3. STRUCTURED REPORTING: You must output strictly valid JSON conforming exactly to the requested schema.
4. PERTINENT NEGATIVES & FOLLOW-UPS: Identify critical missing information (vital signs, duration, associated symptoms) that a physician would require.
5. MANDATORY DISCLAIMER: Always provide the standard disclaimer: "This is an informational AI analysis and not a medical diagnosis. Always consult a licensed healthcare professional for medical evaluation and treatment."
"""

    @classmethod
    def build_system_prompt(cls) -> str:
        return cls.SYSTEM_PROMPT.strip()

    @classmethod
    def build_analysis_prompt(
        cls,
        user_input: str,
        patient_profile: PatientProfile,
        extracted_symptoms: List[str],
        duration: str,
        severity: str,
        context_risks: List[str],
        red_flags: List[str]
    ) -> str:
        """
        Builds user prompt incorporating patient profile, natural language description,
        and automated clinical context markers.
        """
        patient_context = {
            "Age": patient_profile.age if patient_profile.age is not None else "Not provided",
            "Sex": patient_profile.sex or "Not provided",
            "Duration": duration or patient_profile.duration or "Not provided",
            "Reported Severity": severity or patient_profile.severity or "Not specified",
            "Pre-existing Conditions": patient_profile.previous_conditions or ["None reported"],
            "Current Medications": patient_profile.current_medications or ["None reported"],
            "Allergies": patient_profile.allergies or ["None reported"],
            "Pregnancy Status": "Pregnant" if patient_profile.is_pregnant else "Not pregnant / N/A"
        }

        context_notes = []
        if context_risks:
            context_notes.append("Clinical Risk Factors: " + "; ".join(context_risks))
        if red_flags:
            context_notes.append("AUTOMATED RED FLAGS DETECTED: " + "; ".join(red_flags))

        context_flags_str = "\n".join(context_notes) if context_notes else "No immediate red flags detected."

        prompt = f"""Analyze the following patient presentation following the standard Medical Symptom Analysis workflow:

--- PATIENT PROFILE ---
{json.dumps(patient_context, indent=2)}

--- USER NATURAL LANGUAGE INPUT ---
"{user_input}"

--- PRE-ANALYSIS CONTEXT FLAGS ---
{context_flags_str}

--- TASK INSTRUCTIONS ---
Perform comprehensive symptom analysis and output a single JSON object with the following exact keys:

{{
  "reported_symptoms": ["List", "of", "extracted", "symptoms"],
  "duration": "Duration extracted from input (e.g., '2 days')",
  "severity": "Mild | Moderate | Severe | Critical",
  "possible_conditions": [
    {{
      "name": "Condition Name (e.g. Common cold)",
      "likelihood": "High | Moderate | Low",
      "why_considered": "Clear clinical rationale for why this condition is considered based on the reported symptoms",
      "differentiating_factors": "Key distinguishing signs or pertinent negatives",
      "urgency_level": "Routine | Prompt Evaluation | Emergency"
    }}
  ],
  "why_considered": "Summary explanation of why these candidate conditions were considered across the respiratory/systemic spectrum",
  "additional_information_needed": [
    "Specific missing clinical parameter (e.g., Exact temperature)",
    "Severity of symptoms on a numeric scale",
    "Presence of breathing difficulty"
  ],
  "follow_up_questions": [
    "Targeted clinical question 1 to help narrow down the differential",
    "Targeted clinical question 2 regarding progression or warning signs"
  ],
  "medical_attention": "Clear actionable instruction on when and where to seek professional medical attention",
  "triage_level": "Routine / Self-Care | Non-Urgent Medical Consultation | Urgent Care | Emergency",
  "red_flags_detected": ["List of any detected life-threatening signs, or empty list"],
  "emergency_action_required": false,
  "disclaimer": "This is an informational AI analysis and not a medical diagnosis. Seek professional medical evaluation if symptoms are severe, worsening or concerning."
}}

Respond ONLY with valid JSON. Do not include markdown code block syntax if possible, or wrap only in ```json ... ```.
"""
        return prompt.strip()
