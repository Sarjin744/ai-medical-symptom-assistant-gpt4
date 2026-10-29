"""
Data Models for AI-Powered Medical Symptom Assistant
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import datetime


class PatientProfile(BaseModel):
    """Patient demographic and clinical background context."""
    age: Optional[int] = Field(default=None, description="Patient age in years")
    sex: Optional[str] = Field(default=None, description="Biological sex (Male, Female, Other, Not specified)")
    duration: Optional[str] = Field(default="", description="Duration of symptoms e.g. '2 days', '1 week'")
    severity: Optional[str] = Field(default="Mild", description="Reported severity (Mild, Moderate, Severe, Critical)")
    previous_conditions: List[str] = Field(default_factory=list, description="Known medical history/pre-existing conditions")
    current_medications: List[str] = Field(default_factory=list, description="Current prescription or OTC medications")
    allergies: List[str] = Field(default_factory=list, description="Known drug or environmental allergies")
    is_pregnant: Optional[bool] = Field(default=None, description="Pregnancy status where relevant")

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "Age": self.age if self.age is not None else "Not specified",
            "Sex": self.sex or "Not specified",
            "Duration": self.duration or "Not specified",
            "Severity": self.severity or "Not specified",
            "Pre-existing Conditions": ", ".join(self.previous_conditions) if self.previous_conditions else "None reported",
            "Current Medications": ", ".join(self.current_medications) if self.current_medications else "None reported",
            "Allergies": ", ".join(self.allergies) if self.allergies else "None reported",
            "Pregnancy Status": "Yes" if self.is_pregnant is True else ("No" if self.is_pregnant is False else "N/A")
        }


class PossibleCondition(BaseModel):
    """Potential condition identified during differential symptom analysis."""
    name: str = Field(..., description="Condition name (e.g., Common cold, Influenza)")
    likelihood: str = Field(default="Possible", description="Likelihood level (High, Moderate, Low / Possible)")
    why_considered: str = Field(..., description="Clinical rationale explaining why this condition was considered")
    differentiating_factors: Optional[str] = Field(default="", description="Features that distinguish this from other candidates")
    urgency_level: Optional[str] = Field(default="Routine", description="Routine, Prompt Evaluation, or Emergency")


class TriageAssessment(BaseModel):
    """Clinical risk and escalation classification."""
    triage_level: str = Field(default="Routine", description="Routine / Self-Care, Non-Urgent Consultation, Urgent Care, Emergency")
    color_code: str = Field(default="green", description="green, yellow, orange, red")
    red_flags_detected: List[str] = Field(default_factory=list, description="Emergency warning signs detected")
    escalation_recommendation: str = Field(..., description="Actionable instruction on when and where to seek medical care")
    emergency_action_required: bool = Field(default=False, description="True if immediate 911 / emergency evaluation is needed")


class MedicalSymptomAnalysis(BaseModel):
    """
    Standard structured response conforming to the lab specification:
    Reported Symptoms, Duration, Possible Conditions, Why These Were Considered,
    Additional Information Needed, Medical Attention Guidance, Disclaimer, and Triage.
    """
    reported_symptoms: List[str] = Field(default_factory=list, description="Extracted distinct symptoms")
    duration: str = Field(default="", description="Symptom duration")
    severity: str = Field(default="Mild", description="Overall symptom severity")
    patient_profile: Optional[PatientProfile] = Field(default=None, description="Patient clinical background")
    possible_conditions: List[PossibleCondition] = Field(default_factory=list, description="Differential considerations")
    why_considered: str = Field(default="", description="High-level clinical explanation")
    additional_information_needed: List[str] = Field(default_factory=list, description="Missing clinical parameters needed")
    follow_up_questions: List[str] = Field(default_factory=list, description="Interactive questions for patient clarification")
    medical_attention: str = Field(default="", description="Professional medical attention guidance")
    triage: Optional[TriageAssessment] = Field(default=None, description="Triage & escalation assessment")
    disclaimer: str = Field(
        default="This is an informational AI analysis and not a medical diagnosis. Always consult a licensed healthcare professional for clinical diagnosis and treatment.",
        description="Mandatory medical legal disclaimer"
    )
    raw_user_input: str = Field(default="", description="Original natural-language input")
    backend_used: str = Field(default="", description="LLM backend utilized (e.g. GPT-4, Ollama, Deterministic)")
    model_name: str = Field(default="", description="Model identifier")
    execution_time: float = Field(default=0.0, description="Processing time in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_formatted_text(self) -> str:
        """
        Formats report strictly matching the lab manual specification format (Pages 2-3).
        """
        lines = []
        lines.append("MEDICAL SYMPTOM ANALYSIS")
        lines.append("--------------------------------")
        lines.append("Reported Symptoms:")
        for s in self.reported_symptoms:
            lines.append(f"• {s}")
        if not self.reported_symptoms:
            lines.append("• None specified")
        
        lines.append("")
        lines.append("Duration:")
        lines.append(self.duration if self.duration else "Not specified")
        
        if self.patient_profile and self.patient_profile.age is not None:
            lines.append("")
            lines.append("Patient Age:")
            lines.append(f"{self.patient_profile.age} years")

        lines.append("")
        lines.append("Severity:")
        lines.append(self.severity if self.severity else "Mild")

        lines.append("")
        lines.append("Possible Conditions:")
        for idx, cond in enumerate(self.possible_conditions, 1):
            lines.append(f"{idx}. {cond.name} ({cond.likelihood})")
        if not self.possible_conditions:
            lines.append("1. Non-specific symptoms requiring clinical review")

        lines.append("")
        lines.append("Why These Were Considered:")
        lines.append(self.why_considered if self.why_considered else "The reported symptoms can occur with several conditions.")

        lines.append("")
        lines.append("Additional Information Needed:")
        for item in self.additional_information_needed:
            lines.append(f"• {item}")
        if not self.additional_information_needed:
            lines.append("• Exact temperature")
            lines.append("• Severity of symptoms")
            lines.append("• Breathing difficulty")
            lines.append("• Other associated symptoms")

        lines.append("")
        lines.append("Medical Attention:")
        lines.append(self.medical_attention if self.medical_attention else "Seek professional medical evaluation if symptoms are severe, worsening or concerning.")

        if self.triage and self.triage.red_flags_detected:
            lines.append("")
            lines.append("RED FLAG WARNINGS:")
            for rf in self.triage.red_flags_detected:
                lines.append(f"⚠️ {rf}")

        lines.append("")
        lines.append("Disclaimer:")
        lines.append(self.disclaimer)

        return "\n".join(lines)
