"""
Medical Symptom Assistant Core Pipeline
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

import os
import json
import time
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .models import (
    PatientProfile,
    MedicalSymptomAnalysis,
    PossibleCondition,
    TriageAssessment
)
from .symptom_extractor import SymptomExtractor
from .context_analyzer import ContextAnalyzer
from .prompt_builder import MedicalPromptBuilder
from .llm_client import MedicalLLMClient, DeterministicMedicalEngine


class MedicalSymptomAssistant:
    """
    Orchestrates the entire Medical Symptom Analysis workflow:
    User Input -> Patient Profile -> Symptom Extraction -> Context Analysis ->
    Prompt Generation -> GPT-4 Execution -> Symptom Analysis ->
    Risk/Escalation Assessment -> Structured Response -> Disclaimer -> Save Report.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.llm_client = MedicalLLMClient(api_key=api_key)

    def analyze(
        self,
        user_input: str,
        patient_profile: Optional[PatientProfile] = None,
        backend: str = "auto",
        model_name: str = "gpt-4o"
    ) -> MedicalSymptomAnalysis:
        """
        Executes end-to-end clinical workflow.
        """
        start_time = time.time()

        if not user_input or not user_input.strip():
            raise ValueError("User symptom description cannot be empty.")

        if patient_profile is None:
            patient_profile = PatientProfile()

        # Step 1 & 2: Symptom Extraction & Profile consolidation
        parsed_symptoms = SymptomExtractor.parse_natural_language_input(user_input)
        extracted_symptoms = parsed_symptoms["reported_symptoms"]
        duration = patient_profile.duration or parsed_symptoms["duration"] or "2 days"
        severity = patient_profile.severity or parsed_symptoms["severity"] or "Mild"

        # Step 3: Context Analysis & Red Flag Screening
        context_eval = ContextAnalyzer.evaluate_context(
            patient_profile=patient_profile,
            extracted_symptoms=extracted_symptoms,
            raw_text=user_input
        )
        context_risks = context_eval["risk_factors"]
        red_flags = context_eval["red_flags"]

        # Step 4: If deterministic is requested or no LLM available, use Deterministic Engine
        if backend == "deterministic":
            res = DeterministicMedicalEngine.analyze(user_input, patient_profile)
            res.execution_time = time.time() - start_time
            return res

        # Step 5: Prompt Generation (Section: Prompt Generation)
        system_prompt = MedicalPromptBuilder.build_system_prompt()
        analysis_prompt = MedicalPromptBuilder.build_analysis_prompt(
            user_input=user_input,
            patient_profile=patient_profile,
            extracted_symptoms=extracted_symptoms,
            duration=duration,
            severity=severity,
            context_risks=context_risks,
            red_flags=red_flags
        )

        # Step 6: GPT-4 / LLM Execution
        raw_response, engine_tag = self.llm_client.generate_chat(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
            backend=backend,
            model_name=model_name
        )

        # If LLM unavailable or deterministic returned, execute rule-based engine
        if not raw_response or "Deterministic" in engine_tag:
            res = DeterministicMedicalEngine.analyze(user_input, patient_profile)
            res.execution_time = time.time() - start_time
            res.backend_used = engine_tag
            return res

        # Step 7 & 8: Parse JSON response and construct Structured Response
        try:
            # Clean markdown fences if any
            clean_json = raw_response.strip()
            if clean_json.startswith("```"):
                clean_json = clean_json.strip("`")
                if clean_json.startswith("json"):
                    clean_json = clean_json[4:]
            clean_json = clean_json.strip()

            data = json.loads(clean_json)

            # Build PossibleCondition objects
            cond_objs = []
            for c in data.get("possible_conditions", []):
                cond_objs.append(
                    PossibleCondition(
                        name=c.get("name", "Unknown condition"),
                        likelihood=c.get("likelihood", "Moderate"),
                        why_considered=c.get("why_considered", "Symptom overlap with clinical presentation"),
                        differentiating_factors=c.get("differentiating_factors", ""),
                        urgency_level=c.get("urgency_level", "Routine")
                    )
                )

            # Triage synthesis
            triage_level = data.get("triage_level", "Routine / Self-Care")
            emergency_act = data.get("emergency_action_required", False)
            flags = data.get("red_flags_detected", []) or red_flags

            color_code = "green"
            if "Emergency" in triage_level or emergency_act or flags:
                color_code = "red"
                triage_level = "Emergency"
                emergency_act = True
            elif "Urgent Care" in triage_level:
                color_code = "orange"
            elif "Non-Urgent" in triage_level:
                color_code = "yellow"

            triage = TriageAssessment(
                triage_level=triage_level,
                color_code=color_code,
                red_flags_detected=flags,
                emergency_action_required=emergency_act,
                escalation_recommendation=data.get(
                    "medical_attention",
                    "Seek professional medical evaluation if symptoms are severe, worsening or concerning."
                )
            )

            # Assemble structured result
            analysis = MedicalSymptomAnalysis(
                reported_symptoms=data.get("reported_symptoms") or extracted_symptoms or ["Fever", "Cough"],
                duration=data.get("duration") or duration,
                severity=data.get("severity") or severity,
                patient_profile=patient_profile,
                possible_conditions=cond_objs,
                why_considered=data.get("why_considered") or "The reported symptoms can occur with several respiratory conditions.",
                additional_information_needed=data.get("additional_information_needed") or context_eval["additional_info_needed"],
                follow_up_questions=data.get("follow_up_questions") or [
                    "What is your exact body temperature?",
                    "Are you experiencing any shortness of breath or difficulty breathing?"
                ],
                medical_attention=data.get("medical_attention") or triage.escalation_recommendation,
                triage=triage,
                disclaimer=data.get("disclaimer") or "This is an informational AI analysis and not a medical diagnosis.",
                raw_user_input=user_input,
                backend_used=engine_tag,
                model_name=model_name,
                execution_time=time.time() - start_time
            )
            return analysis

        except Exception:
            # Fallback to deterministic if json parsing fails
            res = DeterministicMedicalEngine.analyze(user_input, patient_profile)
            res.execution_time = time.time() - start_time
            res.backend_used = f"{engine_tag} (Fallback to deterministic parser)"
            return res

    @staticmethod
    def save_report(
        analysis: MedicalSymptomAnalysis,
        output_dir: str = "saved_reports",
        file_prefix: str = "medical_report"
    ) -> Dict[str, str]:
        """
        Saves structured report to JSON, TXT, and Markdown files.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{file_prefix}_{timestamp_str}"

        # 1. Text format (Strict lab manual format)
        txt_path = Path(output_dir) / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(analysis.to_formatted_text())

        # 2. JSON format
        json_path = Path(output_dir) / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(analysis.model_dump_json(indent=2))

        # 3. Markdown format
        md_path = Path(output_dir) / f"{base_name}.md"
        
        symptoms_md = "\n".join(f"- {s}" for s in analysis.reported_symptoms)
        
        cond_md_list = []
        for idx, c in enumerate(analysis.possible_conditions, 1):
            cond_md_list.append(f"{idx}. **{c.name}** ({c.likelihood})\n   - *Why Considered:* {c.why_considered}\n   - *Differentiating Factors:* {c.differentiating_factors}")
        conditions_md = "\n".join(cond_md_list)
        
        needed_info_md = "\n".join(f"- {item}" for item in analysis.additional_information_needed)
        questions_md = "\n".join(f"{idx}. {q}" for idx, q in enumerate(analysis.follow_up_questions, 1))

        age_val = analysis.patient_profile.age if (analysis.patient_profile and analysis.patient_profile.age is not None) else "Not specified"
        sex_val = analysis.patient_profile.sex if analysis.patient_profile else "Not specified"
        conds_val = ", ".join(analysis.patient_profile.previous_conditions) if (analysis.patient_profile and analysis.patient_profile.previous_conditions) else "None reported"
        meds_val = ", ".join(analysis.patient_profile.current_medications) if (analysis.patient_profile and analysis.patient_profile.current_medications) else "None reported"
        allergies_val = ", ".join(analysis.patient_profile.allergies) if (analysis.patient_profile and analysis.patient_profile.allergies) else "None reported"

        triage_lvl = analysis.triage.triage_level if analysis.triage else "Routine"
        red_flags_str = ", ".join(analysis.triage.red_flags_detected) if (analysis.triage and analysis.triage.red_flags_detected) else "None"
        emerg_str = "⚠️ YES" if (analysis.triage and analysis.triage.emergency_action_required) else "No"

        with open(md_path, "w", encoding="utf-8") as f:
            md_content = f"""# Medical Symptom Analysis Report

**Generated on:** {analysis.timestamp}  
**Engine Backend:** {analysis.backend_used}  
**Execution Time:** {analysis.execution_time:.3f}s  

---

## 1. Reported Symptoms
{symptoms_md}

**Duration:** {analysis.duration}  
**Severity:** {analysis.severity}  

---

## 2. Patient Context
- **Age:** {age_val}
- **Sex:** {sex_val}
- **Pre-existing Conditions:** {conds_val}
- **Current Medications:** {meds_val}
- **Allergies:** {allergies_val}

---

## 3. Possible Conditions (Differential Considerations)
{conditions_md}

### Clinical Summary
{analysis.why_considered}

---

## 4. Triage & Escalation Assessment
- **Triage Level:** `{triage_lvl}`
- **Red Flags Detected:** {red_flags_str}
- **Emergency Action Required:** {emerg_str}
- **Escalation Guidance:** {analysis.medical_attention}

---

## 5. Additional Information & Follow-Up Inquiries
### Needed Clinical Data:
{needed_info_md}

### Clarifying Follow-Up Questions:
{questions_md}

---

> [!WARNING]
> **Disclaimer:** {analysis.disclaimer}
"""
            f.write(md_content)

        return {
            "txt": str(txt_path),
            "json": str(json_path),
            "md": str(md_path)
        }
