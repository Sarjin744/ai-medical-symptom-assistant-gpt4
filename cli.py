"""
Command-Line Interface (CLI) for AI Medical Symptom Assistant
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

import os
import sys
import json
import argparse
from typing import Optional

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from engine.models import PatientProfile
from engine.assistant import MedicalSymptomAssistant


def load_sample_cases():
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_cases.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def print_banner():
    banner = """
================================================================================
       🏥 24CS2506: AI-POWERED MEDICAL SYMPTOM ASSISTANT (GPT-4) 🏥
               Karunya Institute of Technology and Sciences
================================================================================
    """
    print(banner)


def run_interactive_session(assistant: MedicalSymptomAssistant, backend: str, model_name: str):
    print("\n--- PATIENT INTAKE & PROFILE ---")
    age_str = input("Patient Age (e.g., 22): ").strip()
    age = int(age_str) if age_str.isdigit() else None

    sex = input("Patient Sex (e.g., Female/Male/Other): ").strip() or "Not specified"
    duration = input("Symptom Duration (e.g., 2 days): ").strip() or "2 days"
    severity = input("Reported Severity (Mild / Moderate / Severe / Critical) [Mild]: ").strip() or "Mild"
    conditions = input("Pre-existing Medical Conditions (comma-separated, or press Enter): ").strip()
    medications = input("Current Medications (comma-separated, or press Enter): ").strip()
    allergies = input("Known Allergies (comma-separated, or press Enter): ").strip()

    profile = PatientProfile(
        age=age,
        sex=sex,
        duration=duration,
        severity=severity,
        previous_conditions=[c.strip() for c in conditions.split(",") if c.strip()],
        current_medications=[m.strip() for m in medications.split(",") if m.strip()],
        allergies=[a.strip() for a in allergies.split(",") if a.strip()]
    )

    print("\n--- NATURAL LANGUAGE SYMPTOM DESCRIPTION ---")
    print("Describe your symptoms naturally (e.g., 'I have a sore throat, mild fever and cough'):")
    user_input = input("> ").strip()
    if not user_input:
        user_input = "I have a sore throat, mild fever and cough."
        print(f"Using default input: {user_input}")

    print("\nProcessing symptom analysis pipeline...")
    analysis = assistant.analyze(
        user_input=user_input,
        patient_profile=profile,
        backend=backend,
        model_name=model_name
    )

    print("\n" + "=" * 60)
    print(analysis.to_formatted_text())
    print("=" * 60)

    # Interactive Follow-Up Questions (Refinement Loop)
    if analysis.follow_up_questions:
        print("\n--- CLINICAL FOLLOW-UP QUESTIONS ---")
        for idx, q in enumerate(analysis.follow_up_questions, 1):
            print(f"[{idx}] {q}")
            ans = input("    Your answer (or press Enter to skip): ").strip()

    # Save Report
    paths = assistant.save_report(analysis)
    print("\n[+] Report successfully saved:")
    print(f"    - Text Report:     {paths['txt']}")
    print(f"    - JSON Data:       {paths['json']}")
    print(f"    - Markdown Summary:{paths['md']}")


def run_sample_case(sample_id: str, assistant: MedicalSymptomAssistant, backend: str, model_name: str):
    cases = load_sample_cases()
    matched = None
    for c in cases:
        if sample_id.lower() in c.get("id", "").lower() or sample_id.lower() in c.get("title", "").lower():
            matched = c
            break

    if not matched:
        print(f"Error: Sample case '{sample_id}' not found. Available cases:")
        for c in cases:
            print(f"  - {c.get('id')}: {c.get('title')}")
        return

    print(f"\n[+] Running Sample Case: {matched.get('title')}")
    print(f"    Description: {matched.get('description')}")
    print(f"    User Input:  \"{matched.get('input_text')}\"")

    p_data = matched.get("profile", {})
    profile = PatientProfile(
        age=p_data.get("age"),
        sex=p_data.get("sex"),
        duration=p_data.get("duration"),
        severity=p_data.get("severity"),
        previous_conditions=p_data.get("previous_conditions", []),
        current_medications=p_data.get("current_medications", []),
        allergies=p_data.get("allergies", []),
        is_pregnant=p_data.get("is_pregnant")
    )

    analysis = assistant.analyze(
        user_input=matched.get("input_text"),
        patient_profile=profile,
        backend=backend,
        model_name=model_name
    )

    print("\n" + "=" * 60)
    print(analysis.to_formatted_text())
    print("=" * 60)

    paths = assistant.save_report(analysis, file_prefix=matched.get("id", "case"))
    print("\n[+] Report files generated:")
    print(f"    - Text: {paths['txt']}")
    print(f"    - JSON: {paths['json']}")
    print(f"    - MD:   {paths['md']}")


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Medical Symptom Assistant CLI")
    parser.add_argument("--sample", type=str, help="Run a specific sample case (e.g., 'lab', 'emergency', 'gastro', 'pediatric')")
    parser.add_argument("--list-samples", action="store_true", help="List all available sample clinical cases")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "openai", "gpt-4", "gpt-4o", "ollama", "deterministic"], help="Execution backend")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name for OpenAI or Ollama")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key (or set OPENAI_API_KEY environment variable)")

    args = parser.parse_args()
    print_banner()

    if args.list_samples:
        cases = load_sample_cases()
        print("\nAvailable Sample Cases:")
        for c in cases:
            print(f"  * {c.get('id'):<25} : {c.get('title')}")
        return

    assistant = MedicalSymptomAssistant(api_key=args.api_key)

    if args.sample:
        run_sample_case(args.sample, assistant, args.backend, args.model)
    else:
        run_interactive_session(assistant, args.backend, args.model)


if __name__ == "__main__":
    main()
