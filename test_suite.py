"""
Comprehensive Unit & Integration Test Suite
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

import os
import json
import unittest
from engine.models import PatientProfile, MedicalSymptomAnalysis
from engine.symptom_extractor import SymptomExtractor
from engine.context_analyzer import ContextAnalyzer
from engine.prompt_builder import MedicalPromptBuilder
from engine.llm_client import DeterministicMedicalEngine
from engine.assistant import MedicalSymptomAssistant


class TestMedicalSymptomAssistant(unittest.TestCase):

    def test_01_symptom_extraction(self):
        """Test entity extraction from natural language symptom narratives."""
        text = "I have a sore throat, mild fever and cough since 2 days ago."
        parsed = SymptomExtractor.parse_natural_language_input(text)

        symptoms = [s.lower() for s in parsed["reported_symptoms"]]
        self.assertIn("sore throat", symptoms)
        self.assertIn("fever", symptoms)
        self.assertIn("cough", symptoms)
        self.assertIn("2 days", parsed["duration"])
        self.assertEqual(parsed["severity"], "Mild")

    def test_02_context_analyzer_red_flags(self):
        """Test red-flag detection for life-threatening presentations."""
        profile = PatientProfile(age=62, sex="Male", severity="Critical")
        text = "I am having severe crushing chest pain, shortness of breath, and feeling dizzy."
        extracted = SymptomExtractor.extract_symptoms(text)

        eval_result = ContextAnalyzer.evaluate_context(profile, extracted, text)
        self.assertTrue(len(eval_result["red_flags"]) > 0)
        self.assertTrue(any("Acute Coronary" in rf or "cardiac" in rf.lower() for rf in eval_result["red_flags"]))

        triage = ContextAnalyzer.determine_triage(profile, extracted, "Critical", eval_result["red_flags"])
        self.assertEqual(triage.triage_level, "Emergency")
        self.assertTrue(triage.emergency_action_required)
        self.assertEqual(triage.color_code, "red")

    def test_03_prompt_builder(self):
        """Test clinical prompt synthesis and safety guardrails."""
        profile = PatientProfile(age=22, sex="Female", duration="2 days", severity="Mild")
        sys_prompt = MedicalPromptBuilder.build_system_prompt()
        self.assertIn("NEVER ESTABLISH A DEFINITIVE DIAGNOSIS", sys_prompt)
        self.assertIn("MANDATORY DISCLAIMER", sys_prompt)

        user_prompt = MedicalPromptBuilder.build_analysis_prompt(
            user_input="I have a sore throat, mild fever and cough.",
            patient_profile=profile,
            extracted_symptoms=["Sore throat", "Fever", "Cough"],
            duration="2 days",
            severity="Mild",
            context_risks=[],
            red_flags=[]
        )
        self.assertIn("reported_symptoms", user_prompt)
        self.assertIn("possible_conditions", user_prompt)
        self.assertIn("22", user_prompt)

    def test_04_deterministic_engine_benchmark(self):
        """Test deterministic engine output matching lab manual benchmark."""
        profile = PatientProfile(age=22, sex="Female", duration="2 days", severity="Mild")
        user_input = "I have a sore throat, mild fever and cough."

        res = DeterministicMedicalEngine.analyze(user_input, profile)
        self.assertIsInstance(res, MedicalSymptomAnalysis)
        self.assertTrue(len(res.reported_symptoms) >= 2)
        self.assertEqual(res.duration, "2 days")
        self.assertEqual(res.severity, "Mild")

        # Verify candidate conditions include common cold / influenza / respiratory infection
        cond_names = [c.name.lower() for c in res.possible_conditions]
        self.assertTrue(any("cold" in name for name in cond_names))
        self.assertTrue(any("influenza" in name for name in cond_names) or any("respiratory" in name for name in cond_names))

        # Check disclaimer presence
        self.assertIn("informational AI analysis", res.disclaimer)

        # Check formatted output string conforms to Page 2-3 of manual
        formatted = res.to_formatted_text()
        self.assertIn("MEDICAL SYMPTOM ANALYSIS", formatted)
        self.assertIn("Reported Symptoms:", formatted)
        self.assertIn("Duration:", formatted)
        self.assertIn("Possible Conditions:", formatted)
        self.assertIn("Why These Were Considered:", formatted)
        self.assertIn("Additional Information Needed:", formatted)
        self.assertIn("Medical Attention:", formatted)
        self.assertIn("Disclaimer:", formatted)

    def test_05_assistant_end_to_end(self):
        """Test full assistant execution, fallback, and report persistence."""
        assistant = MedicalSymptomAssistant()
        profile = PatientProfile(age=30, sex="Male", duration="3 days", severity="Moderate")
        user_input = "I've had severe nausea, stomach ache, and diarrhea since yesterday."

        analysis = assistant.analyze(
            user_input=user_input,
            patient_profile=profile,
            backend="deterministic"
        )

        self.assertIsNotNone(analysis)
        self.assertTrue(len(analysis.possible_conditions) > 0)
        self.assertTrue(len(analysis.follow_up_questions) > 0)

        # Test report saving
        saved = assistant.save_report(analysis, output_dir="test_saved_reports", file_prefix="unit_test")
        self.assertTrue(os.path.exists(saved["txt"]))
        self.assertTrue(os.path.exists(saved["json"]))
        self.assertTrue(os.path.exists(saved["md"]))

        # Cleanup test files
        for p in saved.values():
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists("test_saved_reports"):
            os.rmdir("test_saved_reports")


if __name__ == "__main__":
    unittest.main()
