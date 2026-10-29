"""
Natural Language Symptom Extractor
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

import re
from typing import Dict, List, Any, Tuple, Optional


class SymptomExtractor:
    """
    Extracts structured clinical symptoms, duration, and severity indicators
    from unstructured natural language input.
    """

    KNOWN_SYMPTOMS_CATALOG = {
        # Respiratory
        "fever": ["fever", "pyrexia", "high temp", "temperature", "chills", "febrile", "burning up"],
        "cough": ["cough", "coughing", "dry cough", "productive cough", "hacking cough"],
        "sore throat": ["sore throat", "throat pain", "scratchy throat", "pharyngitis", "pain swallowing"],
        "shortness of breath": ["shortness of breath", "dyspnea", "breathing difficulty", "hard to breathe", "breathless", "can't catch breath"],
        "runny nose": ["runny nose", "rhinorrhea", "nasal discharge", "stuffy nose", "congested", "congestion"],
        "sneezing": ["sneezing", "sneeze"],
        "wheezing": ["wheezing", "whistling breath"],
        "chest congestion": ["chest congestion", "tight chest", "chest tightness"],
        
        # Pain & Neurological
        "headache": ["headache", "migraine", "head pain", "head hurts", "throbbing head", "temple pain"],
        "chest pain": ["chest pain", "angina", "pressure in chest", "sternal pain", "crushing pain"],
        "neck stiffness": ["neck stiffness", "stiff neck", "nuchal rigidity"],
        "dizziness": ["dizziness", "lightheaded", "lightheadedness", "vertigo", "spinning"],
        "fatigue": ["fatigue", "tired", "tiredness", "exhaustion", "lethargy", "weakness"],
        "confusion": ["confusion", "disoriented", "altered mental state", "hallucination"],
        
        # Gastrointestinal
        "nausea": ["nausea", "queasy", "feeling sick"],
        "vomiting": ["vomiting", "throwing up", "emesis"],
        "diarrhea": ["diarrhea", "loose stools", "watery stool"],
        "abdominal pain": ["abdominal pain", "stomach ache", "belly pain", "cramps", "stomach cramps"],
        "loss of appetite": ["loss of appetite", "not eating", "poor appetite", "anorexia"],
        
        # Musculoskeletal & Systemic
        "body aches": ["body aches", "muscle aches", "myalgia", "joint pain", "arthralgia"],
        "rash": ["rash", "skin eruption", "hives", "urticaria", "red spots"],
        "ear pain": ["ear pain", "earache", "otitis"],
        "loss of smell or taste": ["loss of smell", "loss of taste", "anosmia", "ageusia"],
    }

    DURATION_PATTERNS = [
        r"(?:for|since|lasting|lasted|over the past|past)\s+(\d+\s+(?:days?|weeks?|months?|hours?|years?))",
        r"(\d+\s+(?:days?|weeks?|months?|hours?|years?))\s+(?:ago|duration|now)",
        r"(since\s+yesterday|since\s+this\s+morning|since\s+last\s+night)",
        r"(a\s+few\s+days|a\s+few\s+hours|several\s+days|couple\s+of\s+days|one\s+day|two\s+days)",
        r"(\d+)\s*days?",
    ]

    SEVERITY_KEYWORDS = {
        "Critical": ["unbearable", "excruciating", "can't breathe", "crushing", "severe and worsening rapidly", "emergency", "passed out", "blacked out"],
        "Severe": ["severe", "intense", "very bad", "extreme", "high fever", "sharp", "terrible", "uncontrolled", "heavy"],
        "Moderate": ["moderate", "medium", "noticeable", "intermittent", "persistent", "bothersome", "aching"],
        "Mild": ["mild", "slight", "minor", "a bit", "a little", "low grade", "tolerable", "manageable", "subtle"]
    }

    @classmethod
    def extract_symptoms(cls, text: str) -> List[str]:
        """Identifies medical symptoms mentioned in natural language text."""
        if not text:
            return []
        
        lower_text = text.lower()
        extracted = []

        for canonical_name, aliases in cls.KNOWN_SYMPTOMS_CATALOG.items():
            for alias in aliases:
                # Word boundary check
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, lower_text):
                    if canonical_name.capitalize() not in extracted:
                        extracted.append(canonical_name.capitalize())
                    break

        return extracted

    @classmethod
    def extract_duration(cls, text: str) -> str:
        """Extracts symptom duration from natural language."""
        if not text:
            return ""
        
        lower = text.lower()
        for pat in cls.DURATION_PATTERNS:
            match = re.search(pat, lower)
            if match:
                duration_str = match.group(1) if match.groups() else match.group(0)
                return duration_str.strip()
        
        return ""

    @classmethod
    def extract_severity(cls, text: str) -> str:
        """Determines reported severity from keywords in text."""
        lower = text.lower()
        for level in ["Critical", "Severe", "Moderate", "Mild"]:
            for keyword in cls.SEVERITY_KEYWORDS[level]:
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, lower):
                    return level
        return "Mild"

    @classmethod
    def parse_natural_language_input(cls, text: str) -> Dict[str, Any]:
        """
        Parses full user narrative into structured symptom context.
        Matches Section: Traditional Input vs Natural Language Input.
        """
        symptoms = cls.extract_symptoms(text)
        duration = cls.extract_duration(text)
        severity = cls.extract_severity(text)

        return {
            "reported_symptoms": symptoms,
            "duration": duration,
            "severity": severity,
            "raw_text": text
        }
