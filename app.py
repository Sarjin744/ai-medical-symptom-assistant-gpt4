"""
Streamlit Web Application for AI-Powered Medical Symptom Assistant
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5/6: AI-Powered Medical Symptom Assistant using GPT-4
"""

import os
import sys
import json
import time
import streamlit as st

# Configure UTF-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from engine.models import PatientProfile, MedicalSymptomAnalysis
from engine.assistant import MedicalSymptomAssistant
from engine.symptom_extractor import SymptomExtractor
from engine.context_analyzer import ContextAnalyzer

# Page configuration
st.set_page_config(
    page_title="AI Medical Symptom Assistant | Exp 5/6",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical UI Theme
st.markdown("""
<style>
    .medical-header {
        background: linear-gradient(135deg, #09203f 0%, #1e3c72 50%, #2a5298 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .medical-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .medical-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin: 0;
    }
    .report-card {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 6px solid #2563eb;
        border-radius: 8px;
        padding: 22px;
        margin-top: 15px;
        color: #0f172a;
    }
    .report-mono {
        background-color: #0f172a;
        color: #38bdf8;
        font-family: 'Consolas', 'Courier New', monospace;
        padding: 18px;
        border-radius: 8px;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    .triage-badge-red {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .triage-badge-orange {
        background-color: #ffedd5;
        color: #9a3412;
        border: 2px solid #f97316;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .triage-badge-yellow {
        background-color: #fef9c3;
        color: #854d0e;
        border: 2px solid #eab308;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .triage-badge-green {
        background-color: #dcfce7;
        color: #166534;
        border: 2px solid #22c55e;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.95rem;
        color: #92400e;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_cases():
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_cases.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "active_sample_loaded" not in st.session_state:
    st.session_state.active_sample_loaded = None

# Header
st.markdown("""
<div class="medical-header">
    <h1>🏥 AI-Powered Medical Symptom Assistant</h1>
    <p><b>Course:</b> 24CS2506 – Building Applications using GPT-4 | <b>Institution:</b> Karunya Institute of Technology and Sciences</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    backend_choice = st.selectbox(
        "LLM Backend Engine",
        ["Auto-Detect", "OpenAI (GPT-4 / GPT-4o)", "Local Ollama", "Deterministic Clinical Engine"],
        index=0,
        help="Select between cloud GPT-4, local Ollama LLM, or the built-in deterministic clinical rule engine."
    )

    backend_param = "auto"
    if "OpenAI" in backend_choice:
        backend_param = "openai"
    elif "Ollama" in backend_choice:
        backend_param = "ollama"
    elif "Deterministic" in backend_choice:
        backend_param = "deterministic"

    api_key = ""
    if backend_param in ("auto", "openai"):
        api_key = st.text_input(
            "OpenAI API Key (Optional)",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Leave blank to use environment OPENAI_API_KEY or fallback to the local deterministic clinical engine."
        )

    model_name = st.selectbox(
        "Model Selection",
        ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "llama3.2", "medllama"],
        index=0
    )

    st.markdown("---")
    st.subheader("📋 Pre-Configured Test Cases")
    sample_cases = load_sample_cases()
    case_titles = ["-- Select a Sample Case --"] + [c["title"] for c in sample_cases]
    selected_sample = st.selectbox("Load Clinical Case", case_titles)

    loaded_case_data = None
    if selected_sample != "-- Select a Sample Case --":
        for c in sample_cases:
            if c["title"] == selected_sample:
                loaded_case_data = c
                break

    st.markdown("---")
    st.markdown("""
    **💡 Key Clinical Guardrails:**
    - Communicates uncertainty (no definitive diagnosis).
    - Emergency red flag screening.
    - Generates follow-up clarifying questions.
    - Standard medical disclaimer.
    """)

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Symptom Intake & Analysis",
    "📄 Structured Medical Report",
    "❓ Interactive Follow-Up & Triage",
    "🔍 System Workflow & Traceability",
    "💾 Export & History"
])

# TAB 1: Symptom Intake & Analysis
with tab1:
    col_input, col_profile = st.columns([1.2, 0.8])

    with col_profile:
        st.subheader("👤 Patient Profile")
        default_age = 22
        default_sex = "Female"
        default_dur = "2 days"
        default_sev = "Mild"
        default_conditions = ""
        default_meds = ""
        default_allergies = ""
        default_preg = False

        if loaded_case_data:
            prof = loaded_case_data.get("profile", {})
            default_age = prof.get("age", 22) or 22
            default_sex = prof.get("sex", "Female") or "Female"
            default_dur = prof.get("duration", "2 days") or "2 days"
            default_sev = prof.get("severity", "Mild") or "Mild"
            default_conditions = ", ".join(prof.get("previous_conditions", []))
            default_meds = ", ".join(prof.get("current_medications", []))
            default_allergies = ", ".join(prof.get("allergies", []))
            default_preg = bool(prof.get("is_pregnant", False))

        p_age = st.number_input("Age", min_value=0, max_value=120, value=int(default_age), step=1)
        p_sex = st.selectbox("Biological Sex", ["Female", "Male", "Other", "Not specified"], index=["Female", "Male", "Other", "Not specified"].index(default_sex) if default_sex in ["Female", "Male", "Other", "Not specified"] else 0)
        p_duration = st.text_input("Duration of Symptoms", value=default_dur, placeholder="e.g. 2 days, 1 week, since yesterday")
        p_severity = st.select_slider("Reported Severity", options=["Mild", "Moderate", "Severe", "Critical"], value=default_sev)
        p_conditions = st.text_input("Pre-existing Medical Conditions", value=default_conditions, placeholder="e.g. Asthma, Hypertension, Diabetes")
        p_meds = st.text_input("Current Medications", value=default_meds, placeholder="e.g. Paracetamol, Inhaler, Metformin")
        p_allergies = st.text_input("Allergies", value=default_allergies, placeholder="e.g. Penicillin, Pollen, Peanuts")
        p_pregnant = st.checkbox("Currently Pregnant", value=default_preg)

    with col_input:
        st.subheader("💬 Natural Language Symptom Input")
        st.caption("Unlike traditional fixed forms (Fever = Yes, Cough = Yes), users describe their symptoms naturally.")

        default_text = "I have a sore throat, mild fever and cough."
        if loaded_case_data:
            default_text = loaded_case_data.get("input_text", default_text)

        user_symptom_text = st.text_area(
            "Describe your health concerns, symptoms, and feelings:",
            value=default_text,
            height=140,
            placeholder="e.g., I've had a fever and cough since yesterday. I also feel tired and have a mild headache..."
        )

        # Quick preset pills
        st.markdown("**Quick Preset Descriptions:**")
        qp_col1, qp_col2, qp_col3 = st.columns(3)
        if qp_col1.button("🤧 Viral Cold / Flu"):
            st.session_state.quick_preset = "I have a sore throat, mild fever and cough since 2 days ago."
            st.rerun()
        if qp_col2.button("🚨 Acute Chest Pain"):
            st.session_state.quick_preset = "I am having sudden crushing chest pain radiating to my left arm, shortness of breath, and cold sweats."
            st.rerun()
        if qp_col3.button("🤢 Stomach Cramps"):
            st.session_state.quick_preset = "I have a stomach ache, intense nausea, vomiting, and watery diarrhea since last night."
            st.rerun()

        if "quick_preset" in st.session_state:
            user_symptom_text = st.session_state.pop("quick_preset")

        # Instant Extractor Preview
        if user_symptom_text.strip():
            quick_parsed = SymptomExtractor.parse_natural_language_input(user_symptom_text)
            st.markdown(f"**⚡ Auto-Detected Entities:** `Symptoms:` {', '.join(quick_parsed['reported_symptoms']) if quick_parsed['reported_symptoms'] else 'None detected'} | `Duration:` {quick_parsed['duration'] or 'Inferred from profile'} | `Severity:` {quick_parsed['severity']}")

        analyze_clicked = st.button("🚀 Analyze Symptoms (Run Pipeline)", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_symptom_text.strip():
            st.error("Please enter a description of your symptoms.")
        else:
            profile = PatientProfile(
                age=int(p_age),
                sex=p_sex,
                duration=p_duration,
                severity=p_severity,
                previous_conditions=[c.strip() for c in p_conditions.split(",") if c.strip()],
                current_medications=[m.strip() for m in p_meds.split(",") if m.strip()],
                allergies=[a.strip() for a in p_allergies.split(",") if a.strip()],
                is_pregnant=p_pregnant
            )

            assistant = MedicalSymptomAssistant(api_key=api_key if api_key else None)

            progress_bar = st.progress(0, text="Initializing workflow pipeline...")
            time.sleep(0.1)
            progress_bar.progress(20, text="Step 1 & 2: Patient Profile & Symptom Extraction...")
            time.sleep(0.1)
            progress_bar.progress(40, text="Step 3: Context Analysis & Red-Flag Screening...")
            time.sleep(0.1)
            progress_bar.progress(60, text="Step 4: Prompt Generation & Medical LLM Reasoning...")

            analysis_result = assistant.analyze(
                user_input=user_symptom_text,
                patient_profile=profile,
                backend=backend_param,
                model_name=model_name
            )

            progress_bar.progress(85, text="Step 5: Triage & Escalation Assessment...")
            time.sleep(0.1)
            progress_bar.progress(100, text="Analysis Complete!")
            time.sleep(0.1)
            progress_bar.empty()

            st.session_state.current_analysis = analysis_result
            st.session_state.history.append(analysis_result)

            st.success(f"Analysis successfully completed using **{analysis_result.backend_used}** in {analysis_result.execution_time:.3f}s!")

# TAB 2: Structured Medical Report
with tab2:
    current = st.session_state.current_analysis
    if not current:
        st.info("No active symptom analysis found. Please enter symptoms in Tab 1 and click 'Analyze Symptoms'.")
    else:
        # Triage Banner
        triage = current.triage
        if triage:
            if triage.color_code == "red":
                st.markdown(f'<div class="triage-badge-red">🚨 TRIAGE LEVEL: {triage.triage_level.upper()}<br><span style="font-weight:normal; font-size:1.0rem;">{triage.escalation_recommendation}</span></div>', unsafe_allow_html=True)
                if triage.red_flags_detected:
                    st.error("⚠️ Red Flag Emergency Signs Detected: " + "; ".join(triage.red_flags_detected))
            elif triage.color_code == "orange":
                st.markdown(f'<div class="triage-badge-orange">⚠️ TRIAGE LEVEL: {triage.triage_level.upper()}<br><span style="font-weight:normal; font-size:1.0rem;">{triage.escalation_recommendation}</span></div>', unsafe_allow_html=True)
            elif triage.color_code == "yellow":
                st.markdown(f'<div class="triage-badge-yellow">ℹ️ TRIAGE LEVEL: {triage.triage_level.upper()}<br><span style="font-weight:normal; font-size:1.0rem;">{triage.escalation_recommendation}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="triage-badge-green">✅ TRIAGE LEVEL: {triage.triage_level.upper()}<br><span style="font-weight:normal; font-size:1.0rem;">{triage.escalation_recommendation}</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_report_std, col_report_raw = st.columns([1.1, 0.9])

        with col_report_std:
            st.subheader("📋 Structured Clinical Analysis")
            st.caption("Conforming strictly to the Lab Manual Report Specification (Pages 2-3)")

            st.markdown(f"**Reported Symptoms:**")
            for s in current.reported_symptoms:
                st.markdown(f"- **{s}**")
            
            st.markdown(f"**Duration:** `{current.duration}`")
            st.markdown(f"**Reported Severity:** `{current.severity}`")

            st.markdown("---")
            st.subheader("🩺 Possible Conditions (Differential Considerations)")
            for idx, cond in enumerate(current.possible_conditions, 1):
                with st.expander(f"{idx}. {cond.name} — Likelihood: {cond.likelihood}", expanded=(idx == 1)):
                    st.markdown(f"**Why Considered:** {cond.why_considered}")
                    if cond.differentiating_factors:
                        st.markdown(f"**Differentiating Factors:** {cond.differentiating_factors}")

            st.markdown(f"**Why These Were Considered:**")
            st.write(current.why_considered)

            st.markdown("---")
            st.subheader("🔍 Additional Information Needed")
            for item in current.additional_information_needed:
                st.markdown(f"• {item}")

            st.markdown("---")
            st.subheader("🏥 Medical Attention")
            st.info(current.medical_attention)

            st.markdown(f"""
            <div class="disclaimer-box">
                <b>Disclaimer:</b> {current.disclaimer}
            </div>
            """, unsafe_allow_html=True)

        with col_report_raw:
            st.subheader("💻 Lab Manual Output Format (ASCII Text)")
            st.caption("Exact monospace reproduction matching Experiment 5/6 sample output")
            st.markdown(f'<div class="report-mono">{current.to_formatted_text()}</div>', unsafe_allow_html=True)

# TAB 3: Interactive Follow-Up & Dynamic Triage
with tab3:
    current = st.session_state.current_analysis
    if not current:
        st.info("Please generate an analysis in Tab 1 first.")
    else:
        st.subheader("💬 Clinical Follow-up Clarification")
        st.write("Answering these clinical follow-up questions allows the assistant to refine risk level and confirm or rule out emergency red flags.")

        with st.form("follow_up_form"):
            answers = {}
            for idx, q in enumerate(current.follow_up_questions, 1):
                answers[q] = st.text_input(f"Q{idx}: {q}", key=f"q_{idx}")

            refine_submitted = st.form_submit_button("Submit Follow-Up Details", type="primary")

        if refine_submitted:
            answered_any = any(v.strip() for v in answers.values())
            if not answered_any:
                st.warning("Please provide answers to at least one question to refine the assessment.")
            else:
                combined_answers = " ".join([f"Q: {k} A: {v}" for k, v in answers.items() if v.strip()])
                st.success("Follow-up data integrated into clinical context.")
                
                # Check for emergency keywords in answers
                if any(w in combined_answers.lower() for w in ["yes", "cannot breathe", "chest pain", "high", "severe", "blood", "stiff neck"]):
                    st.error("⚠️ Follow-up response indicates acute red flag symptoms. Escalating to **Urgent / Emergency Evaluation**.")
                else:
                    st.info("Follow-up responses indicate stable presentation. Routine conservative self-care and monitoring remain appropriate.")

# TAB 4: System Workflow & Traceability
with tab4:
    st.subheader("🔄 End-to-End System Workflow")
    st.markdown("""
    The system strictly executes the multi-stage pipeline defined in the syllabus:
    """)

    st.markdown("""
```
User Input (Natural Language)
    ↓
Patient Profile (Demographics & Medical History)
    ↓
Symptom Extraction (NLP & Clinical Catalog Entity Mapping)
    ↓
Context Analysis (Vulnerability & Red Flag Screening)
    ↓
Prompt Generation (Safety Guardrails, Structured Schema)
    ↓
GPT-4 / Multi-Backend LLM Execution
    ↓
Symptom Analysis:
 ├── Extracted Symptoms
 ├── Possible Conditions (with Rationale & Distinctions)
 └── Follow-up Clarifying Questions
    ↓
Risk / Escalation Triage Assessment
    ↓
Structured Response Formation
    ↓
Disclaimer Enforcement
    ↓
Save Report (TXT, JSON, Markdown)
```
    """)

    current = st.session_state.current_analysis
    if current:
        st.markdown("---")
        st.subheader("📊 Execution Metadata & Raw JSON")
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        meta_col1.metric("Execution Engine", current.backend_used)
        meta_col2.metric("Processing Time", f"{current.execution_time:.3f} s")
        meta_col3.metric("Candidate Conditions", len(current.possible_conditions))

        with st.expander("View Full JSON Model Output"):
            st.json(current.model_dump())

# TAB 5: Export & Session History
with tab5:
    st.subheader("💾 Save & Export Reports")
    current = st.session_state.current_analysis
    if not current:
        st.info("No active report available to export.")
    else:
        st.write("Download this structured consultation report in university laboratory standard formats:")

        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.download_button(
            label="📥 Download as TXT (Lab Format)",
            data=current.to_formatted_text(),
            file_name=f"medical_report_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )
        col_d2.download_button(
            label="📥 Download as JSON Data",
            data=current.model_dump_json(indent=2),
            file_name=f"medical_report_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )
        col_d3.download_button(
            label="📥 Download as Markdown",
            data=f"# Medical Symptom Analysis Report\n\n```\n{current.to_formatted_text()}\n```",
            file_name=f"medical_report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("🕒 Session History")
    if not st.session_state.history:
        st.caption("No consultations in this session yet.")
    else:
        for idx, h in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Consultation #{len(st.session_state.history) - idx + 1} — Symptoms: {', '.join(h.reported_symptoms)} ({h.timestamp})"):
                st.text(h.to_formatted_text())
