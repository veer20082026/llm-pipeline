"""
LLM Pipeline - Streamlit Application.

Interactive visual dashboard demonstrating an end-to-end multi-stage LLM pipeline:
Stage 1: Input Processing
Stage 2: Prompt Engineering
Stage 3: LLM Integration
Stage 4: Response Processing
Stage 5: Output Generation
"""

from __future__ import annotations

import time
import streamlit as st

from config.settings import (
    OUTPUT_STYLES,
    SAMPLE_INPUTS,
    SUPPORTED_MODELS,
    PipelineStage,
    StageStatus,
    get_config,
)
from pipeline.pipeline_runner import PipelineRunner
from pipeline.prompt_builder import PERSONAS

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Pipeline - End-to-End AI Engineering",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern clean aesthetic and status badges
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stage-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border: 1px solid #E2E8F0;
        background-color: #F8FAFC;
    }
    .status-badge {
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-pending {
        background-color: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
    }
    .status-running {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #93C5FD;
    }
    .status-completed {
        background-color: #F0FDF4;
        color: #16A34A;
        border: 1px solid #86EFAC;
    }
    .status-failed {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }
    .metric-box {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_style_safe=True,
)


def get_status_badge_html(status: StageStatus) -> str:
    """Generate HTML badge for stage status."""
    if status == StageStatus.PENDING:
        return '<span class="status-badge status-pending">⏳ Pending</span>'
    elif status == StageStatus.RUNNING:
        return '<span class="status-badge status-running">🔄 Running</span>'
    elif status == StageStatus.COMPLETED:
        return '<span class="status-badge status-completed">✅ Completed</span>'
    elif status == StageStatus.FAILED:
        return '<span class="status-badge status-failed">❌ Failed</span>'
    return '<span class="status-badge status-pending">⏳ Pending</span>'


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "execution_bundle" not in st.session_state:
    st.session_state.execution_bundle = None

if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []

if "stage_states" not in st.session_state:
    st.session_state.stage_states = {
        PipelineStage.INPUT_PROCESSING.value: {
            "status": StageStatus.PENDING,
            "latency": 0.0,
            "error": None,
        },
        PipelineStage.PROMPT_ENGINEERING.value: {
            "status": StageStatus.PENDING,
            "latency": 0.0,
            "error": None,
        },
        PipelineStage.LLM_INTEGRATION.value: {
            "status": StageStatus.PENDING,
            "latency": 0.0,
            "error": None,
        },
        PipelineStage.RESPONSE_PROCESSING.value: {
            "status": StageStatus.PENDING,
            "latency": 0.0,
            "error": None,
        },
        PipelineStage.OUTPUT_GENERATION.value: {
            "status": StageStatus.PENDING,
            "latency": 0.0,
            "error": None,
        },
    }

# -----------------------------------------------------------------------------
# SIDEBAR: API & MODEL CONFIGURATION
# -----------------------------------------------------------------------------
config = get_config()

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Configure LLM credentials and inference hyperparameters.")

    # API Key Input
    env_key = config.api_key or ""
    api_key_input = st.text_input(
        "API Key (OpenAI / Provider)",
        value=env_key,
        type="password",
        help="Reads OPENAI_API_KEY from .env by default, or paste your API key here.",
    )

    # Base URL Input (for Groq, Ollama, DeepSeek, Together, etc.)
    base_url_input = st.text_input(
        "Custom Base URL (Optional)",
        value=config.base_url or "",
        help="Leave blank for OpenAI default (https://api.openai.com/v1). "
        "Supports Groq (https://api.groq.com/openai/v1), Ollama (http://localhost:11434/v1), DeepSeek, etc.",
    )

    st.markdown("---")
    st.subheader("🤖 Model & Inference")

    # Model Selection
    selected_model = st.selectbox(
        "Select Model",
        options=SUPPORTED_MODELS,
        index=0,
        help="Choose standard OpenAI models or provider-compatible model IDs.",
    )

    custom_model_input = st.text_input(
        "Or enter custom model ID:",
        value="",
        placeholder="e.g. gpt-4o, llama3-70b-8192, qwen2.5:7b",
        help="Overrides dropdown if provided.",
    )
    active_model = custom_model_input.strip() if custom_model_input.strip() else selected_model

    # Hyperparameters
    col_temp, col_tokens = st.columns(2)
    with col_temp:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=config.default_temperature,
            step=0.05,
            help="Higher values make output more random, lower values more deterministic.",
        )
    with col_tokens:
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=config.default_max_tokens,
            step=100,
            help="Maximum token count allowed for the LLM response.",
        )

    st.markdown("---")
    st.markdown("### 📘 Pipeline Architecture")
    st.caption(
        "1. **Input Processing**: Clean & validate text\n"
        "2. **Prompt Engineering**: System instructions & style\n"
        "3. **LLM Integration**: API call & latency timing\n"
        "4. **Response Processing**: Clean & extract insights\n"
        "5. **Output Generation**: Metrics & formatting"
    )

# -----------------------------------------------------------------------------
# MAIN DASHBOARD CONTENT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ LLM Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An interactive, multi-stage Large Language Model pipeline demonstrating '
    'input sanitization, prompt construction, live LLM inference, response analysis, and output delivery.</div>',
    unsafe_allow_html=True,
)

# Row for Sample Input loaders and Actions
col_samples, col_actions = st.columns([3, 1])

with col_samples:
    st.write("**Load Sample Input:**")
    sample_cols = st.columns(len(SAMPLE_INPUTS))
    for idx, (sample_name, sample_text) in enumerate(SAMPLE_INPUTS.items()):
        with sample_cols[idx]:
            if st.button(sample_name, use_container_width=True):
                st.session_state.user_input = sample_text
                st.rerun()

with col_actions:
    st.write("**Actions:**")
    if st.button("🧹 Clear All", use_container_width=True):
        st.session_state.user_input = ""
        st.session_state.execution_bundle = None
        st.session_state.pipeline_logs = []
        for s in st.session_state.stage_states:
            st.session_state.stage_states[s] = {
                "status": StageStatus.PENDING,
                "latency": 0.0,
                "error": None,
            }
        st.rerun()

# Text Input Area
user_text = st.text_area(
    "Input Text to Process:",
    value=st.session_state.user_input,
    height=140,
    placeholder="Type or paste your prompt/text here (e.g., questions, meeting notes, code to analyze, or articles to summarize)...",
    key="input_box",
)

# Update state if user edited manually
if user_text != st.session_state.user_input:
    st.session_state.user_input = user_text

# Prompt Engineering Configuration Section
with st.expander("🛠️ Prompt Engineering & Persona Settings", expanded=False):
    pe_col1, pe_col2 = st.columns(2)
    with pe_col1:
        selected_style = st.selectbox(
            "Desired Output Style:",
            options=list(OUTPUT_STYLES.keys()),
            index=0,
            help="Controls the tone, formatting, and conciseness of the LLM response.",
        )
        st.caption(f"ℹ️ {OUTPUT_STYLES[selected_style]['description']}")

    with pe_col2:
        selected_persona = st.selectbox(
            "AI Persona / Role:",
            options=list(PERSONAS.keys()),
            index=0,
            help="Defines the perspective, expertise, and voice of the model.",
        )
        st.caption(f"🎭 {PERSONAS[selected_persona]}")

    task_context_input = st.text_input(
        "Optional Task Context / Goal:",
        placeholder="e.g. Target audience: 10-year-old students / Code must be Python 3.12 compatible",
        help="Injected into the user prompt frame as structured task constraints.",
    )

    custom_system_input = st.text_area(
        "Custom System Instructions (Optional):",
        placeholder="Any additional instructions or guardrails you want to inject into the system prompt...",
        height=80,
    )

# -----------------------------------------------------------------------------
# RUN PIPELINE EXECUTION
# -----------------------------------------------------------------------------
run_col1, run_col2 = st.columns([1, 4])
with run_col1:
    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# PIPELINE STAGES VISUAL REPRESENTATION
# -----------------------------------------------------------------------------
st.markdown("### 📊 Pipeline Stages Execution")

stage_col1, stage_col2, stage_col3, stage_col4, stage_col5 = st.columns(5)
stage_column_map = {
    PipelineStage.INPUT_PROCESSING.value: stage_col1,
    PipelineStage.PROMPT_ENGINEERING.value: stage_col2,
    PipelineStage.LLM_INTEGRATION.value: stage_col3,
    PipelineStage.RESPONSE_PROCESSING.value: stage_col4,
    PipelineStage.OUTPUT_GENERATION.value: stage_col5,
}

stage_placeholders = {}
for stage_name, col in stage_column_map.items():
    with col:
        stage_placeholders[stage_name] = st.empty()


def render_stage_cards(current_stage_states: dict):
    """Render the 5 stage cards with their current status badges and latencies."""
    short_names = {
        PipelineStage.INPUT_PROCESSING.value: "1. Input Processing",
        PipelineStage.PROMPT_ENGINEERING.value: "2. Prompt Engineering",
        PipelineStage.LLM_INTEGRATION.value: "3. LLM Integration",
        PipelineStage.RESPONSE_PROCESSING.value: "4. Response Processing",
        PipelineStage.OUTPUT_GENERATION.value: "5. Output Generation",
    }
    for s_name, placeholder in stage_placeholders.items():
        state = current_stage_states.get(s_name, {"status": StageStatus.PENDING, "latency": 0.0})
        badge = get_status_badge_html(state["status"])
        latency_str = f"{round(state['latency'], 3)}s" if state["latency"] > 0 else "—"

        card_html = f"""
        <div class="stage-card">
            <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 6px;">
                {short_names.get(s_name, s_name)}
            </div>
            <div style="margin-bottom: 6px;">{badge}</div>
            <div style="font-size: 0.8rem; color: #64748B;">⏱️ Time: <b>{latency_str}</b></div>
        </div>
        """
        placeholder.markdown(card_html, unsafe_allow_html=True)


# Render initial cards
render_stage_cards(st.session_state.stage_states)

# Status notification placeholder
status_msg_placeholder = st.empty()

# -----------------------------------------------------------------------------
# RUN PIPELINE LOGIC (When Run button is pressed)
# -----------------------------------------------------------------------------
if run_button:
    if not st.session_state.user_input.strip():
        st.warning("⚠️ Please enter some text or click a sample button before running the pipeline.")
    elif not api_key_input.strip():
        st.error(
            "🔑 API Key is required. Please add OPENAI_API_KEY in `.env` or paste your key into the sidebar."
        )
    else:
        # Reset stage states
        for s in st.session_state.stage_states:
            st.session_state.stage_states[s] = {
                "status": StageStatus.PENDING,
                "latency": 0.0,
                "error": None,
            }
        render_stage_cards(st.session_state.stage_states)
        status_msg_placeholder.info("🚀 Pipeline execution started...")

        # Initialize runner with sidebar config
        app_config = get_config()
        app_config.api_key = api_key_input.strip()
        app_config.base_url = base_url_input.strip() or None

        runner = PipelineRunner(config=app_config)

        # Execute generator
        generator = runner.run_step_by_step(
            user_input=st.session_state.user_input,
            model=active_model,
            style=selected_style,
            persona=selected_persona,
            custom_system_instruction=custom_system_input,
            task_context=task_context_input,
            temperature=temperature,
            max_tokens=max_tokens,
            runtime_api_key=api_key_input.strip(),
        )

        final_bundle = None
        for snapshot in generator:
            cur_stage = snapshot["current_stage"].value
            cur_status = snapshot["stage_status"]
            stage_records = snapshot["stage_records"]

            # Update our session state tracking
            for s_name, record in stage_records.items():
                st.session_state.stage_states[s_name] = {
                    "status": record.status,
                    "latency": record.latency_seconds,
                    "error": record.error_message,
                }

            # Update UI cards
            render_stage_cards(st.session_state.stage_states)
            st.session_state.pipeline_logs = snapshot["logs"]

            if snapshot.get("bundle"):
                final_bundle = snapshot["bundle"]

            time.sleep(0.05)  # Smooth UI transition

        st.session_state.execution_bundle = final_bundle

        if final_bundle and final_bundle.success:
            status_msg_placeholder.success(
                f"✅ Pipeline completed successfully in {round(final_bundle.total_execution_time_seconds, 3)} seconds!"
            )
        else:
            err = final_bundle.error_summary if final_bundle else "Pipeline execution failed."
            status_msg_placeholder.error(f"❌ Pipeline Failed: {err}")

# -----------------------------------------------------------------------------
# FINAL GENERATED OUTPUT & COMPARISON
# -----------------------------------------------------------------------------
bundle = st.session_state.execution_bundle

if bundle:
    st.markdown("---")
    st.markdown("### 🎯 Pipeline Results")

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric(
            label="Total Latency",
            value=f"{round(bundle.total_execution_time_seconds, 2)}s",
        )
    with m_col2:
        tokens_val = bundle.llm_result.total_tokens if bundle.llm_result else 0
        st.metric(label="Total Tokens", value=tokens_val)
    with m_col3:
        word_count = bundle.response_result.word_count if bundle.response_result else 0
        st.metric(label="Output Words", value=word_count)
    with m_col4:
        speed = bundle.response_result.tokens_per_second if bundle.response_result else 0.0
        st.metric(label="Generation Speed", value=f"{speed} t/s")
    with m_col5:
        reading_time = (
            bundle.response_result.estimated_reading_time_minutes
            if bundle.response_result
            else 0.0
        )
        st.metric(label="Reading Time", value=f"~{reading_time} min")

    # Tabs for Output, Comparison, Prompt Inspection, and Export
    tab_output, tab_compare, tab_inspector, tab_logs = st.tabs(
        ["📝 Generated Output", "⚖️ Input vs Output", "🔍 Pipeline Inspector", "📜 Execution Logs"]
    )

    with tab_output:
        if bundle.success:
            st.markdown("#### Final LLM Response:")
            st.markdown(bundle.final_output)

            # Copy and Download options
            st.markdown("---")
            act_col1, act_col2 = st.columns([1, 4])
            with act_col1:
                st.download_button(
                    label="📥 Download Output (.txt)",
                    data=bundle.final_output,
                    file_name="llm_pipeline_output.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with act_col2:
                # Text for easy clipboard copy
                with st.expander("📋 View Raw Text for Copying"):
                    st.code(bundle.final_output, language=None)
        else:
            st.error(f"Execution Error: {bundle.error_summary}")
            if bundle.stage_records:
                failed_stages = [
                    (s, rec.error_message)
                    for s, rec in bundle.stage_records.items()
                    if rec.status == StageStatus.FAILED
                ]
                for stage_name, err in failed_stages:
                    st.write(f"**{stage_name} Failure Details:** `{err}`")

    with tab_compare:
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("**Original User Input:**")
            st.info(bundle.original_input)
            if bundle.input_result:
                st.caption(
                    f"Characters: {bundle.input_result.character_count} | Words: {bundle.input_result.word_count} | "
                    f"Estimated Tokens: {bundle.input_result.estimated_tokens}"
                )
        with comp_col2:
            st.markdown("**Processed LLM Output:**")
            if bundle.success:
                st.success(bundle.final_output)
                if bundle.response_result:
                    st.caption(
                        f"Words: {bundle.response_result.word_count} | Sentences: {bundle.response_result.sentence_count} | "
                        f"Code Blocks: {bundle.response_result.code_blocks_count}"
                    )
            else:
                st.warning("No output produced due to stage failure.")

    with tab_inspector:
        st.markdown("#### Intermediate Stage Artifacts")

        if bundle.input_result:
            with st.expander("Stage 1 Artifact: Sanitized Input & Metrics"):
                st.json(bundle.input_result.to_dict())

        if bundle.prompt_result:
            with st.expander("Stage 2 Artifact: Formatted Chat Messages"):
                st.write("**Messages sent to API:**")
                st.json(bundle.prompt_result.messages)

        if bundle.llm_result:
            with st.expander("Stage 3 Artifact: Raw LLM API Response & Tokens"):
                st.json(bundle.llm_result.to_dict())

        if bundle.response_result:
            with st.expander("Stage 4 Artifact: Extracted Insights & Metrics"):
                st.json(bundle.response_result.to_dict())

        # Full execution report export
        st.markdown("---")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            text_report = runner.output_handler.format_as_text(bundle)
            st.download_button(
                label="📄 Download Full Execution Report (.txt)",
                data=text_report,
                file_name="llm_pipeline_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with exp_col2:
            json_report = runner.output_handler.format_as_json(bundle)
            st.download_button(
                label="📦 Download Execution Bundle (.json)",
                data=json_report,
                file_name="llm_pipeline_bundle.json",
                mime="application/json",
                use_container_width=True,
            )

    with tab_logs:
        st.markdown("#### Real-time Execution Logs")
        if st.session_state.pipeline_logs:
            log_content = "\n".join(st.session_state.pipeline_logs)
            st.code(log_content, language="bash")
        else:
            st.caption("No logs recorded yet.")

elif st.session_state.pipeline_logs:
    # If a failure occurred before bundle completion, still show logs
    st.markdown("---")
    st.markdown("### 📜 Pipeline Execution Logs")
    st.code("\n".join(st.session_state.pipeline_logs), language="bash")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "⚡ LLM Pipeline Architecture & Dashboard &bull; Modular Python &bull; Streamlit &bull; OpenAI SDK"
    "</div>",
    unsafe_allow_html=True,
)
