"""Streamlit frontend for Audit Tool."""

import streamlit as st
import requests
import json
from pathlib import Path

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="Audit Checkpoint Generator",
    page_icon="📋",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .checkpoint-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .checkpoint-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .checkpoint-prompt {
        color: #333;
        line-height: 1.6;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">📋 Audit Checkpoint Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered verification checkpoint generation for audit processes</div>', unsafe_allow_html=True)
    
    # Check backend status
    if not check_backend_health():
        st.error("⚠️ Backend server is not running. Please start the backend with: `python backend/main.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["Generate Checkpoints", "Build Knowledge Base", "Database Info"],
            index=0
        )
        
        st.divider()
        
        # RAG settings
        if mode == "Generate Checkpoints":
            st.subheader("RAG Settings")
            use_rag = st.checkbox("Use RAG (Retrieval)", value=True, help="Use similar documents from knowledge base")
            process_type = st.selectbox(
                "Process Type (Template)",
                options=[
                    "Auto (detect from document)",
                    "Verification",
                    "Joint Review",
                    "Configuration Management",
                ],
                index=0,
                help="Choose a process type to force the original checkpoint template. Auto will infer from filename/content."
            )
            num_checkpoints = st.slider("Number of Checkpoints", 3, 10, 5, help="For templated processes, the system will use the canonical number of checkpoints.")
        
        st.divider()
        st.caption("🔧 Powered by HuggingFace & ChromaDB")
    
    # Main content
    if mode == "Generate Checkpoints":
        generate_checkpoints_mode(use_rag, num_checkpoints, process_type)
    
    elif mode == "Build Knowledge Base":
        build_knowledge_base_mode()
    
    elif mode == "Database Info":
        show_database_info()


def generate_checkpoints_mode(use_rag, num_checkpoints, process_type):
    """Generate checkpoints mode."""
    
    st.header("📤 Upload Process Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Upload a process document (DOCX, PDF, XLSX, XML, PPTX) to generate verification checkpoints.")
    
    with col2:
        st.metric("Checkpoints to Generate", num_checkpoints)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["docx", "pdf", "xlsx", "xls", "xml", "pptx", "txt"],
        help="Supported formats: DOCX, PDF, XLSX, XML, PPTX, TXT"
    )
    
    if uploaded_file is not None:
        # Show file details
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size} bytes)")

        ingest_to_kb = False
        if use_rag:
            ingest_to_kb = st.checkbox(
                "Also add this document to the Knowledge Base",
                value=False,
                help="If enabled, this uploaded document will be chunked/embedded and stored so future runs can retrieve from it."
            )
        
        # Generate button
        if st.button("🚀 Generate Checkpoints", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing document and generating checkpoints..."):
                try:
                    # Prepare request
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    params = {
                        "use_rag": use_rag,
                        "num_checkpoints": num_checkpoints,
                        "ingest_to_kb": ingest_to_kb,
                        "process_type": _map_process_type(process_type),
                    }
                    
                    # Call API
                    response = requests.post(
                        f"{BACKEND_URL}/generate-checkpoints",
                        files=files,
                        params=params,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display success message
                        st.success(f"✅ Successfully generated {result['num_checkpoints']} checkpoints!")
                        
                        # Display info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Checkpoints Generated", result['num_checkpoints'])
                        with col2:
                            st.metric("RAG Used", "Yes" if result['used_rag'] else "No")
                        with col3:
                            st.metric("Context Chunks", result.get('num_context_chunks', 0))

                        detected_profile = result.get("process_profile")
                        detected_type = result.get("process_type")
                        if detected_profile or detected_type:
                            st.caption(f"Template: {detected_profile or 'None'} (process_type={detected_type or 'auto'})")
                        
                        st.divider()
                        
                        # Show retrieved context chunks (for RAG traceability)
                        retrieved = result.get("retrieved_chunks", [])
                        if isinstance(retrieved, list) and retrieved:
                            with st.expander("🔎 Retrieved Context Chunks (RAG Trace)", expanded=False):
                                for i, ch in enumerate(retrieved, 1):
                                    st.markdown(f"**Chunk {i}**")
                                    st.markdown(f"- **source**: {ch.get('source', 'Unknown')}")
                                    st.markdown(f"- **score**: {ch.get('score', '')}")
                                    if ch.get("id"):
                                        st.markdown(f"- **id**: {ch.get('id')}")
                                    st.text_area(
                                        label=f"Preview {i}",
                                        value=ch.get("text_preview", ""),
                                        height=140,
                                        key=f"ctx_preview_{i}"
                                    )

                        # Display checkpoints
                        st.header("📋 Generated Checkpoints")
                        
                        # If no checkpoints were parsed, show raw output
                        if result['num_checkpoints'] == 0:
                            st.warning("⚠️ No checkpoints were parsed. Showing raw LLM output:")
                            with st.expander("View Raw Output", expanded=True):
                                st.text_area("Raw LLM Response", result.get('raw_output', 'No output available'), height=400)
                        
                        for idx, checkpoint in enumerate(result['checkpoints'], start=1):
                            with st.expander(
                                f"**Checkpoint {idx}: {checkpoint.get('process_phase_reference', '').strip() or 'Checkpoint'}**",
                                expanded=True
                            ):
                                # New schema fields (fallback to legacy keys for backward compatibility)
                                st.markdown(f"**Process Phase Reference:** {checkpoint.get('process_phase_reference', checkpoint.get('title', ''))}")
                                st.markdown(f"**Standard Clause Reference:** {checkpoint.get('standard_clause_reference', checkpoint.get('standard_reference', 'TBD'))}")
                                st.markdown(f"**Verification Section:** {checkpoint.get('verification_section', checkpoint.get('verification_objective', ''))}")
                                st.markdown("---")
                                st.markdown(f"**Prompt:**")
                                st.markdown(checkpoint['prompt'])
                        
                        # Export all checkpoints
                        st.divider()
                        def _format_cp(cp, i: int) -> str:
                            phase = cp.get("process_phase_reference", "") or cp.get("title", f"Checkpoint {i}")
                            clause = cp.get("standard_clause_reference", cp.get("standard_reference", "ISO 9001 TBD"))
                            section = cp.get("verification_section", cp.get("verification_objective", ""))
                            prompt = cp.get("prompt", "")
                            return f"{phase}, {clause}: {section}\n\nPrompt: {prompt}"

                        all_checkpoints_text = "\n\n".join(
                            [_format_cp(cp, i + 1) for i, cp in enumerate(result["checkpoints"])]
                        )
                        st.download_button(
                            label="📥 Download All Checkpoints",
                            data=all_checkpoints_text,
                            file_name=f"checkpoints_{uploaded_file.name}.txt",
                            mime="text/plain",
                            type="primary",
                            use_container_width=True
                        )
                        
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Error: {error_detail}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The document might be too large or the server is busy.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def build_knowledge_base_mode():
    """Build knowledge base mode."""
    
    st.header("📚 Build Knowledge Base")

    st.info("Upload reference documents to build a knowledge base. These documents will be used as context when generating checkpoints.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a reference document",
        type=["docx", "pdf", "xlsx", "xls", "xml", "pptx", "txt"],
        help="Upload documents related to audit processes, standards, or templates"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File selected: **{uploaded_file.name}**")
        
        # Ingest button
        if st.button("📥 Ingest Document", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing and ingesting document..."):
                try:
                    # Prepare request
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    
                    # Call API
                    response = requests.post(
                        f"{BACKEND_URL}/ingest-documents",
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Document ingested successfully!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Chunks Created", result['chunks_created'])
                        with col2:
                            st.metric("Total Documents in DB", result['total_documents_in_db'])
                        
                        # Show metadata
                        with st.expander("📊 Document Metadata"):
                            st.json(result['metadata'])
                        
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Error: {error_detail}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def _map_process_type(label: str) -> str:
    """
    Map UI label to backend process_type.
    """
    if label == "Verification":
        return "verification"
    if label == "Joint Review":
        return "joint_review"
    if label == "Configuration Management":
        return "configuration_management"
    return "auto"


def show_database_info():
    """Show database information."""
    
    st.header("🗄️ Database Information")
    
    try:
        response = requests.get(f"{BACKEND_URL}/database-info", timeout=5)
        
        if response.status_code == 200:
            info = response.json()
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Vector Store Documents", info['vector_store_documents'])
                st.metric("Embedding Model", info['embedding_model'])
            
            with col2:
                st.metric("BM25 Indexed Documents", info['bm25_indexed_documents'])
                st.metric("LLM Model", info['llm_model'])
            
            st.divider()
            
            # Reset database button (dangerous action)
            with st.expander("⚠️ Danger Zone"):
                st.warning("**Warning:** Resetting the database will delete all ingested documents.")
                if st.button("🗑️ Reset Database", type="secondary"):
                    with st.spinner("Resetting database..."):
                        reset_response = requests.delete(f"{BACKEND_URL}/reset-database", timeout=10)
                        if reset_response.status_code == 200:
                            st.success("✅ Database reset successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Error resetting database")
        else:
            st.error("❌ Error fetching database info")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
