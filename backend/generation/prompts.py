"""Prompts for checkpoint generation."""

from __future__ import annotations

from typing import Optional

from .process_profiles import ProcessProfile, CheckpointSlot
from backend.config import config


def _extract_headings_and_excerpts(text: str, max_items: int) -> list[tuple[str, str]]:
    """
    Best-effort extraction of (heading, excerpt) pairs from plain text.
    Used to create a generic template for unseen processes.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines()]
    pairs: list[tuple[str, str]] = []

    def is_heading(line: str) -> bool:
        if not line:
            return False
        if len(line) > 80:
            return False
        # Numbered headings: "1.", "1.2", etc.
        if any(line.startswith(p) for p in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            return True
        # Ends with colon and short
        if line.endswith(":") and len(line.split()) <= 8:
            return True
        # All-caps short headings
        if line.isupper() and len(line) >= 4 and len(line.split()) <= 6:
            return True
        return False

    i = 0
    seen = set()
    while i < len(lines) and len(pairs) < max_items:
        line = lines[i]
        if is_heading(line):
            heading = line.rstrip(":").strip()
            key = heading.lower()
            if key in seen:
                i += 1
                continue
            seen.add(key)

            # Take next few non-empty lines as excerpt
            excerpt_lines = []
            j = i + 1
            while j < len(lines) and len(excerpt_lines) < 4:
                if lines[j] and not is_heading(lines[j]):
                    excerpt_lines.append(lines[j])
                j += 1
            excerpt = " ".join(excerpt_lines).strip()
            if excerpt:
                pairs.append((heading, excerpt[:600]))
        i += 1

    return pairs


def get_generic_profiled_structured_output_prompt(
    num_checkpoints: int,
    allowed_phases: list[str],
) -> str:
    """
    Strict JSON output for generic (unseen) processes.
    Locks process_phase_reference values to a detected list of phases/headings.
    """
    phases = "; ".join([f'"{p}"' for p in allowed_phases])
    return f"""You are an expert Quality Assurance auditor.

Return ONLY valid JSON. No markdown. No commentary. No code fences.

The JSON MUST match this schema exactly:
{{
  "checkpoints": [
    {{
      "process_phase_reference": "string",
      "standard_clause_reference": "string",
      "verification_section": "string",
      "prompt": "string"
    }}
  ]
}}

Hard Rules:
- Output exactly {num_checkpoints} items in "checkpoints".
- All values must be strings.
- process_phase_reference MUST be one of: {phases}.
- Do NOT output evidence or artifacts/templates fields.
"""


def build_generic_profiled_structured_user_prompt(
    process_document: str,
    relevant_context: str,
    num_checkpoints: int,
    heading_excerpt_pairs: list[tuple[str, str]],
) -> str:
    """
    Generic user prompt: one checkpoint per detected heading/excerpt.
    """
    items = "\n".join(
        [f'- Phase: "{h}"\n  Excerpt: "{ex}"' for (h, ex) in heading_excerpt_pairs]
    )
    prompt = f"""Generate exactly {num_checkpoints} checkpoints for the uploaded process document.

You MUST produce exactly one checkpoint per phase below (same order as listed).
Each checkpoint must be grounded in the provided excerpt (do not invent requirements).

Phases and excerpts:
{items}

Rules:
- standard_clause_reference: use "ISO 9001 TBD" unless the document explicitly provides a clause.
- verification_section: a particular requirement/statement from the excerpt, phrased as "Verify that ..." or "Show ..." depending on the excerpt wording.
- prompt: expand the excerpt into a short, auditee-facing request describing what to show/demonstrate; do NOT list evidence/templates.

PROCESS DOCUMENT:
{process_document}
"""
    if relevant_context:
        prompt += f"""

RELEVANT CONTEXT (retrieved chunks):
{relevant_context}
"""
    prompt += "\nReturn the JSON now."
    return prompt

def get_system_prompt(num_checkpoints: int = 5) -> str:
    """
    Get system prompt with specified number of checkpoints.
    
    Args:
        num_checkpoints: Number of checkpoints to generate
        
    Returns:
        System prompt string
    """
    return f"""You are an expert Quality Assurance auditor specializing in ISO 9001, ISO 13485, ASPICE, and other quality management standards.

Your task is to analyze process documents and generate exactly {num_checkpoints} verification checkpoints that can be used during audits.

Each checkpoint must follow this structure:

**[Number]. [Title], [Standard Reference]:** [What needs to be verified]

**Prompt:** [Detailed verification instructions explaining HOW to verify, WHAT evidence to look for, and WHAT template/document to use]

Requirements for each checkpoint:
1. Must reference specific standards (ISO 9001, ASPICE, etc.) or mark as "TBD" if standard is unclear
2. Must clearly state WHAT needs to be verified (the verification objective)
3. Must include a detailed Prompt section explaining:
   - How to perform the verification
   - What evidence is required (at least 2 evidences when applicable)
   - What templates or documents should be used
   - Any specific processes or procedures to follow
4. Must be actionable and measurable
5. Should cover different phases of the process (planning, specification, execution, evaluation, etc.)

Focus on:
- Process phase references (e.g., design phase, test phase, review phase)
- Standard clause requirements and compliance
- Verification methodologies (review, testing, inspection)
- Evidence and documentation requirements
- Template usage and formalization
- Issue management and tracking

Generate checkpoints that are comprehensive, professional, and directly applicable for auditing the provided process document."""

SYSTEM_PROMPT = get_system_prompt(5)


FEW_SHOT_EXAMPLES = """
Example 1:
**1. Verification Planning, ISO 9001 TBD:** Verify that the verification strategy has been planned and the proper template has been used for the related evidence

**Prompt:** The verification strategy and the related plan shall be defined for each phase of the product design and development lifecycle. All the necessary information shall be formalized in the Verification Plan.

Example 2:
**2. Verification Specification, ISO 9001 TBD:** Verify that verification review checklists have been specified for each work product foreseen in the design phases (at least 2 evidences) and verify that the proper template has been used for the related evidence

**Prompt:** In the design phases, the verification is aimed to evaluate the work products (i.e. requirements specifications, software code). The verification reviews shall be based on the identification of specific checklist for each work product and those checklists shall be formalized in the Verification Review Specification.

Example 3:
**3. Verification Execution and Evaluation, ISO 9001 TBD:** Verify that the verification review is conducted by a different person than the author(s) of the work product to be verified, that it is clearly formalized using the proper template and that it is tracked in the Issue Management System

**Prompt:** The results of the verification review shall be reported in the Verification Review Report with a clear statement of whether the verification passed or not. In case of failed checks, the issues shall be managed according to the "Issues from Review" process. Verification Review Report shall be formalized in a new ticket on the Issue Management System. It is recommended that the verification is performed by a different person than the author(s) of the work product to be verified.
"""


def build_user_prompt(process_document: str, relevant_context: str = "", num_checkpoints: int = 5) -> str:
    """
    Build user prompt for checkpoint generation.
    
    Args:
        process_document: The main process document text
        relevant_context: Additional relevant context from similar documents
        num_checkpoints: Number of checkpoints to generate
        
    Returns:
        Formatted user prompt
    """
    prompt = f"""Based on the following process document, generate exactly {num_checkpoints} verification checkpoints following the structure and guidelines provided.

{'=' * 80}
PROCESS DOCUMENT:
{'=' * 80}
{process_document}
"""
    
    if relevant_context:
        prompt += f"""

{'=' * 80}
RELEVANT CONTEXT FROM SIMILAR PROCESSES:
{'=' * 80}
{relevant_context}
"""
    
    prompt += f"""

{'=' * 80}
TASK:
{'=' * 80}
Generate exactly {num_checkpoints} verification checkpoints for this process. Ensure each checkpoint:
1. References appropriate standards (ISO 9001, ASPICE, etc.)
2. Covers different phases/aspects of the process
3. Includes clear verification objectives
4. Provides detailed prompts with specific instructions
5. Mentions required evidence and templates

Output the {num_checkpoints} checkpoints now:
"""
    
    return prompt


def get_structured_output_prompt(num_checkpoints: int = 5) -> str:
    """
    System prompt that forces strict JSON output.
    """
    return f"""You are an expert Quality Assurance auditor.

Return ONLY valid JSON. No markdown. No commentary. No code fences.

The JSON MUST match this schema exactly:
{{
  "checkpoints": [
    {{
      "process_phase_reference": "e.g., Verification Planning / Verification Specification / Verification Execution and Evaluation",
      "standard_clause_reference": "e.g., ISO 9001:2015 § 7.5.3 (or TBD)",
      "verification_section": "What needs to be verified (derived from the uploaded document)",
      "prompt": "Detailed how-to verify. Do NOT include evidence/artifacts lists."
    }}
  ]
}}

Rules:
- Output exactly {num_checkpoints} items in "checkpoints".
- All values must be strings.
- Do NOT output evidence or artifacts/templates fields.
- Use the SAME wording style as the examples: concise verification_section + actionable prompt.
"""


def _format_slot_requirements(slots: list[CheckpointSlot]) -> str:
    lines = []
    for s in slots:
        lines.append(
            f'- #{s.number}: process_phase_reference="{s.process_phase_reference}"; '
            f'verification_section="{s.verification_section}"; '
            f'canonical_prompt="{s.canonical_prompt}"'
        )
    return "\n".join(lines)


def get_profiled_structured_output_prompt(profile: ProcessProfile) -> str:
    """
    System prompt that forces strict JSON output AND forces canonical slots.
    This is the main lever to match the original checkpoint templates.
    """
    n = len(profile.slots)
    allowed_starts = ", ".join([f'"{s}"' for s in profile.allowed_prompt_starts])
    slot_requirements = _format_slot_requirements(list(profile.slots))
    clause_lock_line = ""
    if bool(getattr(config, "PROFILE_LOCK_STANDARD_CLAUSE", True)):
        clause_lock_line = '- "standard_clause_reference" MUST be exactly "ISO 9001 TBD".\n'

    return f"""You are an expert Quality Assurance auditor.

Return ONLY valid JSON. No markdown. No commentary. No code fences.

You MUST generate checkpoints for this process type: "{profile.display_name}".
You MUST follow the canonical slot template exactly (same process_phase_reference and verification_section).

The JSON MUST match this schema exactly:
{{
  "checkpoints": [
    {{
      "process_phase_reference": "string",
      "standard_clause_reference": "string",
      "verification_section": "string",
      "prompt": "string"
    }}
  ]
}}

Hard Rules:
- Output exactly {n} items in "checkpoints".
- All values must be strings.
{clause_lock_line.rstrip()}
- "prompt" MUST start with one of: {allowed_starts}.
- "prompt" MUST begin with the canonical_prompt for that checkpoint slot (verbatim; do NOT paraphrase).
- After the canonical_prompt, you MAY append at most 1 additional sentence to connect to the uploaded document (traceability).
- "prompt" MUST NOT be identical to "verification_section".
- Do NOT invent phase names. Use the canonical slot values.
- Do NOT output evidence or artifacts/templates fields.

Canonical slots (use these exact values):
{slot_requirements}
"""


def build_profiled_structured_user_prompt(
    process_document: str,
    relevant_context: str,
    profile: ProcessProfile,
) -> str:
    """
    User prompt for profile-driven structured JSON output.
    """
    n = len(profile.slots)
    slots = "\n".join(
        [
            f"{s.number}. {s.process_phase_reference} ({s.standard_clause_reference}): {s.verification_section}\n"
            f"   Canonical Prompt (must follow the meaning, expand with document specifics): {s.canonical_prompt}"
            for s in profile.slots
        ]
    )

    tone_note = ""
    if any(s.prompt_style == "auditee_request" for s in profile.slots):
        tone_note = (
            "Tone requirement: write prompts as auditee-facing requests (e.g., 'Show…', 'Can you show…', "
            "'Pick an example… and show…'). Include sampling language when the population could be large.\n"
        )

    clause_lock_note = ""
    if bool(getattr(config, "PROFILE_LOCK_STANDARD_CLAUSE", True)):
        clause_lock_note = '- standard_clause_reference: use exactly "ISO 9001 TBD".\n'

    prompt = f"""Generate the checkpoints for the process document below using the canonical slot template.

You MUST return EXACTLY these {n} checkpoints (same phase + verification_section, same numbering):
{slots}

{tone_note}For each checkpoint:
{clause_lock_note.rstrip()}
- The prompt MUST start with the Canonical Prompt text for that slot (verbatim; do NOT paraphrase).
- You MAY append at most 1 extra sentence after the Canonical Prompt to add traceability to the uploaded document (e.g., naming the artifact section/title you saw).
- Do NOT add bullet lists. Do NOT add separate evidence/templates lists. Keep it tight and canonical.
- Prefer traceability: show how artifacts link (plan → specs → reports → tickets/logs → repositories).
- Use hints from canonical prompt / artifacts_hints, but align to the actual document content.

PROCESS DOCUMENT:
{process_document}
"""

    if relevant_context:
        prompt += f"""

RELEVANT CONTEXT (retrieved chunks):
{relevant_context}
"""

    prompt += """

Return the JSON now."""
    return prompt


def build_structured_user_prompt(
    process_document: str,
    relevant_context: str = "",
    num_checkpoints: int = 5
) -> str:
    """
    User prompt for structured JSON output.
    """
    prompt = f"""Generate exactly {num_checkpoints} verification checkpoints for the process document below.

Use this checkpoint structure (examples of style):
- process_phase_reference: use phase names found in the document (e.g., Planning / Execution / Results Handling)
- standard_clause_reference: ISO 9001 TBD (or a precise clause if you can infer it)
- verification_section: a particular requirement/statement from the uploaded document (written as "Verify that ..." or "Show ...")
- prompt: expand the verification_section with a concise, actionable description. Do NOT output evidence or templates lists.

PROCESS DOCUMENT:
{process_document}
"""

    if relevant_context:
        prompt += f"""

RELEVANT CONTEXT (retrieved chunks):
{relevant_context}
"""

    prompt += f"""

Return the JSON now."""
    return prompt


def build_context_from_chunks(chunks: list) -> str:
    """
    Build context string from retrieved chunks.
    
    Args:
        chunks: List of retrieved chunk dictionaries
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return ""
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        source = metadata.get('filename', 'Unknown')
        
        context_parts.append(f"[Context {i} from {source}]:\n{text}")
    
    return "\n\n".join(context_parts)


# Alternative prompt for structured output
STRUCTURED_OUTPUT_PROMPT = get_structured_output_prompt(5)


if __name__ == "__main__":
    # Test prompt building
    sample_doc = "This is a quality process for verification activities..."
    sample_context = "Related processes include review procedures and testing protocols..."
    
    user_prompt = build_user_prompt(sample_doc, sample_context)
    print("User Prompt Length:", len(user_prompt))
    print("\nPrompt Preview:")
    print(user_prompt[:500])

