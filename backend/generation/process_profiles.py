"""
Process profiles and canonical checkpoint templates.

Goal: make checkpoint generation robust across different processes by forcing the
LLM to fill in a known-good template (the "original checkpoints") rather than
free-form inventing phases/structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence


ProcessType = Literal[
    "auto",
    "verification",
    "joint_review",
    "configuration_management",
]


@dataclass(frozen=True)
class CheckpointSlot:
    number: int
    process_phase_reference: str
    standard_clause_reference: str
    verification_section: str
    prompt_style: Literal["auditee_request", "auditor_instruction"]
    # Canonical prompt guidance text (the "original Prompt:" paragraph(s)).
    canonical_prompt: str
    # Optional hints to steer evidence/templates without hardcoding outputs.
    evidence_hints: Sequence[str] = ()
    artifacts_hints: Sequence[str] = ()


@dataclass(frozen=True)
class ProcessProfile:
    process_type: ProcessType
    display_name: str
    # Keywords used for lightweight detection from filename/text
    keywords: Sequence[str]
    # Canonical checkpoint slots (this is the "original template" set)
    slots: Sequence[CheckpointSlot]
    # Allowed leading phrases for the prompt (enforced in validation)
    allowed_prompt_starts: Sequence[str]


ISO_9001_TBD = "ISO 9001 TBD"


VERIFICATION_PROFILE = ProcessProfile(
    process_type="verification",
    display_name="Verification",
    keywords=("verification plan", "verification specification", "verification review", "verification test"),
    allowed_prompt_starts=(
        # Verification originals are more auditor-descriptive than auditee-requests
        "The verification",
        "The results",
        "The verification activities",
        "In the",
        "Verify",
        "Check",
        "Confirm",
        "Ensure",
        "Show",
        "Demonstrate",
    ),
    slots=(
        CheckpointSlot(
            number=1,
            process_phase_reference="Verification Planning",
            standard_clause_reference=ISO_9001_TBD,
            verification_section=(
                "Verify that the verification strategy has been planned and the proper template has been used for the related evidence"
            ),
            prompt_style="auditor_instruction",
            canonical_prompt=(
                "The verification strategy and the related plan shall be defined for each phase of the product design and development lifecycle. "
                "All the necessary information shall be formalized in the Verification Plan."
            ),
            artifacts_hints=("Verification Plan", "Verification Plan template"),
        ),
        CheckpointSlot(
            number=2,
            process_phase_reference="Verification Specification",
            standard_clause_reference=ISO_9001_TBD,
            verification_section=(
                "Verify that verification review checklists have been specified for each work product foreseen in the design phases (at least 2 evidences) and verify that the proper template has been used for the related evidence"
            ),
            prompt_style="auditor_instruction",
            canonical_prompt=(
                "In the design phases, the verification is aimed to evaluate the work products (i.e. requirements specifications, software code). "
                "The verification reviews shall be based on the identification of specific checklist for each work product and those checklists shall be formalized in the Verification Review Specification."
            ),
            artifacts_hints=("Verification Review Specification", "Verification Review Checklist"),
        ),
        CheckpointSlot(
            number=3,
            process_phase_reference="Verification Specification",
            standard_clause_reference=ISO_9001_TBD,
            verification_section=(
                "Verify that product has been evaluated in a test environment and verify that the proper template has been used for the related evidence"
            ),
            prompt_style="auditor_instruction",
            canonical_prompt=(
                "In the test phases, the verification is aimed to evaluate the work products (i.e. HW or SW elements) in a test environment. "
                "The testing activities shall be based on the identification of test cases or test scenarios that shall be formalized in the Verification Test Specification."
            ),
            artifacts_hints=("Verification Test Specification", "Test environment"),
        ),
        CheckpointSlot(
            number=4,
            process_phase_reference="Verification Execution and Evaluation",
            standard_clause_reference=ISO_9001_TBD,
            verification_section=(
                "Verify that the verification review is conducted by a different person than the author(s) of the work product to be verified, that it is clearly formalized using the proper template and that it is tracked in the Issue Management System"
            ),
            prompt_style="auditor_instruction",
            canonical_prompt=(
                "The results of the verification review shall be reported in the Verification Review Report with a clear statement of whether the verification passed or not. "
                "In case of failed checks, the issues shall be managed according to the “Issues from Review” process. "
                "Verification Review Report shall be formalized in a new ticket on the Issue Management System. "
                "It is recommended that the verification is performed by a different person than the author(s) of the work product to be verified."
            ),
            artifacts_hints=("Verification Review Report", "Issue Management System"),
        ),
        CheckpointSlot(
            number=5,
            process_phase_reference="Verification Execution and Evaluation",
            standard_clause_reference=ISO_9001_TBD,
            verification_section=(
                "Verify that the verification testing is conducted by a different person than the author(s) of the work product to be verified, that the testing environment is put under Configuration Management, that the testing activity is clearly formalized using the proper template and that all the failed tests tracked in the Issue Management System"
            ),
            prompt_style="auditor_instruction",
            canonical_prompt=(
                "The results of the verification testing shall be reported in the Verification Test Report with a clear statement of whether the verification passed or not. "
                "In case of failed tests, the issues shall be managed according to the Internal Bug Issue Management process. "
                "It is recommended that the verification is performed by a different person than the author(s) of the work product to be verified. "
                "The Verification environment shall be put under Configuration Management."
            ),
            artifacts_hints=("Verification Test Report", "Configuration Management", "Issue Management System"),
        ),
        CheckpointSlot(
            number=6,
            process_phase_reference="Verification Execution and Evaluation",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Verify that all verification activities have been executed as planned and specified",
            prompt_style="auditor_instruction",
            canonical_prompt=(
                "The verification activities shall be executed as planned in the Verification Plan and as specified in the verification specification "
                "(Verification Review Specification and Verification Test Specification)."
            ),
            artifacts_hints=("Verification Plan", "Verification Review Specification", "Verification Test Specification"),
        ),
    ),
)


JOINT_REVIEW_PROFILE = ProcessProfile(
    process_type="joint_review",
    display_name="Joint Review",
    keywords=("joint review", "joint review report", "review checklist", "minutes", "attendance"),
    allowed_prompt_starts=(
        "Show",
        "Can you show",
        "When",
        "Provide",
        "Demonstrate",
    ),
    slots=(
        CheckpointSlot(
            number=1,
            process_phase_reference="Joint Review Planning",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Show me how you planned and prepared for a joint review that was performed?",
            prompt_style="auditee_request",
            canonical_prompt=(
                "Show the materials prepared before the joint review highlighting scope, schedule of review activities, necessary resources, checklist and inputs for the review."
            ),
            artifacts_hints=("Joint Review Plan", "Checklist"),
        ),
        CheckpointSlot(
            number=2,
            process_phase_reference="Joint Review Planning",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Can you show how materials for the joint review were distributed to involved parties?",
            prompt_style="auditee_request",
            canonical_prompt=(
                "Show evidence of how review materials were communicated to stakeholders – email/link/others."
            ),
            artifacts_hints=("Distribution log", "Email/links"),
        ),
        CheckpointSlot(
            number=3,
            process_phase_reference="Conduct Review",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Can you show that the review was performed according to the plan?",
            prompt_style="auditee_request",
            canonical_prompt=(
                "Show the Joint Review Report highlighting who attended the review and how the checklist was used during the joint review."
            ),
            artifacts_hints=("Joint Review Report", "Attendance", "Checklist"),
        ),
        CheckpointSlot(
            number=4,
            process_phase_reference="Results Handling",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Show how the results from the Joint Review were analysed, tracked and distributed?",
            prompt_style="auditee_request",
            canonical_prompt=(
                "Show the Joint Review Report highlighting actions derived from the review. "
                "Show actions with different statuses highlighting how they are tracked and assigned to relevant stakeholders."
            ),
            artifacts_hints=("Action log", "Issue tickets", "Distribution evidence"),
        ),
    ),
)


CONFIGURATION_MANAGEMENT_PROFILE = ProcessProfile(
    process_type="configuration_management",
    display_name="Configuration Management",
    keywords=("configuration management", "baseline", "change log", "configuration status report", "cqa"),
    allowed_prompt_starts=(
        "Show",
        "When",
        "Pick",
        "Provide",
        "Demonstrate",
        "From",
    ),
    slots=(
        CheckpointSlot(
            number=1,
            process_phase_reference="Setup Configuration Plan",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Show me evidence that the configuration management plan was setup according to the process?",
            prompt_style="auditee_request",
            canonical_prompt=(
                "From the configuration management plan show an item under configuration highlighting it’s properties, mechanisms for controlling the configuration and main baselines."
            ),
            artifacts_hints=("Configuration Management Plan", "Item under configuration"),
        ),
        CheckpointSlot(
            number=2,
            process_phase_reference="Baseline Generation",
            standard_clause_reference=ISO_9001_TBD,
            verification_section=(
                "When generating internal baselines, show me evidence that you controlled modifications to items generated during the Design phase and that the Change Log was updated?"
            ),
            prompt_style="auditee_request",
            canonical_prompt=(
                "Pick a mechanism from the configuration management plan and show how it was applied for a Design item under configuration. "
                "Show where the item is stored and entries related to it in the Change Log."
            ),
            artifacts_hints=("Change Log", "Repository/baseline storage"),
        ),
        CheckpointSlot(
            number=3,
            process_phase_reference="Baseline Generation",
            standard_clause_reference=ISO_9001_TBD,
            verification_section=(
                "When generating internal baselines, show me evidence that you controlled modifications to items generated during the V&V phases?"
            ),
            prompt_style="auditee_request",
            canonical_prompt=(
                "Pick a mechanism from the configuration management plan and show how it was applied for a V&V item under configuration. "
                "Show where the item is stored and entries related to it in the Change Log."
            ),
            artifacts_hints=("Change Log", "Repository/baseline storage"),
        ),
        CheckpointSlot(
            number=4,
            process_phase_reference="Baseline Generation",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Show evidence that internal baselines were generated?",
            prompt_style="auditee_request",
            canonical_prompt=(
                "Show the Configuration Status Report, pick an example for design and V&V items; show where they are stored as part of that baseline."
            ),
            artifacts_hints=("Configuration Status Report", "Baseline contents"),
        ),
        CheckpointSlot(
            number=5,
            process_phase_reference="Baseline Generation",
            standard_clause_reference=ISO_9001_TBD,
            verification_section="Show evidence that external baselines were generated after the configuration quality assurance was completed?",
            prompt_style="auditee_request",
            canonical_prompt=(
                "Show the Configuration Status Report, pick an example on an external baseline. "
                "Show the outcome from the Configuration Quality Assurance Audit and relate to the items included in the external baseline."
            ),
            artifacts_hints=("Configuration Status Report", "CQA audit outcome", "External baseline"),
        ),
    ),
)


PROFILES: Dict[ProcessType, ProcessProfile] = {
    "verification": VERIFICATION_PROFILE,
    "joint_review": JOINT_REVIEW_PROFILE,
    "configuration_management": CONFIGURATION_MANAGEMENT_PROFILE,
}


def detect_process_type(filename: str = "", text: str = "") -> ProcessType:
    """
    Lightweight process-type detection from filename/text.
    """
    haystack = f"{filename}\n{text}".lower()
    if not haystack.strip():
        return "auto"

    # Prefer filename hints (most reliable when user uploads a process doc)
    fn = (filename or "").lower()
    if "joint" in fn and "review" in fn:
        return "joint_review"
    if "verification" in fn:
        return "verification"
    if "configuration" in fn and "management" in fn:
        return "configuration_management"

    # Keyword scoring with weights.
    # Important: many Verification docs mention Configuration Management (e.g., environment under CM),
    # so we should NOT treat a single "configuration management" mention as dominant.
    scores: Dict[ProcessType, int] = {k: 0 for k in PROFILES.keys()}
    for ptype, prof in PROFILES.items():
        for kw in prof.keywords:
            if kw in haystack:
                scores[ptype] += 1

    # Extra boosts for distinctive terms
    if "baseline" in haystack or "change log" in haystack or "configuration status report" in haystack or "cmp" in haystack:
        scores["configuration_management"] += 2
    if "joint review report" in haystack or "results handling" in haystack or "conduct review" in haystack:
        scores["joint_review"] += 2
    if "verification plan" in haystack or "verification review specification" in haystack or "verification test specification" in haystack:
        scores["verification"] += 2

    # Decide with safeguards
    best = max(scores.items(), key=lambda kv: kv[1])
    best_type, best_score = best[0], best[1]
    if best_score <= 0:
        return "auto"

    # Tie-breaker: if Verification is present and CM is only weakly present, choose Verification.
    if scores.get("verification", 0) >= scores.get("configuration_management", 0) and "verification" in haystack:
        return "verification"

    return best_type


def get_profile(process_type: ProcessType, filename: str = "", text: str = "") -> Optional[ProcessProfile]:
    """
    Resolve a ProcessProfile either from explicit process_type or by detection.
    """
    if process_type and process_type != "auto":
        return PROFILES.get(process_type)
    detected = detect_process_type(filename=filename, text=text)
    return PROFILES.get(detected) if detected != "auto" else None


