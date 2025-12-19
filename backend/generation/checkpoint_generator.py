"""Checkpoint generator combining RAG and LLM."""

import json
import re
from typing import List, Dict, Optional
import logging

from backend.config import config
from .llm_client import LLMClient
from .prompts import (
    get_structured_output_prompt,
    build_structured_user_prompt,
    get_system_prompt,
    build_user_prompt,
    build_context_from_chunks,
    get_profiled_structured_output_prompt,
    build_profiled_structured_user_prompt,
    _extract_headings_and_excerpts,
    get_generic_profiled_structured_output_prompt,
    build_generic_profiled_structured_user_prompt,
)
from .process_profiles import ProcessType, get_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointGenerator:
    """Generate audit checkpoints using RAG + LLM."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize checkpoint generator.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.num_checkpoints = config.NUM_CHECKPOINTS
    
    def generate_checkpoints(
        self,
        process_document: str,
        relevant_chunks: List[Dict] = None,
        num_checkpoints: int = None,
        process_type: ProcessType = "auto",
        filename: str = ""
    ) -> Dict:
        """
        Generate verification checkpoints for a process document.
        
        Args:
            process_document: Main process document text
            relevant_chunks: Relevant context chunks from RAG
            num_checkpoints: Number of checkpoints to generate (default: 5)
            
        Returns:
            Dictionary with generated checkpoints and metadata
        """
        profile = get_profile(process_type=process_type, filename=filename, text=process_document)
        if profile is not None:
            num_checkpoints = len(profile.slots)
        else:
            num_checkpoints = num_checkpoints or self.num_checkpoints
        
        try:
            context = ""
            if relevant_chunks:
                context = build_context_from_chunks(relevant_chunks)
                logger.info(f"Using {len(relevant_chunks)} relevant chunks as context")
            
            # Prefer strict JSON output (more reliable parsing)
            if profile is not None:
                system_prompt = get_profiled_structured_output_prompt(profile)
                user_prompt = build_profiled_structured_user_prompt(process_document, context, profile)
            else:
                # Generic unseen process: detect phases/headings and create a stable template.
                # If heading detection is weak, do an LLM-assisted phase extraction that is
                # forced to pick phase titles as exact substrings from the document.
                pairs = _extract_headings_and_excerpts(process_document, max_items=max(num_checkpoints, 3))
                if len(pairs) < 2:
                    score = self._process_document_score(process_document)
                    logger.info("Generic process doc score=%s (higher is more process-like)", score)
                    llm_pairs = self._extract_phases_and_excerpts_via_llm(
                        process_document=process_document,
                        max_items=num_checkpoints,
                    )
                    if llm_pairs:
                        pairs = llm_pairs

                if len(pairs) >= 2:
                    pairs = pairs[:num_checkpoints]
                    allowed_phases = [h for (h, _) in pairs]
                    system_prompt = get_generic_profiled_structured_output_prompt(num_checkpoints, allowed_phases)
                    user_prompt = build_generic_profiled_structured_user_prompt(
                        process_document=process_document,
                        relevant_context=context,
                        num_checkpoints=num_checkpoints,
                        heading_excerpt_pairs=pairs,
                    )
                else:
                    # Last resort fallback: still structured, but without explicit phase lock.
                    system_prompt = get_structured_output_prompt(num_checkpoints)
                    user_prompt = build_structured_user_prompt(process_document, context, num_checkpoints)
            
            # Generate checkpoints using LLM
            logger.info(f"Generating {num_checkpoints} checkpoints with LLM...")
            generated_text = self.llm_client.generate_with_system_prompt(
                system_prompt=system_prompt,
                user_message=user_prompt,
                max_tokens=2600,
                temperature=0.05,
                top_p=1.0
            )

            # Helper: formatting-only JSON repair to recover from malformed/empty outputs
            def _repair_json(raw: str) -> str:
                repair_user = (
                    "Your previous response was not valid JSON.\n"
                    "Convert it into valid JSON that matches the required schema exactly.\n"
                    "Do NOT add commentary. Do NOT use markdown. Do NOT change the wording beyond fixing JSON formatting.\n\n"
                    "RAW OUTPUT:\n"
                    f"{raw}"
                )
                return self.llm_client.generate_with_system_prompt(
                    system_prompt=system_prompt,
                    user_message=repair_user,
                    max_tokens=1800,
                    temperature=0.0,
                    top_p=1.0
                )

            checkpoints = self.parse_checkpoints(generated_text, num_checkpoints=num_checkpoints)
            if not checkpoints:
                # JSON repair pass (formatting-only). This reduces "0 checkpoints parsed" failures.
                generated_text = _repair_json(generated_text)
                checkpoints = self.parse_checkpoints(generated_text, num_checkpoints=num_checkpoints)

            # Enforce template slot fields and prompt tone when a profile is used
            if profile is not None and checkpoints:
                checkpoints = self._apply_profile_template(checkpoints, profile)
                invalid = self._validate_profiled_checkpoints(checkpoints, profile)
                if invalid:
                    # One retry with explicit correction instructions (robustness)
                    logger.warning("Profile validation failed; retrying once. Errors: %s", invalid[:6])
                    # Preserve the last good output so we can fall back if the retry fails/comes back empty.
                    prev_generated_text = generated_text
                    prev_checkpoints = list(checkpoints)
                    retry_user_prompt = (
                        user_prompt
                        + "\n\nIMPORTANT: Your previous output violated these constraints:\n- "
                        + "\n- ".join(invalid[:12])
                        + "\n\nRegenerate the JSON, fully compliant."
                    )
                    generated_text = self.llm_client.generate_with_system_prompt(
                        system_prompt=system_prompt,
                        user_message=retry_user_prompt,
                        max_tokens=2600,
                        temperature=0.0,
                        top_p=1.0
                    )
                    if not (generated_text or "").strip():
                        logger.warning("Retry returned empty output; falling back to previous generation.")
                        generated_text = prev_generated_text
                        checkpoints = prev_checkpoints
                    else:
                        checkpoints = self.parse_checkpoints(generated_text, num_checkpoints=num_checkpoints)
                        if not checkpoints:
                            # Retry can still fail JSON parsing; attempt formatting-only repair once.
                            repaired = _repair_json(generated_text)
                            repaired_checkpoints = self.parse_checkpoints(repaired, num_checkpoints=num_checkpoints)
                            if repaired_checkpoints:
                                generated_text = repaired
                                checkpoints = repaired_checkpoints
                            else:
                                logger.warning("Retry output still unparsable after repair; falling back to previous generation.")
                                generated_text = prev_generated_text
                                checkpoints = prev_checkpoints
                    if checkpoints:
                        checkpoints = self._apply_profile_template(checkpoints, profile)
            elif profile is None and checkpoints:
                # Generic validation (phase-lock adherence / count / schema sanity).
                # If we used the generic phase-locked prompt, we can enforce allowed phases.
                allowed_phases = None
                try:
                    if "allowed_phases" in locals():
                        allowed_phases = list(locals().get("allowed_phases"))  # type: ignore[arg-type]
                except Exception:
                    allowed_phases = None

                invalid_generic = self._validate_generic_checkpoints(
                    checkpoints,
                    num_checkpoints=num_checkpoints,
                    allowed_phases=allowed_phases,
                )
                if invalid_generic:
                    logger.warning("Generic validation failed; retrying once. Errors: %s", invalid_generic[:6])
                    retry_user_prompt = (
                        user_prompt
                        + "\n\nIMPORTANT: Your previous output violated these constraints:\n- "
                        + "\n- ".join(invalid_generic[:12])
                        + "\n\nRegenerate the JSON, fully compliant."
                    )
                    prev_generated_text = generated_text
                    prev_checkpoints = list(checkpoints)
                    generated_text = self.llm_client.generate_with_system_prompt(
                        system_prompt=system_prompt,
                        user_message=retry_user_prompt,
                        max_tokens=2600,
                        temperature=0.0,
                        top_p=1.0
                    )
                    if not (generated_text or "").strip():
                        generated_text = prev_generated_text
                        checkpoints = prev_checkpoints
                    else:
                        checkpoints = self.parse_checkpoints(generated_text, num_checkpoints=num_checkpoints)
                        if not checkpoints:
                            repaired = _repair_json(generated_text)
                            repaired_checkpoints = self.parse_checkpoints(repaired, num_checkpoints=num_checkpoints)
                            if repaired_checkpoints:
                                generated_text = repaired
                                checkpoints = repaired_checkpoints
                            else:
                                generated_text = prev_generated_text
                                checkpoints = prev_checkpoints

            checkpoints = self._post_process_checkpoints(
                checkpoints,
                retrieved_context=context,
                num_checkpoints=num_checkpoints
            )

            # Return strictly the 4 required fields (no number/title/full_text/legacy keys)
            checkpoints_minimal: List[Dict] = []
            for cp in checkpoints:
                checkpoints_minimal.append({
                    "process_phase_reference": str(cp.get("process_phase_reference", "")).strip(),
                    "standard_clause_reference": str(cp.get("standard_clause_reference", "")).strip(),
                    "verification_section": str(cp.get("verification_section", "")).strip(),
                    "prompt": str(cp.get("prompt", "")).strip(),
                })
            
            logger.info(f"Successfully generated {len(checkpoints_minimal)} checkpoints")
            
            # If parsing failed, log the issue
            if not checkpoints_minimal:
                logger.warning("Failed to parse checkpoints! Raw output:")
                logger.warning(generated_text)
            
            return {
                'checkpoints': checkpoints_minimal,
                'raw_output': generated_text,
                'num_checkpoints': len(checkpoints_minimal),
                'process_type': profile.process_type if profile is not None else "auto",
                'process_profile': profile.display_name if profile is not None else None,
                'process_doc_score': self._process_document_score(process_document) if profile is None else None,
                'status': 'success',
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error generating checkpoints: {str(e)}")
            return {
                'checkpoints': [],
                'raw_output': '',
                'num_checkpoints': 0,
                'process_type': process_type,
                'process_profile': None,
                'status': 'error',
                'error': str(e)
            }

    def _validate_generic_checkpoints(
        self,
        checkpoints: List[Dict],
        num_checkpoints: int,
        allowed_phases: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Best-effort validation for generic (non-profiled) outputs.
        Focuses on schema + count + phase-lock adherence (when provided).
        """
        errors: List[str] = []
        if not isinstance(checkpoints, list) or not checkpoints:
            return ["No checkpoints returned."]

        # Count: accept short outputs but encourage exact count
        if len(checkpoints) != num_checkpoints:
            errors.append(f"Must output exactly {num_checkpoints} checkpoints (got {len(checkpoints)}).")

        allowed_set = None
        if allowed_phases:
            allowed_set = {p.strip() for p in allowed_phases if isinstance(p, str) and p.strip()}

        for i, cp in enumerate(checkpoints, start=1):
            if not isinstance(cp, dict):
                errors.append(f"Checkpoint {i}: must be an object.")
                continue
            for k in ("process_phase_reference", "standard_clause_reference", "verification_section", "prompt"):
                v = cp.get(k)
                if not isinstance(v, str) or not v.strip():
                    errors.append(f"Checkpoint {i}: missing/empty {k}.")

            if allowed_set:
                phase = str(cp.get("process_phase_reference", "")).strip()
                if phase and phase not in allowed_set:
                    errors.append(f"Checkpoint {i}: process_phase_reference must be one of the allowed phases.")

            # Basic guardrails to reduce hallucinations:
            # - verification_section should begin with "Verify" or "Show" (audit style)
            vs = str(cp.get("verification_section", "")).strip()
            if vs and not (vs.lower().startswith("verify") or vs.lower().startswith("show")):
                errors.append(f"Checkpoint {i}: verification_section should start with 'Verify' or 'Show'.")

        return errors

    def _process_document_score(self, text: str) -> int:
        """
        Lightweight heuristic score for 'process-likeness'.
        Used only to decide whether to attempt stronger phase extraction.
        """
        if not text:
            return 0
        t = text.lower()
        signals = [
            "purpose",
            "scope",
            "inputs",
            "outputs",
            "roles",
            "responsibilities",
            "activities",
            "procedure",
            "process",
            "workflow",
            "steps",
            "review",
            "approve",
            "verification",
            "validation",
            "baseline",
            "change log",
        ]
        score = 0
        for s in signals:
            if s in t:
                score += 1
        # Bonus for numbered steps
        if re.search(r"(?m)^\s*\d+(\.\d+)*\s+", text):
            score += 2
        # Bonus for obvious section brackets like "[ 1 Process Description ]"
        if "[" in text and "]" in text and "process description" in t:
            score += 2
        return score

    def _extract_phases_and_excerpts_via_llm(
        self,
        process_document: str,
        max_items: int,
    ) -> List[tuple[str, str]]:
        """
        LLM-assisted phase extraction for generic processes.
        Hard constraint: phase titles MUST be exact substrings from the document.
        Returns (phase, excerpt) pairs.
        """
        if not process_document or max_items <= 0:
            return []

        sys = (
            "Return ONLY valid JSON. No markdown. No commentary.\n"
            "Schema: {\"phases\": [\"string\", ...]}\n"
            f"Hard Rules:\n- Output 2 to {max_items} phases.\n"
            "- Each phase MUST be an exact substring copied from PROCESS DOCUMENT (case-insensitive match acceptable).\n"
            "- Each phase should be a short heading-like phrase (max 60 characters).\n"
            "- Prefer phases that represent distinct steps/sections of the process.\n"
        )
        user = f"Extract phase titles from this document.\n\nPROCESS DOCUMENT:\n{process_document}\n\nReturn JSON now."

        try:
            raw = self.llm_client.generate_with_system_prompt(
                system_prompt=sys,
                user_message=user,
                max_tokens=450,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception:
            return []

        blob = self._extract_first_json_object(raw or "")
        if not blob:
            return []
        try:
            data = json.loads(blob)
        except Exception:
            return []

        phases = data.get("phases")
        if not isinstance(phases, list):
            return []

        # Validate + dedupe
        doc_lower = process_document.lower()
        cleaned: List[str] = []
        seen = set()
        for p in phases:
            if not isinstance(p, str):
                continue
            ph = p.strip().strip('"').strip()
            if not ph or len(ph) > 60:
                continue
            key = ph.lower()
            if key in seen:
                continue
            if key not in doc_lower:
                # must be copied from document
                continue
            seen.add(key)
            cleaned.append(ph)
            if len(cleaned) >= max_items:
                break

        if len(cleaned) < 2:
            return []

        # Build excerpts around each phase (grab nearby text)
        pairs: List[tuple[str, str]] = []
        for ph in cleaned:
            idx = doc_lower.find(ph.lower())
            if idx < 0:
                continue
            start = max(0, idx)
            end = min(len(process_document), idx + 800)
            window = process_document[start:end]
            # Use a compact excerpt: take first ~3 lines after the phase occurrence
            lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
            excerpt = " ".join(lines[1:5]) if len(lines) > 1 else " ".join(lines[:4])
            excerpt = (excerpt or window).strip()
            pairs.append((ph, excerpt[:600]))

        # Ensure at least 2
        return pairs[:max_items] if len(pairs) >= 2 else []

    def _apply_profile_template(self, checkpoints: List[Dict], profile) -> List[Dict]:
        """
        Force canonical fields (phase / clause / verification section) from profile slots.
        This prevents phase drift (e.g., CM checkpoints showing 'Verification Planning').
        """
        slots_by_number = {s.number: s for s in profile.slots}
        out: List[Dict] = []
        for cp in checkpoints:
            cp = dict(cp)
            try:
                number = int(cp.get("number", 0) or 0)
            except Exception:
                number = 0
            slot = slots_by_number.get(number)
            if slot:
                cp["process_phase_reference"] = slot.process_phase_reference
                # Hard-lock clause reference for profiles unless explicitly disabled.
                if getattr(config, "PROFILE_LOCK_STANDARD_CLAUSE", True):
                    cp["standard_clause_reference"] = slot.standard_clause_reference or "ISO 9001 TBD"
                else:
                    cp["standard_clause_reference"] = slot.standard_clause_reference or cp.get("standard_clause_reference", "ISO 9001 TBD")
                cp["verification_section"] = slot.verification_section
                # Keep legacy fields aligned
                cp["title"] = slot.process_phase_reference
                cp["standard_reference"] = cp["standard_clause_reference"]
                cp["verification_objective"] = cp["verification_section"]
            out.append(cp)
        out.sort(key=lambda x: int(x.get("number", 9999)))
        return out

    def _validate_profiled_checkpoints(self, checkpoints: List[Dict], profile) -> List[str]:
        """
        Validate a profiled generation run.
        Returns list of error strings (empty means valid).
        """
        errors: List[str] = []
        allowed_starts = [s.lower() for s in profile.allowed_prompt_starts]
        slots_by_number = {s.number: s for s in profile.slots}

        def _norm(s: str) -> str:
            s = (s or "").strip().lower()
            # Normalize smart quotes to plain quotes
            s = (
                s.replace("“", '"')
                .replace("”", '"')
                .replace("’", "'")
                .replace("‘", "'")
            )
            # Collapse whitespace
            s = re.sub(r"\s+", " ", s)
            return s.strip()

        for cp in checkpoints:
            try:
                num = int(cp.get("number", 0) or 0)
            except Exception:
                num = 0
            slot = slots_by_number.get(num)
            if not slot:
                errors.append(f"Unexpected checkpoint number {cp.get('number')} (not in canonical slots).")
                continue

            # Phase + verification_section must match canonical slot
            if str(cp.get("process_phase_reference", "")).strip() != slot.process_phase_reference:
                errors.append(f"Checkpoint {num}: process_phase_reference must be '{slot.process_phase_reference}'.")
            if str(cp.get("verification_section", "")).strip() != slot.verification_section:
                errors.append(f"Checkpoint {num}: verification_section must match canonical text.")

            # Standard clause lock (profiles)
            if getattr(config, "PROFILE_LOCK_STANDARD_CLAUSE", True):
                expected = (slot.standard_clause_reference or "ISO 9001 TBD").strip()
                actual = str(cp.get("standard_clause_reference", "")).strip() or str(cp.get("standard_reference", "")).strip()
                if actual != expected:
                    errors.append(f"Checkpoint {num}: standard_clause_reference must be '{expected}'.")

            # Prompt style
            prompt = str(cp.get("prompt", "")).strip()
            if not prompt:
                errors.append(f"Checkpoint {num}: missing prompt.")
            else:
                p0 = prompt[:40].lower()
                if not any(p0.startswith(a) for a in allowed_starts):
                    errors.append(
                        f"Checkpoint {num}: prompt must start with one of {profile.allowed_prompt_starts}."
                    )
                # Canonical prompt anchoring: must begin with canonical_prompt (verbatim/near-verbatim)
                canonical = (slot.canonical_prompt or "").strip()
                if canonical:
                    p_norm = _norm(prompt)
                    c_norm = _norm(canonical)
                    if not p_norm.startswith(c_norm):
                        errors.append(
                            f"Checkpoint {num}: prompt must begin with the canonical prompt text (do not paraphrase)."
                        )
                    else:
                        # Allow at most 1 extra sentence after canonical prompt
                        remainder_norm = p_norm[len(c_norm):].strip()
                        if remainder_norm:
                            # Count sentences in remainder (simple heuristic)
                            parts = [p.strip() for p in re.split(r"[.!?]+", remainder_norm) if p.strip()]
                            if len(parts) > 1:
                                errors.append(
                                    f"Checkpoint {num}: prompt may only add up to 1 sentence after the canonical prompt."
                                )
                # Avoid trivial copy of verification_section (common failure mode)
                vs = str(cp.get("verification_section", "")).strip()
                if vs and prompt.strip().lower() == vs.strip().lower():
                    errors.append(f"Checkpoint {num}: prompt must expand the verification_section (not identical).")
                # Minimum detail heuristic: discourage single-clause prompts
                if len(prompt) < 80 and "." not in prompt:
                    errors.append(f"Checkpoint {num}: prompt is too short; include how-to steps and traceability.")

        return errors
    
    def parse_checkpoints(self, generated_text: str, num_checkpoints: Optional[int] = None) -> List[Dict]:
        """
        Parse generated text into structured checkpoint objects.
        
        Args:
            generated_text: Raw generated text from LLM
            
        Returns:
            List of parsed checkpoint dictionaries
        """
        # 1) Try strict JSON first
        json_checkpoints = self._parse_checkpoints_json(generated_text, num_checkpoints=num_checkpoints)
        if json_checkpoints:
            return json_checkpoints

        # 2) Fallback to regex-based parsing
        checkpoints: List[Dict] = []
        
        # Split by checkpoint numbers (1., 2., 3., etc.)
        # Pattern: **[digit]. [title], [standard]:** [description]
        pattern = r'\*\*(\d+)\.\s*([^:,]+?)(?:,\s*([^:]+?))?\s*:\*\*\s*(.+?)(?=\*\*Prompt:\*\*|$)'
        matches = re.finditer(pattern, generated_text, re.DOTALL)
        
        for match in matches:
            number = int(match.group(1))
            title = match.group(2).strip()
            standard = match.group(3).strip() if match.group(3) else "TBD"
            objective = match.group(4).strip()
            
            # Extract the prompt section
            prompt = ""
            prompt_pattern = rf'\*\*{number}\..*?\*\*Prompt:\*\*\s*(.+?)(?=\*\*\d+\.|$)'
            prompt_match = re.search(prompt_pattern, generated_text, re.DOTALL)
            if prompt_match:
                prompt = prompt_match.group(1).strip()
            
            checkpoint = {
                'number': number,
                'title': title,
                'standard_reference': standard,
                'verification_objective': objective,
                'prompt': prompt,
                'full_text': f"**{number}. {title}, {standard}:** {objective}\n\n**Prompt:** {prompt}"
            }
            
            checkpoints.append(checkpoint)
        
        # If parsing failed, try alternative parsing
        if not checkpoints:
            checkpoints = self._fallback_parse(generated_text)
        
        return checkpoints

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """
        Extract the first balanced {...} JSON object from a text blob.
        Handles braces inside strings.
        """
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _parse_checkpoints_json(self, text: str, num_checkpoints: Optional[int] = None) -> List[Dict]:
        """
        Parse JSON output of the form {"checkpoints": [...]}.
        """
        json_blob = self._extract_first_json_object(text)
        if not json_blob:
            return []

        try:
            data = json.loads(json_blob)
        except Exception:
            return []

        cps = data.get("checkpoints")
        if not isinstance(cps, list) or not cps:
            return []

        parsed: List[Dict] = []
        for idx, cp in enumerate(cps, start=1):
            if not isinstance(cp, dict):
                continue
            # "number" is optional in the new minimalist schema; keep stable ordering via idx
            number = cp.get("number", idx)
            try:
                number_int = int(number)
            except Exception:
                number_int = idx

            # New schema (preferred)
            process_phase = str(cp.get("process_phase_reference", "")).strip()
            standard_clause = str(cp.get("standard_clause_reference", "")).strip()
            verification_section = str(cp.get("verification_section", "")).strip()
            prompt = str(cp.get("prompt", "")).strip()

            # Backward compatibility with older keys
            if not standard_clause:
                standard_clause = str(cp.get("standard_reference", "TBD")).strip()
            if not verification_section:
                verification_section = str(cp.get("verification_objective", "")).strip()

            # Derive a title for UI compatibility (kept for frontend)
            title = process_phase or f"Checkpoint {number_int}"
            if not process_phase:
                # try infer from verification_section prefix
                process_phase = ""
            if not standard_clause:
                standard_clause = "TBD"

            parsed.append({
                "number": number_int,
                "title": title,
                "process_phase_reference": process_phase,
                "standard_clause_reference": standard_clause,
                "verification_section": verification_section,
                # Keep legacy fields so older UI code keeps working
                "standard_reference": standard_clause,
                "verification_objective": verification_section,
                "prompt": prompt,
                "full_text": (
                    f"**{number_int}. {process_phase or title}, {standard_clause}:** {verification_section}\n\n"
                    f"**Prompt:** {prompt}"
                ),
            })

        # Enforce requested count if provided
        if num_checkpoints is not None:
            parsed = [cp for cp in parsed if 1 <= int(cp.get("number", 0)) <= num_checkpoints]
            parsed.sort(key=lambda x: int(x.get("number", 0)))
            if len(parsed) == num_checkpoints:
                return parsed
            # If model returned more/less, still return what we got (caller/UI can see count)
        return parsed

    def _post_process_checkpoints(
        self,
        checkpoints: List[Dict],
        retrieved_context: str,
        num_checkpoints: int
    ) -> List[Dict]:
        """
        - Clamp ISO clause references unless explicitly present in retrieved_context
        - Keep output minimal: no evidence/artifacts/templates fields (per product requirement)
        """
        context_iso_clauses = self._extract_iso9001_clauses(retrieved_context)

        processed: List[Dict] = []
        for cp in checkpoints:
            cp = dict(cp)

            # Clamp ISO 9001 clause references if not explicitly present in retrieved context
            cp["standard_clause_reference"] = self._clamp_standard_clause(
                str(cp.get("standard_clause_reference") or cp.get("standard_reference") or "TBD"),
                context_iso_clauses
            )
            cp["standard_reference"] = cp["standard_clause_reference"]

            # Keep legacy verification objective key aligned
            cp["verification_objective"] = cp.get("verification_section", cp.get("verification_objective", ""))

            # Refresh full_text
            number = int(cp.get("number", 0) or 0)
            phase = cp.get("process_phase_reference") or cp.get("title") or f"Checkpoint {number}"
            std = cp.get("standard_clause_reference", "TBD")
            ver = cp.get("verification_section", "")
            prompt = cp.get("prompt", "")
            cp["full_text"] = (
                f"**{number}. {phase}, {std}:** {ver}\n\n"
                f"**Prompt:** {prompt}"
            )

            # Ensure we don't leak evidence/template keys even if model returns them
            cp.pop("evidence", None)
            cp.pop("artifacts_templates", None)

            processed.append(cp)

        # Prefer exactly 1..N order if possible
        processed.sort(key=lambda x: int(x.get("number", 9999)))
        if len(processed) > num_checkpoints:
            processed = processed[:num_checkpoints]
        return processed

    def _extract_iso9001_clauses(self, text: str) -> set:
        """
        Extract ISO 9001 clause numbers present in text (e.g., 7.5.3).
        """
        if not text:
            return set()
        clauses = set()
        # Capture "ISO 9001" mentions with clause numbers nearby, and standalone clause numbers
        for m in re.finditer(r'ISO\s*9001(?:\:2015)?[^\d]{0,20}(\d+(?:\.\d+)+)', text, flags=re.IGNORECASE):
            clauses.add(m.group(1))
        for m in re.finditer(r'\b(\d+(?:\.\d+)+)\b', text):
            # Filter to ISO-ish range (avoid random numbers): keep 4.x, 5.x, 6.x, 7.x, 8.x, 9.x, 10.x
            major = int(m.group(1).split(".")[0])
            if 4 <= major <= 10:
                clauses.add(m.group(1))
        return clauses

    def _clamp_standard_clause(self, standard_clause_reference: str, context_iso_clauses: set) -> str:
        """
        If ISO 9001 clause is not explicitly present in retrieved context, return ISO 9001 TBD.
        """
        s = (standard_clause_reference or "").strip()
        if not s:
            return "ISO 9001 TBD"
        if "tbd" in s.lower():
            return "ISO 9001 TBD" if "iso 9001" in s.lower() else s

        # Only clamp ISO 9001 (leave other standards as-is)
        if "iso" in s.lower() and "9001" in s.lower():
            m = re.search(r'(\d+(?:\.\d+)+)', s)
            if not m:
                return "ISO 9001 TBD"
            clause = m.group(1)
            if clause not in context_iso_clauses:
                return "ISO 9001 TBD"
            return s

        return s

    def _derive_evidence_from_prompt(self, prompt: str) -> List[str]:
        """
        Try to extract evidence items from prompt text (e.g., 'Evidence: A and B').
        """
        if not prompt:
            return []
        m = re.search(r'(?i)\bevidence\b\s*[:\-]\s*(.+)', prompt)
        if not m:
            return []
        tail = m.group(1).strip()
        # Split on common separators
        parts = re.split(r'\s*(?:;|,|\band\b|\+)\s*', tail)
        items = [p.strip(" .\n\t") for p in parts if p.strip(" .\n\t")]
        # Deduplicate while preserving order
        seen = set()
        out = []
        for it in items:
            if it.lower() in seen:
                continue
            seen.add(it.lower())
            out.append(it)
        return out

    def _derive_artifacts_from_text(self, text: str) -> List[str]:
        """
        Heuristic extraction of common verification artifacts/templates.
        """
        if not text:
            return []
        candidates = [
            "Verification Plan",
            "Verification Review Specification",
            "Verification Test Specification",
            "Verification Review Report",
            "Verification Test Report",
            "Issue Management System",
            "Issue ticket",
            "Configuration Management",
            "CM log",
            "CM repository",
            "Approval sign-off",
            "Checklist",
            "Mapping matrix",
        ]
        found = []
        lower = text.lower()
        for c in candidates:
            if c.lower() in lower:
                found.append(c)
        # If nothing matched, return empty (UI will still show [])
        return found
    
    def _fallback_parse(self, text: str) -> List[Dict]:
        """
        Fallback parsing method if primary parsing fails.
        
        Args:
            text: Generated text
            
        Returns:
            List of checkpoint dictionaries
        """
        checkpoints = []
        
        # Try multiple parsing strategies
        
        # Strategy 1: Split by numbered items (1., 2., 3., etc.)
        sections = re.split(r'\n\s*(\d+\.)', text)
        
        current_checkpoint = None
        for i, part in enumerate(sections):
            part = part.strip()
            if not part:
                continue
            
            # Check if this is a number
            if re.match(r'^\d+\.$', part):
                # Save previous checkpoint
                if current_checkpoint and current_checkpoint.get('full_text'):
                    checkpoints.append(current_checkpoint)
                
                # Start new checkpoint
                number = int(part.rstrip('.'))
                current_checkpoint = {
                    'number': number,
                    'title': f'Checkpoint {number}',
                    'standard_reference': 'TBD',
                    'verification_objective': '',
                    'prompt': '',
                    'full_text': ''
                }
            elif current_checkpoint is not None:
                # Accumulate text for current checkpoint
                current_checkpoint['full_text'] += part + '\n'
                
                # Try to extract title if not set
                if not current_checkpoint.get('title') or current_checkpoint['title'].startswith('Checkpoint'):
                    first_line = part.split('\n')[0]
                    if len(first_line) < 150:
                        current_checkpoint['title'] = first_line[:100]
                
                # Try to extract verification objective and prompt
                if 'Prompt:' in part or '**Prompt:**' in part:
                    parts = re.split(r'\*\*Prompt:\*\*|Prompt:', part, maxsplit=1)
                    if len(parts) == 2:
                        current_checkpoint['verification_objective'] = parts[0].strip()
                        current_checkpoint['prompt'] = parts[1].strip()
        
        # Don't forget the last checkpoint
        if current_checkpoint and current_checkpoint.get('full_text'):
            checkpoints.append(current_checkpoint)
        
        logger.info(f"Fallback parser found {len(checkpoints)} checkpoints")
        return checkpoints
    
    def format_checkpoints_for_display(self, checkpoints: List[Dict]) -> str:
        """
        Format checkpoints for display.
        
        Args:
            checkpoints: List of checkpoint dictionaries
            
        Returns:
            Formatted string
        """
        formatted = []
        
        for checkpoint in checkpoints:
            formatted.append(
                f"**{checkpoint['number']}. {checkpoint['title']}, "
                f"{checkpoint['standard_reference']}:** "
                f"{checkpoint['verification_objective']}\n\n"
                f"**Prompt:** {checkpoint['prompt']}\n"
            )
        
        return "\n\n".join(formatted)


# Example usage
if __name__ == "__main__":
    try:
        # Initialize
        llm_client = LLMClient()
        generator = CheckpointGenerator(llm_client)
        
        # Sample document
        sample_doc = """
        Verification Process
        
        This process describes the approach to conduct verification activities.
        
        1. Verification Planning: Define verification strategy and plan
        2. Verification Specification: Specify review checklists
        3. Verification Execution: Conduct reviews and testing
        4. Verification Evaluation: Analyze results and track issues
        """
        
        # Generate checkpoints
        result = generator.generate_checkpoints(sample_doc)
        
        if result['status'] == 'success':
            print(f"Generated {result['num_checkpoints']} checkpoints:\n")
            formatted = generator.format_checkpoints_for_display(result['checkpoints'])
            print(formatted)
        else:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"Error: {e}")

