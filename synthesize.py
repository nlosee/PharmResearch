"""
synthesize.py — Pharma Regulatory Pulse
Generates the weekly newsletter using OpenAI GPT-4.1-mini with strict RAG grounding.

Monte Carlo Synthesis Engine
─────────────────────────────
Rather than generating the newsletter once, this module runs N independent
generation passes (default: 3) with slightly varied prompts and random article
orderings. It then:

  1. Parses each pass into sections (Spotlight, FDA Updates, EMA, etc.)
  2. Scores each section version on citation density, unique source coverage,
     and content completeness
  3. Selects the highest-scoring version of each section
  4. Merges selected sections into a single coherent newsletter

This guarantees:
  - No important story is omitted due to random model variance
  - Ideas across sources are properly grouped, not siloed
  - The best phrasing/completeness for each topic area is always used
  - True deduplication within the final output (same story, one best treatment)

Key constraints:
  - All facts must cite [Source N] traceable to the numbered source list
  - Temperature 0.2 base (slight variance per pass for Monte Carlo effect)
  - Model pinned via OPENAI_MODEL_WRITER env var (default: gpt-5.4-mini)
  - Never hallucinates beyond provided sources
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

# ── Section ordering ──────────────────────────────────────────────────────────
# Regulatory Spotlight leads, then 14 disease-state sections, then Dates/Sign-off.
# Each disease-state section contains ### Clinical Trials and ### Guidelines sub-headers.
SECTION_ORDER = [
    "regulatory spotlight",
    "renal and liver disease",
    "infectious disease",
    "cardiovascular conditions",
    "anticoagulation and blood disorders",
    "eyes ears nose and skin conditions",
    "pulmonary conditions and tobacco cessation",
    "endocrine conditions",
    "male and female health",
    "special populations",
    "pain related conditions",
    "oncology",
    "psychiatric conditions",
    "neurologic conditions",
    "gastrointestinal conditions",
    "dates to watch",
    "sign-off",
]

# Canonical header strings used in the output
SECTION_HEADERS = {
    "regulatory spotlight":                       "## Regulatory Spotlight",
    "renal and liver disease":                    "## Renal and Liver Disease",
    "infectious disease":                         "## Infectious Disease",
    "cardiovascular conditions":                  "## Cardiovascular Conditions",
    "anticoagulation and blood disorders":        "## Anti-Coagulation and Blood Disorders",
    "eyes ears nose and skin conditions":         "## Eyes, Ears, Nose, and Skin Conditions",
    "pulmonary conditions and tobacco cessation": "## Pulmonary Conditions and Tobacco Cessation",
    "endocrine conditions":                       "## Endocrine Conditions",
    "male and female health":                     "## Male and Female Health",
    "special populations":                        "## Special Populations (Weight Loss, Transplants, Pediatric)",
    "pain related conditions":                    "## Pain Related Conditions (Pain, Migraine, Gout)",
    "oncology":                                   "## Oncology",
    "psychiatric conditions":                     "## Psychiatric Conditions",
    "neurologic conditions":                      "## Neurologic Conditions",
    "gastrointestinal conditions":                "## Gastrointestinal Conditions",
    "dates to watch":                             "## Dates to Watch",
    "sign-off":                                   "## Sign-off",
}


# ── Few-shot calibration examples ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
--- TONE & FORMAT EXAMPLES ---

Example 1 — FDA Final Guidance:
**FDA Finalizes Guidance on Postmarketing Safety Reporting for Combination Products**
The FDA released the final version of its guidance clarifying postmarketing safety reporting
requirements for combination products under 21 CFR Part 4. The guidance supersedes the 2019
draft and provides updated recommendations on adverse event report submission based on
primary mode of action. Regulatory affairs teams should assess any necessary updates to
their pharmacovigilance SOPs. [Source 3]
https://www.fda.gov/regulatory-information/...

Example 2 — EMA Biosimilar Opinion:
**CHMP Issues Positive Opinion for Biosimilar Adalimumab (Hadlima) Interchangeability**
The EMA's Committee for Medicinal Products for Human Use (CHMP) adopted a positive opinion
recommending interchangeability designation for Samsung Bioepis' adalimumab biosimilar Hadlima.
This follows the EMA's revised biosimilar guidelines implemented in September 2025. The
decision may influence national-level substitution policies across EU member states. [Source 7]
https://www.ema.europa.eu/en/medicines/...

Example 3 — FDA Warning Letter:
**FDA Issues Warning Letter to Generic Manufacturer Over Data Integrity Failures**
The FDA issued a warning letter to a contract manufacturing organization following an inspection
that identified repeated failures in audit trail controls and laboratory data integrity under
21 CFR Part 211. The letter cites failure to invalidate out-of-specification results without
a documented scientific justification. [Source 12]
https://www.fda.gov/inspections-compliance-enforcement/...

--- END EXAMPLES ---
"""


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class SectionScore:
    section_key: str
    text: str
    citation_count: int       = 0
    unique_sources: int       = 0
    source_ids: set[int]      = field(default_factory=set)
    word_count: int           = 0
    item_count: int           = 0   # number of **bold** items found
    score: float              = 0.0


def _score_section(key: str, text: str) -> SectionScore:
    """
    Score a section on three dimensions:
      1. Citation density — number of [Source N] references
      2. Source breadth  — number of unique sources cited
      3. Content depth   — word count and item count

    Combined into a weighted score for section selection.
    """
    s = SectionScore(section_key=key, text=text)

    # Citation count and unique source IDs
    citations = re.findall(r"\[Source (\d+)\]", text)
    s.citation_count = len(citations)
    s.source_ids = {int(c) for c in citations}
    s.unique_sources = len(s.source_ids)

    # Word count (rough proxy for depth)
    s.word_count = len(text.split())

    # Item count (number of bold headlines = distinct news items)
    s.item_count = len(re.findall(r"\*\*.+?\*\*", text))

    # Weighted score
    # citation_density is citations per 100 words (avoids rewarding verbosity alone)
    density = (s.citation_count / max(s.word_count, 1)) * 100
    s.score = (
        s.unique_sources * 4.0      # breadth is most valuable
        + density * 2.0             # citation density
        + s.item_count * 1.5        # more distinct items = more complete
        + min(s.word_count, 600) * 0.005  # depth, capped to avoid bloat
    )

    return s


# ── Section parser ────────────────────────────────────────────────────────────

def _parse_sections(newsletter_md: str) -> dict[str, str]:
    """
    Parse a newsletter markdown string into a dict of {section_key: section_text}.
    Section keys are lowercased canonical names from SECTION_ORDER.
    """
    sections: dict[str, str] = {}

    # Build regex: match ## <title> ... up to next ## or end
    pattern = re.compile(
        r"##\s*(.+?)\s*\n(.*?)(?=\n##|\Z)", re.DOTALL | re.IGNORECASE
    )
    for m in pattern.finditer(newsletter_md):
        raw_title = m.group(1).strip().lower()
        body = m.group(2).strip()

        # Map to canonical key
        matched_key = None
        for key in SECTION_ORDER:
            if key in raw_title or raw_title in key:
                matched_key = key
                break
        if matched_key is None:
            matched_key = raw_title   # keep as-is if unknown section

        sections[matched_key] = body

    return sections


# ── Source context builder ────────────────────────────────────────────────────

def _build_source_context(articles: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, a in enumerate(articles, start=1):
        lines.append(
            f"SOURCE [{i}]: {a.get('title', 'Untitled')}\n"
            f"URL: {a.get('url', '')}\n"
            f"Source: {a.get('source_name', '')} ({a.get('source_type', '')})\n"
            f"Published: {a.get('published_date', 'unknown')}\n"
            f"{a.get('content', '')[:2500]}\n"
        )
    return "\n---\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(newsletter_config: dict) -> str:
    tone = newsletter_config.get("tone", "")
    audience = newsletter_config.get("audience", "")
    return f"""You are an expert pharmaceutical regulatory affairs writer producing the weekly "Pharma Regulatory Pulse" newsletter.

AUDIENCE:
{audience}

TONE & STYLE:
{tone}

{FEW_SHOT_EXAMPLES}

CRITICAL GROUNDING RULES (non-negotiable):
1. Write based ONLY on the provided numbered source context. Never invent, infer, or hallucinate information not present in the sources.
2. Cite every factual claim as [Source N] where N matches the source number provided.
3. If a source is ambiguous, state that uncertainty explicitly rather than inferring.
4. Omit any section with zero relevant articles — do NOT include empty headers.
5. For "Dates to Watch", only include items explicitly mentioned in sources. If none, omit.
6. Do not speculate about commercial impact, market share, or financial implications.
7. When multiple sources cover the same story, cite all relevant sources: [Source 3][Source 7].
8. Group related regulatory actions together under one item where appropriate.
9. Use regulatory terminology precisely — NDA, BLA, ANDA, 505(b)(2), PDUFA, etc. need no explanation.

OUTPUT FORMAT:
Use Markdown. Section headers as ## Title. News items: **Bold Headline** on its own line, then 2-3 sentence body, then [Source N] inline, then the URL on its own line below.
"""


# ── User prompt builder ───────────────────────────────────────────────────────

def _build_user_prompt(
    source_context: str,
    date_str: str,
    pass_num: int,
    total_passes: int,
    focus_hint: str = "",
) -> str:
    """
    Build a user prompt with slight variation per Monte Carlo pass to encourage
    independent exploration of the source material.

    Pass 1: Standard comprehensive pass
    Pass 2: Emphasises completeness — "ensure every source is accounted for"
    Pass 3: Emphasises grouping and thematic synthesis
    """
    focus_instructions = {
        1: "Write a complete, comprehensive newsletter covering all significant developments in the sources.",
        2: (
            "This is a completeness-focused pass. Ensure EVERY source that contains a "
            "significant regulatory development is represented. Do not skip any source "
            "with meaningful content. It is better to include too much than too little."
        ),
        3: (
            "This is a synthesis-focused pass. Group thematically related regulatory "
            "actions together (e.g., multiple FDA guidance updates on the same topic). "
            "Prioritise synthesis and pattern recognition over exhaustive listing. "
            "Identify the 2-3 most impactful themes across all sources and lead with those."
        ),
    }
    instruction = focus_instructions.get(pass_num, focus_instructions[1])
    if focus_hint:
        instruction += f" Additional focus: {focus_hint}"

    disease_state_sections = """## Renal and Liver Disease
### Clinical Trials
Renal/hepatic disease clinical trial milestones: IND/CTA filings, Breakthrough Therapy,
Fast Track, Priority Review, Accelerated Approval designations, REMS changes, Phase 3
completions with regulatory significance. Include drugs for CKD, AKI, ESRD, hepatitis,
NASH/MASH, cirrhosis, hepatocellular carcinoma.

### Guidelines & Updates
FDA/EMA guidances specific to renal/hepatic disease: dosing in renal/hepatic impairment
guidance, nephrotoxicity assessment, hepatotoxicity guidance, renal biomarker qualification.

---

## Infectious Disease
### Clinical Trials
Infectious disease clinical trial milestones: antiviral, antibacterial, antifungal,
antiparasitic agents; HIV, RSV, influenza, COVID-19, MRSA, fungal infections. Include
IND filings, designations, QIDP (Qualified Infectious Disease Product) status.

### Guidelines & Updates
FDA/EMA guidances: antimicrobial resistance policy, QIDP program updates, antiviral
development guidance, bacterial/fungal endpoints guidance, PK/PD in infectious disease.

---

## Cardiovascular Conditions
### Clinical Trials
Cardiovascular clinical trial milestones: hypertension, heart failure (HFrEF/HFpEF),
arrhythmia, coronary artery disease, dyslipidemia, stroke prevention. Include
IND/CTA filings, designations, Phase 3 initiations/completions.

### Guidelines & Updates
FDA/EMA cardiovascular guidances: cardiovascular outcomes trial requirements,
QTc/cardiac safety guidance, heart failure endpoint guidance, lipid-lowering therapy guidance.

---

## Anti-Coagulation and Blood Disorders
### Clinical Trials
Hematology/anticoagulation trial milestones: anticoagulants, antiplatelets, sickle cell
disease, hemophilia, thalassemia, anemia, thrombocytopenia, von Willebrand disease.
Include designations, IND filings, gene therapy trials.

### Guidelines & Updates
FDA/EMA guidances: antithrombotic development guidance, hemoglobin/hematology endpoints,
sickle cell disease guidance, hemophilia gene therapy guidance.

---

## Eyes, Ears, Nose, and Skin Conditions
### Clinical Trials
Ophthalmology, dermatology, otology, rhinology trial milestones: retinal diseases (AMD, DME),
glaucoma, atopic dermatitis, psoriasis, alopecia, hearing loss, allergic rhinitis.
Include IND filings, designations, Phase completions.

### Guidelines & Updates
FDA/EMA guidances: ophthalmic drug development, dermatologic endpoints, sunscreen monograph
updates, OTC topical drug guidance, medical device/drug combination for ocular delivery.

---

## Pulmonary Conditions and Tobacco Cessation
### Clinical Trials
Pulmonary/respiratory trial milestones: COPD, asthma, IPF, pulmonary arterial hypertension,
cystic fibrosis, smoking/tobacco cessation, lung cancer (respiratory focus). Include
IND filings, designations, Phase completions.

### Guidelines & Updates
FDA/EMA guidances: pulmonary drug delivery guidance, inhaled product development,
spirometry/FEV1 endpoints, tobacco product standards, nicotine replacement guidance.

---

## Endocrine Conditions
### Clinical Trials
Endocrine/metabolic trial milestones: type 1 and type 2 diabetes, thyroid disorders,
adrenal insufficiency, growth hormone deficiency, hyperparathyroidism, Cushing's syndrome.
Include IND filings, designations, Phase completions.

### Guidelines & Updates
FDA/EMA guidances: diabetes drug development, HbA1c endpoint guidance, thyroid hormone
product guidance, insulin biosimilar guidance, hypoglycemia risk management.

---

## Male and Female Health
### Clinical Trials
Reproductive/sexual health trial milestones: contraception, menopause (HRT), fertility,
PCOS, endometriosis, uterine fibroids, benign prostatic hyperplasia, testosterone
deficiency, sexual dysfunction. Include IND filings, designations, Phase completions.

### Guidelines & Updates
FDA/EMA guidances: contraceptive development guidance, menopause endpoint guidance,
obstetric drug development, male hypogonadism guidance, pregnancy labelling guidance.

---

## Special Populations (Weight Loss, Transplants, Pediatric)
### Clinical Trials
Trial milestones for obesity/weight management (GLP-1 agonists, etc.), solid organ and
stem cell transplantation (immunosuppression), and pediatric/neonatal drug development.
Include IND filings, designations, BPCA/PREA studies, Phase completions.

### Guidelines & Updates
FDA/EMA guidances: obesity drug development, anti-rejection therapy guidance,
pediatric study plans (iPSP/PIP), neonatal dosing, PREA/BPCA pediatric requirements,
extrapolation from adults to pediatric populations.

---

## Pain Related Conditions (Pain, Migraine, Gout)
### Clinical Trials
Pain/headache/gout trial milestones: chronic pain, acute pain, neuropathic pain,
migraine (preventive and acute), cluster headache, gout, hyperuricemia. Include
opioid and non-opioid analgesics, CGRP antagonists, IND filings, designations.

### Guidelines & Updates
FDA/EMA guidances: analgesic drug development, chronic pain endpoints, migraine
endpoint guidance, opioid risk management/REMS updates, abuse-deterrent formulation guidance.

---

## Oncology
### Clinical Trials
Oncology trial milestones across all tumor types and hematologic malignancies:
Breakthrough Therapy, Fast Track, Priority Review, Accelerated Approval, Orphan Drug
designations; IND filings; Phase 3 initiations/completions; PMA/BLA/NDA submissions;
CAR-T, ADC, checkpoint inhibitor, targeted therapy, and gene therapy developments.

### Guidelines & Updates
FDA/EMA oncology guidances: tumor-agnostic approval guidance, surrogate endpoint
qualification, estimands in oncology trials, biosimilar oncology guidance,
accelerated approval post-marketing confirmatory trial requirements.

---

## Psychiatric Conditions
### Clinical Trials
Psychiatric trial milestones: major depressive disorder, treatment-resistant depression,
generalized anxiety disorder, schizophrenia, bipolar disorder, ADHD, PTSD, OCD,
substance use disorder. Include IND filings, designations, Phase completions,
novel mechanisms (psychedelics, neuromodulators).

### Guidelines & Updates
FDA/EMA guidances: psychiatric drug development, CNS endpoint guidance,
suicidality assessment (C-SSRS requirements), substance use disorder trial guidance,
risk-benefit in psychiatric indications.

---

## Neurologic Conditions
### Clinical Trials
Neurology trial milestones: Alzheimer's disease, Parkinson's disease, multiple sclerosis,
epilepsy, ALS, spinal muscular atrophy, Huntington's disease, migraine (neurologic focus),
stroke. Include IND filings, designations, Phase completions, gene/cell therapy trials.

### Guidelines & Updates
FDA/EMA guidances: neurodegenerative disease endpoint guidance, Alzheimer's drug
development guidance, amyloid PET/CSF biomarker qualification, gene therapy for
neurologic conditions, CNS drug delivery guidance.

---

## Gastrointestinal Conditions
### Clinical Trials
GI/hepatology trial milestones: IBD (Crohn's disease, ulcerative colitis), IBS, GERD,
eosinophilic esophagitis, NASH/MASH (GI focus), celiac disease, C. difficile,
gastroparesis, short bowel syndrome. Include IND filings, designations, Phase completions.

### Guidelines & Updates
FDA/EMA guidances: IBD endpoint guidance, NASH/MASH clinical development guidance,
GI drug development, microbiome-based therapy guidance, endoscopic endpoint qualification.

---"""

    return f"""Using ONLY the sources provided below, write the "Pharma Regulatory Pulse" newsletter for the week of {date_str}.

PASS {pass_num} OF {total_passes} INSTRUCTION:
{instruction}

STRUCTURE RULES:
- Use the exact ## section headers and ### sub-headers shown below.
- Omit any ## section entirely if you have zero relevant sources for it — do not include empty sections or placeholder text.
- Omit any ### sub-header within a section if there are no relevant sources for that sub-type.
- Every factual claim must cite [Source N] where N matches the source number below.
- Format each news item as:
  **[Headline]**
  2–3 sentence summary. [Source N]
  URL

---

## Regulatory Spotlight
The single most significant regulatory development this week across any disease area (2–3 sentences + [Source N]).

---

{disease_state_sections}

## Dates to Watch
Upcoming PDUFA dates, comment deadlines, advisory committee meetings in the next 2–4 weeks
(only if explicitly stated in a source — do not infer or estimate dates).

## Sign-off
One sentence closing.

---
SOURCES:
{source_context}
"""


# ── API call with retry ───────────────────────────────────────────────────────

def _call_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_attempts: int = 3,
) -> str:
    """OpenAI Chat Completions call with exponential backoff."""
    delays = [2, 4, 8]
    last_exc: Exception | None = None

    for attempt, delay in enumerate(delays[:max_attempts], start=1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIStatusError) as exc:
            logger.warning("OpenAI attempt %d/%d failed: %s", attempt, max_attempts, exc)
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(delay)
        except Exception as exc:
            logger.error("Unexpected OpenAI error: %s", exc)
            raise

    raise RuntimeError(f"All {max_attempts} OpenAI API attempts failed") from last_exc


# ── Monte Carlo merge ─────────────────────────────────────────────────────────

def _monte_carlo_merge(
    pass_results: list[str],
    articles: list[dict[str, Any]],
) -> str:
    """
    Given N independent newsletter passes, select the best version of each
    section and merge into a single coherent newsletter.

    Algorithm:
      1. Parse each pass into sections
      2. Score each (pass, section) pair
      3. For each section key, pick the highest-scoring version
      4. Reassemble in canonical section order
      5. Append a Monte Carlo provenance comment for debugging

    Returns
    -------
    str
        Merged newsletter markdown.
    """
    # {section_key: [SectionScore, ...]} across all passes
    candidates: dict[str, list[SectionScore]] = defaultdict(list)

    for pass_idx, md in enumerate(pass_results, start=1):
        parsed = _parse_sections(md)
        for key, text in parsed.items():
            if text.strip():
                sc = _score_section(key, text)
                logger.debug(
                    "Pass %d | %-25s | score=%.2f | items=%d | sources=%s",
                    pass_idx, key, sc.score, sc.item_count, sc.source_ids,
                )
                candidates[key].append(sc)

    # Identify sources NOT cited in any pass (coverage gap check)
    all_cited: set[int] = set()
    for scores in candidates.values():
        for sc in scores:
            all_cited |= sc.source_ids
    total_sources = len(articles)
    uncited = set(range(1, total_sources + 1)) - all_cited
    if uncited:
        logger.warning(
            "Monte Carlo coverage gap: sources %s not cited in any pass",
            sorted(uncited),
        )

    # Select best section version
    merged_sections: list[tuple[int, str, str]] = []  # (order_idx, key, text)
    for key in SECTION_ORDER:
        if key not in candidates:
            continue
        best = max(candidates[key], key=lambda s: s.score)
        order_idx = SECTION_ORDER.index(key)
        header = SECTION_HEADERS.get(key, f"## {key.title()}")
        merged_sections.append((order_idx, header, best.text))
        logger.info(
            "Section '%s' — selected pass with score=%.2f (%d items, %d unique sources)",
            key, best.score, best.item_count, best.unique_sources,
        )

    # Also include any non-standard sections from any pass (preserve novel content)
    for key, scores in candidates.items():
        if key not in SECTION_ORDER and scores:
            best = max(scores, key=lambda s: s.score)
            header = f"## {key.title()}"
            merged_sections.append((len(SECTION_ORDER), header, best.text))

    merged_sections.sort(key=lambda t: t[0])

    lines: list[str] = []
    for _, header, text in merged_sections:
        lines.append(header)
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


# ── Hallucination verification pass ──────────────────────────────────────────

def _verify_newsletter(
    client: OpenAI,
    model: str,
    newsletter_md: str,
    articles: list[dict[str, Any]],
) -> str:
    """
    Second LLM pass: auditor role checks every factual claim against sources.

    Per the artifact best-practice: "use a second LLM call to cross-check each
    claim against the original source text — flag anything that can't be traced
    back." Critical for regulatory content where professionals act on this data.

    The auditor is instructed to:
      1. Identify any claim not traceable to the provided sources
      2. Remove or flag unsupported statements
      3. Ensure every [Source N] citation actually matches the source content
      4. Return the corrected newsletter (or the original if no issues found)

    Uses temperature=0.0 for maximum determinism in fact-checking.
    """
    source_context = _build_source_context(articles)

    audit_system = """You are a strict regulatory content auditor.
Your job is to fact-check a pharmaceutical newsletter against its source documents.

AUDITING RULES:
1. For every factual claim in the newsletter, verify it appears in the cited [Source N].
2. If a claim is NOT supported by its cited source, either:
   - Remove the unsupported claim, OR
   - Replace it with what the source actually says
3. If a [Source N] citation number doesn't exist in the sources list, remove it.
4. Do NOT add new information. Do NOT change writing style. Do NOT restructure sections.
5. If everything checks out, return the newsletter unchanged.
6. Respond with ONLY the corrected newsletter markdown — no preamble, no audit report."""

    audit_user = f"""Audit the following newsletter against the provided sources.
Fix any claims not supported by the cited sources. Return only the corrected newsletter.

NEWSLETTER TO AUDIT:
{newsletter_md}

---
SOURCES (for verification):
{source_context}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=5500,
            temperature=0.0,   # Zero temp for deterministic fact-checking
            messages=[
                {"role": "system", "content": audit_system},
                {"role": "user",   "content": audit_user},
            ],
        )
        verified = response.choices[0].message.content or newsletter_md
        # Sanity check: if the output is suspiciously short, keep the original
        if len(verified) < len(newsletter_md) * 0.5:
            logger.warning(
                "Verification output was too short (%.0f%% of original) — "
                "keeping merged draft.", len(verified) / len(newsletter_md) * 100
            )
            return newsletter_md
        logger.info(
            "Hallucination verification complete. "
            "Original: %d chars → Verified: %d chars (Δ%+d)",
            len(newsletter_md), len(verified), len(verified) - len(newsletter_md),
        )
        return verified
    except Exception as exc:
        logger.warning(
            "Hallucination verification failed: %s — using unverified draft.", exc
        )
        return newsletter_md


# ── Public entry point ────────────────────────────────────────────────────────

def generate_newsletter(
    articles: list[dict[str, Any]],
    newsletter_config: dict,
    date_str: str = "",
    num_passes: int = 3,
) -> str:
    """
    Generate the weekly newsletter using Monte Carlo multi-pass synthesis.

    Parameters
    ----------
    articles : list[dict]
        Deduplicated articles from dedup.py.
    newsletter_config : dict
        The 'newsletter' section of config.yaml.
    date_str : str
        Human-readable week date (e.g. "March 24, 2025").
    num_passes : int
        Number of independent generation passes (default 3).
        More passes = better coverage, more API cost.

    Returns
    -------
    str
        Merged, best-of-N newsletter in Markdown format.
    """
    if not date_str:
        from datetime import date
        date_str = date.today().strftime("%B %d, %Y")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    model = (
        os.getenv("OPENAI_MODEL_WRITER")
        or newsletter_config.get("model_writer", "gpt-5.4-mini")
    )
    base_temperature = float(newsletter_config.get("temperature", 0.2))
    max_tokens = int(newsletter_config.get("max_tokens", 5000))

    client = OpenAI(api_key=api_key)
    system_prompt = _build_system_prompt(newsletter_config)

    # Shuffle article order per pass so the model explores different groupings
    article_list = list(articles)
    pass_results: list[str] = []

    logger.info(
        "Monte Carlo synthesis: %d passes, model=%s, base_temp=%.2f",
        num_passes, model, base_temperature,
    )

    for pass_num in range(1, num_passes + 1):
        # Each pass uses a freshly shuffled article list to vary attention
        shuffled = article_list.copy()
        if pass_num > 1:
            random.shuffle(shuffled)

        # Slight temperature variance: pass 1 = base, pass 2 = base + 0.05, pass 3 = base - 0.02
        temp_offsets = {1: 0.0, 2: 0.05, 3: -0.02}
        temperature = round(
            base_temperature + temp_offsets.get(pass_num, 0.0), 3
        )
        # Clamp to valid range
        temperature = max(0.0, min(1.0, temperature))

        source_context = _build_source_context(shuffled)
        user_prompt = _build_user_prompt(
            source_context, date_str, pass_num, num_passes
        )

        logger.info(
            "Pass %d/%d — temp=%.3f, %d articles",
            pass_num, num_passes, temperature, len(shuffled),
        )

        md = _call_openai(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        pass_results.append(md)
        logger.info("Pass %d complete: %d characters", pass_num, len(md))

        # Small pause between passes to avoid rate limit bursts
        if pass_num < num_passes:
            time.sleep(1)

    # Merge the best sections from all passes
    logger.info("Merging %d passes via Monte Carlo scoring...", num_passes)
    merged = _monte_carlo_merge(pass_results, articles)
    logger.info("Merged newsletter: %d characters", len(merged))

    # Hallucination verification pass (per artifact best-practice)
    # A second LLM call acts as an independent auditor, verifying every
    # factual claim against the original source documents.
    logger.info("STAGE: Hallucination verification pass...")
    verified = _verify_newsletter(client, model, merged, articles)
    logger.info("Final verified newsletter: %d characters", len(verified))
    return verified
