"""
format_email.py — Pharma Regulatory Pulse
Converts the Markdown newsletter into a visually polished, responsive HTML email.

Design principles (revised):
  - Full-width card on desktop (800 px), fluid/stacked on mobile
  - Disease-state ## headers as full-width coloured dividers
  - ### sub-headers (Clinical Trials / Guidelines) as smaller indented labels
  - Each news item indented under its sub-header, with a source-labelled button
  - No bare URLs anywhere — every link is a descriptor button inside the item card
  - Article-count summary banner above the Regulatory Spotlight hero
  - Spotlight: large title + smaller subtitle, visually separated
  - Larger base fonts throughout (16 px body, 18 px item headlines)
"""

from __future__ import annotations

import logging
import re
from datetime import date

import mjml

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
BRAND_NAVY      = "#0D2B45"   # Deep navy — primary headers, hero bg
BRAND_BLUE      = "#1B4F72"   # Mid pharma blue — section dividers
BRAND_ACCENT    = "#2E86C1"   # Bright blue — links, badges, buttons
BRAND_LIGHT_BG  = "#EBF5FB"   # Very light blue — summary banner, disclaimer
BRAND_RULE      = "#AED6F1"   # Pale blue — horizontal rules, borders
TEXT_BODY       = "#1A252F"   # Near-black — body text
TEXT_SECONDARY  = "#566573"   # Mid-grey — meta text, dates
TEXT_MUTED      = "#909AA5"   # Light grey — footer
BG_PAGE         = "#F0F3F4"   # Off-white — page wrapper
BG_CARD         = "#FFFFFF"   # Pure white — email body

DISCLAIMER = (
    "This newsletter is generated using AI-assisted research and synthesis. "
    "Always verify regulatory information against official FDA, EMA, and ICH "
    "primary sources before taking compliance or submission actions."
)

# ── Section icons ─────────────────────────────────────────────────────────────
SECTION_META: dict[str, tuple[str, str]] = {
    "regulatory spotlight":                       ("🔦", BRAND_NAVY),
    "renal and liver disease":                    ("🫘", "#1A5276"),
    "infectious disease":                         ("🦠", "#1A5276"),
    "cardiovascular conditions":                  ("❤️",  "#922B21"),
    "anti-coagulation and blood disorders":       ("🩸", "#922B21"),
    "eyes, ears, nose, and skin conditions":      ("👁️",  "#6E2F19"),
    "pulmonary conditions and tobacco cessation": ("🫁", "#117A65"),
    "endocrine conditions":                       ("⚗️",  "#117A65"),
    "male and female health":                     ("🧬", "#6C3483"),
    "special populations (weight loss, transplants, pediatric)": ("👶", "#6C3483"),
    "pain related conditions (pain, migraine, gout)":            ("💊", "#4A235A"),
    "oncology":                                   ("🎗️",  "#145A32"),
    "psychiatric conditions":                     ("🧠", "#4A235A"),
    "neurologic conditions":                      ("⚡", "#1F618D"),
    "gastrointestinal conditions":                ("🏥", "#784212"),
    "dates to watch":                             ("📅", "#4A235A"),
    "sign-off":                                   ("✉️",  TEXT_SECONDARY),
}


# ── URL → human-readable source label ────────────────────────────────────────
def _url_to_source_label(url: str) -> str:
    """
    Map a URL to a descriptive button label so readers know where they're going
    without ever seeing the raw URL.
    """
    rules = [
        # ── Regulatory agencies ──────────────────────────────────────────────
        ("fda.gov/drugs",               "Full Guidance — FDA Drugs"),
        ("fda.gov/biologics",           "Full Guidance — FDA Biologics"),
        ("fda.gov/safety",              "FDA Safety Communication"),
        ("fda.gov/inspections",         "FDA Enforcement Action"),
        ("fda.gov/medical-devices",     "FDA Medical Devices"),
        ("fda.gov",                     "Read on FDA.gov"),
        ("ema.europa.eu",               "Read on EMA.europa.eu"),
        ("ich.org",                     "Read ICH Guideline"),
        ("who.int",                     "Read on WHO.int"),
        ("clinicalinfo.hiv.gov",        "HHS HIV Guidelines"),
        # ── Cardiovascular ───────────────────────────────────────────────────
        ("acc.org",                     "ACC Guideline / Statement"),
        ("heart.org",                   "AHA Guideline / Statement"),
        # ── Hematology / Anticoagulation ─────────────────────────────────────
        ("chestnet.org",                "CHEST / ACCP Guideline"),
        ("hematology.org",              "ASH Guideline"),
        # ── Endocrine / Diabetes ─────────────────────────────────────────────
        ("diabetes.org",                "ADA Standards of Care"),
        ("aace.com",                    "AACE Clinical Guideline"),
        # ── Infectious Disease ───────────────────────────────────────────────
        ("idsociety.org",               "IDSA Practice Guideline"),
        ("survivingsepsis.org",         "Surviving Sepsis Campaign"),
        # ── Pulmonary ────────────────────────────────────────────────────────
        ("ginasthma.org",               "GINA Asthma Guidelines"),
        ("goldcopd.org",                "GOLD COPD Guidelines"),
        ("thoracic.org",                "ATS Clinical Practice Guideline"),
        # ── Neurology ────────────────────────────────────────────────────────
        ("aan.com",                     "AAN Practice Guideline"),
        # ── Psychiatry ───────────────────────────────────────────────────────
        ("psychiatry.org",              "APA Practice Guideline"),
        # ── Pediatrics / Women's Health ──────────────────────────────────────
        ("aap.org",                     "AAP Clinical Practice Guideline"),
        ("acog.org",                    "ACOG Practice Bulletin"),
        ("auanet.org",                  "AUA Guideline"),
        # ── Oncology ─────────────────────────────────────────────────────────
        ("nccn.org",                    "NCCN Clinical Practice Guidelines"),
        # ── Renal ────────────────────────────────────────────────────────────
        ("kdigo.org",                   "KDIGO Clinical Practice Guideline"),
        # ── Gastroenterology ─────────────────────────────────────────────────
        ("gi.org",                      "ACG Clinical Guideline"),
        # ── Rheumatology / Dermatology ───────────────────────────────────────
        ("rheumatology.org",            "ACR Guideline"),
        ("aad.org",                     "AAD Clinical Practice Guideline"),
        # ── Vaccines / Public Health ─────────────────────────────────────────
        ("cdc.gov/vaccines/acip",       "ACIP Vaccine Recommendation"),
        ("cdc.gov",                     "CDC Guideline / Advisory"),
        # ── Literature / Data ────────────────────────────────────────────────
        ("pubmed.ncbi.nlm.nih.gov",     "Read on PubMed"),
        ("ncbi.nlm.nih.gov",            "Read on NIH/NCBI"),
        ("clinicaltrials.gov",          "ClinicalTrials.gov Record"),
        # ── Industry / News ──────────────────────────────────────────────────
        ("drugs.com",                   "Read on Drugs.com"),
        ("rxlist.com",                  "Read on RxList"),
        ("medscape.com",                "Read on Medscape"),
        ("fiercepharma.com",            "Read on FiercePharma"),
        ("biopharmadive.com",           "Read on BioPharma Dive"),
        ("statnews.com",                "Read on STAT News"),
        ("reuters.com",                 "Read on Reuters"),
        ("bloomberg.com",               "Read on Bloomberg"),
        ("pharmatimes.com",             "Read on PharmaTimes"),
        ("news.google.com",             "Read on Google News"),
    ]
    lower = url.lower()
    for fragment, label in rules:
        if fragment in lower:
            return label
    return "Read Article →"


# ── Date formatter ────────────────────────────────────────────────────────────
def _format_pub_date(raw: str) -> str:
    """
    Convert a raw date string (YYYY-MM-DD, YYYYMMDD, or similar) to
    a readable abbreviated form like 'Mar 15, 2026'.
    Returns empty string for unknown/missing dates.
    """
    if not raw or raw.lower() in ("unknown", "none", ""):
        return ""
    m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", raw)
    if m:
        try:
            from datetime import date as _date
            d = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.strftime("%b %d, %Y")
        except ValueError:
            pass
    # Plain YYYYMMDD (openFDA format)
    m2 = re.match(r"^(\d{4})(\d{2})(\d{2})$", raw)
    if m2:
        try:
            from datetime import date as _date
            d = _date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
            return d.strftime("%b %d, %Y")
        except ValueError:
            pass
    return raw[:10]  # fallback: first 10 chars as-is


# ── Inline transforms ─────────────────────────────────────────────────────────
def _citation_badge(match: re.Match) -> str:
    """Render [Source N] as a small inline badge."""
    n = match.group(1)
    return (
        f'<span style="display:inline-block;background:{BRAND_ACCENT};color:#fff;'
        f'font-size:11px;font-weight:bold;padding:2px 8px;border-radius:10px;'
        f'vertical-align:middle;margin-left:5px;">[{n}]</span>'
    )


def _apply_inline(text: str) -> str:
    """Apply citation badges and markdown links to a string."""
    text = re.sub(r"\[Source (\d+)\]", _citation_badge, text)
    text = re.sub(
        r"\[(.+?)\]\((https?://[^\)]+)\)",
        rf'<a href="\2" style="color:{BRAND_ACCENT};font-weight:500;">\1</a>',
        text,
    )
    return text


# ── News item card ─────────────────────────────────────────────────────────────
def _render_item(bold_title: str, body: str, url: str = "", pub_date: str = "") -> str:
    """
    Render a news item as an indented card.
    Indented via margin-left so it visually belongs under the sub-header above it.
    Source link is a labelled button with the publication date appended.
    No bare URLs are shown anywhere in the output.
    """
    button_html = ""
    if url:
        label = _url_to_source_label(url)
        date_suffix = f"  ·  {pub_date}" if pub_date else ""
        button_html = (
            f'<div style="margin-top:12px;">'
            f'<a href="{url}" style="display:inline-block;background:{BRAND_ACCENT};'
            f'color:#fff;font-size:14px;font-weight:600;padding:7px 18px;'
            f'border-radius:4px;text-decoration:none;letter-spacing:0.2px;">'
            f'{label}{date_suffix}</a>'
            f'</div>'
        )
    return (
        f'<div style="border-left:3px solid {BRAND_ACCENT};'
        f'padding:14px 18px;margin:0 0 22px 20px;'
        f'background:#FAFCFF;border-radius:0 5px 5px 0;">'
        f'<p style="margin:0 0 8px;font-size:18px;font-weight:700;color:{TEXT_BODY};">'
        f'{bold_title}</p>'
        f'<p style="margin:0;font-size:16px;line-height:1.75;color:{TEXT_BODY};">'
        f'{body}</p>'
        f'{button_html}'
        f'</div>'
    )


# ── Section headers ────────────────────────────────────────────────────────────
def _render_section_header(title: str, level: int = 2) -> str:
    """
    level=2 → full-width coloured bar (## disease-state headers)
    level=3 → smaller indented sub-label (### Clinical Trials / Guidelines)
    """
    if level == 3:
        return (
            f'<p style="margin:20px 0 10px 0;font-size:14px;font-weight:800;'
            f'color:{BRAND_BLUE};text-transform:uppercase;letter-spacing:1px;'
            f'border-bottom:2px solid {BRAND_RULE};padding-bottom:5px;">'
            f'{title}</p>'
        )
    # level 2 — full coloured bar
    lower = title.lower().strip()
    icon, colour = SECTION_META.get(lower, ("📋", BRAND_BLUE))
    return (
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="margin:32px 0 14px;">'
        f'<tr>'
        f'<td style="background:{colour};padding:11px 18px;border-radius:4px;">'
        f'<span style="font-size:14px;font-weight:800;color:#ffffff;'
        f'letter-spacing:0.8px;text-transform:uppercase;">'
        f'{icon}&nbsp;&nbsp;{title}</span>'
        f'</td>'
        f'</tr>'
        f'</table>'
    )


# ── Markdown → structured HTML ────────────────────────────────────────────────
def _md_to_html(md: str, url_to_date: dict[str, str] | None = None) -> str:
    """
    Convert Markdown newsletter body to rich HTML for MJML injection.

    Key behaviours:
      - ## → coloured full-width bar (disease state)
      - ### → smaller sub-header label (Clinical Trials / Guidelines)
      - **Bold** at line start → opens a news item card
      - Bare URL line immediately after item body → stored as item button + date lookup
      - [Source N] → inline badge
      - Empty line → flushes the current item
      - Empty sections/sub-sections (header with no items) → silently removed
    """
    if url_to_date is None:
        url_to_date = {}

    html_parts: list[str] = []
    current_item_title: str | None = None
    current_item_body_lines: list[str] = []
    current_item_url: str = ""

    # ── Empty-section tracking ────────────────────────────────────────────────
    # Stores the index in html_parts where the most recent ## / ### header was
    # appended. If nothing is added after that index before the next header of
    # the same or higher level, the header is retroactively removed.
    last_h2_start: int = -1
    last_h3_start: int = -1

    def has_content_after(idx: int) -> bool:
        """True if at least one element exists in html_parts beyond position idx."""
        return len(html_parts) > idx + 1

    def flush_item() -> None:
        nonlocal current_item_title, current_item_body_lines, current_item_url
        if current_item_title:
            body = " ".join(current_item_body_lines).strip()
            pub_date = _format_pub_date(url_to_date.get(current_item_url, ""))
            html_parts.append(
                _render_item(current_item_title, body, current_item_url, pub_date)
            )
        current_item_title = None
        current_item_body_lines = []
        current_item_url = ""

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Section headers (## and ###) ─────────────────────────────────────
        h3_match = re.match(r"^###\s+(.+)$", stripped)
        h2_match = re.match(r"^##\s+(.+)$", stripped) if not h3_match else None
        h1_match = re.match(r"^#\s+(.+)$", stripped) if not h3_match and not h2_match else None

        if h3_match:
            flush_item()
            # Remove previous ### sub-header if it produced no content
            if last_h3_start >= 0 and not has_content_after(last_h3_start):
                del html_parts[last_h3_start:]
            last_h3_start = len(html_parts)
            html_parts.append(_render_section_header(h3_match.group(1), level=3))
            i += 1
            continue

        if h2_match or h1_match:
            flush_item()
            title_text = (h2_match or h1_match).group(1)
            # Remove previous ### sub-header if it produced no content
            if last_h3_start >= 0 and not has_content_after(last_h3_start):
                del html_parts[last_h3_start:]
                last_h3_start = -1
            # Remove previous ## section header if it produced no content at all
            if last_h2_start >= 0 and not has_content_after(last_h2_start):
                del html_parts[last_h2_start:]
            last_h2_start = len(html_parts)
            last_h3_start = -1
            html_parts.append(_render_section_header(title_text, level=2))
            i += 1
            continue

        # ── Horizontal rule ──────────────────────────────────────────────────
        if re.match(r"^-{3,}$", stripped):
            flush_item()
            html_parts.append(
                f'<hr style="border:none;border-top:1px solid {BRAND_RULE};margin:24px 0;"/>'
            )
            i += 1
            continue

        # ── Empty line: flush current item ───────────────────────────────────
        if not stripped:
            flush_item()
            i += 1
            continue

        # ── Bare URL line — attach to current item as a button, never display ──
        if re.match(r"^https?://\S+$", stripped):
            if current_item_title is not None and not current_item_url:
                current_item_url = stripped
            # All other bare URLs silently dropped — no raw URLs in output
            i += 1
            continue

        # ── Apply inline transforms ──────────────────────────────────────────
        processed = _apply_inline(stripped)

        # ── Bold item title at line start → opens a news item card ──────────
        bold_start = re.match(r"^\*\*(.+?)\*\*(.*)$", processed)
        if bold_start:
            flush_item()
            current_item_title = bold_start.group(1)
            remainder = bold_start.group(2).strip()
            if remainder:
                current_item_body_lines.append(remainder)
            i += 1
            continue

        # ── Continuation line for current item ───────────────────────────────
        if current_item_title is not None:
            current_item_body_lines.append(processed)
            i += 1
            continue

        # ── Regular paragraph ─────────────────────────────────────────────────
        html_parts.append(
            f'<p style="margin:0 0 16px;font-size:16px;line-height:1.75;color:{TEXT_BODY};">'
            f'{processed}</p>'
        )
        i += 1

    flush_item()

    # ── Final trailing empty-section cleanup ──────────────────────────────────
    if last_h3_start >= 0 and not has_content_after(last_h3_start):
        del html_parts[last_h3_start:]
        last_h3_start = -1  # noqa: F841 (kept for symmetry)
    if last_h2_start >= 0 and not has_content_after(last_h2_start):
        del html_parts[last_h2_start:]

    return "\n".join(html_parts)


# ── Plain-text fallback ───────────────────────────────────────────────────────
def md_to_plain(md: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    text = re.sub(r"\[(.+?)\]\((https?://[^\)]+)\)", r"\1: \2", text)
    text = re.sub(r"^#{1,6} ", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[Source \d+\]", "", text)
    return text.strip()


# ── Spotlight extraction ──────────────────────────────────────────────────────
def _extract_spotlight(md: str) -> tuple[str, str, str]:
    """
    Extract the Regulatory Spotlight section and split it into:
      - title_html: the first **bold** line rendered as a large heading
      - body_html:  the remaining body text (smaller, subtitle treatment)
      - remaining_md: the rest of the newsletter with the spotlight block removed

    This gives the hero section a two-tier typographic hierarchy.
    """
    pattern = re.compile(
        r"##\s*Regulatory Spotlight\s*\n(.*?)(?=\n##|\Z)", re.DOTALL | re.IGNORECASE
    )
    m = pattern.search(md)
    if not m:
        return "", "", md

    spotlight_raw = m.group(1).strip()
    remaining = pattern.sub("", md).strip()

    # Split off the first **bold** headline as the large title
    title_html = ""
    body_lines: list[str] = []

    for line in spotlight_raw.split("\n"):
        s = line.strip()
        if not title_html:
            bold_match = re.match(r"^\*\*(.+?)\*\*", s)
            if bold_match:
                title_html = bold_match.group(1)
                # Anything after the bold on the same line goes into body
                rest = s[bold_match.end():].strip()
                if rest:
                    body_lines.append(rest)
                continue
        # Non-bold first line or lines after title
        if s:
            body_lines.append(s)

    # If no bold title found, treat entire text as body
    if not title_html and body_lines:
        title_html = body_lines.pop(0) if body_lines else ""

    # Clean up body: remove [Source N], apply bold
    body_text = " ".join(body_lines).strip()
    body_text = re.sub(r"\[Source \d+\]", "", body_text).strip()
    body_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body_text)

    return title_html, body_text, remaining


# ── MJML template ─────────────────────────────────────────────────────────────
def _build_mjml(
    spotlight_title: str,
    spotlight_body: str,
    body_html: str,
    newsletter_config: dict,
    date_str: str,
    article_count: int = 0,
    guideline_count: int = 0,
) -> str:
    title: str = newsletter_config.get("title", "Pharma Regulatory Pulse")
    subtitle: str = newsletter_config.get("subtitle", "Your Weekly Briefing")
    year = date.today().year

    # ── Summary banner (above hero) ──────────────────────────────────────────
    summary_text = ""
    if article_count > 0:
        guideline_note = ""
        if guideline_count > 0:
            guideline_note = (
                f" · <strong>{guideline_count} guideline/guidance "
                f"{'change' if guideline_count == 1 else 'changes'}</strong> detected"
            )
        summary_text = (
            f"This edition synthesized <strong>{article_count} articles</strong> "
            f"across 14 disease and condition areas{guideline_note}."
        )
    summary_section = f"""
    <!-- ===== ARTICLE COUNT SUMMARY ===== -->
    <mj-section background-color="{BRAND_LIGHT_BG}" padding="14px 0">
      <mj-column>
        <mj-text
          font-size="15px"
          color="{BRAND_BLUE}"
          align="center"
          font-weight="500"
          padding="0 32px"
        >{summary_text}</mj-text>
      </mj-column>
    </mj-section>""" if summary_text else ""

    # ── Regulatory Spotlight hero ─────────────────────────────────────────────
    hero_section = ""
    if spotlight_title:
        hero_section = f"""
    <!-- ===== SPOTLIGHT HERO ===== -->
    <mj-section background-color="{BRAND_NAVY}" padding="36px 0 30px">
      <mj-column>
        <mj-text
          font-size="12px"
          font-weight="800"
          color="{BRAND_ACCENT}"
          letter-spacing="2px"
          align="center"
          padding-bottom="14px"
          text-transform="uppercase"
        >🔦 Regulatory Spotlight</mj-text>
        <mj-text
          font-size="26px"
          font-weight="800"
          color="#FFFFFF"
          line-height="1.35"
          align="center"
          padding="0 32px 14px"
        >{spotlight_title}</mj-text>
        <mj-divider
          border-color="{BRAND_ACCENT}"
          border-width="1px"
          width="80px"
          padding="0 0 14px"
        />
        <mj-text
          font-size="17px"
          font-weight="400"
          color="#C8D6E0"
          line-height="1.65"
          align="center"
          padding="0 40px 0"
        >{spotlight_body}</mj-text>
      </mj-column>
    </mj-section>
    <mj-section background-color="{BRAND_ACCENT}" padding="3px 0" />"""

    return f"""<mjml>
  <mj-head>
    <mj-title>{title} — {date_str}</mj-title>
    <mj-preview>Pharma Regulatory Pulse | Week of {date_str} — FDA, EMA &amp; ICH updates for regulatory professionals</mj-preview>
    <mj-attributes>
      <mj-all font-family="'Helvetica Neue', Arial, sans-serif" />
      <mj-text font-size="16px" line-height="1.7" color="{TEXT_BODY}" />
      <mj-body width="800px" />
      <mj-section background-color="{BG_CARD}" padding="0" />
    </mj-attributes>
    <mj-style>
      a {{ color: {BRAND_ACCENT}; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      @media only screen and (max-width: 480px) {{
        .mj-column-full {{ width: 100% !important; }}
      }}
    </mj-style>
  </mj-head>

  <mj-body background-color="{BG_PAGE}">

    <!-- ===== HEADER BAR ===== -->
    <mj-section background-color="{BRAND_NAVY}" padding="22px 0 0">
      <mj-column>
        <mj-text
          font-size="11px"
          color="{BRAND_RULE}"
          align="center"
          letter-spacing="3px"
          font-weight="600"
          padding-bottom="6px"
        >PHARMACEUTICAL REGULATORY INTELLIGENCE</mj-text>
        <mj-text
          font-size="30px"
          font-weight="800"
          color="#FFFFFF"
          align="center"
          padding="0 0 4px"
          letter-spacing="-0.5px"
        >{title.upper()}</mj-text>
        <mj-text
          font-size="13px"
          color="{BRAND_RULE}"
          align="center"
          padding="0 0 18px"
        >Week of {date_str}</mj-text>
      </mj-column>
    </mj-section>

    {summary_section}

    {hero_section}

    <!-- ===== INTRO LINE ===== -->
    <mj-section background-color="{BG_CARD}" padding="22px 32px 6px">
      <mj-column>
        <mj-text
          font-size="15px"
          color="{TEXT_SECONDARY}"
          font-style="italic"
        >Your curated briefing on FDA guidances, EMA decisions, ICH harmonization, and enforcement actions — sourced from official channels and industry publications.</mj-text>
      </mj-column>
    </mj-section>

    <!-- ===== MAIN BODY ===== -->
    <mj-section background-color="{BG_CARD}" padding="10px 32px 32px">
      <mj-column>
        <mj-text padding="0">
          {body_html}
        </mj-text>
      </mj-column>
    </mj-section>

    <!-- ===== DISCLAIMER CALLOUT ===== -->
    <mj-section background-color="{BRAND_LIGHT_BG}" padding="0 32px">
      <mj-column>
        <mj-table padding="16px 0">
          <tr>
            <td style="padding:16px 18px;border:1px solid {BRAND_RULE};border-radius:6px;background:#fff;">
              <span style="font-size:20px;">⚠️</span>&nbsp;
              <span style="font-size:14px;color:{TEXT_SECONDARY};font-style:italic;line-height:1.6;">
                {DISCLAIMER}
              </span>
            </td>
          </tr>
        </mj-table>
      </mj-column>
    </mj-section>

    <!-- ===== DIVIDER ===== -->
    <mj-section background-color="{BG_CARD}" padding="0 32px">
      <mj-column>
        <mj-divider border-color="{BRAND_RULE}" border-width="1px" padding="26px 0 0" />
      </mj-column>
    </mj-section>

    <!-- ===== FOOTER ===== -->
    <mj-section background-color="{BG_CARD}" padding="14px 32px 30px">
      <mj-column>
        <mj-text font-size="13px" color="{TEXT_MUTED}" align="center">
          You're receiving this because you subscribed to <strong>Pharma Regulatory Pulse</strong>.<br/>
          <a href="#" style="color:{TEXT_MUTED};">Unsubscribe</a>
          &nbsp;·&nbsp;
          <a href="mailto:regulatory-team@yourdomain.com" style="color:{TEXT_MUTED};">Contact us</a>
          &nbsp;·&nbsp;
          <a href="#" style="color:{TEXT_MUTED};">View in browser</a>
        </mj-text>
        <mj-text font-size="12px" color="{TEXT_MUTED}" align="center" padding-top="8px">
          © {year} Pharma Regulatory Pulse. For regulatory professionals.
        </mj-text>
      </mj-column>
    </mj-section>

  </mj-body>
</mjml>"""


# ── Fallback HTML (if MJML fails) ─────────────────────────────────────────────
def _fallback_html(
    spotlight_title: str,
    spotlight_body: str,
    body_html: str,
    newsletter_config: dict,
    date_str: str,
    guideline_count: int = 0,
) -> str:
    title = newsletter_config.get("title", "Pharma Regulatory Pulse")
    year = date.today().year
    hero = ""
    if spotlight_title:
        hero = (
            f'<div style="background:{BRAND_NAVY};color:#fff;padding:36px 32px;text-align:center;">'
            f'<div style="font-size:12px;color:{BRAND_ACCENT};letter-spacing:2px;font-weight:800;'
            f'margin-bottom:14px;text-transform:uppercase;">🔦 Regulatory Spotlight</div>'
            f'<div style="font-size:26px;font-weight:800;line-height:1.35;margin-bottom:14px;">'
            f'{spotlight_title}</div>'
            f'<hr style="border:none;border-top:1px solid {BRAND_ACCENT};width:80px;margin:0 auto 14px;"/>'
            f'<div style="font-size:17px;color:#C8D6E0;line-height:1.65;">{spotlight_body}</div>'
            f'</div>'
            f'<div style="background:{BRAND_ACCENT};height:3px;"></div>'
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title} — {date_str}</title>
<style>
body{{font-family:'Helvetica Neue',Arial,sans-serif;background:{BG_PAGE};margin:0;padding:20px;color:{TEXT_BODY};}}
.wrap{{max-width:800px;margin:0 auto;background:{BG_CARD};border-radius:6px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}
.hdr{{background:{BRAND_NAVY};color:#fff;padding:22px;text-align:center;}}
.hdr h1{{margin:0;font-size:28px;letter-spacing:-0.5px;}}
.hdr small{{color:{BRAND_RULE};font-size:13px;}}
.body{{padding:28px 36px;font-size:16px;line-height:1.75;}}
.disc{{background:{BRAND_LIGHT_BG};border:1px solid {BRAND_RULE};border-radius:6px;padding:16px 18px;font-size:14px;color:{TEXT_SECONDARY};font-style:italic;margin:20px 0;}}
.footer{{padding:16px 36px 24px;font-size:12px;color:{TEXT_MUTED};text-align:center;border-top:1px solid {BRAND_RULE};}}
a{{color:{BRAND_ACCENT};}}
</style></head>
<body><div class="wrap">
<div class="hdr"><small style="letter-spacing:3px;font-size:11px;color:{BRAND_RULE};">PHARMACEUTICAL REGULATORY INTELLIGENCE</small>
<h1>{title.upper()}</h1><small>Week of {date_str}</small></div>
{f'<div style="background:{BRAND_LIGHT_BG};padding:10px 18px;text-align:center;font-size:14px;color:{BRAND_BLUE};">{guideline_count} guideline/guidance change{"" if guideline_count == 1 else "s"} detected this week</div>' if guideline_count > 0 else ''}
{hero}
<div class="body">{body_html}</div>
<div class="disc">⚠️ {DISCLAIMER}</div>
<div class="footer">© {year} Pharma Regulatory Pulse &nbsp;·&nbsp;
<a href="#">Unsubscribe</a> &nbsp;·&nbsp; <a href="#">View in browser</a></div>
</div></body></html>"""


# ── Public entry point ────────────────────────────────────────────────────────
def format_newsletter_html(
    newsletter_md: str,
    newsletter_config: dict,
    date_str: str = "",
    article_count: int = 0,
    articles: list[dict] | None = None,
) -> str:
    """
    Convert the Markdown newsletter to a polished responsive HTML email via MJML.

    Parameters
    ----------
    newsletter_md : str
        Full newsletter markdown from synthesize.py.
    newsletter_config : dict
        The 'newsletter' section of config.yaml.
    date_str : str
        Human-readable week date (e.g. "March 24, 2025").
    article_count : int
        Number of deduplicated articles that went into synthesis (shown in summary banner).
    articles : list[dict] | None
        Deduplicated article dicts from the pipeline. Used to build a URL→date
        lookup so each source button shows the article's publication date.
    """
    if not date_str:
        date_str = date.today().strftime("%B %d, %Y")

    # Build URL → formatted publication date lookup from the articles list
    url_to_date: dict[str, str] = {}
    guideline_count = 0
    if articles:
        for art in articles:
            url = art.get("url", "")
            raw_date = art.get("published_date", "")
            if url and raw_date:
                url_to_date[url] = raw_date
            if art.get("_guideline_boost"):
                guideline_count += 1

    spotlight_title, spotlight_body, remaining_md = _extract_spotlight(newsletter_md)
    body_html = _md_to_html(remaining_md, url_to_date=url_to_date)
    mjml_source = _build_mjml(
        spotlight_title, spotlight_body, body_html,
        newsletter_config, date_str, article_count, guideline_count,
    )

    try:
        result = mjml.mjml_to_html(mjml_source)
        if result.errors:
            for err in result.errors:
                logger.warning("MJML warning: %s", err)
        logger.info("HTML rendered: %d characters", len(result.html))
        return result.html
    except Exception as exc:
        logger.error("MJML rendering failed: %s — falling back to plain HTML", exc)
        return _fallback_html(
            spotlight_title, spotlight_body, body_html,
            newsletter_config, date_str, guideline_count,
        )
