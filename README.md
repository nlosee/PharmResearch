# Pharma Regulatory Pulse

**Automated weekly newsletter for pharmaceutical regulatory affairs professionals.**

Monitors FDA guidances, drug approvals, EMA decisions, ICH harmonization updates, enforcement actions, and clinical trial milestones — then synthesizes them into a professional HTML email newsletter delivered every Monday morning.

---

## What It Does

1. **Researches** — Pulls from FDA RSS feeds, EMA RSS feeds, Google News, openFDA REST API, PubMed, and ClinicalTrials.gov
2. **Deduplicates** — URL normalization + TF-IDF cosine similarity to remove duplicate stories
3. **Synthesizes** — Anthropic Claude generates the newsletter, grounded strictly in sourced articles (no hallucination)
4. **Formats** — MJML renders a responsive, inbox-compatible HTML email with pharma-blue branding
5. **Delivers** — Resend (or SMTP) sends to all subscribers in `subscribers.csv`
6. **Archives** — Saves each newsletter as `drafts/YYYY-MM-DD.md` for audit trail

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourorg/pharma-regulatory-pulse.git
cd pharma-regulatory-pulse
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your actual keys (see "API Keys" section below)
```

### 3. Add subscribers

Edit `subscribers.csv`:
```csv
email,name
john@yourcompany.com,John Smith
regulatory-team@yourcompany.com,Regulatory Affairs
```

### 4. Test with a dry run

```bash
python main.py --dry-run      # Generates HTML, prints to stdout — no emails sent
python main.py --draft-only   # Generates Markdown draft only
```

### 5. Full run

```bash
python main.py
```

---

## API Keys

| Key | Required | Free Tier | Get It |
|-----|----------|-----------|--------|
| `ANTHROPIC_API_KEY` | **Yes** | Pay-per-use | [console.anthropic.com](https://console.anthropic.com/) |
| `RESEND_API_KEY` | **Yes** (or SMTP) | 3,000/month | [resend.com](https://resend.com/) |
| `OPENFDA_API_KEY` | Optional | 120k req/day | [open.fda.gov](https://open.fda.gov/apis/authentication/) |
| `PUBMED_API_KEY` | Optional | Higher rate limits | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) |

FDA RSS feeds, EMA RSS feeds, Google News RSS, ClinicalTrials.gov, and openFDA base endpoints are all **free with no key required**.

---

## GitHub Actions Setup

1. Push this repo to GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Add these repository secrets:
   - `ANTHROPIC_API_KEY`
   - `RESEND_API_KEY`
   - `OPENFDA_API_KEY` (optional)
   - `PUBMED_API_KEY` (optional)
4. The workflow runs automatically every **Monday at 7 AM PT** (2 PM UTC)
5. Trigger manually: **Actions → Pharma Regulatory Pulse → Run workflow**

---

## Configuration

All settings live in `config.yaml`:

- **Topics** — search queries for Google News RSS
- **RSS feeds** — FDA, EMA, and industry feeds (add/remove as needed)
- **Email settings** — from address, reply-to, subject template
- **Model settings** — Claude model, temperature, max tokens
- **Minimum threshold** — abort if fewer than N articles survive dedup

---

## Email Delivery Providers

**Resend** (recommended): Set `email.provider: resend` in `config.yaml` and set `RESEND_API_KEY`. You'll need a verified sending domain.

**SMTP** (alternative): Set `email.provider: smtp` and configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` in `.env`.

---

## Project Structure

```
pharma-regulatory-pulse/
├── .github/workflows/newsletter.yml  # Monday 7AM PT automated delivery
├── config.yaml          # Topics, sources, email settings, model config
├── main.py              # Pipeline orchestrator (research→dedup→synthesize→format→deliver)
├── research.py          # FDA/EMA/Google News RSS + openFDA + PubMed + ClinicalTrials.gov
├── dedup.py             # URL normalization + TF-IDF cosine similarity dedup
├── synthesize.py        # Anthropic Claude newsletter generation (RAG grounded)
├── format_email.py      # MJML → responsive HTML email (pharma-blue branding)
├── deliver.py           # Resend/SMTP delivery with per-recipient error isolation
├── subscribers.csv      # Recipient list
├── drafts/              # Auto-saved markdown newsletters (audit trail)
├── generation_log.txt   # Auto-updated each run (keeps GitHub Actions alive)
├── requirements.txt     # Pinned Python dependencies
├── .env.example         # API key template
└── .gitignore
```

---

## Important Disclaimer

> This newsletter is generated using AI-assisted research and synthesis. Always verify regulatory information against official FDA, EMA, and ICH primary sources before taking compliance or submission actions.

This disclaimer appears in every newsletter footer and is non-negotiable per project requirements.

---

## Troubleshooting

**"Only N articles (minimum: 3)" abort** — Normal behavior when no news matches the past 7 days. The workflow exits cleanly (exit 0). Check your RSS feeds and topics in `config.yaml`.

**MJML rendering errors** — The email formatter falls back to plain HTML automatically. Check that `mjml` is installed correctly (`pip install mjml`).

**Resend "domain not verified"** — You must verify your sending domain at [resend.com/domains](https://resend.com/domains) before sending. Use SMTP as a fallback during setup.

**PubMed rate limits** — Run with `--no-pubmed` flag if hitting NCBI quotas: `python main.py --no-pubmed`
