# AlexDoor-XAS Technical Wiki Rules

Root `AGENTS.md` governs implementation, Git and testing. This file governs
`knowledge/`; explicit user instructions take precedence.

## Ownership

- `raw/` is user-owned: read/inventory only; never edit, rename, move or delete it.
- `wiki/` is agent-maintained. Preserve user edits and experimental decisions.
- Keep the existing hierarchy: `implementation_phases/`, `topics/`, `decisions/`,
  `experiments/`, `sources/`, plus `index.md`, `status.md` and `log.md`.

## Canonical Responsibilities

- `status.md`: current state, material limits and next action; no intake chronicle.
- Phase pages: implemented boundaries and future requirements, clearly separated.
- Topics: one explanation per reusable technical subject.
- Decisions: rationale and consequences of material choices.
- Experiments: durable question, method, result, limits and Git reference.
- Sources: provenance for material actually ingested from `raw/`.
- `log.md`: concise milestones, **not append-only**. Merge repeated updates and
  remove superseded detail; retain significant outcomes and references.

Use US English, concise paragraphs and meaningful Obsidian links. Link to the
canonical explanation instead of copying it. Keep per-asset operational details
in its JSON records. Preserve scientific limits when compressing historical text.
Git provides ordinary source/attempt history; removed ignored payloads are not
recoverable from Git and must never be described as if they were.

## Evidence and Synchronization

Current code/tests establish executable behavior; configs/records establish
contracts and acquired results; Git establishes historical phase attribution.
Do not infer implementation from a filename or plan. Label planned work and
separate observations, simulation truth and interpretation.

Update affected canonical pages for material behavior/interface changes or when
requested. Formatting/comments do not require a wiki entry. Do not copy transient
logs, routine test counts or repeated lint reports into the wiki.

For ingestion, inventory/read only the relevant raw material, inspect meaningful
images, record claims/provenance in a source page and update affected canonical
pages. Do not invent a source page for un-ingested material.

## Verification

Start queries at `wiki/index.md`. Keep every content page indexed except `log.md`.
Resolve internal links, check active file references and remove stale operational
claims. Verify important statements against code, records or Git. Preserve `raw/`
and future specifications, run whitespace/link checks and inspect the final diff.
