"""Regression checks for the canonical `knowledge/wiki` documentation surface."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = REPO_ROOT / "knowledge" / "wiki"
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _wiki_pages() -> list[Path]:
    return sorted(WIKI_ROOT.rglob("*.md"))


def test_documentation_links_resolve_and_index_is_complete() -> None:
    pages = _wiki_pages()
    by_stem: dict[str, list[Path]] = {}
    for page in pages:
        by_stem.setdefault(page.stem, []).append(page)

    failures = []
    for page in pages:
        for raw_target in WIKILINK_RE.findall(page.read_text()):
            target = raw_target.split("|", 1)[0].split("#", 1)[0]
            relative = Path(target if target.endswith(".md") else f"{target}.md")
            if (WIKI_ROOT / relative).is_file():
                continue
            if len(by_stem.get(relative.stem, [])) != 1:
                failures.append(f"{page.relative_to(REPO_ROOT)}: {target}")
    for document in (REPO_ROOT / "README.md", REPO_ROOT / "datasets" / "README.md"):
        for raw_target in MARKDOWN_LINK_RE.findall(document.read_text()):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / target).resolve().exists():
                failures.append(f"{document.relative_to(REPO_ROOT)}: {raw_target}")
    assert failures == []

    index = (WIKI_ROOT / "index.md").read_text()
    missing = []
    for page in pages:
        if page.name in {"index.md", "log.md"}:
            continue
        target = page.relative_to(WIKI_ROOT).with_suffix("").as_posix()
        if f"[[{target}|" not in index and f"[[{target}]]" not in index:
            missing.append(target)
    assert missing == []
