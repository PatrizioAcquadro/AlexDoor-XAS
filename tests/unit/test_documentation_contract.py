"""Links and navigation in the published documentation must work in a clean checkout."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_ROOT = REPO_ROOT / "knowledge/wiki"


def _wiki_destination(page, target, pages):
    if not target:
        return page
    name = target if target.endswith(".md") else f"{target}.md"
    for directory in (page.parent, WIKI_ROOT):
        candidate = (directory / name).resolve()
        if candidate in pages:
            return candidate
    if "/" not in name:
        matches = [path for path in pages if path.name == name]
        if len(matches) == 1:
            return matches[0]
    return None


def _has_heading(page, section):
    for title in re.findall(r"^#{1,6}[ \t]+(.+)$", page.read_text(), re.MULTILINE):
        title = title.strip().casefold()
        slug = re.sub(r"[^\w -]", "", title).replace(" ", "-")
        if section.casefold() in (title, slug):
            return True
    return False


def test_wiki_links_resolve_and_index_is_complete():
    pages = set(WIKI_ROOT.rglob("*.md"))
    index = WIKI_ROOT / "index.md"
    indexed = set()
    for page in sorted(pages):
        for link in re.findall(r"\[\[([^\]\n]+)\]\]", page.read_text()):
            target, _, section = link.split("|", 1)[0].partition("#")
            destination = _wiki_destination(page, target, pages)
            assert destination, f"{page.name}: {link}"
            if section:
                assert _has_heading(destination, section), f"{page.name}: {link}"
            if page == index:
                indexed.add(destination)

    assert not pages - indexed - {index, WIKI_ROOT / "log.md"}


def test_local_markdown_links_resolve():
    guides = (
        "README.md",
        "datasets/README.md",
        "outputs/README.md",
        "outputs/generated/README.md",
        "outputs/evidence/README.md",
    )
    for page in [*(REPO_ROOT / guide for guide in guides), *WIKI_ROOT.rglob("*.md")]:
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", page.read_text()):
            if "://" in link or link.startswith("mailto:"):
                continue
            target, _, section = link.partition("#")
            destination = (page.parent / target).resolve() if target else page
            assert destination.exists(), f"{page.name}: {link}"
            if section and destination.suffix == ".md":
                assert _has_heading(destination, section), f"{page.name}: {link}"
