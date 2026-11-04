"""Apply a genome to the world.

The SEO actuator renders static HTML into a directory. Dry-run, the
default, writes a manifest and no HTML. There is no deploy method and
no network call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

from evoml.domains.seo_pages.genome import PageRecipe


class Actuator(Protocol):
    def apply(
        self,
        recipe: PageRecipe,
        pages: Sequence[str],
        *,
        dry_run: bool,
        max_pages: int,
    ) -> dict:
        """Apply `recipe` to at most `max_pages` pages. Return a manifest."""


class SeoActuator:
    """Write a static page per id. Never copies the files anywhere else."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def apply(
        self,
        recipe: PageRecipe,
        pages: Sequence[str],
        *,
        dry_run: bool = True,
        max_pages: int = 8,
    ) -> dict:
        chosen = [str(page) for page in pages][: max(0, max_pages)]
        manifest = {
            "dry_run": dry_run,
            "deployed": False,
            "pages": chosen,
            "pages_requested": len(list(pages)),
            "pages_changed": len(chosen),
            "max_pages": max_pages,
            "recipe": recipe.serialize(),
            "files": [],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            files = []
            for page in chosen:
                path = self.output_dir / f"{_slug(page)}.html"
                path.write_text(_render(recipe, page))
                files.append(str(path.name))
            manifest["files"] = files
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest


def _slug(page: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in page)
    return cleaned.strip("-") or "page"


def _render(recipe: PageRecipe, page: str) -> str:
    keyword = page.replace("-", " ")
    title = recipe.title_template.replace("{keyword}", keyword)
    description = recipe.meta_template.replace("{keyword}", keyword)
    sections = "\n".join(
        f'    <section id="{_slug(section)}"><h2>{section.replace("_", " ")}</h2></section>'
        for section in recipe.sections
    )
    schema = {
        "@context": "https://schema.org",
        "@type": list(recipe.schema_types) or ["Service"],
        "name": title,
    }
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{_escape(title)}</title>\n"
        f'  <meta name="description" content="{_escape(description)}">\n'
        f"  <script type=\"application/ld+json\">{json.dumps(schema)}</script>\n"
        "</head>\n"
        "<body>\n"
        f"  <p>Call to action: {_escape(recipe.cta_placement)}. "
        f"Links: {_escape(recipe.link_strategy)}.</p>\n"
        f"{sections}\n"
        "</body>\n"
        "</html>\n"
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
