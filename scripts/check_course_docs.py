"""Check local Markdown links, SVG diagrams, and Python syntax in the course."""

from __future__ import annotations

import ast
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "site-packages"}
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)")


def project_files(suffix: str) -> list[Path]:
    """Return course files while excluding environments and generated files."""
    return [
        path
        for path in ROOT.rglob(f"*{suffix}")
        if not any(part in SKIP_DIRS for part in path.parts)
    ]


def check_markdown_links() -> list[str]:
    """Find inline links and images whose local target does not exist."""
    failures: list[str] = []
    for markdown in project_files(".md"):
        content = markdown.read_text(encoding="utf-8")
        for match in INLINE_LINK.finditer(content):
            target = match.group(1).strip("<>").strip()
            target = target.split(maxsplit=1)[0] if target else ""
            if not target:
                continue

            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue

            relative_path = unquote(parsed.path)
            if not relative_path:  # Same-file fragment link.
                destination = markdown
            else:
                destination = (markdown.parent / relative_path).resolve()

            if not destination.exists():
                line = content.count("\n", 0, match.start()) + 1
                failures.append(f"{markdown.relative_to(ROOT)}:{line}: missing {target}")
    return failures


def check_python_syntax() -> list[str]:
    """Parse Python source without importing optional course dependencies."""
    failures: list[str] = []
    for source in project_files(".py"):
        try:
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (SyntaxError, UnicodeDecodeError) as error:
            failures.append(f"{source.relative_to(ROOT)}: {error}")
    return failures


def check_svg() -> list[str]:
    """Parse SVG files as XML to catch malformed vector graphics."""
    failures: list[str] = []
    for graphic in project_files(".svg"):
        try:
            ET.parse(graphic)
        except ET.ParseError as error:
            failures.append(f"{graphic.relative_to(ROOT)}: {error}")
    return failures


def main() -> int:
    """Run all checks and return a non-zero status when a check fails."""
    checks = [
        ("Markdown links", check_markdown_links()),
        ("Python syntax", check_python_syntax()),
        ("SVG XML", check_svg()),
    ]
    failed = False
    for label, errors in checks:
        if errors:
            failed = True
            print(f"FAIL {label} ({len(errors)} issue(s))")
            for error in errors:
                print(f"  {error}")
        else:
            print(f"PASS {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
