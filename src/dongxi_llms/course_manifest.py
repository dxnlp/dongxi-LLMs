"""Declarative course routing, independent of Torch and notebook runtimes.

Registration describes location/dependencies, not completed learning or a passed
experiment. Every extension remains inside the original fifteen/28-day route.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


class CourseManifestError(ValueError):
    """A routing contract is malformed or disagrees with the repository."""


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _relative_path(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and "\\" not in value
            and not any(ord(character) < 32 or ord(character) == 127 for character in value)
            and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts
            and str(PurePosixPath(value)) == value)


def validate_manifest(document: Any, root: Path | None = None) -> list[str]:
    """Return all schema/route issues; optional root adds inventory/content checks."""
    problems: list[str] = []
    if not isinstance(document, dict):
        return ["Manifest must be an object"]
    expected = {"schema_version", "scope", "chapters", "days", "notebooks"}
    if (set(document) != expected or not _integer(document.get("schema_version"))
            or document.get("schema_version") != 1):
        problems.append("Unknown manifest version/fields")
    if not isinstance(document.get("scope"), str) or not document.get("scope", "").strip():
        problems.append("Missing manifest scope")
    tables = {}
    for name in ("chapters", "days", "notebooks"):
        value = document.get(name)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            return problems + [f"{name} must be a list of objects"]
        tables[name] = value

    def table_ids(rows, key, expected_ids):
        ids = [row.get(key) for row in rows]
        if (not all(_integer(i) for i in ids) or sorted(ids) != list(expected_ids)):
            problems.append(f"{key} must cover exactly {list(expected_ids)} without duplicates")

    table_ids(tables["chapters"], "id", range(1, 16))
    table_ids(tables["days"], "day", range(1, 29))
    path_fields: list[str] = []
    for row in tables["chapters"]:
        if set(row) != {"id", "chapter", "solutions", "lab"}:
            problems.append("Unknown chapter fields")
        for key, directory in (("chapter", "chapters"), ("solutions", "solutions"), ("lab", "labs")):
            value = row.get(key)
            if not _relative_path(value) or not _integer(row.get("id")):
                problems.append(f"Invalid chapter path: {value!r}")
            elif not value.startswith(f"book/{directory}/{row['id']:02d}-") or not value.endswith(".md"):
                problems.append(f"Chapter ID/path mismatch: {value}")
            else:
                path_fields.append(value)
    day_chapters = {}
    for row in tables["days"]:
        if set(row) != {"day", "chapter", "index"}:
            problems.append("Unknown day fields")
        day, chapter = row.get("day"), row.get("chapter")
        if not _integer(chapter) or chapter not in range(1, 16):
            problems.append(f"Unknown chapter on day {day}")
        if _integer(day):
            day_chapters[day] = chapter
            if row.get("index") != f"notebooks/day-{day:02d}/README.md":
                problems.append(f"Day/index mismatch: {day}")
            else:
                path_fields.append(row["index"])
    # All chapters need at least one daily route, not a dangling book-only entry.
    if set(v for v in day_chapters.values() if _integer(v)) != set(range(1, 16)):
        problems.append("Daily routes must cover all fifteen chapters")

    entries = {}
    for row in tables["notebooks"]:
        if set(row) != {"path", "day", "chapter", "lane", "execution", "improvement", "requires"}:
            problems.append("Unknown notebook fields")
        path, day = row.get("path"), row.get("day")
        if not _relative_path(path):
            problems.append(f"Unsafe/invalid notebook path: {path!r}")
            continue
        if path in entries:
            problems.append(f"Duplicate notebook: {path}")
        entries[path] = row
        matches_path = (_integer(day) and bool(re.fullmatch(
            rf"notebooks/day-{day:02d}/[^/]+\.ipynb", path)))
        if (not _integer(day) or day not in day_chapters
                or not _integer(row.get("chapter"))
                or row.get("chapter") != day_chapters.get(day)
                or not matches_path):
            problems.append(f"Notebook day/chapter/path mismatch: {path}")
        lane = row.get("lane")
        if lane not in ("core", "optional", "extension") or row.get("execution") != "cpu-offline":
            problems.append(f"Unknown lane/execution: {path}")
        improvement = row.get("improvement")
        if lane == "extension":
            if not isinstance(improvement, str) or not re.fullmatch(r"DXI-(0[1-9]|1[0-8])", improvement):
                problems.append(f"Extension needs a known improvement ID: {path}")
        elif improvement is not None:
            problems.append(f"Non-extension improvement ID: {path}")
        requires = row.get("requires")
        if (not isinstance(requires, list) or not all(_relative_path(p) for p in requires)
                or len(set(requires)) != len(requires)):
            problems.append(f"Invalid dependencies: {path}")
        path_fields.append(path)
    for day in range(1, 29):
        if not any(row.get("day") == day and row.get("lane") == "core" for row in entries.values()):
            problems.append(f"Day {day} has no core notebook")
    for path, row in entries.items():
        for dependency in row.get("requires", []) if isinstance(row.get("requires"), list) else []:
            if not isinstance(dependency, str) or dependency not in entries:
                problems.append(f"Unknown dependency of {path}: {dependency!r}")
    visiting, visited = set(), set()

    def visit(path):
        if path in visiting:
            problems.append(f"Notebook dependency cycle: {path}")
            return
        if path in visited:
            return
        visiting.add(path)
        dependencies = entries[path].get("requires", [])
        for dependency in dependencies if isinstance(dependencies, list) else []:
            if isinstance(dependency, str) and dependency in entries:
                visit(dependency)
        visiting.remove(path)
        visited.add(path)

    for path in entries:
        visit(path)
    if root is None:
        return problems
    root = Path(root).resolve()
    for path in path_fields:
        actual = root / path
        if not actual.resolve().is_relative_to(root):
            problems.append(f"Path escapes repository: {path}")
        elif not actual.is_file() or not actual.stat().st_size:
            problems.append(f"Missing/empty artifact: {path}")
    physical = {p.relative_to(root).as_posix() for p in (root / "notebooks").glob("day-*/*.ipynb")}
    for path in sorted(physical - set(entries)):
        problems.append(f"Unregistered notebook: {path}")
    for path in entries:
        actual = root / path
        if not actual.is_file() or not actual.resolve().is_relative_to(root):
            continue
        try:
            notebook = json.loads(actual.read_text())
            cells = notebook["cells"]
            if not isinstance(cells, list) or not all(isinstance(c, dict) for c in cells):
                raise ValueError("cells must be objects")

            def source(cell):
                value = cell.get("source", "")
                if isinstance(value, list) and all(isinstance(s, str) for s in value):
                    return "".join(value)
                if isinstance(value, str):
                    return value
                raise ValueError("invalid cell source")

            code = [source(c).strip() for c in cells if c.get("cell_type") == "code"]
            prose = "\n".join(source(c) for c in cells if c.get("cell_type") == "markdown")
            if not any(code) or not prose.strip() or all(c in ("", "...", "pass") for c in code):
                problems.append(f"Empty/placeholder notebook: {path}")
            if not re.search(r"!\[[^\]]*\]\([^)]*\)", prose):
                problems.append(f"No saved explanatory preview: {path}")
            metadata = notebook.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("invalid notebook metadata")
            local = metadata.get("dongxi", {})
            if not isinstance(local, dict):
                raise ValueError("invalid dongxi metadata")
            assertions = [("chapter", metadata.get("chapter")),
                          ("chapter", metadata.get("dongxi_chapter")),
                          ("chapter", local.get("chapter")), ("day", local.get("day"))]
            for key, value in assertions:
                if value is not None and (not _integer(value) or value != entries[path][key]):
                    problems.append(f"Notebook metadata/{key} mismatch: {path}")
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"Invalid notebook {path}: {error}")
    return problems


def load_manifest(root: Path, *, check_files: bool = True) -> dict:
    path = Path(root) / "docs/course_manifest.json"
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise CourseManifestError(f"Cannot load course manifest: {error}") from error
    problems = validate_manifest(document, Path(root) if check_files else None)
    if problems:
        raise CourseManifestError("\n".join(problems))
    return document


def select_notebooks(document: dict, *, days=None, paths=None, lane=None) -> list[dict]:
    """Select registered routes; unknown/empty selectors fail, never disappear."""
    problems = validate_manifest(document)
    if problems:
        raise CourseManifestError("\n".join(problems))
    entries = document["notebooks"]
    if days is not None:
        if (not isinstance(days, (list, tuple)) or not days
                or any(not _integer(d) or d not in range(1, 29) for d in days)):
            raise CourseManifestError("Days must be nonempty and in1..28")
        entries = [row for row in entries if row["day"] in days]
    if paths is not None:
        known = {row["path"] for row in document["notebooks"]}
        if (not isinstance(paths, (list, tuple)) or not paths
                or any(not _relative_path(p) or p not in known for p in paths)):
            raise CourseManifestError("Unknown/empty explicit notebook selector")
        entries = [row for row in entries if row["path"] in paths]
    if lane is not None:
        if lane not in ("core", "optional", "extension"):
            raise CourseManifestError("Unknown notebook lane")
        entries = [row for row in entries if row["lane"] == lane]
    if not entries:
        raise CourseManifestError("No matching registered notebooks")
    return sorted(entries, key=lambda row: row["path"])
