"""docker/gym/wire_pg_routers.py applies cleanly to the gym submodule's MCP server sources,
including the Notion response-shape patch to pg-client.ts. Text only: no Docker, no Node, no tsc."""

import importlib.util
import sys

import pytest

from evcarry.paths import repo_root

ROOT = repo_root()
LOCAL_SERVERS = ROOT / "third_party" / "toolathlon_gym" / "local_servers"


def _wire():
    spec = importlib.util.spec_from_file_location("wire_pg_routers", ROOT / "docker" / "gym" / "wire_pg_routers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WIRE = _wire()
FILES = list(dict.fromkeys(path for path, _, _ in WIRE.EDITS))


def _source(relative: str) -> str:
    path = LOCAL_SERVERS / relative
    if not path.is_file():
        pytest.skip(f"gym submodule file missing: {path}")
    return path.read_text(encoding="utf-8")


def _balance(code: str) -> tuple[int, ...]:
    return tuple(code.count(opening) - code.count(closing) for opening, closing in ("()", "{}", "[]"))


def test_the_pg_client_edits_are_exactly_the_versioned_notion_shape_edits() -> None:
    assert WIRE.NOTION_SHAPE_PATCH == "evcarry-notion-shape-v1"
    assert [edit for edit in WIRE.EDITS if edit[0] == WIRE.PG_CLIENT] == WIRE.NOTION_SHAPE_EDITS


def test_every_notion_anchor_matches_exactly_once_in_the_unpatched_source() -> None:
    text = _source(WIRE.PG_CLIENT)
    for _, anchor, _ in WIRE.NOTION_SHAPE_EDITS:
        assert text.count(anchor) == 1, f"anchor does not match exactly once: {anchor[:80]!r}"


def test_patched_pg_client_carries_the_marker_and_shapes_every_write() -> None:
    patched = WIRE.apply_edits(WIRE.PG_CLIENT, _source(WIRE.PG_CLIENT))
    assert patched.count(f'/*! NOTION_SHAPE_PATCH = "{WIRE.NOTION_SHAPE_PATCH}"') == 1
    for call in (
        "JSON.stringify(this.shapePageProperties(properties))",  # post-page
        "this.mergedPageProperties(page_id, body[field])",  # patch-page merges
        "JSON.stringify(this.shapeRichText(title))",  # create-a-database
        "JSON.stringify(this.shapeDatabaseProperties(properties))",
        "this.shapeDatabaseField(field, body[field])",  # update-a-database
        "const child = this.parseBlockChild(rawChild)",  # insertBlocks: string children
        "JSON.stringify(this.shapeBlockData(blockType, blockData))",
    ):
        assert patched.count(call) == 1, call
    for unshaped in (
        "JSON.stringify(properties), JSON.stringify(icon)",
        "values.push(isJsonField ? JSON.stringify(body[field]) : body[field])",
        "JSON.stringify(blockData), position",
        "for (const child of children)",
    ):
        assert unshaped not in patched, unshaped


def test_each_notion_edit_keeps_brackets_balanced() -> None:
    for _, anchor, replacement in WIRE.NOTION_SHAPE_EDITS:
        assert _balance(replacement) == _balance(anchor), replacement[:80]


@pytest.mark.parametrize("relative", FILES)
def test_every_file_patches_cleanly(relative: str) -> None:
    text = _source(relative)
    assert WIRE.apply_edits(relative, text) != text


@pytest.mark.parametrize("relative", FILES)
def test_patching_an_already_patched_file_is_refused(relative: str) -> None:
    patched = WIRE.apply_edits(relative, _source(relative))
    with pytest.raises(SystemExit):
        WIRE.apply_edits(relative, patched)


def test_refusing_a_patched_pg_client_names_the_marker() -> None:
    patched = WIRE.apply_edits(WIRE.PG_CLIENT, _source(WIRE.PG_CLIENT))
    with pytest.raises(SystemExit, match="already carries evcarry-notion-shape-v1"):
        WIRE.apply_edits(WIRE.PG_CLIENT, patched)


def test_main_patches_a_copy_of_the_local_servers_tree_once(tmp_path, monkeypatch, capsys) -> None:
    sources = {relative: _source(relative) for relative in FILES}
    for relative, text in sources.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["wire_pg_routers.py", str(tmp_path)])

    WIRE.main()

    for relative, text in sources.items():
        assert (tmp_path / relative).read_text(encoding="utf-8") == WIRE.apply_edits(relative, text)
    assert capsys.readouterr().out.count("patched ") == len(FILES)
    with pytest.raises(SystemExit):
        WIRE.main()
