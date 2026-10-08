"""devzone.py: the two write primitives every orin generator uses, so that the campaign's content stays
separable from the other devices':
  write_new(path, text)             only files whose name carries the device tag, only under a sarc_dev directory;
  put_block(path, anchor, id, text) inserts `text` immediately before `anchor` in a shared dev-zone file, between
                                    `// >>> 4070ti <id>` and `// <<< 4070ti <id>`; a second run replaces the text
                                    between its own markers and touches nothing else."""
import pathlib
TAG = "orin"

def write_new(path, text):
    path = pathlib.Path(path)
    assert TAG in path.name.lower() and "sarc_dev" in path.parts, f"refusing to write {path}: not a {TAG} dev-zone file"
    path.write_text(text)

def put_block(path, anchor, block_id, text, indent=""):
    path = pathlib.Path(path); assert "sarc_dev" in path.parts, path
    t = path.read_text()
    a, b = f"{indent}// >>> {TAG} {block_id}\n", f"{indent}// <<< {TAG} {block_id}\n"
    if a in t:
        assert t.count(a) == 1 and t.count(b) == 1, f"markers of {block_id} are not unique"
        i, j = t.index(a), t.index(b) + len(b)
        t = t[:i] + a + text + b + t[j:]
    else:
        assert t.count(anchor) == 1, f"anchor occurs {t.count(anchor)} times: {anchor[:60]!r}"
        t = t.replace(anchor, a + text + b + anchor)
    path.write_text(t)
