"""
Final assembly: resolve image links, consolidate picture files, and concatenate
the ordered AST into a single markdown document.

This runs after the LangGraph pipeline, once every stage has filled the segment
backbone. It walks segments in document order while each segment's picture
context is still in hand, so an IMAGE node's `![N]()` placeholder resolves against
its own segment's `pictures` (see the note in image resolution: the node's page is
its containing segment — no per-node page index is needed).

For each resolved placeholder the surviving picture is copied into a single
`<output_dir>/images/` directory under a collision-free name
(`seg<segment>_img<picture>.png`), and the placeholder is rewritten to a relative
link (`![](images/seg0003_img002.png)`) so the emitted document is self-contained
and portable. A placeholder whose index has no matching picture is left untouched
rather than raising — the pipeline should not produce these, but a stray one must
not abort assembly.
"""

import re
import shutil
from pathlib import Path

from .state import Picture, Segment, NodeType

# Tolerant match for an image placeholder: ![N]() with optional surrounding
# whitespace and an empty link target.
_PLACEHOLDER = re.compile(r"!\[\s*(\d+)\s*\]\(\s*\)")

IMAGES_DIRNAME = "images"


def _consolidate_picture(picture: Picture, segment_index: int, images_dir: Path) -> str:
    """Copy one surviving picture into the consolidated images dir and return its
    markdown-relative link (`images/segNNNN_imgYYY.png`)."""
    suffix = Path(picture.image_path).suffix or ".png"
    dest_name = f"seg{segment_index:04d}_img{picture.index:03d}{suffix}"
    shutil.copyfile(picture.image_path, images_dir / dest_name)
    # Forward-slashed regardless of platform so the link is valid markdown.
    return f"{IMAGES_DIRNAME}/{dest_name}"


def _resolve_content(
    content: str,
    segment_index: int,
    pictures_by_index: dict[int, Picture],
    images_dir: Path,
) -> str:
    """Rewrite every `![N]()` placeholder in one IMAGE node's content into a
    relative link, copying the matched picture as a side effect. Unmatched
    placeholders pass through unchanged."""
    def _replace(match: re.Match) -> str:
        picture = pictures_by_index.get(int(match.group(1)))
        if picture is None:
            return match.group(0)
        return f"![]({_consolidate_picture(picture, segment_index, images_dir)})"

    return _PLACEHOLDER.sub(_replace, content)


def assemble(
    segments: list[Segment],
    output_dir: str | Path = "output",
    filename: str = "document.md",
) -> Path:
    """Resolve image links and write the ordered AST to a single markdown file.

    Only pictures that are actually referenced by a surviving placeholder are
    copied into `<output_dir>/images/`; anything filtered out upstream simply never
    gets linked. Returns the path of the written document.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    for segment in segments:
        pictures_by_index = {p.index: p for p in segment.pictures}
        for node in segment.nodes:
            if node.content is None:
                continue
            if node.type == NodeType.IMAGE:
                parts.append(_resolve_content(node.content, segment.index, pictures_by_index, images_dir))
            else:
                parts.append(node.content)

    document = "\n\n".join(parts) + "\n"
    output_path = output_dir / filename
    output_path.write_text(document, encoding="utf-8")
    return output_path
