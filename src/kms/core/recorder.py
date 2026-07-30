"""
Record DSPy module inputs and outputs as a JSONL corpus, one directory per
module.

Each recorded call appends one line to
``<corpus_root>/<module>/<run_id>/examples.jsonl``, and ``load_examples`` reads
those lines back as ``dspy.Example`` objects — so a recorded run is a reusable
corpus of the material the pipeline actually produced, and the input side of a
future training set.

Layout::

    examples/
      corrector/
        20260730T142205Z-9f3ab1/
          examples.jsonl
          images/
            3f2a1c9d8b7e6f50.png

A record has three parallel sections, each answering one question: ``metadata``
how it was produced, ``inputs`` what went in, ``outputs`` what came out.
Provenance lives in its own namespace rather than beside the payload sections,
so it cannot leak into the field space DSPy sees when a record is rebuilt as an
Example.

**Append-only.** One line per call, never a rewrite, so the cost of recording
does not grow with the size of the corpus and a torn write costs one line
instead of the whole file (``load_records`` skips unparseable lines rather than
failing). Note the concurrency limit: appending a long line is not atomic, so
concurrent writers — threads, or DSPy's parallel evaluation — can still
interleave. That is a bounded, per-line risk rather than the lost-update
rewrite it replaces.

**Records are self-describing**, and image references are relative to the
corpus root rather than to the run directory, so concatenating two runs'
``examples.jsonl`` files keeps both the model and the images resolvable.

**Images are written out as files and referenced by path.** A page render
inlined as base64 is a half-megabyte string that makes the corpus unreadable
and is duplicated by every repeated sample of the same page — and repeated
sampling is exactly what a stable metric needs. Sidecars are named by content
hash, so N samples of one page share one file. Keeping the bytes (rather than
only a render recipe) means the corpus survives its source PDF being deleted.
Callers may attach the recipe as well via ``image_provenance``; re-rendering
from it when a sidecar is missing is left to the caller.
"""

import base64
import binascii
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import dspy

EXAMPLES_FILENAME = 'examples.jsonl'
IMAGES_DIRNAME = 'images'

# Marks a serialized image field, so the format is self-describing and the
# loader needs no out-of-band list of which fields hold images.
IMAGE_KEY = '$image'

_SECTIONS = ('inputs', 'outputs')

# An image reference that is already a url dspy can resolve itself, rather than
# a corpus-relative sidecar path.
_URL_SCHEMES = ('data:', 'http://', 'https://')


def new_run_id() -> str:
    """A sortable, unique id for one recording run.

    Returns:
        A UTC timestamp plus a short random suffix, safe as a directory name.
    """
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    return f'{stamp}-{uuid.uuid4().hex[:6]}'


# One run directory per process, so a pipeline run's records land together
# without every call site having to thread a run id through.
RUN_ID = new_run_id()


def _decode_data_url(url: str) -> bytes | None:
    """The bytes carried by a base64 data URL.

    Args:
        url: A ``dspy.Image`` url.

    Returns:
        The decoded payload, or None when the url is not an inlined base64
        image (e.g. a remote http url, which has no bytes to write out).
    """
    header, _, encoded = url.partition(',')
    if not header.startswith('data:') or 'base64' not in header:
        return None
    try:
        return base64.b64decode(encoded)
    except (binascii.Error, ValueError):
        return None


def _store_image(image: dspy.Image, corpus_root: Path, run_dir: str) -> str:
    """Write one image beside the corpus and return its reference.

    The file is named by content hash, so repeated samples of the same page
    share a single sidecar. The returned reference is relative to
    ``corpus_root`` (not the run directory) so it survives concatenating runs.

    Args:
        image: The image to write out.
        corpus_root: The corpus root directory.
        run_dir: The record's ``<module>/<run>`` directory, corpus-relative.

    Returns:
        The corpus-relative path of the written file, or the image's url
        unchanged when it carries no inlined bytes.
    """
    payload = _decode_data_url(image.url)
    if payload is None:
        return image.url
    digest = hashlib.sha256(payload).hexdigest()[:16]
    reference = f'{run_dir}/{IMAGES_DIRNAME}/{digest}.png'
    path = corpus_root / reference
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    return reference


def _deserialize(value: object, corpus_root: Path) -> object:
    """The in-memory form of one recorded value.

    Args:
        value: The value as it was stored.
        corpus_root: The corpus root the image reference is relative to.

    Returns:
        A ``dspy.Image`` for an image reference — rebuilt from the sidecar's
        bytes, or handed straight to dspy when the reference is already a url —
        otherwise the value unchanged.

    Raises:
        FileNotFoundError: If a sidecar the corpus references is missing. This
            is data loss, so it is raised rather than papered over: returning
            some stand-in would quietly feed a training run an image that is
            not the page.
    """
    if not isinstance(value, dict) or IMAGE_KEY not in value:
        return value
    reference = value[IMAGE_KEY]
    if reference.startswith(_URL_SCHEMES):
        return dspy.Image(url=reference)
    path = corpus_root / reference
    if not path.is_file():
        raise FileNotFoundError(
            f'recorded image {reference!r} is missing from the corpus at '
            f'{corpus_root}'
        )
    encoded = base64.b64encode(path.read_bytes()).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


def record_example(
    module_name: str,
    inputs: dict,
    outputs: dict,
    model: str,
    corpus_root: str | Path = 'examples',
    run_id: str | None = None,
    image_provenance: dict[str, dict] | None = None,
) -> str:
    """Append one call to a module's corpus.

    Args:
        module_name: Stable key for the module (e.g. ``'corrector'``). Becomes
            a directory name, so it must be filesystem-safe and should outlive
            refactors — a rename forks the corpus.
        inputs: Keyword args the module was called with.
        outputs: Keyword args the module returned.
        model: The LM the call ran on, recorded on the record itself so the
            file stays self-describing when runs are concatenated.
        corpus_root: Root directory the per-module corpora live under.
        run_id: The run to record under. Defaults to this process's ``RUN_ID``.
        image_provenance: Per-image-field extras merged into that field's
            reference object — e.g. ``{'page_image': {'source_pdf': …,
            'page_index': 0, 'render_scale': 2.5}}``.

    Returns:
        The record's id, so an adjudication or score can be attached to it
        later.
    """
    root = Path(corpus_root)
    provenance = image_provenance or {}
    record_id = uuid.uuid4().hex
    run_dir = f'{module_name}/{run_id or RUN_ID}'

    def serialize(field: str, value: object) -> object:
        """One value's JSON-safe form, writing images out as it goes."""
        if not isinstance(value, dspy.Image):
            return value
        return {
            IMAGE_KEY: _store_image(value, root, run_dir),
            **provenance.get(field, {}),
        }

    record: dict = {'metadata': {'id': record_id, 'model': model}}
    for section, fields in zip(_SECTIONS, (inputs, outputs), strict=True):
        record[section] = {
            name: serialize(name, value) for name, value in fields.items()
        }

    path = root / run_dir / EXAMPLES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record_id


def load_records(
    module_name: str,
    corpus_root: str | Path = 'examples',
    run_id: str | None = None,
) -> list[dict]:
    """Read a module's raw records, newest run last.

    Unparseable lines are skipped: an append cut short by a crash costs its own
    line and nothing else.

    Args:
        module_name: The key the records were written under.
        corpus_root: Root directory the per-module corpora live under.
        run_id: Read only this run. Defaults to every run, in run-id order.

    Returns:
        One dict per recorded call, each with its ``metadata``, ``inputs``, and
        ``outputs`` sections. Empty when nothing was recorded.
    """
    root = Path(corpus_root) / module_name
    pattern = f'{run_id or "*"}/{EXAMPLES_FILENAME}'

    records: list[dict] = []
    for path in sorted(root.glob(pattern)):
        # Streamed rather than slurped: the point of appending is that the
        # corpus outgrows what you want to hold in memory at once.
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_examples(
    module_name: str,
    corpus_root: str | Path = 'examples',
    run_id: str | None = None,
) -> list[dspy.Example]:
    """Read a module's records back as ``dspy.Example`` objects.

    Image fields are rebuilt from their sidecars. Note what the returned
    examples are: the ``outputs`` section holds what the module *said*, not a
    verified label, so using them as training targets without adjudication is
    self-distillation.

    Args:
        module_name: The key the records were written under.
        corpus_root: Root directory the per-module corpora live under.
        run_id: Read only this run. Defaults to every run.

    Returns:
        One example per record, each with ``with_inputs`` set from the recorded
        input keys so a consumer can split ``inputs()`` from ``labels()``.

    Raises:
        FileNotFoundError: If a sidecar image the corpus references is missing.
    """
    root = Path(corpus_root)
    return [
        dspy.Example(
            **{
                name: _deserialize(value, root)
                for section in _SECTIONS
                for name, value in record[section].items()
            }
        ).with_inputs(*record['inputs'].keys())
        for record in load_records(module_name, corpus_root, run_id)
    ]
