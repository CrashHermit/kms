# Multimodal Content Propagation Fix Plan

## Purpose

Make source-document inputs to DSPy consistently multimodal. Any source material that may contain prose, mathematical notation, diagrams, figures, or image-only content must reach the model through the canonical `Content` → `ContentParts` path.

The smaller implementation model should follow this plan literally. Make the smallest coherent change. Do not add compatibility wrappers, silent fallbacks, heuristic image recovery, or malformed-output repair.

## Core contract

The canonical flow is:

```text
models.Node[] / walker.WindowNode[]
    -> content.labeled_content(...)
    -> content.Content
    -> content.ContentParts
    -> ContentParts.format()
    -> ordered OpenAI-style text/image blocks
```

The LLM must receive the source material in document order. Text and images must remain separate content blocks; do not flatten them to a string containing `[image]` when the image is available.

Use `ContentParts` for DSPy fields that represent source-document material. Keep fields that are intrinsically derived text—such as `fact_text`, `statement`, and `canonical_knowledge`—as `str` unless they are deliberately expanded to include source images.

Malformed model positions, spans, split results, and ownership outputs must continue to fail at the boundary. Do not silently skip or repair invalid LLM output.

## Current state and known problems

The following work is already present in the working tree and must be preserved unless a test demonstrates a necessary correction:

- `splitter.Signature.current_nodes` uses `content.ContentParts`.
- `splitter.Signature.context_before` has been changed to `content.ContentParts`.
- `Splitter.encode()` wraps both target nodes and preceding context with `content.labeled_content_parts(...)`.
- `triplet_extractor._FactSignature.current_nodes` uses `content.ContentParts` and `_FactExtractor.encode()` wraps its window.
- `walker.nodes_before(...)` has been added to return preceding source nodes while retaining image nodes.
- Splitter reconstruction preserves `image_path`, provenance, and governing-instruction metadata.
- Splitter malformed-output validation must remain fail-fast. Do not restore the temporary warning-and-skip behavior.

The remaining multimodal gaps are now addressed:

1. [x] `walker.fixed_windows_with_context()` retains image nodes in selected ranges and returns multimodal `before`/`after` context.
2. [x] `semantic.window_content()` preserves target image order instead of appending it after following context.
3. [x] Semantic context handling preserves neighboring images.
4. [x] `content.labeled_content()` documentation and emitted image labels agree on the colon-terminated image label.
5. [x] Tests assert actual serialized blocks, not only Python object types.

## Phase 1: Establish and test the canonical content adapter ✅

### File: `src/kms/core/content.py`

Review the existing implementation before editing. Preserve these behaviors:

- [x] `Content.parts` is an ordered list of `TextPart | ImagePart`.
- [x] `load_image()` loads the referenced file and applies the configured image downsampling cap.
- [x] `labeled_content()` emits a text label for every node and an image block immediately after the label for an image-bearing node.
- [x] `ContentParts.format()` returns OpenAI-style blocks.

Make the image label contract explicit and consistent:

- [x] For an image node, the text block is exactly equivalent to:

  ```text
  [position] (image):
  ```

  where the type is rendered from the node's type value.
- [x] The image payload follows that label immediately.
- [x] For a text node, the existing format is retained:

  ```text
  [position] (type): content
  ```

Update either the implementation documentation or the implementation so they agree. Prefer the implementation already used by the prompts: image labels should include the colon. Do not add a second renderer or a second node-to-content format.

Add or extend focused tests for `labeled_content()`/`ContentParts.format()` that assert:

1. [x] text-only nodes produce one text block;
2. [x] an image-bearing node produces a label text block followed by an `image_url` block;
3. [x] a sequence of text → image → text preserves that exact block order;
4. [x] the image block contains a data URI when given a local image path;
5. [x] an empty node list produces an empty `ContentParts` block list.

Use an existing small image fixture/helper if one already exists. Do not duplicate a large base64 fixture unnecessarily.

## Phase 2: Add a shared multimodal context-window primitive ✅

### File: `src/kms/core/walker.py`

The repository already centralizes window construction in `kms.core.walker`. Keep that responsibility there rather than adding per-stage context assembly.

### 2.1 Preserve source nodes when collecting context

`nodes_before(nodes, cursor, budget)` should:

- [x] treat `cursor` as an exclusive list index;
- [x] walk backward from `cursor - 1`;
- [x] count each node using the existing token estimate;
- [x] stop before adding the next node when the budget would be exceeded;
- [x] retain image nodes even if their textual content is empty;
- [x] retain empty-content nodes when they carry an image path;
- [x] reverse the selected nodes before returning them so the result is oldest-first/document order;
- [x] return source `models.Node` objects, not hand-built `WindowNode` objects;
- [x] avoid inventing or copying fields that do not exist on `models.Node` or `WindowNode`.

The caller should convert the result once with the existing `node_views()` helper. Do not duplicate the `WindowNode` projection logic.

### 2.2 Decide how images count against the budget

For this initial fix, use the existing text token estimate for node-window selection unless the repository already has a defined image-token accounting policy. An image node must still be retained even when it has no text. Do not estimate image cost by arbitrary magic numbers in this change.

Document this limitation in the function docstring if necessary: the budget bounds textual node selection; image payload size is controlled by the configured image resize cap.

### 2.3 Upgrade `fixed_windows_with_context()`

Current behavior filters out image nodes from `eligible` and builds `before`/`after` using text-only helpers. Change the window contract so multimodal consumers can receive images:

- [x] Do not treat an image-only node as eligible text content for starting or filling the primary text budget; image-only nodes are retained in selected ranges and context.
- [x] Include image nodes that occur within the selected source range in the returned `Window.items`.
- [x] Build context from source nodes, not joined strings, when the window is intended for multimodal DSPy input.
- [x] Preserve document order across text and image nodes.
- Keep the existing `Window` API compatible if possible. If changing `Window.before` and `Window.after` from `str | None` to multimodal values would affect unrelated callers, introduce a clearly named multimodal context field or a parallel helper rather than silently changing unrelated text consumers.

[x] Traced all callers of `fixed_windows_with_context()`; there are no production callers outside `walker.py`, and focused tests cover the returned multimodal `Window` contract.

Preferred design if callers are limited to multimodal stages:

```text
Window.items: list[WindowNode]
Window.before: list[WindowNode]
Window.after: list[WindowNode]
```

The corresponding DSPy encoder then calls `content.labeled_content_parts(...)` for each field.

If text-only consumers require the current strings, keep `content_before()` and `content_after()` as text-only compatibility primitives and add explicit node-returning helpers for multimodal contexts. Do not make a text-only helper pretend to contain images.

Add tests covering:

- an image between two text nodes remains in `Window.items` in the correct position;
- an image immediately before a window appears in multimodal `before` context;
- an image immediately after a window appears in multimodal `after` context;
- text-only behavior remains unchanged for callers that explicitly use `content_before()`/`content_after()`;
- budget selection is deterministic and does not reorder nodes.

## Phase 3: Fix semantic enrichment ordering and neighboring images ✅

### File: `src/kms/core/semantic.py`

Inspect the existing `window_content()` implementation and all callers before editing.

The required ordering rule is:

```text
before node 1
before node 2
anchor/target node text
anchor/target node image, if any
after node 1
after node 2
```

More generally, each source node must be converted to its content parts at its original position. Never collect all text first and append images later.

Implement this with one ordered traversal or one ordered assembly operation. Do not maintain separate text and image lists and concatenate them afterward.

For every node in the semantic context window:

- [x] emit its label and text in the normal labeled-node format;
- [x] if it has `image_path`, load the image using the same configured image-size policy used by `content.labeled_content()`;
- [x] emit the image immediately after that node's label/text block;
- [x] retain images from neighboring context nodes as well as the target node;
- [x] preserve target markers/anchor markers in the text label where the current API supports them.

Do not recover image identity by parsing `![N]()` text. `Node.image_path` is already the authoritative resolved path, and picture resolution is positional upstream.

Add regression tests that construct nodes in these arrangements:

1. text before → image-bearing target → text after;
2. image before → text target → image after;
3. image-bearing target with empty textual content;
4. multiple neighboring image nodes.

Assert the resulting `Content.parts` or formatted blocks by type and order. A test that only checks `render()` is insufficient because `render()` intentionally replaces images with `[image]`.

## Phase 4: Make DSPy source inputs consistently multimodal ✅

Search all construction-stage DSPy signatures and encoders for fields typed as:

- `list[walker.WindowNode]`;
- raw `models.Node` lists;
- `str` fields that are actually source-document windows or passages;
- direct passing of node DTOs into `aforward()`.

For each occurrence, classify the field:

- If it is source material that may contain an image, use `content.ContentParts` in the signature and `content.labeled_content_parts(...)` or `content.ContentParts(content=...)` in the encoder.
- If it is genuinely derived text, leave it as `str`.
- If it is a structured output or metadata field, do not convert it to content.

Known required correction:

- [x] `src/kms/construction/triplet_extractor.py` fact-window input remains on the `ContentParts` path.

Known correct examples to follow:

- `src/kms/construction/splitter.py` target window and preceding context;
- `src/kms/construction/instruction_finder.py` node inputs;
- `src/kms/construction/pedagogical_component_finder.py` node windows;
- `src/kms/construction/statement_procedure_builder.py` source contents;
- `src/kms/construction/governance_judge.py` context windows;
- `src/kms/core/search.py` multimodal search fields.

[x] No new `MultimodalWindow`, `ImageAwareNode`, or stage-specific DTO was introduced; the implementation reuses `Content`, `ContentParts`, `WindowNode`, and `node_views()`.

## Phase 5: Splitter-specific validation and tests ✅

### File: `src/kms/construction/splitter.py`

Preserve the current architecture and validation rules:

- [x] `current_nodes` is `content.ContentParts`.
- [x] `context_before` is `content.ContentParts`.
- [x] The encoder converts node views to labeled content parts.
- [x] An empty preceding context is represented by an empty `ContentParts`, not a raw empty string or an omitted incompatible field.
- [x] Invalid positions, duplicate positions, fewer than two exercise items, and empty exercise items raise `ValueError` at the boundary.

Check the formatting of the encoder for project style, but do not change behavior merely to reduce line count.

Ensure `_rebuild()` preserves all source metadata needed by downstream processing:

- [x] node type;
- [x] source content replacement;
- [x] document index;
- [x] stable UUID policy already used by the splitter;
- [x] image path;
- [x] provenance;
- [x] governing instruction UUIDs.

If splitting an image-bearing node creates multiple derived nodes, explicitly decide and test whether the image should be copied to every resulting piece or associated with only one piece. The implementation must follow the project’s source-preservation semantics and must not silently discard the image. If the image is the visual source for the entire packed block, copying it to each derived piece is acceptable only if downstream semantics already treat the image as block-level evidence; otherwise retain it on the designated source piece and document that choice.

Add/retain tests for:

- [x] text target plus image context reaching `ContentParts`;
- [x] empty context producing empty `ContentParts`;
- [x] image paths surviving splitter rebuild according to the chosen policy;
- [x] malformed splitter output failing fast.

## Phase 6: Decide router/executor scope separately ✅

Do not combine a splitter router/executor redesign with this multimodal propagation fix unless the implementation model finds that the existing splitter cannot express the corrected input contract.

The immediate objective is to ensure every splitter call receives complete ordered multimodal evidence. A router/executor design is a separate architectural change:

```text
multimodal window
    -> candidate/need judge
    -> per-candidate split executor
```

If pursued later, follow existing stage executor/gate patterns, pass the stage’s own LM explicitly into every internal call, and keep the splitter’s current fail-fast output validation. Do not add a redundant router if one existing judge already performs routing.

## Verification requirements

Run these checks after implementation:

1. [x] Focused content and walker tests.
2. [x] Focused semantic enrichment tests.
3. [x] Focused splitter and triplet extractor tests.
4. [x] Full non-live test suite: 383 passed, 3 skipped.
5. [x] Python compile/import check for all changed modules.
6. [x] A deterministic encoder-level inspection asserts `ContentParts.format()` for a text/image/text sequence.

The minimum observable assertion is that the outgoing formatted payload contains ordered blocks such as:

```python
[
    {'type': 'text', 'text': '[0] (paragraph): ...'},
    {'type': 'text', 'text': '[1] (image):'},
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,...'}},
    {'type': 'text', 'text': '[2] (paragraph): ...'},
]
```

Do not use `Content.render()` as the only verification because it intentionally hides image payloads.

Before reporting success, inspect the final diff and confirm:

- [x] no raw `list[WindowNode]` is passed directly to a DSPy source-material field;
- [x] no image is appended out of document order;
- [x] no image-bearing context is flattened into text;
- [x] no malformed LLM output is silently dropped or repaired;
- [x] no duplicate node-view or content-format abstraction was introduced;
- [x] all changed files have focused test coverage or an explicit reason why a test is not applicable.

## Out of scope

- Do not change image resolution from positional OCR mapping.
- Do not add deterministic OCR/image heuristics.
- Do not alter embedding behavior; embeddings already use the multimodal Voyage path.
- Do not redesign the Neo4j graph schema.
- Do not clear or migrate the live graph.
- Do not introduce a generic router abstraction solely for this fix.
