# Semantic Stage Research Report: Accuracy and Efficiency

**Research cutoff:** 2026-09-18  
**Scope:** KMS2 semantic-stage fact extraction, triplet decomposition, typed descriptions, source hubs, provenance, and inference cost.  
**Evidence policy:** 2025–2026 literature is prioritized because LLM extraction and serving methods are changing quickly. Older papers are retained only when they provide a distinct method or evaluation baseline. Paper-reported results are not KMS2 results. Any transfer to KMS2 is an applicability hypothesis that requires a contract-preserving experiment.

## Executive findings

1. **The highest-confidence accuracy opportunity is evidence-preserving decomposition.** KMS2 already separates fact extraction from fact-to-triplet decomposition. Recent work supports testing atomic propositions, complexity-aware decomposition, and pair-specific context selection, but also shows that decomposition can improve relation recall while harming entity recall or increasing unsupported inferences.
2. **The highest-confidence efficiency opportunity is selective computation, not more universal verification.** Recent DocRE work routes hard pairs to stronger refiners, while ICML 2025 routing research identifies quality estimation as the prerequisite for successful cascades. KMS2 should first measure calibrated semantic quality and actual local-model role-transition cost.
3. **The source-hub stage needs direct evaluation before threshold changes.** KMS2 has explicit vector retrieval, reranking, borderline judging, weighted community detection, and typed synthesis. Recent entity-matching work supports richer relational serialization; it does not establish that a new serialization or threshold improves KMS2’s five typed hub families.
4. **The current evidence base lacks KMS2-domain labels.** Existing tests prove ordering, provenance, schema, and persistence contracts. They do not measure fact precision/recall, triplet support, typed-description fidelity, community purity, or end-to-end cost. A current-domain semantic-stage benchmark still needs to be created.
5. **Do not begin with a triple refiner, QA expansion, ontology lock-in, or model cascade.** Those methods require labels, a quality estimator, or a domain ontology that KMS2 does not yet have. Establish the frozen baseline and an error taxonomy first.

## 1. KMS2 semantic-stage baseline

### 1.1 Pipeline and boundaries

The semantic graph is a two-pass source-faithful extraction pipeline followed by typed descriptions, embeddings, source hubs, and exact triplet-hub synthesis:

```text
ordered source blocks
        |
        v
fact extraction: one bounded context window per block
        |
        v
atomic facts
        |
        v
triplet decomposition: one call per extracted fact
        |
        v
source facts + typed entity/event endpoints + predicates + triplets
        |
        v
+---------------- sequential typed phases ----------------+
| entity -> event -> predicate -> statement -> procedure |
| description -> embed -> persistence for each type      |
+--------------------------------------------------------+
        |
        v
+------------------- typed hub cascade ------------------+
| entity -> event -> predicate -> statement -> procedure |
+--------------------------------------------------------+
        |
        v
exact source triplet hubs

Authoritative topology: `src/kms2/langgraph/semantic/graph.py::SemanticGraph.build_graph`.

- `TripletSourceLoadNode` loads ordered persisted source blocks.
- `FactExtractionNode` dispatches one `FactExtractionRequest` per block, selects bounded backward/forward context, and collects results in source order.
- The fact model boundary is `FactExtractionInput(context_before, target_block, context_after)`. UUIDs and embeddings are not sent to the model. The fact signature allows only the target block to support an extracted fact; neighboring blocks provide reference context.
- `TripletDecompositionNode` dispatches one request per extracted fact. The model emits `TripletCandidate(subject, predicate, object, subject_kind, object_kind)`, where endpoint kinds are `ENTITY` or `EVENT`.
- Collection creates fresh source-local endpoint, predicate, triplet, and occurrence identities while preserving source, source-block, and source-fact provenance.
- Five typed description phases run sequentially after raw triplet persistence. Each phase completes description generation, embedding, and persistence before the next type begins. Entity, event, and predicate descriptions use one target block; statement and procedure descriptions can use ordered multi-block targets. Description workers may still fan out internally, but the graph-level phase boundary is sequential.
- Each typed source hub retrieves same-source vector candidates, reranks them, sends borderline pairs to a type-specific Boolean judge, forms weighted SLLPA communities, synthesizes fixed memberships, embeds definitions, and persists hubs.
- Source triplet hubs group exact combinations of subject, predicate, and object hubs and synthesize from fixed, sorted evidence. They do not re-decide membership.

### 1.2 Current accuracy and efficiency controls

Read from `src/kms2/config/semantic.py` and related runtime modules; these are frozen baseline settings for experiments, not literature-validated optima:

- Fact extraction and triplet decomposition use no-retry inference settings.
- Semantic context defaults to 400 estimated tokens backward and forward.
- Typed hub defaults include candidate limit `32`, minimum vector similarity `0.86`, direct reranker acceptance `0.90`, borderline lower threshold `0.60`, reranker token budget `4096`, judge token budget `8192`, judge batch size `16`, SLLPA maximum iterations `100`, minimum association strength `0.2`, and minimum community size `2`.
- The local runtime holds a global lock across activation and inference. Only one LLM, embedding, or reranker role is resident at a time. LangGraph fan-out therefore expresses logical work decomposition, not necessarily concurrent GPU execution.
- Embeddings are batched internally, while typed description phases and typed hub stages are intentionally serialized entity -> event -> predicate -> statement -> procedure, followed by the triplet hub.

### 1.3 Existing measured contracts

The current focused tests establish structural behavior, including:

- one fact dispatch per persisted block and source-order collection;
- UUID-free context projection at the model boundary;
- fresh triplet and endpoint IDs with source-block provenance;
- source-scoped persistence and deterministic query parameters;
- ordered typed description and embedding identity preservation;
- directed predicate evidence and type-specific judge semantics;
- borderline-only judging, indexed decisions, weighted community inputs, and typed synthesis boundaries;
- serial typed-description and typed-hub phase ordering, plus exact triplet-hub dependencies;
- strict vector/result alignment.

These tests do **not** establish semantic quality or efficiency. The missing measures are fact precision/recall, atomicity, unsupported-fact rate, directed triplet F1, endpoint-kind accuracy, relation direction, qualifier and negation preservation, typed-description groundedness, candidate recall, community purity, hub coverage, model calls, token cost, phase latency, role transitions, and peak memory.

## 2. Existing local literature

The local corpus provides complementary evidence rather than one validated architecture.

| Paper | What it evaluates | Relevant mechanism | Reported evidence | KMS2 applicability and limitation |
|---|---|---|---|---|
| [ATOM](atom.md) | Continuous news to dynamic temporal KG | Atomic fact decomposition, parallel temporal 5-tuple extraction, parallel entity/relation merging | On the 2020-COVID-NYT setting, atomic facts improved stability and factual/temporal exhaustivity over lead paragraphs; factual hallucination increased. Entity-resolution F1 was reported as 0.994 and relation-resolution F1 as 1.000. | Strongest local evidence for short fact units and parallel work. KMS2 is not extracting temporal 5-tuples from news, and ATOM does not establish a non-temporal source-span policy. Test atomicity and support, not blind adoption. |
| [DIAL-KG](dial-kg.md) | Static and streaming KG construction | Meta-Knowledge Base, dual-track triple/event extraction, schema evolution, evidence verification, entity/event canonicalization | Reports batch and streaming F1 gains over EDC, high precision for incremental additions and evidence-backed deprecations, fewer relation types, and lower redundancy. | Relevant to typed descriptions, schema/domain-range constraints, and lifecycle-aware semantic memory. Its governance relies on LLM judgments and its streaming data is not KMS2’s scientific-source workload. |
| [RAGA](raga.md) | Scientific-paper QA with graph, vector, and fusion retrieval | Read–Search–Verify–Construct loop, source-linked evidence, typed tools, graph/vector synchronization, RRF fusion | On a small QASPER subset, fusion improved answer and evidence metrics over no-KG controls and reduced one tested paper’s construction time from 77 to 54 minutes. Retrieved Evidence F1 remained low, and extraction-quality/provenance-completeness evaluation was deferred. | Best local source-hub/provenance analogue. Its QA gains do not prove better semantic extraction, and its results are preliminary. |
| [AutoGraph-R1](autograph-r1.md) ([full text](autograph-r1.txt)) | Downstream RAG utility of constructed graphs | GRPO construction rewards for knowledge-carrying graphs versus source-index graphs | Reports downstream QA gains and a relation-vocabulary trade-off: knowledge-carrying rewards produce richer relation vocabularies, while indexing rewards produce more compact graphs. Training requires multi-hour GPU runs and retrieval/evaluation loops. | Useful for testing downstream utility and source-span indexing, not a first-line KMS2 extraction method. Substantive claims must use `autograph-r1.txt`, not the abstract-only page. |
| [EvoRAG](evorag.md) | KG-RAG response accuracy and graph refinement | Response feedback → path utility → triplet contribution scores; retrieval optimization | Reports downstream accuracy gains and prompt-length reduction, with degradation under flipped feedback. It does not establish source-grounded truth or typed-description quality. | Relevant only after KMS2 can measure downstream usefulness of facts. It is not direct evidence for extraction, hub identity, or provenance correctness. |

### Synthesis of the local corpus

- ATOM covers upstream semantic units and parallel extraction.
- DIAL-KG covers typed normalization, schema governance, and incremental evolution.
- RAGA covers source-linked evidence and graph/vector retrieval.
- AutoGraph-R1 and EvoRAG cover downstream utility and feedback-driven efficiency.

No local paper jointly evaluates KMS2-domain fact accuracy, typed descriptions, exact source-hub provenance, and end-to-end local-model cost. Cross-paper transfers are therefore hypotheses.

## 3. Recent primary research

### 3.1 Atomic decomposition and context selection

| Paper and status | Finding | KMS2 question |
|---|---|---|
| [Pommeret et al., “LLM-based Atomic Propositions help weak extractors”](pommeret-atomic-propositions.md) ([source](https://arxiv.org/abs/2604.02866)), arXiv, 2026 | A small multilingual propositioner distilled from Qwen3-32B into Qwen3-0.6B improves relation recall for weaker extractors. For stronger LLMs, a fallback combination recovers entity recall while retaining relation gains. Evaluations use SMiLER, FewRel, DocRED, and CaRB. | Compare atomic-proposition preprocessing against the current fact contract. Measure support, entity preservation, relation recall, and unsupported inference. Do not assume a decomposition stage helps the current `gemma-text-32k` profile. |
| [Choi et al., “SocraticKG”](socratic-kg.md) ([source](https://arxiv.org/abs/2601.10003)), arXiv, 2026 | Uses 5W1H question-answer expansion as a document-grounded intermediate representation before triple extraction. It targets factual retention, structural cohesion, and multi-hop QA. | Test whether QA expansion recovers implicit references or cross-block relations without violating the rule that the target block—not neighboring context—supports a fact. It is high-cost and should not precede a simpler atomic baseline. |
| [Anuyah et al., “CoDe-KG”](code-kg.md) ([source](https://aclanthology.org/2025.emnlp-main.783/)), EMNLP 2025 | Combines coreference resolution and sentence-complexity-aware syntactic decomposition. The paper reports 65.8 macro-F1 on REBEL, 75.7 micro-F1 on WebNLG2, and over 20% rare-relation recall improvement in decomposition/coreference ablations. | Test complexity-aware context or decomposition on blocks containing long syntax, pronouns, and rare relations. The results are sentence-level and do not validate KMS2’s source-block evidence policy. |
| [Zhang et al., “EP-RSR”](ep-rsr.md) ([source](https://aclanthology.org/2025.findings-naacl.224/)), Findings NAACL 2025 | Selects potentially related entity pairs first, then performs pair-specific relation summarization and retrieval. The paper reports large candidate reductions and improved DocRE results, while warning that early filtering can propagate errors. | Apply pair-specific selection to hub candidate retrieval or cross-block evidence selection only after measuring selector recall. It is a candidate-reduction hypothesis, not permission to broaden fact evidence. |
| [Yang and Tan, “SegDRE”](segdre.md) ([source](https://aclanthology.org/2026.findings-acl.1192/)), Findings ACL 2026 | Uses salient-entity structure to handle dense pairs first and sparse pairs later, targeting long-document multi-hop relation recovery. | Test whether multi-topic source documents benefit from structured candidate ordering. Validate the single-salient-entity assumption against KMS2 documents before considering it. |

### 3.2 Source-grounded quality, refinement, and normalization

| Paper and status | Finding | KMS2 question |
|---|---|---|
| [Kim et al., “GraphRefine”](graphrefine.md) ([source](https://aclanthology.org/2026.acl-long.1353/)), ACL 2026 | Categorizes factual inconsistencies through human evaluation and trains a triple-level refiner that can delete, edit, or rewrite draft triples. | First create a KMS2 error taxonomy and labels. Then compare selective refinement against the frozen decomposition baseline. A refiner must preserve source text, direction, qualifiers, endpoint kinds, and provenance; it must not become a generic repair layer. |
| [Huang et al., “GraphJudge”](graphjudge.md) ([source](https://aclanthology.org/2025.emnlp-main.554/)), EMNLP 2025 | Combines entity-centric document filtering with a fine-tuned graph-quality judge and reports strong results on two general and one domain-specific text–graph data set. | Test whether a judge can detect unsupported facts, direction errors, duplicate relations, and type errors on KMS2 labels. Judge precision and calibration matter more than a vague quality score. |
| [Chepurova et al., “Wikontic”](wikontic.md) ([source](https://aclanthology.org/2026.eacl-long.388/)), EACL 2026 | Extracts qualifier-bearing triples, enforces Wikidata type/relation constraints, and normalizes entities. It reports 86% MINE-1 information retention and fewer than 1,000 output tokens for graph construction in its setting. | Use as evidence for evaluating qualifiers, typed constraints, and normalization. Do not impose Wikidata’s closed ontology on KMS2’s open-domain semantic model without a user-approved schema decision. |
| [Yin et al., “How to Talk to Language Models”](relational-serialization.md) ([source](https://aclanthology.org/2025.findings-naacl.437/)), Findings NAACL 2025 | Shows that structured-entity matching depends on serialization and proposes random-walk serialization for relational context with lightweight open models. | Compare current hub candidate text with relation-aware serialization on same-source hard negatives. Measure pair recall, judge precision, throughput, and prompt size. |
| [Zhang and Soh, “EDC”](edc.md) ([source](https://aclanthology.org/2024.emnlp-main.548/)), EMNLP 2024 | Separates open extraction, schema definition, and schema canonicalization with a schema retriever and verification stage. | Retain as the ontology-canonicalization baseline. Test only if KMS2 relation-label drift or duplicate predicates is demonstrated; it is older and multi-call. |

### 3.3 Selective verification and calibration

| Paper and status | Finding | KMS2 question |
|---|---|---|
| [Zhang et al., “Rethinking the Role of LLMs for DocRE”](rethinking-docre.md) ([source](https://aclanthology.org/2025.naacl-long.319/)), NAACL 2025 | Routes hard entity pairs near the no-relation boundary to an LLM refiner and combines model distributions. | Can a calibrated KMS2 score route only ambiguous facts, triplets, or hub pairs to a stronger verifier? Measure risk-coverage and false-positive/false-negative asymmetry first. |
| [Li et al., “Supervised Rationale Verification and Feedback”](rationale-verification.md) ([source](https://ojs.aaai.org/index.php/AAAI/article/view/34631)), AAAI 2025 | Trains a rationale supervisor and regenerates only flagged relation predictions. | Test whether a lightweight evidence supervisor predicts actual source support. Fluent rationales are not evidence entailment; the label must be the source-grounded decision. |
| [Xu et al., “ATGL”](atgl.md) ([source](https://aclanthology.org/2026.acl-long.1603/)), ACL 2026 | Stabilizes a discriminative relation threshold while exposing a precision/recall trade-off. | The loss is not directly applicable to KMS2’s generative API. The transferable question is whether calibrated abstention and threshold selection improve risk-coverage. |
| [Dekoninck et al., “A Unified Approach to Routing and Cascading for LLMs”](routing-cascading.md) ([source](https://proceedings.mlr.press/v267/dekoninck25a.html)), ICML 2025 | Derives routing/cascading policies and identifies good quality estimators as the critical factor for the success of model selection paradigms. | Profile KMS2 and build a semantic quality estimator before testing cascades. Compare one-shot routing, selective verification, and fixed baseline under equal quality targets. |

Rethinking, EP-RSR, and ATGL share a related DocRE/no-relation-imbalance lineage. Their results must not be counted as independent confirmation of one candidate-routing hypothesis.

### 3.4 Historical comparison references

- [GenIE](https://aclanthology.org/2022.naacl-main.342/) remains a useful constrained-generation baseline for schema-consistent triple output. It assumes a closed schema and therefore is not a direct replacement for KMS2’s open predicate wording.
- [Eider](https://aclanthology.org/2022.findings-acl.23/) and [DREEAM](https://aclanthology.org/2023.eacl-main.145/) remain evidence-selection baselines for document-level relation extraction. Their evidence retrieval ideas can inform KMS2 context experiments, but their supervised architectures are not drop-in semantic modules.
- [Chain-of-Verification](https://aclanthology.org/2024.findings-acl.212/) is a verification baseline. Its multi-step cost is a reason to test selective verification rather than apply it to every KMS2 output.

## 4. Accuracy and efficiency synthesis mapped to KMS2

| Research axis | KMS2 decision point | Accuracy risk | Efficiency risk | First measurement |
|---|---|---|---|---|
| Atomic propositions / complexity-aware decomposition | `FactExtractionNode`, `FactExtractionSignature` | Omitted qualifiers, invented implications, loss of entity identity | More preprocessing calls and tokens | Fact support, atomicity, recall, unsupported-fact rate, calls, tokens, latency |
| Fact-to-triplet extraction | `TripletDecompositionNode`, `TripletDecompositionSignature` | Wrong direction, entity/event kind, duplicate or tautological relations | One call per fact | Directed triple F1, kind accuracy, support, duplicates, per-fact cost |
| Evidence-selected relation context | Fact context window and typed description loaders | Cross-block evidence leakage or missing references | Retrieval overhead | Recall of supported facts, evidence sufficiency, context tokens, latency |
| Typed descriptions | Five sequential `source_*_description` phases | Unsupported glosses, lost order/qualifiers, type conflation | Five phases share one locked runtime and may still fan out internally within each phase | Groundedness, type-specific field accuracy, retrieval utility, tokens, latency |
| Entity/predicate serialization | Five embedding nodes and hub candidate queries | Missed aliases, relation-direction errors, poor hard-negative separation | Larger prompts and embedding cost | Candidate recall, pair precision, prompt size, embedding latency/storage |
| Reranker and borderline judge | `source_*_hub.py` cascades | Threshold drift, false merges/splits, uncalibrated Boolean decisions | Expensive reranker/judge calls | Pair precision/recall, risk-coverage, calls, tokens, role transitions |
| Community formation | SLLPA queries/repositories | Impure hubs, singleton loss, unstable overlaps | Projection and graph algorithm cost | Purity, coverage, rerun stability, exact triplet-hub coverage |
| Canonical synthesis | Hub synthesis workers | Unsupported canonical definitions or membership re-decisions | One call per retained community and retries for synthesis profiles | Definition faithfulness, evidence coverage, output tokens, latency |
| Runtime scheduling | `SemanticGraph.build_graph`, `LocalModelRuntime` | Quality regressions from altered phase order | Repeated LLM/reranker/embed activation | Wall time by phase, role transitions, server startup, peak memory, throughput |

## 5. Ranked experiment portfolio

Each experiment compares one intervention against the frozen KMS2 baseline. A result is actionable only if it improves the named accuracy target without violating source, type, provenance, ordering, or strict-alignment contracts.

| Rank | Experiment | Fixed baseline | Required labels / data | Accuracy measures | Efficiency measures | Decision criterion |
|---:|---|---|---|---|---|---|
| 1 | Establish a current-domain fact-extraction benchmark | Current fact signature, ±400 estimated-token context, no retry | Stratified current-domain source sample with manual labels for support, atomicity, references, negation, modality, and qualifiers | Exact/normalized fact precision, recall, F1; source support; atomicity; negation, modality, qualifier retention; unsupported-fact rate | Calls, input/output tokens, wall time | Freeze a reproducible baseline and error taxonomy before architecture changes |
| 2 | Compare source-faithful atomic or complexity-aware decomposition | Current one-call-per-block fact extraction | Current-domain fact benchmark plus labels for coreference, atomicity, and target-block support | Fact recall and precision by block type/complexity; unsupported implications; entity preservation | Added preprocessing calls/tokens and end-to-end latency | Adopt only if support and recall improve without unacceptable added cost |
| 3 | Compare fact-to-triplet strategies | Current one-call-per-fact decomposition | Manually labeled directed triplets, endpoint kind, relation support, qualifiers, duplicate/tautology status | Subject/predicate/object precision, recall, F1; direction accuracy; `ENTITY|EVENT` accuracy; source support | Per-fact calls, tokens, latency, malformed/empty output rate | Require a clear gain in directed support and kind accuracy; preserve fresh IDs and provenance |
| 4 | Calibrate typed-description quality | Current five type-specific description modules and contexts | Typed descriptions with source-groundedness, roles, order, scope, conditions, aliases | Groundedness and field-level accuracy by type; retrieval utility; qualifier/order retention | Tokens, calls, description length, latency | Do not change prompts or context budgets without type-specific evidence |
| 5 | Evaluate source-hub candidate recall and serialization | Candidate limit 32, similarity 0.86, current embedding text | Same-source positive/negative pairs, aliases, relation-direction hard negatives, event/process pairs | Candidate recall, pair precision/recall, hard-negative separation, per-type calibration | Embedding size, prompt size, reranker calls, tokens | Tune candidate retrieval only when recall is measured before reranker/judge changes |
| 6 | Evaluate reranker and Boolean judge routing | Direct threshold .90, borderline lower threshold .60, judge batch size 16 | Pair equivalence labels and judge correctness; include abstention/uncertainty labels if possible | Pair precision/recall, risk-coverage, false merge/split rates, calibration | Reranker/judge calls, tokens, batch utilization, role transitions | Add selective verification only when quality estimates predict errors and cost is lower at equal quality |
| 7 | Compare community algorithms and parameters | Weighted SLLPA, max iterations 100, association strength .2, minimum size 2 | Gold hub memberships or expert cluster labels | Purity, coverage, overlapping-membership quality, stability, exact triplet-hub coverage | Projection memory/time, graph runtime, persistence size | Change graph formation only if source-local identity and downstream triplet coverage both improve |
| 8 | Evaluate canonical hub synthesis | Fixed memberships and current type-specific synthesis | Human labels for canonical name/description support, aliases, conditions, and evidence coverage | Definition faithfulness, unsupported additions, alias precision, evidence coverage | Calls, retries, tokens, latency | Prefer extractive/evidence-linked synthesis only when it preserves type-specific information at lower or equal cost |
| 9 | Profile real runtime and scheduling | Current serialized typed-description and hub order plus locked role manager | Recorded runs with per-phase instrumentation | Quality must remain within baseline confidence intervals | Wall time, model calls, tokens, role transitions, activation time, peak GPU memory, throughput | Only then test batching, same-role grouping, or phase reordering; prove triplet coverage equivalence |
| 10 | Selective refiner, graph judge, QA expansion, or cascade | Best frozen baseline from experiments 1–9 | Training/validation labels, quality estimator, source-support annotations | Quality at fixed risk/coverage; fact/triplet/hub metrics; human agreement | Expected extra calls, escalation rate, end-to-end latency, GPU cost | Defer unless a measured estimator identifies a high-value subset and the intervention wins on KMS2 data |

## 6. Evidence prerequisites and decision sequence

KMS2 currently lacks complete labels for:

- source-supported atomic facts with explicit atomicity, negation, modality, and qualifier judgments;
- directed triplets and endpoint kinds;
- entity, event, predicate, statement, and procedure description fields;
- same-source hub memberships and hard negatives;
- exact source spans or evidence sufficiency labels;
- phase-level timing, token, memory, and resident-role transition records.

The recommended sequence is therefore:

1. **Create and freeze a current-domain benchmark.** Use a stratified source sample and document the annotation protocol, label definitions, and split policy before prompt or architecture changes.
2. **Build the error taxonomy.** Separate unsupported extraction, omission, bad atomicity, reference failure, direction error, endpoint-kind error, duplicate/tautology, type-description drift, false merge, false split, community impurity, and synthesis hallucination.
3. **Add labels at the earliest high-leverage boundary.** Fact and triplet labels precede typed descriptions and hubs; source-hub labels precede threshold or serialization changes; runtime profiles precede scheduling changes.
4. **Run one intervention at a time.** Keep source-block support policy, provenance, deterministic ordering, fresh occurrence IDs, typed boundaries, fixed membership, and strict vector alignment unchanged unless a separate contract change is explicitly approved.
5. **Use quality–cost Pareto decisions.** A method that improves recall by increasing unsupported facts or doubles role-transition cost is not an efficiency improvement. Report quality at fixed cost and cost at fixed quality.
6. **Escalate only on calibrated evidence.** Refiners, rationale supervisors, graph judges, and cascades require a quality estimator whose risk-coverage behavior is measured on KMS2 data. Recent literature supports selective computation; it does not supply KMS2 calibration for free.

## 7. Primary references

### Local corpus

- [ATOM](atom.md)
- [DIAL-KG](dial-kg.md)
- [EvoRAG](evorag.md)
- [RAGA](raga.md)
- [AutoGraph-R1 metadata](autograph-r1.md) and [full text](autograph-r1.txt)

### 2025–2026 primary sources

- [Pommeret et al., Atomic Propositions, arXiv 2026](pommeret-atomic-propositions.md) — [source](https://arxiv.org/abs/2604.02866)
- [Choi et al., SocraticKG, arXiv 2026](socratic-kg.md) — [source](https://arxiv.org/abs/2601.10003)
- [Kim et al., GraphRefine, ACL 2026](graphrefine.md) — [source](https://aclanthology.org/2026.acl-long.1353/)
- [Chepurova et al., Wikontic, EACL 2026](wikontic.md) — [source](https://aclanthology.org/2026.eacl-long.388/)
- [Xu et al., ATGL, ACL 2026](atgl.md) — [source](https://aclanthology.org/2026.acl-long.1603/)
- [Yang and Tan, SegDRE, Findings ACL 2026](segdre.md) — [source](https://aclanthology.org/2026.findings-acl.1192/)
- [Anuyah et al., CoDe-KG, EMNLP 2025](code-kg.md) — [source](https://aclanthology.org/2025.emnlp-main.783/)
- [Huang et al., GraphJudge, EMNLP 2025](graphjudge.md) — [source](https://aclanthology.org/2025.emnlp-main.554/)
- [Yin et al., structured entity matching, Findings NAACL 2025](relational-serialization.md) — [source](https://aclanthology.org/2025.findings-naacl.437/)
- [Zhang et al., EP-RSR, Findings NAACL 2025](ep-rsr.md) — [source](https://aclanthology.org/2025.findings-naacl.224/)
- [Zhang et al., Rethinking DocRE, NAACL 2025](rethinking-docre.md) — [source](https://aclanthology.org/2025.naacl-long.319/)
- [Li et al., supervised rationale verification, AAAI 2025](rationale-verification.md) — [source](https://ojs.aaai.org/index.php/AAAI/article/view/34631)
- [Dekoninck et al., routing and cascading, ICML 2025](routing-cascading.md) — [source](https://proceedings.mlr.press/v267/dekoninck25a.html)
- [SLOT, EMNLP Industry 2025](slot.md) — [source](https://aclanthology.org/2025.emnlp-industry.32/)

### Historical baselines

- [GenIE](https://aclanthology.org/2022.naacl-main.342/)
- [Eider](https://aclanthology.org/2022.findings-acl.23/)
- [DREEAM](https://aclanthology.org/2023.eacl-main.145/)
- [Chain-of-Verification](https://aclanthology.org/2024.findings-acl.212/)
- [EDC](https://aclanthology.org/2024.emnlp-main.548/)

## Conclusion

The current research does not justify replacing KMS2’s semantic stage with a single fashionable architecture. It does justify a disciplined evaluation program: preserve source-faithful atomic facts, measure directed triplet and endpoint-kind quality, label typed descriptions and source-hub memberships, calibrate selective verification, then optimize the locked local runtime against measured quality and cost. Recent 2025–2026 work makes hard-pair routing, evidence-aware decomposition, graph refinement, relational serialization, and calibrated cascading promising hypotheses. KMS2 should adopt none of them until the corresponding intervention wins on KMS2 data without weakening provenance or type contracts.