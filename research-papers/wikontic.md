            Wikontic: Constructing Wikidata-Aligned, Ontology-Aware
                Knowledge Graphs with Large Language Models
          Alla Chepurova1,2           Aydar Bulatov1,2           Mikhail Burtsev3          Yuri Kuratov1,2

                          Cognitive AI Systems Lab, Moscow, Russia
                                1
      2
          Moscow Independent Research Institute of Artificial Intelligence, Moscow, Russia
                   3
                     London Institute for Mathematical Sciences, London, UK
               {chepurova,bulatov,kuratov}@cogailab.com, mb@lims.ac.uk


                       Abstract                               answering compositional questions), making KGs
    Knowledge graphs (KGs) provide structured,
                                                              a reliable complement to both LLMs and retrieval-
    verifiable grounding for large language mod-              augmented generation (RAG) systems. Therefore,
    els (LLMs), but current LLM-based systems                 creating high-quality KGs directly from raw text
    commonly use KGs as auxiliary structures for              provides reliable, transparent knowledge that com-
    text retrieval, leaving their intrinsic quality un-       plements LLMs and RAG systems.
    derexplored. In this work, we propose Wikon-                 Extracting structured knowledge from text is a
    tic, a multi-stage pipeline that constructs KGs           long-standing challenge in Information Retrieval.
    from open-domain texts by extracting candi-
                                                              A common formulation is closed information ex-
    date triplets with qualifiers, enforcing Wikidata-
    based type and relation constraints, and normal-          traction (cIE), which assumes predefined fixed sets
    izing entities to reduce duplication. The result-         of entities and relation predicates drawn from an ex-
    ing KGs are compact, ontology-consistent, and             isting KG and seeks to recover all triplets that con-
    well-connected; on MuSiQue, the correct an-               form to that schema. Classical cIE pipelines decom-
    swer entity appears in 96% of generated triplets.         pose the task into stages such as named entity recog-
    On HotpotQA, our triplets-only setup achieves             nition and relation classification (Zeng et al., 2014;
    76.0 F1, and on MuSiQue 59.8 F1, match-
                                                              Zhang et al., 2020). However, separating these
    ing or surpassing several retrieval-augmented
    generation baselines that still require textual
                                                              stages leads to error accumulation and prevents the
    context. In addition, Wikontic attains state-             sharing of information between tasks. More re-
    of-the-art information-retention performance              cent end-to-end approaches approach extraction as
    on the MINE-1 benchmark (86%), outperform-                a sequence-to-sequence problem, training models
    ing prior KG construction methods. Wikon-                 to directly generate triplets from text (Distiawan
    tic is also efficient at build time: KG con-              et al., 2019; Cabot and Navigli, 2021; Josifoski
    struction uses less than 1,000 output tokens,             et al., 2022) or complete missing links in KG (Yao
    about 3× fewer than AriGraph and <1/20 of
                                                              et al., 2019). While this reduces error propagation,
    GraphRAG. The proposed pipeline improves
    the quality of the generated KG and offers                such models remain difficult to adapt to new do-
    a scalable solution for leveraging structured             mains, requiring costly retraining on high-quality
    knowledge in LLMs. Wikontic is available at               annotated corpora that remain scarce. LLMs of-
    https://github.com/screemix/Wikontic.                     fer a promising alternative: their broad knowledge
                                                              and strong prompting abilities enable open-domain
1   Introduction
                                                              extraction without expensive task-specific train-
A substantial amount of the world’s knowledge ex-             ing (Wang et al., 2020; Josifoski et al., 2023; Chep-
ists solely in unstructured textual form, such as             urova et al., 2024).
news, scientific articles, blogs, and posts on social            In contrast to cIE, open information extraction
networks. While large language models (LLMs)                  (oIE) does not impose predefined entity and rela-
are capable of extracting insights from these data,           tion names, ontology constraints, and constructs
their internal representations are latent and often           KGs from scratch. This flexibility makes oIE an
unverifiable, making them susceptible to hallu-               attractive tool for augmenting LLMs and RAG
cinations. In contrast, knowledge graphs (KGs)                systems, with recent work demonstrating reduced
record information as explicit subject-relation-              inference costs and more reliable retrieval (Chen
object triplets, supporting verifiable queries, incre-        et al., 2024; Gutierrez et al., 2024; Guo et al., 2024;
mental updates, and multi-step reasoning (e.g., in            Li et al., 2024; Han et al., 2024; Gutiérrez et al.,
                                                          8304
       Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics
                                        Volume 1: Long Papers, pages 8304–8319
                           March 24-29, 2026 ©2026 Association for Computational Linguistics
2025). For example, AriGraph (Anokhin et al.,             text, Wikontic extracts triplets with LLMs, types,
2025) learns KGs to create long-term semantic             and validates them against Wikidata, and dedupli-
memory, while Distill-SynthKG (Choubey et al.,            cates entities to yield compact, consistent KGs.
2024) integrates mixed text–triplet structures to im-        The resulting KGs are both interpretable and
prove question answering. However, most current           effective. When used as the sole knowledge source
oIE pipelines rely on KGs mainly as auxiliary scaf-       for multi-hop QA, Wikontic achieves competitive
folds to structure text retrieval, rather than treating   performance with RAG and KG-based methods
the KG itself as a high-quality knowledge resource.       that rely on raw text as context (Lee et al., 2024;
This perspective overlooks the potential of compact       Li et al., 2024; Anokhin et al., 2025; Panda et al.,
and non-redundant KGs to represent information            2024; Gutiérrez et al., 2025). Moreover, our graphs
directly, and leaves their quality and reasoning ca-      exhibit superior coverage of salient information
pabilities underexploited. As a result, the practical     and strong internal connectivity between relevant
use of oIE KGs remains limited. Extracted triplets        nodes. In summary, our main contributions are:
often contain heterogeneous surface forms—for ex-         1. We introduce Wikontic, which (1) extracts
ample, “NYC located in USA” versus “New York              candidate triplets, (2) enforces schema and ontol-
City in-country United States”—fragmenting the            ogy constraints on entity types and domain–range,
KG into redundant or inconsistent representations.        and (3) performs alias-aware entity normalization
Synonymy, coreference, and predicate variation            and deduplication to reduce redundancy, yielding
accumulate with scale, eroding the very strengths         ontology-consistent KGs.
that motivate KG construction in the first place:         2. We show that Wikontic’s KGs are ontology-
precision, interpretability, and logical consistency.     consistent, have low redundancy, strong coverage,
   To address this, we combine the flexibility of oIE     and connectivity; on MuSiQue, the correct answer
with the structural rigor of cIE by leveraging exter-     entity is present in 96% of generated triplets.
nal ontologies. Wikidata (Vrandečić, 2012), one         3. Using a KG as the sole knowledge source (no
of the largest community-maintained knowledge             access to the original text) on multi-hop question
bases, offers rich entity classes, relation schemas,      answering, Wikontic attains 76.0 F1 on HotpotQA
and domain-range constraints across more than             and 59.8 F1 on MuSiQue, matching or surpassing
100M entities. Its breadth allows coverage from           several RAG/KG baselines that still rely on text.
common sense to specialized domains, while its            4. Wikontic achieves state-of-the-art results on the
formal constraints provide principled supervision         MINE-1 benchmark, reaching 86% information-
for validating LLM outputs. Yet, integrating such         retention score.
ontology guidance into a fully automated pipeline         5. Wikontic’s KG construction uses less than 1,000
poses key challenges: (i) extracting candidate            output tokens, which is ∼3× fewer than AriGraph
triplets without predefined labels, (ii) typing and       and <1/20 of GraphRAG.
disambiguating entities under ontology classes de-
spite lexical ambiguity, and (iii) refining nodes and     2   Methods: Wikontic
edges iteratively while preserving alignment.
   In this paper, we address these challenges with        Wikontic is a multi-stage pipeline for constructing
Wikontic, a multi-stage framework that constructs         high-quality, ontology-aware KGs directly from un-
Wikidata-aligned, ontology-aware KGs directly             structured text (Figure 1). Unlike prior approaches
from texts using LLMs. Unlike prior works that            that directly map text to graph form and often
apply Wikidata ontology only for evaluation or            yield noisy, redundant, or inconsistent outputs, our
entity linking (Polat et al., 2025), we integrate a       pipeline explicitly integrates LLMs with Wikidata-
large-scale ontology from Wikidata directly into the      derived ontological constraints, entity normaliza-
information extraction pipeline. Wikontic includes        tion, and iterative refinement.
six components: (i) a curated ontology database              To enable triplet validation and alignment, the
derived from Wikidata, (ii) candidate triplet extrac-     pipeline stores ontology rules and the current KG
tion with qualifiers, (iii) ontology-aware triplet re-    (Section 2.1). The triplet extraction pipeline (Sec-
finement enforcing schema constraints, (iv) subjec-       tion 2.2) consists of three main stages: (i) triplet
t/object name refinement for entity deduplication,        candidate extraction with contextual metadata, (ii)
(v) KG storage, and (vi) retrieval for multi-hop          ontology-aware triplet refinement, and (iii) entity
question answering. Starting from unstructured            normalization and deduplication. These stages aim
                                                      8305
                                                      Wikidata

                         LLM                 LLM + Ontology                      LLM + Current KG

                          (1)                           (2)                             (3)




                                                                                                                          KG view
                                                                        x

                                                                                                             nodes from
                                                                      entity types
                                      entities                                                               current KG
                                                              relations contraints
                                      to be merged                                            Deduplicated triplets
             Text               Candidate Triplets             Ontology aligned KG             added into the KG

                                                                                                             Nolan
                                                                                               Christopher
                                  Nolan                            Nolan                         Nolan
                                  human                            human                         human
                                                                            directed
                                       directed                         director                      director



                                                                                                                          Triplet view
       In 2010, Nolan
        directed the                           2010              Inception       2010                            2010
                                 Inception                                                      Inception
      science fiction                                               film                           film
                                   movie                             movie
      movie Inception.
                                       genre                            genre                         genre

                                  Science                          Science                       Science
                                   fiction                           fiction                       fiction
                                    genre                         film genre                    film genre
                                                                      genre



Figure 1: Overview of Wikontic: an ontology-guided pipeline that constructs a Wikidata-aligned KG from text.
(1) An LLM extracts candidate (subject, relation, object) triplets (gray). (2) The extracted triplets are then refined
using Wikidata’s ontology: entity types are assigned (colored nodes), and relations that violate ontology constraints
are corrected or removed. (3) Finally, entity names are normalized, and duplicated surface forms are merged. The
resulting graph is de-duplicated, ontology-consistent, and ready for downstream tasks.


to enforce structural validity and reduce redun-                  (P31) and ’subclass of’ (P279) relations, building
dancy, to produce a cleaner and more semantically                 full taxonomies from each type up to the root. Such
coherent KG that can replace raw text in RAG for                  a hierarchy is essential because relation constraints
multi-hop QA tasks (Section 2.3).                                 are defined at different levels of abstraction. For
                                                                  example, a relation may allow connections between
2.1 Ontology and KG Databases                                     instances of the broader class ’audiovisual work’,
                                                                  even if the entity is typed more specifically as ’film’.
We built a custom ontology schema database de-
                                                                  By propagating the allowed properties of each par-
rived from Wikidata. The schema database in-
                                                                  ent type downwards to its children entity type, we
cludes properties (i.e., relations) and their com-
                                                                  ensure that entities can still be matched to valid re-
patible entity types. Properties required solely for
                                                                  lations whenever the constraint applies to its parent
linking external data (e.g., multimedia or external
                                                                  class.
identifiers) were excluded, leaving 2,464 factual
properties with suitable datatypes (e.g., Wikiba-                    We collected labels and aliases for all entity
seItem, Quantity, Point in time).                                 types and relations. Dense retrieval indexes for
   For each property, we retrieved subject and                    relation and entity type names, as well as their re-
object type constraints from Wikidata (e.g.,                      spective aliases, support semantic search. These
Q21503250 for subject, Q21510865 for object).                     indexes allow us to semantically align relation and
These constraints define type compatibility rules,                entity type names from extracted triplets with Wiki-
specifying which entity classes a relation can                    data definitions, even when surface forms differ.
logically connect and thereby guiding ontology-                      The KG database stores triplets, canonical en-
consistent triplet construction.                                  tity names, and aliases. A dense retrieval index
   To support constraint generalization, we recur-                over aliases supports efficient linking and dedupli-
sively expanded entity types using ’instance of’                  cation. As extraction proceeds, new entities are
                                                              8306
inserted with canonical labels and aliases, keeping      Stage 2: Ontology-aware Refinement. At this
the KG compact yet incrementally extensible.             stage, each candidate triplet is refined using the
   Dense retrieval indexes used in both databases        schema and constraints of the Wikidata ontology:
were built with Contriever embeddings (Izacard              Entity typing: For both subject and object, we
et al., 2022) and Atlas MongoDB vector search1 .         retrieve the top-10 candidate types from the dense
MongoDB’s hybrid support for structured queries          retrieval index. The LLM then selects the most
and dense retrieval enables both efficient graph and     plausible type. We then add supertypes from the
semantic search.                                         taxonomy to ensure coverage when constraints are
                                                         defined at higher abstraction levels.
2.2 Ontology-aware Triplet Extraction
                                                            Relation validation: Using Wikidata constraints,
Stage 1: Candidate Triplet Extraction. We ex-            we identify all relations that can legally connect
tract factual triplet candidates from unstructured       selected subject and object types, including inverse
text with an LLM, capturing subject-relation-object      combinations of subject and object types (e.g., di-
triplets, along with contextual qualifiers that en-      rected vs. director). These candidate relations are
hance the semantic meaning of the triplet. LLM is        ranked by cosine similarity to the originally ex-
prompted with instruction and in-context examples        tracted relation.
to extract triplets that include entity types for both      Triplet backbone reconstruction: The text, triplet,
the subject and object, as well as additional meta-      and valid relations are passed to the LLM, which
data that mirrors the structure of Wikidata quali-       selects the most plausible ontology-valid configu-
fiers. These qualifiers are essential because they       ration, yielding a refined triplet backbone.
capture contextual information such as time, lo-            This stage enforces structural validity, seman-
cation, or conditions. While such details usually        tic alignment, and consistency with Wikidata’s on-
cannot be expressed as standalone facts, they are        tology. A worked example is shown in Figure 5
critical for preserving factual precision and avoid-     (Appendix A.6).
ing loss of accurate knowledge during knowledge
extraction.                                              Stage 3: Entity Normalization and Alias-aware
   For instance, given the text "In 2010, Christo-       Deduplication. While the focus of the previous
pher Nolan directed the science fiction movie In-        step is validating triplet structure and semantics,
ception", the extracted triplet would be: (Nolan,        this step aligns entity names to a unified vocabulary
directed, Inception) with entities types                 of existing KG entries to reduce duplication and
(human, film) and the qualifier: {point in               ensure consistency of the constructed KG.
time: 2010}. Further details and examples are                For each refined triplet, we link its subject and
provided in Appendix A.4.                                object names to existing entities in the KG that
   However, LLM outputs may be semantically re-          share the same entity type or a compatible parent
dundant or structurally inconsistent. The entity and     type from the taxonomy. Using precomputed em-
relation names may not align with existing enti-         beddings of entity aliases from the KG, we retrieve
ties and relations already present in the KG. For        top-10 candidates and rank them by cosine similar-
example, an LLM might extract "Nolan" as the             ity to the surface forms of the extracted mentions.
subject from one input text and extract "Christo-        The top-10 candidates, together with their types,
pher Nolan" from the other one; or relations like        are passed to the LLM to determine whether the
"directed" vs. "director" might appear in different      extracted entity is synonymous with one of the ex-
grammatical forms or inverse directions in differ-       isting entries. On a match, we replace the mention
ent input texts (see Figure 1, bottom). Without          with the canonical KG label and store the surface
additional correction, these inconsistencies lead to     form as an alias; otherwise, we preserve a new en-
entity duplication and increased KG size, which          tity and add its surface form to the alias collection.
degrades both storage efficiency and downstream              This step aims to ensure that the resulting KG
reasoning. Thus, to improve consistency and re-          is compact by avoiding redundant entities with dif-
duce redundancy, the next steps of the pipeline val-     ferent surface forms and evolving by supporting
idate extracted triplets using Wikidata’s ontology       incremental updates with the discovery of new en-
and normalize entity names.                              tities. A detailed example for the second step is
   1
    https://www.mongodb.com/products/platform/           provided in Figure 6 in Appendix A.6.
atlas-vector-search                                          We implement a final ontology verification step
                                                     8307
to ensure that extracted triplets comply with the        performance. However, while MINE provides a
structural and semantic constraints of the target        useful and scalable proxy for KG quality, it does
KG. A triplet is verified if (i) its subject and ob-     not include complete ground-truth triplets. There-
ject types, together with the relation, are defined      fore, it cannot measure classical precision or recall
in the ontology, and (ii) the relation’s domain and      and instead captures only the degree to which a
range constraints are satisfied. Triplets that fail      constructed KG preserves factual information.
these checks are flagged as ontology-misaligned
                                                            To further address this, we adopt an alterna-
but retained, as they remain linked to the main
                                                         tive evaluation strategy. We measure KG qual-
KG through the entity name refinement step. Pre-
                                                         ity through (1) structural compactness (i.e., non-
serving them enables computation of an ontology
                                                         redundancy and deduplication) and (2) perfor-
alignment score and provides interpretable cues for
                                                         mance in downstream multi-hop QA. In this setup,
identifying or revising schema-inconsistent facts.
                                                         the LLM must answer factual questions in a text-
2.3 Retrieval for QA                                     free setting using only the constructed KG, without
                                                         access to the original source texts, unlike retrieval-
We address multi-hop question answering with the
                                                         augmented methods such as HippoRAG (Gutierrez
constructed KG via an iterative retrieval that de-
                                                         et al., 2024), AriGraph (Anokhin et al., 2025), and
composes the question into subquestions, ground-
                                                         Holmes (Panda et al., 2024). This design makes
ing each step in the retrieved KG context.
                                                         QA a functional proxy for two key properties of an
   Given a question, the LLM decomposes it into
                                                         extracted KG: (a) factual correctness, since noisy
the first 1-hop subquestion. For each subquestion
                                                         or invalid triplets directly impede correct answers,
the LLM (1) identifies explicitly mentioned or po-
                                                         and (b) coverage and completeness, since incom-
tentially relevant entities; (2) links extracted enti-
                                                         plete graphs restrict multi-hop reasoning. Despite
ties to KG nodes and selects those most relevant for
                                                         existing advances in aligning KGs with LLMs and
the current step; (3) given the retrieved subgraph
                                                         adapting for QA (Han and Shareghi, 2022; Dai
formed by the neighborhood of the selected enti-
                                                         et al., 2025; Sui et al., 2025; Pan et al., 2024), we
ties, generates an answer to the subquestion; (4)
                                                         deliberately refrain from training additional models
conditioned on the previous answer, the LLM for-
                                                         to estimate the KG quality itself.
mulates the next subquestion. This iterative process
continues for up to five subquestions, after which          To compare predicted answers with ground truth,
the LLM produces the final answer. Implementa-           we apply a normalization procedure that lowercases
tion details and prompt templates are provided in        all strings and removes punctuation. To account for
Appendix A.5.                                            lexical variation, we further expand entity matching
                                                         using the alias mappings stored in the KG. If the
2.4 Evaluation                                           model’s predicted answer matches any canonical
Existing benchmarks for triplet extraction suffer        entity or one of its aliases, the corresponding alias
from substantial limitations: annotated closed IE        set is treated as the set of valid candidate answers.
datasets are small-scale, noisy, and often incom-           We perform evaluations on KGs extracted us-
plete (Josifoski et al., 2023, 2022; Huguet Cabot        ing different LLMs: gpt-4.1, gpt-4.1-mini,
and Navigli, 2021), while open IE corpora are diffi-     gpt-4o-mini2 , and Llama-3.3-70b-Instruct3 ,
cult to align with real-world KGs and provide unre-      and assess the quality of the resulting KG on two
liable ground truth for evaluation (Stanovsky et al.,    multi-hop QA datasets: MuSiQue (Trivedi et al.,
2018). Constructing high-quality datasets is costly,     2022) and HotpotQA (Yang et al., 2018). We used
as annotators must not only identify all explicit and    the same questions and candidate passages, in-
implicit entity and relation mentions but also align     cluding both supporting and distractor passages,
them to the complex schemas of large KGs such as         used in HippoRAG (Gutierrez et al., 2024) and Ari-
Wikidata, which contain thousands of entity types        Graph (Anokhin et al., 2025) to compare the results
and relations. Recently, the MINE benchmark (Mo          with existing methods.
et al., 2025) was introduced to address some of
these issues by evaluating KGs through informa-
tion retention rather than exact triplet-level super-       2
                                                             https://platform.openai.com/docs/models
vision. We evaluated Wikontic on the MINE-1 task            3
                                                             https://huggingface.co/meta-llama/
and observed much higher information retention           Llama-3.3-70B-Instruct

                                                     8308
                                     35       KGGen, gpt4o
                                              GraphRAG, gpt4o
                                     30       Wikontic, gpt4o




              Frequency (Articles)
                                     25
                                     20
                                     15
                                     10
                                      5
                                      0
                                          0   10      20        30   40       50      60       70          80         90        100
                                                                      Facts captured, %

Figure 2: Distribution of MINE-1 scores across 100 articles for GraphRAG, KGGen, and Wikontic. Dotted vertical
lines are averaged scores. Wikontic scored 84% on average, substantially outperforming GraphRAG 47.80% and
KGGen 66%.


3   Results                                                                    graphs created by Wikontic in a challenging infor-
                                                                               mation extraction setting. Given the limited avail-
3.1 Evaluations on MINE-1
                                                                               ability of approaches that assess the KG quality
We evaluated Wikontic on the MINE-1 bench-                                     directly, we use a proxy evaluation methodology
mark, which measures how much factual informa-                                 based on the MuSiQue QA dataset to examine how
tion from the source text is retained in the con-                              effectively the knowledge is represented in the re-
structed KGs using an LLM-as-a-judge protocol                                  sulting KG.
from the original study (Mo et al., 2025). Fig-                                   A KG can be formally represented as G =
ure 2 displays the retention scores distribution in                            (T , E, R), where E is the set of entities e, R is
articles of MINE-1 for KGGen, GraphRAG, and                                    the set of relations r and T is the set of triplets:
Wikontic. Table 1 demonstrates the results for                                 T ∈ E × R × E. For efficient knowledge storage
both KGGen and Wikontic with different LLM                                     and retrieval, the KG should satisfy the size, den-
backbones. Wikontic consistently outperforms                                   sity, and diversity desiderata, which can be directly
KGGen, reaching 84% with gpt-4o and 86% with                                   evaluated using graph statistics (Table 2).
gpt-4.1-mini, compared to KGGen’s best score
of 73% (Claude Sonnet 3.5). These results demon-                                 Method                         |E|    |R|
                                                                                                                              Avg. e Unique r diversity
                                                                                                                              degree e per r per 2 × e
strate that Wikontic effectively preserves factual                               HippoRAG                   234.9     130.1    4.0     1.8      1.1
information during the construction of the KG.                                   AriGraph                   228.0     115.6    3.9     2.0     1.01
                                                                                 Wikontic (1-3)             248.8     104.8    4.3     2.5     1.03
                                                                                 w/o ontology (2)           232.4     106.7    4.4     2.6     1.06
    Method                   MINE-1 Score (%)
                                                                                 w/o ontology (2) and
    KGGen, Claude Sonnet 3.5               73                                                               273.0 140.9        4.2     2.3      1.09
                                                                                    normalization (3)
    KGGen, GPT-4o                          66                                    w/o ontology-misaligned
                                                                                                            239.9 99.5         4.3     2.6      1.0
                                                                                    triplets
    KGGen, Gemini 2.0 Flash                44
    GraphRAG, gpt4o                        48
                                                                               Table 2: KGs structural statistics for MuSiQue QA cor-
    Wikontic, gpt4o                        84
                                                                               pus: the number of unique entities (|E|) and relations
    Wikontic, gpt4.1-mini                  86
                                                                               (|R|), average entity degree (Avg. e degree), the number
Table 1: MINE-1 information-retention scores for                               of unique entities per relation (Unique e per r) and the
KGGen, GraphRAG, and Wikontic. Wikontic achieves                               average relation diversity per two entities (r diversity
the highest retention performance across all evaluated                         per 2 × e).
LLMs.
                                                                                  Graph size is directly connected to the number
                                                                               of stored facts, and can be represented by the aver-
3.2 Graph Quality Analysis                                                     age number of entities and relations per MuSiQue
We aim to comprehensively evaluate the structure,                              sample, denoted by |E| and |R|. Wikontic with-
information content, and usability of knowledge                                out ontology (Stage 2) and normalization (Stage
                                                                          8309
                                                                  3.3       Answer Coverage
                                                                  In the task of question answering, the primary pur-
            3-hop                 2-hop
                                                                  pose of the extracted KG is to extract as much
                                                                  relevant information from the context as possible.
                                          0.3
                                                0.4               Ideally, the resulting graph should contain all enti-
                                   0.2                            ties mentioned in the question as well as the answer
                            0.1
   5-hop                                        1-hop             to the question. In multi-hop question answering,
                                     HippoRAG
                                     AriGraph                     entities in question will unlikely be in the same
                                     Wikontic                     context as the answer entity, meaning they may not
                                     w/o ontology
                                      and normalization           be direct neighbors in the resulting KG. However,
                                      (Stage 1)
                                     w/o ontology-misaligned      in a KG with sufficient coverage, there has to be a
                                     triplets
           10-hop                    w/o qualifiers               reasonably short path connecting these two entities.
                                    Main                             To measure the overall coverage of various KG
                                  connected
                                  component                       construction methods, we estimate whether the an-
Figure 3: Wikontic produces the most dense KGs for                swer to the question is present in the KG as an
MuSiQue questions. For each question, subgraphs are               entity and whether the path from the question to
constructed around its entities, and their sizes are re-          the answer exists in the graph. Due to differences
ported relative to the full KG. The figure shows the              in pipelines, we cast all entity and relation names to
relative sizes of 1– to 10-hop neighborhoods and the              lowercase and remove punctuation to standardize
entire connected component containing the question,               their format. To account for possible differences in
defined as all nodes reachable from any question node.
                                                                  entity naming, we consider two entities matching
                                                                  if one is a substring of the other.
                                                                     Table 3 presents coverage and size metrics for
3) yields diverse KGs with the highest number
                                                                  KGs built by our pipeline, AriGraph, and Hip-
of unique entities and relations, followed by Hip-
                                                                  poRAG on the MuSiQue dataset. Since baseline
poRAG. However, the sheer volume of relations
                                                                  KGs are available only for the gpt4o-mini model
does not necessarily make retrieval more informa-
                                                                  and 80 common test samples, we use the same
tive. Each relation should also be well-represented
                                                                  configuration to ensure a fair comparison. We esti-
across various unique entities to ensure that relation
                                                                  mated the standard deviation using bootstrapping.
names are standardized and meaningful. This prop-
                                                                  "Contains Answer" represents the percentage of
erty is reflected by the average number of unique
                                                                  cases when the answer entity is present in the neigh-
entities per relation, which is significantly higher
                                                                  borhood of entities from the MuSiQue question.
in refinement-augmented Wikontic versions.
                                                                  We ablate Wikontic by removing one or multiple
   High KG connectivity, represented by the aver-
                                                                  pipeline steps at a time. "Ontology Entailment" is
age entity degree, ensures efficient retrieval, espe-
cially with limited search depth. Entity normaliza-
                                                                                                  Contains Answer (%)      Ontology
tion (Stage 3) is a key to building KGs with the                      Method
                                                                                                 Total    5-hop 10-hop Entailment (%)
highest density among the compared methods. The                       HippoRAG                  96.3±2.1 67.5±5.3 68.8±5.2    -
                                                                      AriGraph                  79.9±4.5 40.0±5.5 41.3±5.5    -
relation diversity, or the number of unique relations                 Wikontic                  96.2±2.1 66.3±5.3 68.8±5.2   96.5
per two entities, captures a variety of information                    w/o ontology (2)         97.5±1.7 66.3±5.3 70.0±5.2   15.2
                                                                       w/o ontology (2) and
stored in the KG. Here, Wikontic versions with re-                       normalization (3)
                                                                                               96.2±2.1 65.0±5.4 68.8±5.2   12.4

laxed ontology constraints remain the most diverse.                    w/o ontology-misaligned
                                                                                               93.8±2.7 63.8±5.4 66.3±5.3   100.0
                                                                         triplets
   In a dense graph, important nodes should have                       w/o qualifiers          85.0±4.0 51.3±5.7 53.8±5.6   96.5
many neighbors. We select the entities from
MuSiQue questions and assess their neighborhood                   Table 3: Knowledge coverage of graphs built using dif-
                                                                  ferent extraction methods on the MuSiQue dataset with
in graphs built by various methods (Figure 3). The
                                                                  mini-sized models. (Left) Percentage of cases where
largest possible neighborhood is the main con-                    the correct answer to a question appears in the full con-
nected component containing the given entities. In                structed graph or within the 5-and 10-hop neighbor-
KGs built by the full Wikontic pipeline, each neigh-              hoods of the question nodes. Wikontic pipelines provide
borhood contains the greatest number of entities,                 the best answer coverage while maintaining high ontol-
again underscoring strong KG connectivity and the                 ogy agreement. (Right) Average percentage of triplets in
importance of ontology.                                           each sample that are entailed by the Wikidata ontology.

                                                               8310
             Method                                                      MuSiQue                   HotpotQA
                                                                  EM            F1          EM          F1
             Wikontic, gpt4.1                                     46.8±0.8      59.8±0.3    64.5±0.4    76.0±0.4
             Wikontic, gpt4.1-mini                                42.6±0.7      55.9±0.3    59.7±0.6    71.7±0.8
             Wikontic, gpt4o-mini                                 42.1±0.1      53.3±0.1    53.7±1.2    65.8±1.0
             Full context, gpt4                                   33.5          42.7        53.0        68.4
             Supporting facts, gpt4                               45.0          56.0        57.0        73.8
             ReadAgent (Lee et al., 2024), gpt4                   35.0          45.1        48.9        62.0
             GraphReader (Li et al., 2024), gpt4                  38.0          47.4        55.0        70.0
             GraphRAG (Edge et al., 2024), gpt4o-mini             40.0          53.5        58.7        63.3
             AriGraph (Anokhin et al., 2025), gpt4o-mini          36.5          47.9        60.0        68.0
             AriGraph (Anokhin et al., 2025), gpt4                45.0          57.0        68.0        74.7
             HOLMES (Panda et al., 2024), gpt4                    48.0          58.0        66.0        78.0
             Wikontic, Llama 3.3                                  37.7±0.6      49.7±0.4    55.1±0.5    67.4±0.5
             HippoRAG v2 (Gutiérrez et al., 2025), Llama 3.3      37.2          48.6        62.7        75.5

Table 4: Exact Match (EM) and F1 scores on the MuSiQue and HotpotQA. Wikontic operates solely on KG triplets
without accessing the source text, yet achieves performance comparable to or even exceeding both raw-text baselines
(Full Context, Supporting Facts) and retrieval-augmented KG approaches that still rely on source text access.

the percentage of triplets where subject, object, and          the estimated token-based costs for Wikontic, Ari-
relation match the ontology.                                   Graph, and GraphRAG, based on publicly available
   Both Wikontic and HippoRAG achieve over 96%                 data and original implementations.
answer coverage, surpassing AriGraph’s 79.9%.
Notably, only 3.5% of triples in Wikontic are                          Tokens          Wikontic AriGraph GraphRAG
ontology-misaligned, confirming that the gener-                        Prompt            12,687      11,000        115,000
ated knowledge is largely schema-consistent; only                      Completion           881       2,500         20,000
a small portion of produced triplets requires correc-
                                                               Table 5: Mean token efficiency for KG construction per
tion to fully satisfy ontology constraints. Without            text paragraph of Wikontic compared with AriGraph
ontology constraints, Wikontic reaches the highest             and GraphRAG on the MuSiQue dataset.
answer coverage (97.5%), avoiding mismatches be-
tween Wikidata and MuSiQue entities and making                    A key indicator of computational cost is the
it particularly effective for open-domain QA. When             number of completion tokens, which are typically
ontology consistency is required, the Wikontic                 around 3-5 times more expensive4,5 and compu-
variant that excludes ontology-misaligned triplets             tationally intensive than input tokens (Zhou et al.,
(100% ontology entailment) attains a competitive               2024). Under this metric, Wikontic is more ef-
answer coverage of 93.8%.                                      ficient, producing KGs with roughly three times
   These findings suggest that Wikontic variants               fewer output tokens than AriGraph (881 vs 2,500)
provide the best solution both with and without the            and about twenty times fewer than GraphRAG (881
ontology. Ontology and misaligned triplets exclu-              vs 20,000). This shows that Wikontic achieves
sion help to achieve the most standardized graph               comparable KG construction quality while using
and strong answer coverage by thorough triplet re-             significantly fewer output tokens.
finement and deduplication strategies. Wikontic
achieves the best coverage of information essential            3.5       Performance on QA Tasks
for question answering, while maintaining the high-            Table 4 reports Exact Match (EM) and F1 scores
est connectivity levels, crucial for enabling graph            on the MuSiQue and HotpotQA datasets. Un-
search.                                                        like retrieval-augmented approaches such as Hip-
                                                               poRAG and AriGraph, which use KGs primarily
3.4 Computational Efficiency                                   to retrieve and process relevant text passages, our
We evaluated the computational efficiency of dif-              method performs reasoning directly over structured
ferent KG-construction methods by counting the                 triplets without accessing the original documents.
number of input (prompt) and output (completion)                  Despite this constraint, using our triplet-only
tokens required to build a KG from a single para-                  4
                                                                       https://claude.com/platform/api/
graph in the MuSiQue dataset. Table 5 presents                     5
                                                                       https://openai.com/api/pricing/

                                                           8311
contexts for gpt-4.1, it achieves strong results,            4   Conclusions
reaching 64.5 EM and 76.0 F1 on HotpotQA and
46.8 EM and 59.8 F1 on MuSiQue. These scores                 We introduced Wikontic, a fully automated pipeline
surpass several retrieval-based methods, including           that uses LLMs to construct KGs from unstruc-
ReadAgent and GraphReader, and are comparable                tured text. The pipeline produces compact, in-
to more resource-intensive systems such as Ari-              ternally consistent graphs by aligning extracted
Graph and HOLMES that rely on richer textual                 triplets with the Wikidata ontology and dedupli-
context. This demonstrates that complete and well-           cating entities and relations. While using KGs
structured symbolic representations of KGs can               as the sole knowledge source for multi-hop ques-
serve as a sufficient and reliable information source        tion answering, Wikontic achieves competitive per-
for multi-hop reasoning.                                     formance with retrieval-augmented and KG-based
                                                             baselines that still rely on source texts. Thus,
3.6 Ablation Study                                           ontology-guided KG construction is a viable al-
                                                             ternative to passage-level retrieval. On MuSiQue,
                                                             Wikontic includes 38–45 more unique entities than
        Method variant             EM        F1              HippoRAG and AriGraph and contains the cor-
        Wikontic (gpt4.1-mini) 42.6±0.7 55.9±0.3             rect answer entity in 97.5% of cases. Within a
         w/o qualifiers        23.9±0.2 39.4±0.4             10-hop subgraph, it maintains 70% answer cov-
         w/o aliases           36.5±0.8 50.0±1.4             erage and denser local connectivity. For QA, it
         w/o ontology (2)      36.3±1.2 48.8±1.0             attains 64.5 EM / 76.0 F1 on HotpotQA and 46.8
         w/o ontology (2) and
            normalization (3)
                               27.0±1.2 36.9±2.2             EM / 59.8 F1 on MuSiQue, outperforming text-
         Single-step QA        31.3±0.6 43.4±0.6             reliant systems (ReadAgent, GraphReader) and
                                                             approaching larger text-dependent methods (Ari-
Table 6: Ablations of the Wikontic pipeline on               Graph, HOLMES). Moreover, Wikontic achieves
MuSiQue. “Single-step QA” omits iterative subquestion        state-of-the-art results on the MINE-1 benchmark,
reasoning. Removing ontology or entity normalization
                                                             achieving 84–86% information-retention scores
yields the largest drop, highlighting their importance for
accurate reasoning over the constructed KG.                  and surpassing GraphRAG and KGGen. These
                                                             findings indicate that the proposed pipeline pre-
                                                             serves a substantial amount of factual information
   To evaluate the contribution of individual Wikon-         from source texts.
tic components, we conducted ablations (Table 6).
Removing qualifiers leads to a substantial perfor-              Our approach is also token-efficient: during KG
mance drop (–15.9 EM, –15.7 F1) on MuSiQue,                  construction, Wikontic uses under 1,000 output
indicating that qualifier information is essential           tokens, about 3× fewer than AriGraph and 1/20
for capturing fine-grained relational context. Ex-           of GraphRAG, while preserving accuracy. More-
cluding aliases (introduced at Stage 3) moderately           over, only 3.5% of extracted triplets are flagged as
decreases performance, confirming that alias ex-             ontology-misaligned, indicating that nearly all gen-
pansion improves entity matching in question an-             erated knowledge is schema-consistent, minimiz-
swering. Eliminating ontology integration (Stage 2)          ing the need for manual correction and significantly
reduces both EM and F1, demonstrating the impor-             reducing annotation overhead. Beyond efficiency,
tance of type and schema constraints for consistent          the pipeline is adaptable: it can operate without an
KG construction. When both ontology and en-                  ontology or integrate domain-specific ontologies,
tity normalization are removed (Stages 2 and 3),             enabling applications across specialized domains.
performance degrades most severely. The single-              Moreover, as LLMs increasingly serve as data gen-
step QA variant also performs significantly worse,           erators, Wikontic provides a principled foundation
confirming that multi-hop question decomposition             for producing verified, structured KG data suitable
is essential for effective reasoning over the con-           for fine-tuning smaller task-specific models.
structed KG. Overall, these findings show that each             Our findings show that ontology-aware KG con-
component contributes meaningfully to Wikontic’s             struction enables scalable, interpretable, and verifi-
performance, with ontology-guided refinement and             able transformation of unstructured text into struc-
iterative retrieval being the most critical for down-        tured knowledge, bridging symbolic reasoning and
stream reasoning over the constructed KG.                    generative language modeling.
                                                         8312
Limitations                                                Xinbang Dai, Yuncheng Hua, Tongtong Wu, Yang
                                                             Sheng, Qiu Ji, and Guilin Qi. 2025. Large language
Our experiments are restricted to proprietary Ope-           models can better understand knowledge graphs than
nAI models (GPT-4.1, GPT-4.1-mini, GPT-4o-                   we thought. Knowledge-Based Systems, 312:113060.
mini) and the open-source Llama 3.3-70B. To-               Bayu Distiawan, Gerhard Weikum, Jianzhong Qi, and
ken efficiency is measured as model-generated                Rui Zhang. 2019. Neural relation extraction for
tokens during KG construction. This metric re-               knowledge base enrichment. In Proceedings of the
flects provider billing but not end-to-end latency           57th Annual Meeting of the Association for Compu-
                                                             tational Linguistics, pages 229–240.
or throughput. Every stage of the pipeline cur-
rently uses LLMs with instructions and in-context          Darren Edge, Ha Trinh, Newman Cheng, Joshua
examples prompting. Because the pipeline now                 Bradley, Alex Chao, Apurva Mody, Steven Truitt,
                                                             Dasha Metropolitansky, Robert Osazuwa Ness, and
yields its own annotated data, several stages could          Jonathan Larson. 2024. From local to global: A
be replaced in future work by smaller, task-specific         graph rag approach to query-focused summarization.
models fine-tuned on this data, thereby improving            arXiv preprint arXiv:2404.16130.
efficiency and lowering computational cost.                Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
   In the current work, we focus only on Wikidata             Huang. 2024. Lightrag: Simple and fast retrieval-
due to its size, quality, and rich ontology. However,         augmented generation.
the proposed pipeline is flexible and can be adapted       Bernal Gutierrez, Yiheng Shu, Yu Gu, Michihiro Ya-
to any domain and ontology with a matching format            sunaga, and Yu Su. 2024. Hipporag: Neurobiologi-
of triplet constraints and entity types.                     cally inspired long-term memory for large language
                                                             models. volume 37, pages 59532–59569.
                                                           Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
References                                                   Sizhe Zhou, and Yu Su. 2025. From rag to memory:
Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry         Non-parametric continual learning for large language
  Evseev, Andrey Kravchenko, Mikhail Burtsev, and            models. arXiv preprint arXiv:2502.14802.
  Evgeny Burnaev. 2025. Arigraph: Learning knowl-          Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan
  edge graph world models with episodic memory for           Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A
  llm agents. In Proceedings of the Thirty-Fourth Inter-     Rossi, Subhabrata Mukherjee, Xianfeng Tang, and 1
  national Joint Conference on Artificial Intelligence,      others. 2024. Retrieval-augmented generation with
  IJCAI-25, pages 12–20. International Joint Confer-         graphs (graphrag). arXiv preprint arXiv:2501.00309.
  ences on Artificial Intelligence Organization. Main
  Track.                                                   Jiuzhou Han and Ehsan Shareghi. 2022. Self-supervised
                                                              graph masking pre-training for graph-to-text gener-
Pere-Lluís Huguet Cabot and Roberto Navigli. 2021.            ation. In Proceedings of the 2022 Conference on
  Rebel: Relation extraction by end-to-end language           Empirical Methods in Natural Language Processing,
  generation. In Findings of the Association for Com-         pages 4845–4853, Abu Dhabi, United Arab Emirates.
  putational Linguistics: EMNLP 2021, pages 2370–             Association for Computational Linguistics.
  2381.
                                                           Pere-Lluís Huguet Cabot and Roberto Navigli. 2021.
Hanzhu Chen, Xu Shen, Qitan Lv, Jie Wang, Xiaoqi             Rebel: Relation extraction by end-to-end language
  Ni, and Jieping Ye. 2024. Sac-kg: Exploiting large         generation. In Findings of the Association for Com-
  language models as skilled automatic constructors for      putational Linguistics: EMNLP 2021, Online and in
  domain knowledge graph. In Proceedings of the 62nd         the Barceló Bávaro Convention Centre, Punta Cana,
  Annual Meeting of the Association for Computational        Dominican Republic. Association for Computational
  Linguistics (Volume 1: Long Papers), pages 4345–           Linguistics.
  4360.
                                                           Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
Alla Chepurova, Yurii Kuratov, Aydar Bulatov, and            tian Riedel, Piotr Bojanowski, Armand Joulin, and
  Mikhail Burtsev. 2024. Prompt me one more                  Edouard Grave. 2022. Unsupervised dense informa-
  time: A two-step knowledge extraction pipeline             tion retrieval with contrastive learning. Transactions
  with ontology-based verification. In Proceedings           on Machine Learning Research.
  of TextGraphs-17: Graph-based Methods for Natural
  Language Processing, pages 61–77.                        Martin Josifoski, Nicola De Cao, Maxime Peyrard,
                                                            Fabio Petroni, and Robert West. 2022. GenIE: Gen-
Prafulla Kumar Choubey, Xin Su, Man Luo, Xiangyu            erative information extraction. In Proceedings of
  Peng, Caiming Xiong, Tiep Le, Shachar Rosenman,           the 2022 Conference of the North American Chap-
  Vasudev Lal, Phil Mui, Ricky Ho, and 1 others. 2024.      ter of the Association for Computational Linguistics:
  Distill-synthkg: Distilling knowledge graph synthe-       Human Language Technologies, pages 4626–4643,
  sis workflow for improved coverage and efficiency.        Seattle, United States. Association for Computational
  arXiv preprint arXiv:2410.16597.                          Linguistics.
                                                       8313
Martin Josifoski, Marija Sakota, Maxime Peyrard, and      Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
 Robert West. 2023. Exploiting asymmetry for syn-           and Ashish Sabharwal. 2022. Musique: Multi-
 thetic training data generation: SynthIE and the case      hop questions via single-hop question composition.
 of information extraction. In Proceedings of the 2023      Transactions of the Association for Computational
 Conference on Empirical Methods in Natural Lan-            Linguistics, 10:539–554.
 guage Processing, pages 1555–1574, Singapore. As-
 sociation for Computational Linguistics.                 Denny Vrandečić. 2012. Wikidata: a new platform for
                                                            collaborative data collection. In Proceedings of the
Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John            21st International Conference on World Wide Web,
  Canny, and Ian Fischer. 2024. A human-inspired            page 1063–1064, New York, NY, USA. Association
  reading agent with gist memory of very long contexts.     for Computing Machinery.
  In Forty-first International Conference on Machine
  Learning.                                               Chenguang Wang, Xiao Liu, and Dawn Song. 2020.
                                                            Language models are open knowledge graphs. arXiv
Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu,           preprint arXiv:2010.11967.
  Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yang-
  guang Li, Wanli Ouyang, Wenbo Su, and Bo Zheng.         Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
  2024. GraphReader: Building graph-based agent to          William Cohen, Ruslan Salakhutdinov, and Christo-
  enhance long-context abilities of large language mod-     pher D Manning. 2018. Hotpotqa: A dataset for
  els. In Findings of the Association for Computational     diverse, explainable multi-hop question answering.
  Linguistics: EMNLP 2024, pages 12758–12786, Mi-           In Proceedings of the 2018 Conference on Empiri-
  ami, Florida, USA. Association for Computational          cal Methods in Natural Language Processing, pages
  Linguistics.                                              2369–2380.

Belinda Mo, Kyssen Yu, Joshua Kazdan, Proud Mpala,        Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. Kg-
  Lisa Yu, Charilaos I. Kanatsoulis, and Sanmi Koyejo.       bert: Bert for knowledge graph completion. arXiv
  2025. KGGen: Extracting knowledge graphs from              preprint arXiv:1909.03193.
  plain text with language models. In The Thirty-ninth    Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou,
  Annual Conference on Neural Information Process-          and Jun Zhao. 2014. Relation classification via con-
  ing Systems.                                              volutional deep neural network. In Proceedings of
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-          COLING 2014, the 25th international conference on
  apu Wang, and Xindong Wu. 2024. Unifying large            computational linguistics: technical papers, pages
  language models and knowledge graphs: A roadmap.          2335–2344.
  IEEE Transactions on Knowledge and Data Engi-           Ranran Haoran Zhang, Qianying Liu, Aysa Xuemo
  neering, 36(7):3580–3599.                                 Fan, Heng Ji, Daojian Zeng, Fei Cheng, Daisuke
                                                            Kawahara, and Sadao Kurohashi. 2020. Minimize
Pranoy Panda, Ankush Agarwal, Chaitanya Devagup-
                                                            exposure bias of seq2seq models in joint entity and
  tapu, Manohar Kaul, and Prathosh Ap. 2024.
                                                            relation extraction. In Findings of the Association
  Holmes: Hyper-relational knowledge graphs for
                                                            for Computational Linguistics: EMNLP 2020, pages
  multi-hop question answering using llms. In Pro-
                                                            236–246.
  ceedings of the 62nd Annual Meeting of the Associa-
  tion for Computational Linguistics (Volume 1: Long      Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Ji-
  Papers), pages 13263–13282.                               aming Xu, Shiyao Li, Yuming Lou, Luning Wang,
                                                            Zhihang Yuan, Xiuhong Li, and 1 others. 2024. A
Fina Polat, Ilaria Tiddi, and Paul Groth. 2025. Testing     survey on efficient inference for large language mod-
  prompt engineering methods for knowledge extrac-          els. arXiv preprint arXiv:2404.14294.
  tion from text. Semantic Web, 16(2):SW–243719.

Gabriel Stanovsky, Julian Michael, Luke Zettlemoyer,
  and Ido Dagan. 2018. Supervised open information
  extraction. In Proceedings of the 2018 Conference
  of the North American Chapter of the Association
  for Computational Linguistics: Human Language
  Technologies, Volume 1 (Long Papers), pages 885–
  895.

Yuan Sui, Yufei He, Zifeng Ding, and Bryan Hooi. 2025.
  Can knowledge graphs make large language models
  more trustworthy? an empirical study over open-
  ended question answering. In Proceedings of the
  63rd Annual Meeting of the Association for Compu-
  tational Linguistics (Volume 1: Long Papers), pages
  12685–12701, Vienna, Austria. Association for Com-
  putational Linguistics.
                                                      8314
A        Appendix                                          Prompt 1. Candidate Triplet Extraction
                                                           You are an algorithm designed to extract structured knowl-
A.1 Reproducibility                                        edge from texts to build a Wikidata-like knowledge graph.
                                                           A knowledge graph consists of triplets in the format
                                                           (subject, relation, object), where:
We release6 : (1) prompts and source code for each
                                                            • Subject: A named entity or concept describing a group
pipeline stage; (2) scripts to build Wikidata-derived         of people, events, or abstract objects that serves as the
ontology used for ontology-aware stages; (3) the              source of the relation.
KG-only multi-hop QA component.                            • Relation: A Wikidata-style predicate connecting the
                                                             subject and the object.
   We reported metrics averaged across multiple
runs (Tables 4, 6) with standard deviation. For             • Object: A named entity or concept describing a group of
                                                              people, events, or abstract objects related to the subject.
Table 3, standard deviations are estimated via boot-       Additionally, some triplets may have qualifiers that
strap resampling. The exact prompts used at each           provide more context (e.g., date, place, or other attributes).
stage are provided in the Appendix A.4, A.5.               Qualifiers should have relations and objects like triplets
                                                           do, but instead of a subject, their relation connects an
                                                           object and the triplet they qualify. Qualifiers must always
                                                           be attached to a triplet and never exist as standalone triplets.
A.2 Dataset Statistics
                                                           You will receive a text labeled “Text:”. Your task is to ex-
The test splits of HotpotQA and MuSiQue consist            tract meaningful triplets that represent factual relationships.
of 1000 samples each. We provide the QA evalu-
                                                           Output Format. Return only triplets in JSON format as
ation results for Wikontic and HippoRAG for the            a list of dictionaries:
whole evaluation set, and for AriGraph, we report           • "subject": Subject entity.
the openly available results for 200 test samples. To      • "relation": Relation connecting subject and object.
compare the statistics of KGs, we use 80 MuSiQue           • "object": Object entity.
samples that are commonly available for all com-
                                                           • "qualifiers": List of dictionaries, each with:
pared methods.
                                                              – "relation": Relation connecting triplet and object.
                                                              – "object": Object entity connected to the main
A.3 Computational resources                                     triplet.
                                                           • "subject_type": Class that describes the subject.
All experiments on knowledge graph (KG) con-
                                                           • "object_type": Class that describes the object.
struction and question answering (QA) were con-
ducted using the OpenAI and OpenRouter APIs.
Across all datasets, models, and ablation studies,
KG construction and QA (each QA experiment re-
peated three times to compute mean and standard
deviation) required a total cost of approximately
$500.


A.4 Prompts for triplet extraction

Here we provide excerpts of prompts that were
used in our KG construction pipeline. Prompt 1
was used for candidate triplet extraction. Subse-
quently, Prompt 2 was used to refine entity types
for both subject and object entities. Prompt 3 was
used for choosing relations among those that can
legally connect chosen entity types by Wikidata
constraints. Finally, Prompt 6 was used to refine
surface forms of subject and object. All prompts,
instructions, and in-context examples are available
with the code.

    6
        https://github.com/screemix/Wikontic

                                                    8315
Prompt 2. Triplet Backbone Refinement -                                 Prompt 3. Triplet Backbone Refinement -
choosing Relevant Entity Types                                          choosing Relevant Relation
You are given a factual triplet extracted from text. The                You are given a factual triplet extracted from text. The
triplet follows the format (subject, relation, object), where:          triplet follows the format (subject, relation, object), where:

    • Subject: A named entity or concept that represents a
      person, group, event, or abstract entity serving as the               • Subject: A named entity or concept that represents a
      source of the relation.                                                 person, group, event, or abstract entity serving as the
                                                                              source of the relation.
    • Relation: A Wikidata-style predicate that defines the
      connection between the subject and the object.                        • Relation: A Wikidata-style predicate that defines the
                                                                              connection between the subject and the object.
    • Object: A named entity or concept that represents a
      person, group, event, or abstract entity related to the               • Object: A named entity or concept that represents a
      subject.                                                                person, group, event, or abstract entity related to the
                                                                              subject.
    • Subject type: a class that describes the object.
                                                                            • Subject type: a class that describes the object.
    • Object type: a class that describes the subject.
                                                                            • Object type: a class that describes the subject.
The extracted entity types of both subject and object were
mapped to a set of similar Wikidata-style entity types                  The extracted relation has been mapped to a set of similar
based on semantic similarity.                                           Wikidata-style relations based on semantic similarity and
                                                                        the entity types they can connect.
Your Task:
                                                                        Your Task:
You will be provided with the following:
                                                                        You will be provided with the following:
    • Text: The original sentence or passage from which
      the triplet was extracted.                                            • Text: The original sentence or passage from which
                                                                              the triplet was extracted.
    • Extracted Triplet: The factual triplet derived from
      the text.                                                             • Extracted Triplet: The factual triplet derived from
                                                                              the text.
    • Candidate subject types: similar entity types for sub-
      ject type of extracted triplet retrieved from Wikidata.               • Candidate relations: list of relation (or in other words
                                                                              property) names similar to the extracted relation from
    • Candidate object types: similar entity types for object                 triplet retrieved from Wikidata.
      type of extracted triplet retrieved from Wikidata.
                                                                            • Candidate relations: list of relation (or in other words
Select the most appropriate candidate entity types for both                   property) names similar to the extracted relation from
subject and object from the provided candidates that best                     triplet retrieved from Wikidata.
match the meaning of previously extracted triplet and
original text.                                                          Select the most appropriate relation candidate from the
                                                                        provided candidate triplets that best match the meaning of
Provide ONLY an answer in JSON format with the follow-                  previously extracted triplet and original text.
ing keys:
                                                                        Provide only an answer in JSON format with the following
    • "subject_type": Selected subject type candidate.                  keys:
    • "object_type": Selected object type candidate.                        • "relation": Relation for the selected triplet.




                                                                 8316
 Prompt 4. Entity Names Refinement                                    Propmt 5. Entity extraction for question an-
 In the previous step, there was extracted a triplet akin to          swering
 one in Wikidata knowledge graph from the text. Triplet               Extract wikidata-like entities from the question below. It is
 contains two entities (subject and object) and one relation          guaranteed that there is at least one mentioned entity.
 that connects these subject and object. Using semantic
 similarity, we linked subject name with top similar exact            Extract any entity, whether name entity or an abstract
 names from the knowledge graph built from previously                 entity, that might help retrieve the information to answer
 seen texts.                                                          the question.
 You will be provided with the following:                             Provide output in json format, no additional symbols.
                                                                      Output should be represented as a LIST of extracted
                                                                      entities’ names.
    • Text: The original sentence or passage from which
      the triplet was extracted.

    • Extracted Triplet: A structured representation in the
      format "subject": "...", "relation": "...", "object":
      "..." .

    • Original Subject: A subject name that needs refine-
      ment.

    • Candidate Subjects: A list of possible entity names
      from previously seen texts.

 Your Task:

 Select the most contextually appropriate subject name from
 the Candidate Subjects list that best matches subject from
 extracted triplet and context of the given Text.

    • If an exact or semantically appropriate match is
      found, return the corresponding name exactly as it
      appears in the list.

    • If no suitable match exists, return the string "None".
                                                                      Prompt 6. Entity linking for question answer-
                                                                      ing
    • Do not modify name from the candidate list in case
      of match, add explanations, or provide any additional           Task: Identify relevant entities from a pre-constructed
      text.                                                           knowledge graph that might help to answer a provided
                                                                      question.

                                                                      Input Structure:

                                                                          • The question will be labeled as "Question:".

                                                                          • A list of entities from the knowledge graph will be
                                                                            labeled as "Entities:".
A.5 Prompts for question answering
                                                                      Selection Criteria:

                                                                          • Relevance means an entity is directly or indirectly
                                                                            useful for answering the question. Look for names,
                                                                            events, dates, and other related concepts or entities
Here, we provide excerpts of prompts that                                   that match or connect to key concepts in the question.
were used for KG grounded question answering.
                                                                          • Do not ignore possible indirect relevance (e.g., if the
Prompt 5 was used to extract entities relevant to                           question asks about a competition, teams or winners
the question. Then, with Prompt 6 the LLM was in-                           of that competition may be useful).
structed to choose relevant entities among entities                   Response Format:
similar to the extracted ones in the KG. Prompt 7
was used to decompose the question on a single-                           • Always return at least one relevant entity. It is guar-
                                                                            anteed that there is at least one.
hop subquestion conditioned on extracted entities
or previously answered subquestions. Prompt 8                             • The output must be a JSON list of dictionaries, where
                                                                            each dictionary contains a key "entity": the name of
was used to check if a question is answered by a se-                        the chosen relevant entity
quence of subquestions and corresponding answers.
All prompts, instructions, and in-context examples                        • Do not return an empty list. Select the best possible
                                                                            options.
are available with the code.
                                                               8317
Prompt 7. Question decomposition                                      Prompt 8. Check if a question is answered
You are an assistant for stepwise question decomposition.             You are a reasoning assistant for multi-hop question
                                                                      answering.
You will be given three inputs:
                                                                      Your task: Decide whether a list of subquestions and
    • An original multi-hop question.                                 their answers fully resolves the original multi-hop question.

    • A 1-hop sub-question that has already been an-                  Input format:
      swered.
                                                                          • Original multi-hop question: <text>
    • The answer to that 1-hop sub-question.
                                                                          • Question->answer sequence: [a list of subquestions
Your task:                                                                  and their answers, ending with the most recent one]
   Reformulate the original multi-hop question by
integrating obtained answer from sub-question, so the new             Output rules:
question has (n-1) hops.
                                                                          • If the sequence of subquestions and answers com-
Rules:                                                                      pletely and directly resolves the original multi-hop
                                                                            question, output only the final answer to the original
    • Only perform one reasoning hop at a time. Do not                      multi-hop question (not just the last subanswer, i.e.
      generate additional reasoning steps beyond this hop.                  answer the original question).

    • Do not include explanations or text, just reformulated              • If the sequence is not sufficient and more reasoning
      question.                                                             or hops are needed, output exactly: NOT FINAL

                                                                      Do not include any prefixes like "Final answer:",
                                                                      "Answer:", suffixes, formatting, original questions or
                                                                      explanations.

                                                                      Output must be a single line: either string with the final
                                                                      answer to the original multi-hop question or the exact
                                                                      string NOT FINAL.

                                                                      <example>

                                                                        Original multi-hop question: Who was the spouse of the
                                                                      person who wrote The Iron Heel?

                                                                        Question->answer sequence:

                                                                           Who wrote The Iron Heel? → Jack London

                                                                          Who was the spouse of Jack London? → Charmian
                                                                      London

                                                                        Expected output:

                                                                           Charmian London
                                                                      </example>

                                                                      <example>

                                                                         Original multi-hop question: Which country’s capital is
                                                                      closest to the birthplace of Nikola Tesla?

                                                                        Question->answer sequence:

                                                                           Where was Nikola Tesla born? → Smiljan, Croatia

                                                                      Expected output:

                                                                           NOT FINAL
                                                                      </example>

                                                                  A.6       Triplet extraction pipeline examples




                                                               8318
                                                            LLM triplet extraction

                                                  [                                                                                                                                                   Subject and object
                                                      {                                                                                                                                               names refinement
                                                           "subject": "Nolan",                                                                                                       [
                                                           "relation": "directed",                                            Triplet’s backbone refinement                              {
                                                           "object": "Inception",                                                based on ontology rules                                  'subject': 'Inception',
                                                           "qualifiers": [                                                                                                                    'relation': director,
                                                              {                                                               [                                                               'object': 'Christopher Nolan',
                                                                "relation": "point in time",                                    {                                                             'qualifiers': [{'relation': 'point in time',
               Input text                                       "object": "2010"                                                   "subject_type": "film",                                                    'object': '2010'}],
                                                              }                                                                    "relation": "director",                                    'subject_type': 'film',
     In 2010, Nolan directed the                           ],                                                                      "object_type": "human"                                     'object_type': 'human'
        science fiction movie                              "subject_type": "human",                                             },                                                       },
                                                           "object_type": "film"                                                {
              Inception                                                                                                            "relation": "genre",
                                                      },                                                                                                                                       {
                                                      {                                                                            "subject_type": "film",                                            'subject': 'Inception',
                                                           "subject": "Inception",                                                 "object_type": "film genre"                                        'relation': 'genre',
                                                           "relation": "genre",                                                 }                                                                     'object': 'science fiction',
                                                           "object": "science fiction",                                       ]                                                                       'qualifiers': [],
                                                           "qualifiers": [],                                                                                                                          'subject_type': 'film',
                                                           "subject_type": "film",                                                                                                                    'object_type': 'film genre'
                                                           "object_type": "film genre"                                                                                                         }
                                                      }                                                                                                                              ]
                                                  ]




Figure 4: Overview of the multi-stage pipeline for KG extraction from unstructured text. The process consists
of (1) LLM-based triplet extraction, (2) ontology-based validation of triplet structure, and (3) entity linking and
normalization.




                                                                                                      Triplet’s backbone refinement based on ontology rules
                         Input triplet
 {                                                                Finding top-K similar entity types
        "subject": "Nolan",                                                                                        Filtering candidate relations by allowed object
        "relation": "directed",                            "human": [                                                             and subject types                                                                           LLM choosing among potential valid
        "object": "Inception",                                  "human", "human biblical figure",                                                                      Ranking filtered candidate relations                              backbones
        "qualifiers": [                                         "hypothetical person",                             "director": [                                             by semantic similarity
           {                                                    "group of humans","fictional human"                                                                                                                       {
             "relation": "point in time",                  ]                                                              "assistant director",                        1. "director"
             "object": "2010"                                                                                                                                          2. "director / manager"                                   "subject_type": "film",
           }                                                                                                              "director",                                  3. "assistant director"                                   "relation": "director",
        ],                                                 "film": [                                                      "director / manager",                        4. "chief executive officer"                              "object_type": "human"
       "subject_type": "human"                                    "photographic film", "film", "part of a work",          "chief executive officer"                                                                       }
       "object_type": "film"                                      "film award", "dubbing of film'"                 ]
      }                                                    ]




Figure 5: Ontology-based triplet refinement process. For each extracted triplet, we retrieve and extend candidate
entity types using Wikidata’s type hierarchy, identify valid relations allowed to use between extracted entities based
on ontology constraints, and re-rank relation candidates using semantic similarity. The final triplet configuration is
selected by an LLM.




                                                                                                            Subject and object names refinement


             Input triplet with valid structure                        Filtering existing candidate subject & object
 {                                                                                         entities
                                                                                                                                         Ranking filtered candidate entities by semantic               LLM choosing relevant names for subject and
      "subject": "Inception",                                        "film": [                                                                                                                                object entity if there is one
      "relation": "director",                                                                                                                                 similarity
      "object": "Nolan",                                                    "Mission Impossible",
                                                                            "Jurassic Park",                                              "film": [                                                   {
      "qualifiers": [                                                       "John Wick",                                                         "Initiation (2020)", "I Origins",                            'subject': 'Inception',
         {                                                                  ...                                                                  "Mission Impossible", "Jurassic Park",                           'relation': director,
           "relation": "point in time",                              ]                                                                    ]                                                                       'object': 'Christopher Nolan',
           "object": "2010"                                                                                                                                                                                       'qualifiers': [{'relation': 'point in time',
         }                                                           "human": [                                                            "human": [                                                                             'object': '2010'}],
      ],                                                                  "David Baker",                                                        "'Christopher Nolan'", "Jonathan Nolan ",                         'subject_type': 'film',
      "subject_type": "film"                                              "Robert Pattinson",                                                   "Martin McDonagh", "Robert Pattinson"                             'object_type': 'human'
 }                                                                        "Martin McDonagh",                                               ]                                                              }
           'object_type': "human"                                         ...
     },                                                              ]



Figure 6: Entity refinement step for KG construction. For each refined triplet, candidate subject and object entities
are retrieved from the existing KG based on their type and semantic similarity. An LLM determines whether the
extracted entity matches an existing one or should be preserved as a new entry. This process reduces redundancy
and supports incremental KG updates.




                                                                                                                                   8319
