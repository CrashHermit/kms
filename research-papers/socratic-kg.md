                                                                   SocraticKG: Knowledge Graph Construction
                                                                          via QA-Driven Fact Extraction
                                                  Sanghyeok Choi1 *           Woosang Jeon1,2 *          Kyuseok Yang1         Taehyeong Kim1,2,3,4†
                                                             1
                                                               Department of Biosystems Engineering, Seoul National University
                                                                   2
                                                                     Artificial Intelligence Institute, Seoul National University
                                                      3
                                                        Interdisciplinary Program in Artificial Intelligence, Seoul National University
                                                         4
                                                           Interdisciplinary Program in Cognitive Science, Seoul National University
                                                        {cholsang83, jwoosang1, kyuseok0603, taehyeong.kim}@snu.ac.kr

                                                                   Abstract                              However, the reliance on manual curation has his-
                                                                                                         torically limited the availability of domain-specific
                                                 Constructing Knowledge Graphs (KGs) from




arXiv:2601.10003v2 [cs.CL] 23 Apr 2026
                                                 unstructured text provides a structured frame-          KGs, thereby motivating growing interest in auto-
                                                 work for knowledge representation and reason-           mated construction methods that can scale to di-
                                                 ing, yet current LLM-based approaches strug-            verse and large-scale text sources (Ren et al., 2024).
                                                 gle with a fundamental trade-off: factual cov-             Recent advances in LLMs have enabled more
                                                 erage often leads to relational fragmentation,          semantically grounded approaches to knowledge
                                                 while premature consolidation causes informa-           graph construction, moving beyond rule-based pat-
                                                 tion loss. To address this, we propose Socrat-          tern matching toward methods that leverage neural
                                                 icKG, an automated KG construction method
                                                                                                         reasoning to interpret unstructured text (Zhu et al.,
                                                 that introduces question-answer pairs as a struc-
                                                 tured intermediate representation to systemati-         2024). Current approaches address the construction
                                                 cally unfold document-level semantics prior to          challenge through different strategies. Some meth-
                                                 triple extraction. By employing 5W1H-guided             ods focus on capturing explicit factual mentions
                                                 QA expansion, SocraticKG captures contex-               in a single pass, extracting triples directly from
                                                 tual dependencies and implicit relational links         text (Cabot and Navigli, 2021; Shang et al., 2022;
                                                 typically lost in direct KG extraction pipelines,       Zhang and Soh, 2024). Others adopt consolidation-
                                                 providing explicit grounding in the source doc-
                                                                                                         centric strategies, organizing extracted facts around
                                                 ument that helps mitigate implicit reasoning
                                                 errors. Evaluation on the MINE benchmark                pre-identified entity structures to improve graph co-
                                                 and HotpotQA downstream task demonstrates               herence (Zhong and Chen, 2021; Ye et al., 2022a;
                                                 that our approach effectively addresses the             Wei et al., 2023; Mo et al., 2025).
                                                 coverage-connectivity trade-off, achieving su-             However, these approaches face a persistent chal-
                                                 perior factual retention and structural cohesion        lenge: fully externalizing the narrative logic of
                                                 while supporting complex multi-hop reasoning.           source documents into structured graphs. The re-
                                                                                                         sulting knowledge graphs often struggle with a
                                         1       Introduction
                                                                                                         fundamental tension between factual coverage and
                                         As large language models (LLMs) are widely used                 structural coherence. Graphs may contain many
                                         in knowledge-intensive applications, concerns sur-              facts but remain fragmented with weak semantic
                                         rounding factual reliability, interpretability, and             connectivity, or they may be well-organized yet
                                         grounding have become more pronounced (Ji et al.,               incomplete, having filtered out contextual nuances
                                         2023; Huang et al., 2025). While Retrieval-                     that do not conform to predefined structures. At
                                         Augmented Generation (RAG) addresses these con-                 the core of this challenge lies the difficulty of bal-
                                         cerns by anchoring models to external sources, it               ancing comprehensive information extraction with
                                         often struggles with fragmented contexts and shal-              meaningful connectivity across the graph.
                                         low integration of complex facts (Lewis et al., 2020;              To address this limitation, we draw inspiration
                                         Gao et al., 2023). In response, Knowledge Graphs                from how humans naturally process and organize
                                         (KGs) have re-emerged as a complementary solu-                  information from text. Rather than attempting to
                                         tion, providing a structured and verifiable backbone            extract structured knowledge in a single step, hu-
                                         for explicit knowledge representation and reason-               man comprehension is fundamentally interrogative:
                                         ing (Pan et al., 2023; Rajabi and Etminani, 2024).              readers construct understanding by progressively
                                             *    Equal contribution                                     clarifying salient concepts through active inquiry
                                             †
                                                  Corresponding author                                   (Graesser and Person, 1994; Ambrose et al., 2010).

                                                                                                     1
Figure 1: The overall architecture of the SocraticKG framework. Given unstructured text, the method first
generates atomic QA pairs through 5W1H-guided questioning, then extracts triples from these QA pairs, and finally
canonicalizes the triples to produce a cohesive knowledge graph.


This process of interrogative learning serves as a          connected and less fragmented graphs, yielding
natural scaffold for organizing complex informa-            further gains on complex multi-hop reasoning.1
tion. Question-Answering (QA), in particular, facil-           In summary, we make the following contribu-
itates focused attention and explicit articulation of       tions in this work:
relationships that might otherwise remain implicit
                                                                • We propose SocraticKG, a QA-mediated
in direct extraction (Wu et al., 2020).
                                                                  method for knowledge graph construction that
   Building on this insight, we propose SocraticKG                formalizes question-answering as a semantic
(SoKG), a method that treats QA not merely as                     scaffold for unfolding document narratives
a retrieval mechanism, but as a structured inter-                 and explicitly articulating implicit connec-
mediate representation that systematically unfolds                tions prior to structural extraction.
document-level semantics prior to graph construc-
tion (FitzGerald et al., 2018; Cohen et al., 2023).             • We introduce 5W1H-guided QA expansion
SoKG employs a structured interrogative frame-                    as a systematic approach for surfacing latent
work based on the 5W1H framework (who, what,                      dependencies typically overlooked in direct
when, where, why, and how) to generate document-                  extraction, thereby improving factual cover-
grounded QA pairs that capture key concepts, re-                  age while reducing implicit reasoning errors.
lationships, and contextual dependencies. This
                                                                • We demonstrate that our approach mitigates
QA-mediated expansion articulates implicit con-
                                                                  structural fragmentation and information loss,
nections and contextual nuances in explicit natural
                                                                  achieving superior factual retention and recov-
language format. The resulting intermediate repre-
                                                                  erability across various LLMs.
sentation facilitates more consistent and complete
triple extraction by providing well-defined seman-          2       Related Work
tic units rather than requiring simultaneous reso-
lution of semantics and structure. These extracted          2.1      Direct Triple Extraction
triples are then unified through a canonicalization         Knowledge Graph (KG) construction has evolved
process (Mo et al., 2025) that resolves surface-form        from conventional Open Information Extraction
variations and consolidates the graph into a coher-         (OpenIE) (Etzioni et al., 2008; Fader et al., 2011)
ent structure.                                              to modern approaches that extract triples directly
   We evaluate our proposed method on the MINE              via LLMs (Cabot and Navigli, 2021; Bi et al.,
(Measure of Information in Nodes and Edges)                 2024; Zhang and Soh, 2024). While OpenIE is
benchmark (Mo et al., 2025) and HotpotQA (Yang              constrained by surface linguistic patterns (Niklaus
et al., 2018) as a downstream multi-hop reasoning           et al., 2018), such direct extraction methods lever-
task. Our results demonstrate that SocraticKG con-          age LLM reasoning capabilities to bridge semantic
sistently outperforms state-of-the-art counterparts         gaps without explicit intermediate representations.
across multiple LLM backbones, achieving supe-                  1
                                                                Our code is publicly available at https://github.com/
rior factual retention while producing more densely         LABA-SNU/SocraticKG.


                                                        2
   However, this direct extraction approach often            syntactic normalization rather than document-level
limits the model to capturing surface-level, explicit        semantic organization. While effective for resolv-
mentions while overlooking the latent logical ties           ing surface-level ambiguities within individual sen-
that bind them. As noted by Zhu et al. (2024);               tences, they do not systematically capture cross-
Meher et al. (2025), this approach often yields shal-        sentence dependencies or contextual relationships
low factual coverage, often producing fragmented             that span the document. This limits their abil-
subgraphs that lack the connectivity required for            ity to externalize the broader narrative structure
effective graph-based reasoning.                             and global semantics required for comprehensive
                                                             knowledge graph construction.
2.2   Consolidation-Centric Strategies
To address the fragmentation issues, various                 2.4    QA for Knowledge Extraction
pipelines emphasize structural coherence through
post-extraction consolidation. These entity-first            Question-answering has been widely used to elicit
approaches organize extracted facts by first identi-         structured information from text (Levy et al., 2017;
fying key entities, then structuring relations around        Li et al., 2019; Du and Cardie, 2020), by leverag-
this pre-established entity framework. GraphRAG              ing the cognitive process of interrogative inquiry,
(Edge et al., 2024) builds a global index of en-             which facilitates the construction of situation mod-
tities and relationships partitioned into hierarchi-         els (Chi et al., 1989; Graesser and Person, 1994).
cal communities for query-focused summarization,             Recent extraction methods, such as StoryNet (Na-
whereas KGGen (Mo et al., 2025) emphasizes                   gireddy, 2021) and ChatIE (Wei et al., 2023), incor-
clustering-based canonicalization of entities and            porate QA-driven prompting as a core component
relations to produce compact and reusable knowl-             of their extraction pipelines.
edge graphs. Similarly, CLARE (Henry and Gong,                  However, these approaches treat QA pairs as
2025) anchors its relational extraction on initial           transient artifacts, generating and consuming them
entity identification to ensure semantic precision           within a single extraction pass, without formalizing
within consolidated text.                                    them as an intermediate representation for organiz-
    Despite their effectiveness in organizing triples,       ing document-level semantics. As a result, they
these consolidation-focused strategies can act as            lack systematic question generation and struggle
a representational bottleneck (Ye et al., 2022b).            to surface implicit relational and contextual depen-
When entity sets are fixed early in the pipeline,            dencies prior to triple extraction.
relations or contextual dependencies that do not                While recent work has explored QA as an inter-
conform to the initial entity structure may be ex-           mediate step for interpretable knowledge construc-
cluded. This sequencing effectively prioritizes              tion (Aneja et al., 2025), it primarily emphasizes
structural utility over factual density, potentially         retrieval utility through factual restatement, rather
under-representing the document’s latent relations.          than semantic organization. Collectively, these
                                                             gaps suggest that formalizing QA as a structured
2.3   Transform-Then-Extract Approaches                      intermediate representation provides a more robust
To reduce extraction complexity, various ap-                 foundation for construction, particularly when sys-
proaches employ a two-stage process: first trans-            tematic inquiry is used to proactively externalize
forming raw text through intermediate representa-            latent relational and causal dependencies.
tions, then extracting triples from the transformed
output. Common transformation strategies include             3     Methods
coreference resolution to handle referential expres-
sions (Manning et al., 2014; Cetto et al., 2018) and         SoKG introduces QA pairs as a structured inter-
syntactic sentence decomposition to simplify com-            mediate representation for LLM-based KG con-
plex structures (Niklaus et al., 2019; Niklaus, 2022).       struction. Rather than prompting LLMs to extract
CoDe-KG (Anuyah et al., 2025), for instance, lever-          triples directly from raw text, our approach first de-
ages human-guided prompt intervention to incorpo-            composes the document into explicit QA pairs that
rate these transformation tasks, ensuring structural         resolve contextual dependencies and referential am-
clarity prior to extraction.                                 biguities in natural language. These QA pairs are
   These transformation-based approaches operate             then mapped to atomic triples and unified through
primarily at the sentence level, focusing on local           canonicalization to produce the final KG.

                                                         3
3.1   5W1H-Guided QA Generation                                3.3    Graph Construction from Triples
This stage transforms document into a collection of
                                                               The final stage unifies discrete triples into a cohe-
discrete, self-contained QA pairs. To ensure com-
                                                               sive graph structure. Since extraction occurs across
prehensive coverage of the factual content in the
                                                               independent QA units, the raw set often contains re-
text, we design a prompt strategy based on two core
                                                               dundant or synonymous mentions for the same con-
principles: systematic questioning and contextual
                                                               cept. To resolve these redundancies, we adopt the
independence (detailed prompt in Appendix B.1).
                                                               canonicalization procedure from Mo et al. (2025),
Detailed Questioning via 5W1H We leverage                      which combines embedding-based clustering with
the 5W1H framework to guide systematic ques-                   LLM-based refinement.
tion formulation. The LLM generates multiple                      The canonicalization process is performed in-
questions spanning all six categories and diverse              dependently on entities and relations through a
aspects of the document. As a result, the resulting            cluster-then-refine process. First, semantic embed-
QA pairs capture both surface-level entities and               dings are generated for all unique entities and rela-
complex dependencies, including causal rationales              tions using a text embedding model. To narrow the
(why) and procedural details (how).                            search space, these embeddings are partitioned into
Contextual Independence To ensure each QA                      clusters of a manageable size for entities and rela-
pair functions as a standalone unit, we instruct the           tions respectively via K-means clustering. Within
LLM to generate answers that are fully understand-             each cluster, the top-k potential matches for each
able without referencing the original source text.             anchor are identified by balancing dense semantic
Specifically, the model is required to replace pro-            similarity with sparse lexical overlap (BM25). Fi-
nouns (e.g., it, they) with their explicit entity names,       nally, synonyms and abbreviations are resolved by
resolving referential ambiguities. This constraint             an LLM, which maps these variants to a single rep-
prevents information loss when each QA pair is pro-            resentative form to consolidate fragmented triples
cessed individually in the triple extraction phase.            into a cohesive, canonicalized graph.

3.2   Triple Extraction from QA
                                                               4     Experiments
This stage transforms the QA pairs into structured
triples by treating each pair as an independent ex-
                                                               We evaluate SoKG from two complementary per-
traction unit. Operating on these logically self-
                                                               spectives: source-information preservation and
contained units allows the extraction process to fo-
                                                               downstream reasoning utility. For the former, we
cus on well-defined semantic boundaries, reducing
                                                               utilize the MINE benchmark (Mo et al., 2025), de-
errors common in direct extraction from long, com-
                                                               signed to quantify the information gap between
plex texts. To achieve this, the LLM is instructed to
                                                               raw text and its graph representation by measuring
follow three specific constraints (detailed prompt
                                                               how much source information is recoverable. The
in Appendix C.1).
                                                               benchmark comprises 100 diverse articles, each
Atomic Decomposition The model decomposes                      paired with 15 verified atomic facts, providing a
each QA pair into separate, atomic triples, captur-            rigorous evaluation framework across 1,500 inde-
ing fine-grained facts from both the inquiry and the           pendent factual instances. Following the bench-
response to maximize factual richness.                         mark protocol, we constructed one KG per article
                                                               and evaluated each graph in terms of factual reten-
Entity Clarity All entities are expressed as spe-              tion and structural characteristics.
cific noun phrases, and any triple containing am-
                                                                  For downstream reasoning utility, we exam-
biguous pronouns is discarded. This ensures that
                                                               ine whether SoKG’s structural advantages trans-
every extracted fact is self-contained and grounded
                                                               late into practical gains beyond source-information
in clear evidence.
                                                               preservation. To this end, we conducted experi-
Simplified Relations Predicates are distilled into             ments on HotpotQA (Yang et al., 2018) using 800
concise verb phrases to reduce surface-form varia-             “Hard” Bridge samples. These samples require
tions, facilitating subsequent canonicalization. The           identifying an intermediate entity that connects dis-
model is instructed to skip extraction if the relatio          parate pieces of evidence, making them well suited
nship remains ambiguous.                                       for evaluating graph-based multi-hop reasoning.

                                                           4
Figure 2: Comparison of extraction pipelines using an example output from Gemini-2.5-flash-lite. While baseline
pipelines often miss the syntactic connection in complex sentences, failing to recover the causal link between bees
and genetic diversity, SoKG leverages QA-driven reasoning to explicitly reconstruct the intermediate concept. As
a result, SoKG successfully recovers the complete causal chain (bees → cross-pollination → genetic diversity),
whereas baselines tend to simplify or fragment this relationship.


4.1   Evaluation Metrics                                         pute the average degree as
Factual Retention Score As the primary metric,
                                                                                           2E
we measured the proportion of ground-truth facts                                   Deg =      ,
                                                                                           N
successfully recovered from the constructed KGs.
Following the MINE benchmark protocol, we re-                   where N denotes the number of nodes and E
trieved a local subgraph for each fact, consisting              the number of edges.
of the top-8 nodes most semantically similar to
the target statement and their 2-hop neighbors. An            • Triple Count (#Tri): The total number of
LLM-judge then determined whether the fact was                  atomic facts externalized in the graph.
logically supported by the retrieved subgraph con-
text. The score represents the percentage of verifi-          • Normalized Fragmentation Index (NFI):
able facts, reflecting how well the graph preserves             Motivated by the notion of graph fragmen-
information from the source text for downstream                 tation as the decomposition of a network into
tasks such as retrieval and reasoning.                          disconnected components (Borgatti, 2002),
                                                                we define a component-based metric as:
Structural Cohesion and Density To analyze
the organization and coherence of the KGs, we                                            C −1
                                                                                 NFI =        ,
investigated:                                                                            N −1

   • Average Degree (Deg): The average number                   where C denotes the number of connected
     of unique neighboring nodes per node, cap-                 components and N is the total number of
     turing the local connectivity density of the               nodes (N ≥ 2). This formulation normal-
     graph (Barabási, 2013). It reflects how many               izes fragmentation to the unit interval [0, 1],
     distinct entities a node is connected to, irre-            where 0 corresponds to a fully connected
     spective of relation direction. Following stan-            graph (C = 1) and 1 indicates a completely
     dard practice for undirected graphs, we com-               fragmented network (C = N ).

                                                        5
Method                     Qwen-2.5          GPT-4o-mini              GPT-4o         Gemini-2.5          Claude-4
Direct Extraction            66.5                 68.5                 78.1             84.6               86.8
GraphRAG                     59.7                 49.5                 49.3             48.5               52.3
KGGen                        56.7                 44.3                 66.4             62.5               69.1
SoKG (w/o 5W1H)              67.1                 80.5                 83.5             85.6               94.6
SoKG (Ours)                  73.4                 83.9                 89.3             87.7               96.3

Table 1: Comparison of factual retention scores (%) on the MINE benchmark. SoKG consistently achieves the
highest performance across all evaluated models. The vanilla variant (i.e., SoKG w/o 5W1H) shows how the 5W1H
scaffold captures procedural and causal facts to improve factual consistency even on smaller models like Qwen-2.5.


Multi-hop Reasoning Accuracy For down-                             parative method for evaluating factual reten-
stream evaluation on HotpotQA, we measured an-                     tion and structural cohesion in open-domain.
swer accuracy on the Hard Bridge samples. For
each question, we constructed a KG from the as-                 • SoKG (w/o 5W1H): An ablated variant of
sociated source documents, retrieved a top-1 seed                 SoKG that retains QA pairs as its intermediate
triple via embedding similarity, and expanded its                 representation but replaces the 5W1H-guided
neighborhood to 2-hop and 3-hop depths. An an-                    inquiry with generic QA. This design isolates
swer was marked correct when an LLM answering                     the contribution of the 5W1H-guided scaffold
model, given only the retrieved subgraph context,                 to evaluate its impact.
produced a response matching the gold answer.
                                                                • SoKG (Ours): Our proposed approach utiliz-
4.2   Comparative Analysis Design                                 ing 5W1H-guided QA generation to systemat-
                                                                  ically construct KGs from source documents.
KG Construction Methods We compared
                                                                  Unless otherwise specified, SoKG refers to
SoKG against three representative approaches in
                                                                  this complete implementation.
the current landscape of LLM-based KG con-
struction. To ensure a valid comparison, we se-              Evaluation across LLMs To assess robustness
lected the comparative methods that operated in              across varying LLM architectures and scales, we
autonomous and open-domain settings without pre-             evaluated the selected KG construction methods
defined schemas or human intervention. Figure 2              on five LLMs: GPT-4o, GPT-4o-mini, Gemini-2.5
summarizes the procedures of these methods.                  (Gemini-2.5-Flash-Lite), Qwen-2.5 (Qwen2.5-7B-
   • Direct Extraction: A single-pass extraction             Instruct), and Claude-4 (Claude-4-Sonnet).
     strategy where triples are generated directly           Downstream Baseline Comparison For the Hot-
     from raw text (Appendix C.2). For fair com-             potQA experiments, we include a Naive RAG base-
     parison, we apply the identical canonicaliza-           line that retrieves flat text chunks via embedding
     tion procedure used in KGGen and SoKG to                similarity. Chunk sizes were matched to the aver-
     consolidate the extracted triples. It serves as         age character length of graph-based retrieval con-
     a primary benchmark for the LLM’s implicit              texts produced by SoKG for each backbone, ensur-
     reasoning capability without the benefit of in-         ing both methods operate on comparable context
     termediate semantic scaffolding.                        lengths: 100 characters × Top-3 for Qwen-2.5 and
                                                             500 characters × Top-3 for Claude-4.
   • GraphRAG: A prominent solution across in-
     dustry and academia for global, query-focused           4.3   Implementation Details
     entity indexing. We utilize Microsoft’s offi-
     cial implementation for hierarchical commu-             For all LLMs, we set the decoding temperature to
     nity detection and aggregation, providing a             0 to ensure reproducibility, except for GraphRAG,
     benchmark against the widely adopted text-              which follows the default stochastic configuration
     summary-based method.                                   of its official implementation.
                                                                We adopted the canonicalization and factual re-
   • KGGen: A recent state-of-the-art method fo-             tention evaluation protocol proposed by Mo et al.
     cusing on entity-centric extraction and struc-          (2025). For the semantic clustering mentioned in
     tural consolidation. It serves as a primary com-        Section 3.3, we partitioned entities and relations

                                                         6
                            Qwen-2.5            GPT-4o-mini                   GPT-4o                Gemini-2.5                   Claude-4
 Method
                      N         E      Deg     N        E     Deg         N      E     Deg      N        E     Deg        N           E      Deg
 Direct Extraction   21.7     17.3     1.60   33.5     28.1   1.69    33.9      27.4   1.62    58.4     64.1   2.20      46.4      40.8      1.77
 GraphRAG            19.8     19.0     2.00   11.2     10.2   1.84    11.3      9.70   1.75    15.4     17.7   2.35      14.6      16.2      2.20
 KGGen               28.1     22.1     1.56   19.3     16.7   1.75    33.2      28.9   1.74    38.1     43.2   2.23      57.2      58.9      2.07
 SoKG (w/o 5W1H)     28.0     25.4     1.81   49.2     50.5   2.06    51.9      49.1   1.89    58.0     67.8   2.34       84.2     94.5      2.25
 SoKG (Ours)         34.9     34.1     1.96   57.9     62.2   2.16    62.3      60.5   1.95    65.7     80.8   2.47      104.2     128.4     2.48

Table 2: Topological characteristics averaged over the 100 articles in the MINE benchmark. N, E, and Deg denote
the mean count of Nodes, Edges, and Average Degree per graph, respectively. SoKG consistently expands the
knowledge scale while maintaining high connectivity density across all backbones.

                            Qwen-2.5                 GPT-4o-mini                 GPT-4o                 Gemini-2.5                 Claude-4
Method
                          NFI        #Tri       NFI           #Tri            NFI       #Tri          NFI        #Tri           NFI          #Tri
Direct Extraction      0.162         1,955     0.145          3,100           0.172    2,941          0.038      7,315        0.127         4,417
GraphRAG               0.084         1,981     0.038          1,076           0.083    1,009          0.036      1,848        0.067         1,590
KGGen                  0.187         2,375     0.091          1,942           0.112    3,089          0.030      5,301        0.052         6,391
SoKG (w/o 5W1H)        0.106         2,871     0.059          5,646           0.092    5,345          0.034      7,875        0.056         10,511
SoKG (Ours)            0.078         3,958     0.047          7,069           0.086    6,627          0.023      9,612        0.039         14,849

Table 3: Comparison of graph fragmentation averaged over the 100 articles (NFI; lower is better) and total extracted
information volume summed over the 100 articles (#Tri). The results demonstrate that SoKG effectively resolves
the trade-off between knowledge coverage and structural connectivity, maintaining high graph cohesion even as the
volume of extracted facts increases.


into clusters containing at most 128 elements. For                           Notably, both GraphRAG and KGGen under-
the identification of potential matches, we set the                       perform Direct Extraction in terms of factual re-
candidate retrieval size to k = 16, which defines                         tention. GraphRAG prioritizes hierarchical com-
the number of top-ranked duplicates evaluated by                          munity structures and query-focused summariza-
the LLM. All embedding-based processes used the                           tion over comprehensive fact preservation, result-
all-MiniLM-L6-v2 model, and factual verification                          ing in lower coverage of atomic facts. KGGen’s
was performed via an LLM-as-a-judge protocol                              entity-first bottleneck similarly leads to fact omis-
using GPT-4o.                                                             sion when initial entity identification fails, showing
   For the downstream HotpotQA experiments,                               inconsistent performance across models.
GPT-4.1 served as the answering model, and all-                              This relative advantage of Direct Extraction re-
MiniLM-L6-v2 model was used for KG’s triple                               flects its lack of structural constraints: by avoid-
retrieval and Naive RAG’s chunk retrieval.                                ing early filtering or consolidation, it preserves a
                                                                          larger volume of raw triples. However, as shown
5     Results and Discussion                                              in Tables 2 and 3, this comes at the cost of higher
                                                                          fragmentation, limiting the resulting graph’s utility
5.1    Factual Retention Performance                                      for downstream reasoning.
Table 1 summarizes the factual retention perfor-                             In contrast, SoKG with 5W1H guidance further
mance on the MINE benchmark. Across all com-                              enhances performance by systematically surfacing
pared methods and evaluated LLMs, SoKG consis-                            procedural and causal dimensions. This interroga-
tently achieves the highest scores, peaking at 96.3%                      tive framework ensures that latent dependencies are
with Claude-4.                                                            explicitly captured, maintaining high factual con-
   The comparison between Direct Extraction and                           sistency regardless of the underlying LLM model’s
SoKG (w/o 5W1H) illustrates the benefit of intro-                         inherent reasoning capacity.
ducing QA as an intermediate representation. Even                            To further validate our triple extraction strat-
without 5W1H guidance, SoKG outperforms Di-                               egy, we conducted additional experiments in Ap-
rect Extraction on all LLM models. This advantage                         pendix A. By isolating the impact of the QA scaf-
stems from decomposing documents into discrete,                           fold from the extraction strategy, these studies re-
self-contained QA pairs prior to triples extraction.                      veal that entity-first approaches persist as a per-

                                                                      7
formance bottleneck even when applied to QA-                                       Qwen-2.5          Claude-4
                                                            Method
preprocessed inputs.                                                            2-hop    3-hop    2-hop    3-hop

5.2   Graph Scale and Connectivity                          Direct Ext.         19.50    23.88    37.87    39.88
                                                            GraphRAG            23.00    25.00    46.62    52.12
The superior factual retention shown in Table 1             KGGen               16.50    18.50    38.25    46.75
raises a critical question: is SoKG simply extract-         SoKG (w/o 5W1H)     20.12    24.75    48.00    53.12
ing more triples, or is it building a fundamentally         SoKG                23.62    27.00    48.50    56.38
better graph? To address this, we examine graph             Naive RAG             –      20.13      –      47.88
scale and connectivity in Table 2.
   SoKG significantly expands graph scale while             Table 4: Multi-hop reasoning accuracy (%) on Hot-
                                                            potQA Hard Bridge samples. SoKG consistently per-
maintaining or improving connectivity density
                                                            forms best across backbones and retrieval depths, in-
across all evaluated LLMs. In contrast, Direct              dicating the benefit of connected graph structure for
Extraction produces smaller graphs with lower               multi-hop fact integration.
connectivity, while GraphRAG generates compact
structures that sacrifice comprehensive fact cover-
age for hierarchical organization. The comparison           ness of 5W1H guidance. Adding 5W1H consis-
between SoKG (w/o 5W1H) and SoKG reveals that               tently increases triple extraction volume while re-
5W1H guidance substantially increases the number            ducing or maintaining similar fragmentation levels.
of extracted entities and relations while enhancing         This pattern indicates that 5W1H not only surfaces
connectivity density. This indicates that 5W1H sys-         additional facts but also enhances their integration
tematically surfaces additional facts without frag-         into the graph structure.
menting the graph structure.
                                                            5.4   Downstream Multi-hop Reasoning
   Importantly, SoKG achieves higher connectivity
than both Direct Extraction and KGGen despite               Table 4 reports multi-hop reasoning accuracy on
using the same canonicalization procedure. This             HotpotQA Hard Bridge samples. SoKG achieves
confirms that the structural advantage originates           the best overall performance across both back-
from the QA-mediated intermediate representation,           bones and retrieval depths. Notably, while sev-
enabling relevant evidence to co-locate within 2-           eral graph-based methods such as KGGen and Di-
hop neighborhoods and directly supporting the high          rect Extraction fall below Naive RAG in certain
fact recoverability in Table 1.                             settings, SoKG consistently outperforms it across
   Moreover, this increase in graph scale does              all evaluated conditions, demonstrating that graph-
not appear to reflect merely redundant expansion.           structured retrieval is not inherently advantageous
Supplementary analysis following recent quality-            but becomes so when the graph is sufficiently con-
oriented evaluation perspectives for generative re-         nected and factually complete.
lation extraction (Jiang et al., 2024) confirms that           The largest gain is observed with Claude-4 under
SoKG maintains competitive uniqueness, granular-            3-hop retrieval, where SoKG reaches 56.38%, out-
ity, and factual precision even as the number of ex-        performing Naive RAG by 8.5 percentage points
tracted triples substantially increases (Appendix E).       and Direct Extraction by 16.5 percentage points.
                                                            GraphRAG also performs competitively under
5.3   Factual Volume and Structural Cohesion                Claude-4, likely benefiting from its hierarchical
To further examine the relationship between knowl-          community structure being better leveraged by
edge coverage and graph fragmentation, we ana-              stronger LLMs. On Qwen-2.5, SoKG similarly
lyze triple counts and the NFI in Table 3. SoKG             leads at 27.00% under 3-hop retrieval, outperform-
substantially expands knowledge volume while si-            ing Naive RAG by 6.9 percentage points.
multaneously reducing fragmentation across all                 A comparison between SoKG and SoKG (w/o
evaluated LLMs. While alternative methods ei-               5W1H) further indicates that 5W1H-guided expan-
ther limit fact extraction (GraphRAG) or exhibit            sion improves not only factual retention but also the
higher fragmentation (KGGen and Direct Extrac-              downstream usefulness of the resulting graph. By
tion), SoKG extracts substantially more triples             explicitly externalizing latent connections as navi-
while maintaining lower NFI values.                         gable edges, SoKG provides a more effective rep-
   The comparison between SoKG (w/o 5W1H)                   resentation for tasks that require reasoning across
and SoKG (Ours) further illustrates the effective-          disparate facts.

                                                        8
Figure 3: Comparison of extracted graphs for the example sentence: “Volunteers provide essential services and
support to vulnerable populations, such as the homeless, the elderly, and individuals with disabilities.” The nested
relational path implied by this text (Volunteers → Vulnerable Populations → {Homeless, Elderly, Individuals with
disabilities}) is emphasized to assess relational completeness. Specifically, nodes corresponding to this path are
enlarged for clear visibility, connected by thick dark blue arrows to indicate the sequence of triples, while the
remaining background graph elements are displayed in light blue.


5.5    Qualitative Analysis                                   diate representation for document-level semantic
The cases in Figures 2 and 3 provide concrete exam-           expansion prior to triple extraction. By employing
ples of how SoKG’s interrogative process resolves             5W1H-guided QA generation, SoKG resolves ref-
the structural deficiencies and information loss ob-          erential ambiguities and surfaces implicit relational
served in alternative methods.                                dependencies, ensuring that subsequent structural
   Figure 2 illustrates SoKG’s capacity to preserve           mapping is grounded in explicit, contextualized
logical coherence in complex participle phrases,              entities rather than underspecified inferences.
such as facilitating cross-pollination. GraphRAG,                Evaluation on the MINE benchmark demon-
relying on entity summarization, fails to capture the         strates that SoKG achieves superior factual cov-
causal structure entirely, while Direct Extraction            erage while simultaneously improving structural
and KGGen fragment or simplify the relationship.              cohesion across diverse LLMs. This performance
In contrast, SoKG articulates the mediating concept           stems from the QA-mediated scaffold, which sys-
to ensure a cohesive causal chain: bees → cross-              tematically externalizes latent causal and relational
pollination → genetic diversity.                              dependencies that enhance graph connectivity even
   Similarly, Figure 3 demonstrates how SoKG                  as the volume of extracted facts increases.
resolves relational fragmentation in nested entity               Furthermore, downstream experiments on Hot-
structures. GraphRAG isolates key entities as dis-            potQA confirm that these structural advantages
connected nodes, while KGGen introduces impre-                translate into practical reasoning gains. While
cise predicates such as make a difference, losing the         other KG-based methods show inconsistent im-
nested relational structure. SoKG fully reconstructs          provements over Naive RAG, SoKG outperforms
the relational tree by identifying all key entities and       it across all evaluated settings, demonstrating that
linking them via precise predicates such as provide           QA-mediated construction produces KGs that are
services to and include.                                      useful for complex multi-hop reasoning.
   These examples demonstrate that QA-mediated                   Our findings indicate that explicit semantic orga-
semantic scaffolding, guided by 5W1H inquiry, sys-            nization through QA generation is not merely an
tematically addresses both causal reconstruction              auxiliary preprocessing step but a key component
and relational completeness. We further provide an            for maintaining graph fidelity in LLM-based KG
qualitative analysis of failure cases in Appendix F,          construction, enabling more structured knowledge
focusing on the QA mediation stage where the pri-             extraction. By addressing the inherent trade-off
mary bottleneck lies.                                         between factual coverage and structural connectiv-
                                                              ity, SoKG provides a more reliable foundation for
6     Conclusion                                              document-grounded knowledge representation and
                                                              structured reasoning.
We present SoKG, LLM-based KG construction
method that uses QA pairs as a structured interme-

                                                          9
Limitations                                                   by the Korean government (MSIT) [RS-2021-
                                                              II211343, Artificial Intelligence Graduate School
While SoKG’s multi-stage pipeline naturally in-               Program (Seoul National University)]; and the
volves higher token consumption than single-pass              Creative-Pioneering Researchers Program through
extraction, we provide a detailed cost analysis in            Seoul National University.
Appendix D. Improving efficiency while maintain-
ing factual density remains an important direction
for future work.                                              References
   Furthermore, as the graph quality depends on               Susan A Ambrose, Michael W Bridges, Michele DiPi-
the reasoning depth of the underlying LLM, per-                 etro, Marsha C Lovett, and Marie K Norman. 2010.
formance may vary in domains requiring highly                   How learning works: Seven research-based princi-
                                                                ples for smart teaching. John Wiley & Sons.
specialized interrogative logic. In particular, our
failure analysis (Appendix F) reveals that the pri-           Kartikeya Aneja, Manasvi Srivastava, Subhayan Das,
mary bottleneck lies in uni-dimensional queries                 and Nagender Aneja. 2025. Interpretable question
                                                                answering with knowledge graphs. arXiv preprint
that lack sufficient specificity, suggesting that more
                                                                arXiv:2510.19181.
targeted questioning strategies could further en-
hance extraction quality.                                     Sydney Anuyah, Mehedi Mahmud Kaushik, Sri Rama
                                                                Krishna Reddy Dwarampudi, Rakesh Shiradkar, Ar-
   Regarding graph representation, our current use              jan Durresi, and Sunandan Chakraborty. 2025. Au-
of binary triples may simplify multidimensional                 tomated knowledge graph construction using large
qualifiers (e.g., temporal or spatial data) that could          language models and sentence complexity modelling.
be more compactly encoded via n-ary relations.                  In Proceedings of the 2025 Conference on Empiri-
                                                                cal Methods in Natural Language Processing, page
Finally, our evaluation focuses on factual recov-               15526–15550. Association for Computational Lin-
erability and downstream multi-hop reasoning.                   guistics.
While this aligns with our objective of preserving
                                                              Albert-László Barabási. 2013.     Network science.
document-level semantics, other dimensions—such                 Philosophical Transactions of the Royal Society A:
as schema-alignment and relation-type fidelity—                 Mathematical, Physical and Engineering Sciences,
are left as promising avenues for the community to              371(1987):20120375.
explore as KG evaluation standards evolve.                    Zhen Bi, Jing Chen, Yinuo Jiang, Feiyu Xiong, Wei Guo,
                                                                Huajun Chen, and Ningyu Zhang. 2024. Codekgc:
Ethical Considerations                                          Code language model for generative knowledge
                                                                graph construction. ACM Transactions on Asian
This study utilizes the publicly available MINE                 and Low-Resource Language Information Process-
benchmark and LLMs. We acknowledge that the                     ing, 23(3):1–16.
benchmark and underlying LLMs may possess in-                 Steve Borgatti. 2002. The key player problem. SSRN
herent biases, which could be reflected in the con-             Electronic Journal.
structed graphs. Additionally, automated extraction           Pere-Lluís Huguet Cabot and Roberto Navigli. 2021.
carries a risk of hallucinating facts not present in            Rebel: Relation extraction by end-to-end language
source documents. We recommend human verifica-                  generation. In Findings of the association for compu-
tion and validation for applications in sensitive or            tational linguistics: emnlp 2021, pages 2370–2381.
high-stakes domains.                                          Matthias Cetto, Christina Niklaus, André Freitas,
                                                               and Siegfried Handschuh. 2018. Graphene: a
Acknowledgements                                               context-preserving open information extraction sys-
                                                               tem. arXiv preprint arXiv:1808.09463.
This work was supported by the Technology Inno-               Michelene TH Chi, Miriam Bassok, Matthew W Lewis,
vation Program funded by the Ministry of Trade,                 Peter Reimann, and Robert Glaser. 1989. Self-
Industry and Energy (MOTIE, Korea) (RS-2025-                    explanations: How students study and use examples
25453780); the National Research Foundation of                  in learning to solve problems. Cognitive science,
                                                                13(2):145–182.
Korea (RS-2023-00302123) funded by the Korean
government, as part of the European Commission’s              William W Cohen, Wenhu Chen, Michiel De Jong,
Horizon Europe framework programme (Grant                       Nitish Gupta, Alessandro Presta, Pat Verga, and
                                                                John Wieting. 2023. Qa is the new kr: Question-
Agreement No. 101135576, INTEND); the Insti-                    answer pairs as knowledge bases. In Proceedings of
tute of Information and Communications Technol-                 the AAAI Conference on Artificial Intelligence, vol-
ogy Planning and Evaluation (IITP) grant funded                 ume 37, pages 15385–15392.


                                                         10
Xinya Du and Claire Cardie. 2020. Event extraction by             Omer Levy, Minjoon Seo, Eunsol Choi, and Luke
  answering (almost) natural questions. In Proceedings             Zettlemoyer. 2017.    Zero-shot relation extrac-
  of the 2020 Conference on Empirical Methods in                   tion via reading comprehension. arXiv preprint
  Natural Language Processing (EMNLP), pages 671–                  arXiv:1706.04115.
  683.
                                                                  Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Darren Edge, Ha Trinh, Newman Cheng, Joshua                         Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
  Bradley, Alex Chao, Apurva Mody, Steven Truitt,                   rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
  Dasha Metropolitansky, Robert Osazuwa Ness, and                   täschel, and 1 others. 2020. Retrieval-augmented gen-
  Jonathan Larson. 2024. From local to global: A                    eration for knowledge-intensive nlp tasks. Advances
  graph rag approach to query-focused summarization.                in neural information processing systems, 33:9459–
  arXiv preprint arXiv:2404.16130.                                  9474.

Oren Etzioni, Michele Banko, Stephen Soderland, and               Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan,
  Daniel S Weld. 2008. Open information extrac-                     Duo Chai, Mingxin Zhou, and Jiwei Li. 2019. Entity-
  tion from the web. Communications of the ACM,                     relation extraction as multi-turn question answering.
  51(12):68–74.                                                     arXiv preprint arXiv:1905.05529.

Anthony Fader, Stephen Soderland, and Oren Etzioni.               Christopher D Manning, Mihai Surdeanu, John Bauer,
  2011. Identifying relations for open information ex-              Jenny Rose Finkel, Steven Bethard, and David Mc-
  traction. In Proceedings of the 2011 conference on                Closky. 2014. The stanford corenlp natural language
  empirical methods in natural language processing,                 processing toolkit. In Proceedings of 52nd annual
  pages 1535–1545.                                                  meeting of the association for computational linguis-
                                                                    tics: system demonstrations, pages 55–60.
Nicholas FitzGerald, Julian Michael, Luheng He, and
  Luke Zettlemoyer. 2018. Large-scale qa-srl parsing.             Dipak Meher, Carlotta Domeniconi, and Guadalupe
  arXiv preprint arXiv:1805.05377.                                  Correa-Cabrera. 2025.   Link-kg: Llm-driven
                                                                    coreference-resolved knowledge graphs for
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,                    human smuggling networks.       arXiv preprint
  Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen                arXiv:2510.26486.
  Wang, and Haofen Wang. 2023. Retrieval-augmented
  generation for large language models: A survey.                 Belinda Mo, Kyssen Yu, Joshua Kazdan, Joan Cabezas,
  arXiv preprint arXiv:2312.10997, 2(1).                            Proud Mpala, Lisa Yu, Chris Cundy, Charilaos Kanat-
                                                                    soulis, and Sanmi Koyejo. 2025. Kggen: Extracting
Arthur C Graesser and Natalie K Person. 1994. Ques-                 knowledge graphs from plain text with language mod-
  tion asking during tutoring. American educational                 els. arXiv preprint arXiv:2502.09956.
  research journal, 31(1):104–137.
                                                                  Srichakradhar Reddy Nagireddy. 2021. StoryNet: A
Ryan Henry and Jiaqi Gong. 2025. Clare: Context-                     5W1H-Based Knowledge Graph to Connect Stories.
  aware, interactive knowledge graph construction                    University of Missouri-Kansas City.
  from transcripts. Information, 16(10):866.
                                                                  Christina Niklaus. 2022. From complex sentences to
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,                   a formal semantic representation using syntactic
  Zhangyin Feng, Haotian Wang, Qianglong Chen,                      text simplification and open information extraction.
  Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-                 Springer Nature.
  ers. 2025. A survey on hallucination in large lan-
  guage models: Principles, taxonomy, challenges, and             Christina Niklaus, Matthias Cetto, André Freitas, and
  open questions. ACM Transactions on Information                   Siegfried Handschuh. 2018. A survey on open infor-
  Systems, 43(2):1–55.                                              mation extraction. arXiv preprint arXiv:1806.05599.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan              Christina Niklaus, Matthias Cetto, André Freitas, and
  Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea                     Siegfried Handschuh. 2019. Transforming complex
  Madotto, and Pascale Fung. 2023. Survey of hal-                   sentences into a semantic hierarchy. arXiv preprint
  lucination in natural language generation. ACM com-               arXiv:1906.01038.
  puting surveys, 55(12):1–38.
                                                                  Jeff Z. Pan, Simon Razniewski, Jan-Christoph Kalo,
Pengcheng Jiang, Jiacheng Lin, Zifeng Wang, Jimeng                   Sneha Singhania, Jiaoyan Chen, Stefan Dietze, Hajira
  Sun, and Jiawei Han. 2024. Genres: Rethinking                      Jabeen, Janna Omeliyanenko, Wen Zhang, Matteo
  evaluation for generative relation extraction in the era           Lissandrini, Russa Biswas, Gerard de Melo, Angela
  of large language models. In Proceedings of the 2024               Bonifati, Edlira Vakaj, Mauro Dragoni, and Damien
  Conference of the North American Chapter of the                    Graux. 2023. Large Language Models and Knowl-
  Association for Computational Linguistics: Human                   edge Graphs: Opportunities and Challenges. Trans-
  Language Technologies (Volume 1: Long Papers),                     actions on Graph Data and Knowledge, 1(1):2:1–
  pages 2820–2837.                                                   2:38.


                                                             11
Enayat Rajabi and Kobra Etminani. 2024. Knowledge-
  graph-based explainable ai: A systematic review. J.
  Inf. Sci., 50(4):1019–1029.
Xubin Ren, Jiabin Tang, Dawei Yin, Nitesh Chawla,
  and Chao Huang. 2024. A survey of large language
  models for graphs. In Proceedings of the 30th ACM
  SIGKDD Conference on Knowledge Discovery and
  Data Mining, pages 6616–6626.
Yu-Ming Shang, Heyan Huang, and Xianling Mao. 2022.
  Onerel: Joint entity and relation extraction with one
  module in one step. In Proceedings of the AAAI con-
  ference on artificial intelligence, volume 36, pages
  11285–11293.
Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang,
  Xin Zhang, Shen Huang, Pengjun Xie, Jinan Xu,
  Yufeng Chen, Meishan Zhang, and 1 others. 2023.
  Chatie: Zero-shot information extraction via chatting
  with chatgpt. arXiv preprint arXiv:2302.10205.
Wei Wu, Fei Wang, Arianna Yuan, Fei Wu, and Jiwei
 Li. 2020. Corefqa: Coreference resolution as query-
 based span prediction. In Proceedings of the 58th
 annual meeting of the association for computational
 linguistics, pages 6953–6963.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
  William Cohen, Ruslan Salakhutdinov, and Christo-
  pher D Manning. 2018. Hotpotqa: A dataset for
  diverse, explainable multi-hop question answering.
  In Proceedings of the 2018 conference on empiri-
  cal methods in natural language processing, pages
  2369–2380.
Deming Ye, Yankai Lin, Peng Li, and Maosong Sun.
  2022a. Packed levitated marker for entity and rela-
  tion extraction. In Proceedings of the 60th annual
  meeting of the association for computational linguis-
  tics (volume 1: long papers), pages 4904–4917.
Hongbin Ye, Ningyu Zhang, Hui Chen, and Huajun
  Chen. 2022b. Generative knowledge graph construc-
  tion: A review. arXiv preprint arXiv:2210.12714.
Bowen Zhang and Harold Soh. 2024. Extract, define,
  canonicalize: An LLM-based framework for knowl-
  edge graph construction. In Proceedings of the 2024
  Conference on Empirical Methods in Natural Lan-
  guage Processing, pages 9820–9836. Association for
  Computational Linguistics.
Zexuan Zhong and Danqi Chen. 2021. A frustrat-
  ingly easy approach for entity and relation extrac-
  tion. In Proceedings of the 2021 conference of the
  North American chapter of the association for com-
  putational linguistics: human language technologies,
  pages 50–61.
Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao,
  Yixin Ou, Yunzhi Yao, Shumin Deng, Huajun Chen,
  and Ningyu Zhang. 2024. Llms for knowledge graph
  construction and reasoning: Recent capabilities and
  future opportunities. World Wide Web, 27(5):58.



                                                          12
A     Ablation Studies
                               Qwen-2.5        GPT-4o-mini              GPT-4o            Gemini-2.5             Claude-4
Prompt Archetype
                            w/o 5W1H   Full   w/o 5W1H   Full      w/o 5W1H      Full   w/o 5W1H        Full   w/o 5W1H     Full
Role-Oriented (RO)             67.1    73.4     80.5     83.9          83.5      89.3     85.6          87.7     94.6       96.3
Procedural-Step (PS)           60.7    64.5     79.6     83.0          82.1      87.1     86.7          87.3     93.5       95.8
Instructional-Direct (ID)      65.3    68.6     77.9     83.5          79.7      81.2     83.9          88.7     91.3       96.3
Average                        64.4    68.8     79.3     83.5          81.8      85.9     85.4          87.9     93.1       96.1

Table 5: Effectiveness of 5W1H-guided expansion across prompt archetypes. Scores represent factual retention (%)
on the MINE benchmark. The results demonstrate that the 5W1H framework is a robust cognitive guide independent
of stylistic framing.


Method                      Qwen-2.5          GPT-4o-mini               GPT-4o             Gemini-2.5                   Claude-4
KGGen                         56.7               44.3                     66.4                   62.5                     69.1
SoKG-EF                       72.1               79.7                     76.9                   87.0                     92.5
SoKG (Ours)                   73.4               83.9                     89.3                   87.7                     96.3

Table 6: Ablation study on input representation and triple extraction strategy. While SoKG-EF demonstrates the
foundational impact of the QA scaffold, SoKG achieves peak performance by removing the entity-first bottleneck to
maximize factual retention across all LLMs.


A.1    Robustness of Prompt Designs                              A.2    The Entity-First Constraint
To evaluate the contribution of the 5W1H frame-                  We evaluate the respective impacts of the QA scaf-
work, Table 5 compares the full 5W1H-integrated                  fold and extraction strategy by comparing three
pipeline (Full) against the version omitting 5W1H-               configurations: KGGen, SoKG-EF (Entity-First),
guided expansion (w/o 5W1H) across three distinct                and SoKG. SoKG-EF incorporates both the extrac-
prompt archetypes:                                               tion and consolidation logic of KGGen, applying
                                                                 this entity-centric pipeline to our QA-mediated
    • Role-Oriented (RO): Assigns a specific per-
                                                                 scaffold. This setup allows us to evaluate the bene-
      sona (e.g., Knowledge Archivist) and uses
                                                                 fit of the scaffold independently while preserving
      5W1H as analytical lenses to guide deep ex-
                                                                 the underlying entity-first logic.
      ploration. This prompt design was adopted as
                                                                     As shown in Table 6, the superior performance
      the primary setting for our main experiments.
                                                                 of SoKG-EF over KGGen confirms that a QA scaf-
    • Procedural-Step (PS): Defines a systematic                 fold effectively mitigates the complexity of raw
      workflow (Read → Segment → Generate) to                    text. However, SoKG’s even greater success re-
      ensure atomic factual extraction.                          veals that rigid entity-first filtering acts as a re-
                                                                 strictive bottleneck, limiting the model’s ability to
    • Instructional-Direct (ID): Employs standard
                                                                 capture full relational depth even when supported
      task-based instructions without complex role-
                                                                 by a comprehensive QA scaffold.
      play or multi-step procedures.
                                                                     Although intermediate stages introduce over-
   As shown in Table 5, the 5W1H framework                       head, this rich interrogative structure justifies the in-
provides a universal performance lift across all                 vestment by providing a dense semantic foundation
LLMs regardless of the underlying prompt struc-                  for superior factual recoverability. Unlike isolated
ture. While the RO archetype generally yields the                entity extraction which often incurs information
highest retention, peaking at 96.3% with Claude-4,               loss through restrictive filtering, the QA-mediated
even the more concise PS and ID templates show                   scaffold preserves a richer semantic context. These
significant improvements once the interrogative                  results demonstrate that for structural refinement, a
scaffold is present. These results indicate that the             QA-driven approach offers a more systematic and
5W1H constraint functions as a fundamental cog-                  inclusive foundation for knowledge construction
nitive guide that systematically surfaces procedural             than traditional entity-centric methods.
and causal dimensions.

                                                            13
B     QA Generation Prompt Details
B.1    Role-Oriented (RO), w/ 5W1H
## ROLE
You are a **Comprehensive Knowledge Archivist** who converts the [Full Document] into detailed,
    document-grounded QA pairs.

## OBJECTIVE
Extract as many meaningful Question-Answer pairs as possible from the document.
Use the 5W1H perspectives (Who, What, When, Where, Why, How) **as analytical lenses** to help you
    identify and expand potential questions, but do NOT restrict yourself to producing only
    5W1H-type questions.
Your goal is to maximize informational coverage, capturing every explicit fact, relation, event,
    definition, rationale, and process described in the document.

## INPUT
Full Document: "{document_text}"

## CONSTRAINTS
1. **Context-Independent**
   - Each QA must be self-contained and understandable without referencing the original text.
   - Replace pronouns with explicit entities.

2. **No Hallucination**
   - Use only facts explicitly stated in the document.

3. **Expansion-Oriented Thinking**
   - For each sentence or factual unit, consider the 5W1H perspectives as prompts to explore:
     - WHO is involved?
     - WHAT happened or is described?
     - WHEN did it occur?
     - WHERE did it occur?
     - WHY did it occur?
     - HOW was it carried out?
   - These perspectives are **guides** to inspire multiple possible QA pairs, even if they are
    implicit or only partially expressed.

4. **Coverage**
   - Extract all possible QA pairs that can be reasonably derived from the document.

## OUTPUT FORMAT
Return a JSON list of QA objects:

[
    {{"question": "...", "answer": "..."}},
    ...
]




                                                14
B.2   Role-Oriented (RO), w/o 5W1H
## ROLE
You are a **Comprehensive Knowledge Archivist** who converts the [Full Document] into precise and
    meaningful QA pairs.

## OBJECTIVE
Extract as many high-quality Question-Answer pairs as needed to fully represent the document’s
    explicit information.
Use the following analytical perspectives as guides to discover potential questions, but do NOT
    restrict your output to only these categories:

1. **Entities & Definitions** - Identify and clarify key terms, objects, roles, or concepts.
2. **Properties & Characteristics** - Extract notable features, attributes, components, or
    qualities.
3. **Events & Stated Facts** - Capture actions, processes, or explicit factual statements.
4. **Relationships & Dependencies** - Identify connections, comparisons, or dependencies between
    entities or ideas.

These perspectives are **guides for expanding coverage**, not mandatory categories.

## INPUT
Full Document: "{document_text}"

## CONSTRAINTS
1. **Context-Independent**
   - Each QA must be self-contained and understandable without referencing the original text.
   - Replace pronouns with explicit entities when needed.

2. **No Hallucination**
   - Use only facts explicitly stated in the document.

3. **Coverage without Inflation**
   - Extract all meaningful QA pairs that can be reasonably derived from the document.

## OUTPUT FORMAT
Return a JSON list:

[
    {{"question": "...", "answer": "..."}},
    ...
]




                                                15
B.3   Procedural-Step (PS), w/ 5W1H
## ROLE
You are a **Document-Grounded QA Extractor**.

## OBJECTIVE
Convert the full document into high-coverage, explicit-fact QA pairs.

## PROCEDURE
1. Read the document end-to-end.
2. Segment into atomic factual units.
3. For each unit:
   - Generate QAs that capture all explicit information it contains.
   - When forming questions, view the unit through the 5W1H angles (Who, What, When, Where, Why,
    How) so that different aspects of the same fact can be covered.
4. Merge duplicates and keep the most precise wording.

## INPUT
Full Document: "{document_text}"

## CONSTRAINTS
- Context-Independent QAs only.
- No Hallucination.
- Prefer concise but complete answers.

## OUTPUT FORMAT
Return a JSON list:
[
  {{"question": "...", "answer": "..."}},
  ...
]


B.4   Procedural-Step (PS), w/o 5W1H
## ROLE
You are a **Document-Grounded QA Extractor**.

## OBJECTIVE
Convert the full document into high-coverage, explicit-fact QA pairs.

## PROCEDURE
1. Read the document end-to-end.
2. Segment into atomic factual units.
3. For each unit, generate QAs that capture all explicit information it contains.
4. Merge duplicates and keep the most precise wording.

## INPUT
Full Document: "{document_text}"

## CONSTRAINTS
- Context-Independent QAs only.
- No Hallucination.
- Prefer concise but complete answers.

## OUTPUT FORMAT
Return a JSON list:
[
  {{"question": "...", "answer": "..."}},
  ...
]




                                                16
B.5   Instructional-Direct (ID), w/ 5W1H
Read the following document and generate question-answer pairs based on its content.
Generate as many high-quality questions as needed to cover the information explicitly stated in
    the document.
For the same piece of information, consider the 5W1H dimensions (Who, What, When, Where, Why, How)
    and generate separate questions whenever different aspects are supported by the text.
Do not stop at a single question if multiple 5W1H aspects can be identified.
If different parts of the document support different questions, include all of them.
Each question should be answerable using information explicitly stated in the document and written
    in a clear and self-contained manner.

Input Document:
"{document_text}"

Output Format:
Return a JSON list of objects in the following form:
[
  {{"question": "...", "answer": "..."}},
  ...
]


B.6   Instructional-Direct (ID), w/o 5W1H
Read the following document and generate question-answer pairs based on its content.
Generate as many high-quality questions as needed to cover the information explicitly stated in
    the document.
If different parts of the document support different questions, include all of them.
Each question should be answerable using information explicitly stated in the document and written
    in a clear and self-contained manner.

Input Document:
"{document_text}"

Output Format:
Return a JSON list of objects in the following form:
[
  {{"question": "...", "answer": "..."}},
  ...
]




                                                17
C     Triple Extraction Prompt Details
C.1    Triple Extraction from QA Pairs
## ROLE
You are a Semantic Knowledge Graph Builder.
Extract every structured triples (entity1, relation, entity2) from the Q&A pair, following the
    rules below.

## GOAL
From the question-answer pair, extract only useful, knowledge-ready triples that can serve as
    entries in a semantic knowledge graph.

## RULES
Extract clean (subject, relation, object) triples following the rules:

1. Split every stated or clearly implied fact into minimal triples; integrate question and answer
    context when needed.

2. Entities (entity1, entity2) must be short, concrete noun phrases.
   - No pronouns (this, that, it, its, these, those, etc.).
   - Entities must not be unresolved or reference-based pronouns (\eg those, they, someone,
    anyone, whoever); if such a pronoun appears, rewrite it into a specific, explicit noun phrase
    or skip the triple.
   - No clauses or relative clauses (no "who/that/which/what/as it ..." inside an entity).
   - No long gerund or sentence-like phrases. If a phrase contains a verb or clause marker,
    rewrite it into a concise noun concept or skip the triple.

3. Relations must be short, canonical verbs or verb phrases.
   - Express a single semantic link between the two entities (\eg causes, leads to, supports,
    believes, opposes).
   - Must be a compact predicate, not a sentence fragment.
   - No pronouns or clause markers inside the relation (no "its", "that", "as it", "what", etc.).
   - If the source uses an idiomatic or long expression, rewrite it into a simple canonical
    relation without pronouns or embedded clauses, or skip the triple.

4. Include a fact if it can be clearly rewritten into a concise, explicit triple that fits the
    rules above; otherwise skip it.

5. Output only concise, interpretable, knowledge-ready triples.

## INPUT
Q: {question}
A: {answer}

## OUTPUT FORMAT (JSON List)
- Return a list of JSON objects.
- Return [] if no valid triples exist.

[
    {{"entity1": "Specific_Noun", "relation": "precise_verb_phrase", "entity2": "Specific_Noun"}}
]




                                                  18
C.2    Triple Extraction from Raw Text (Direct Extraction)
## ROLE
You are a Semantic Knowledge Graph Builder.
Extract every structured triples (entity1, relation, entity2) from the text, following the rules
    below.

## GOAL
From the given text, extract only useful, knowledge-ready triples that can serve as entries in a
    semantic knowledge graph.

## RULES
Extract clean (subject, relation, object) triples following the rules:

1. Split every stated or clearly implied fact into minimal triples.

2. Entities (entity1, entity2) must be short, concrete noun phrases.
   - No pronouns (this, that, it, its, these, those, etc.).
   - Entities must not be unresolved or reference-based pronouns (\eg those, they, someone,
    anyone, whoever); if such a pronoun appears, rewrite it into a specific, explicit noun phrase
    or skip the triple.
   - No clauses or relative clauses (no "who/that/which/what/as it ..." inside an entity).
   - No long gerund or sentence-like phrases. If a phrase contains a verb or clause marker,
    rewrite it into a concise noun concept or skip the triple.

3. Relations must be short, canonical verbs or verb phrases.
   - Express a single semantic link between the two entities (\eg causes, leads to, supports,
    believes, opposes).
   - Must be a compact predicate, not a sentence fragment.
   - No pronouns or clause markers inside the relation (no "its", "that", "as it", "what", etc.).
   - If the source uses an idiomatic or long expression, rewrite it into a simple canonical
    relation without pronouns or embedded clauses, or skip the triple.

4. Include a fact if it can be clearly rewritten into a concise, explicit triple that fits the
    rules above; otherwise skip it.

5. Output only concise, interpretable, knowledge-ready triples.

## INPUT
Text: {document_text}

## OUTPUT FORMAT (JSON List)
- Return a list of JSON objects.
- Return [] if no valid triples exist.

[
    {{"entity1": "Specific_Noun", "relation": "precise_verb_phrase", "entity2": "Specific_Noun"}}
]




                                                  19
D    Computational Efficiency Analysis
                                         Qwen-2.5                                           Claude-4
Method
                              Ttotal         Ntri              τ               Ttotal              Ntri            τ
Direct Extraction          145,262         1,955              74.30         150,569            4,417             34.09
GraphRAG                   333,646         1,981             168.42         331,313            1,590            208.37
KGGen                      231,762         2,375              97.58         256,691            6,391             40.16
SoKG (w/o 5W1H)            275,137         2,871              95.83         392,284           10,511             37.32
SoKG (Ours)                333,917         3,958              84.36         553,530           14,849             37.28

Table 7: Computational cost analysis on the MINE benchmark for Qwen-2.5 (weakest backbone) and Claude-
4 (strongest backbone). Ttotal : total tokens consumed up to triple extraction (summed over 100 articles); Ntri :
canonicalized triple count; τ = Ttotal /Ntri : tokens per triple.
   The multi-stage QA pipeline naturally incurs              mated metrics from the GenRES framework (Jiang
higher token consumption than single-pass extrac-            et al., 2024)—Uniqueness Score (US) and Gran-
tion. To quantify this overhead, Table 7 reports the         ularity Score (GS)—alongside human-evaluated
total token consumption (input and output) accu-             Factual Precision (FP). We conduct this analysis
mulated up to the triple extraction stage for each           on triples extracted from the MINE benchmark us-
method, along with the resulting triple count after          ing Qwen2.5-7B-Instruct, the lightest backbone in
canonicalization.                                            our evaluation and thus most susceptible to quality
   SoKG consumes approximately 2–3× more to-                 degradation.
tal tokens than Direct Extraction across both back-
                                                             Uniqueness Score (US) measures pairwise di-
bones, reflecting the additional cost of QA gener-
                                                             versity among extracted triples within each doc-
ation and per-pair triple extraction. However, a
                                                             ument. For each triple ti , we compute its maxi-
substantial portion of this investment is converted
                                                             mum cosine similarity to all other triples in the
into a larger volume of canonicalized triples: SoKG
                                                             same document and define uniqueness as US(ti ) =
produces 2.0× more triples than Direct Extraction
                                                             1 − maxj̸=i cos(ti , tj ). The document-level US is
on Qwen-2.5 and 3.4× more on Claude-4.
                                                             then averaged over all triples.
   Direct Extraction achieves the lowest absolute to-
ken consumption, but as shown in the main results            Granularity Score (GS) evaluates the atomicity
(Tables 1–3), the resulting graphs exhibit higher            of each triple by penalizing overly broad facts that
fragmentation and lower factual retention, limiting          could be further decomposed. Following Jiang et al.
their downstream utility. GraphRAG and KGGen                 (2024), an LLM judge assesses whether each triple
consume comparable or higher total tokens than               represents a single, indivisible fact.
SoKG on certain backbones while producing sub-
stantially fewer triples, resulting in less favorable        Factual Precision (FP) was assessed through
cost-to-knowledge ratios.                                    manual verification. Five human annotators in-
   These results indicate that while the QA-                 dependently evaluated all triples extracted from
mediated pipeline introduces meaningful computa-             three randomly sampled MINE document against
tional overhead, the additional cost is offset by a          the source text, labeling each triple as factually
proportionally larger gain in extracted knowledge            supported or not.
volume and structural quality. Reducing this over-
head through selective QA generation or adaptive             Method                Ntri   US (%)      GS      FP (%)
questioning depth remains a promising direction              Direct Ext.         1,955    88.02      69.09   86.42±4.21
for future work.                                             GraphRAG            1,981    86.86      54.24   83.15±7.38
                                                             KGGen               2,375    83.66      87.72   83.93±8.67
                                                             SoKG (w/o 5W1H)     2,871    90.62      82.03   75.12±9.45
E   Triple Quality and Hallucination                         SoKG                3,958    91.75      82.59   84.21±6.33
    Analysis
                                                             Table 8: Triple quality analysis on Qwen-2.5. US:
To verify that SoKG’s expanded extraction vol-               Uniqueness Score (%, higher is better); GS: Granularity
ume does not introduce redundancy or hallucina-              Score (higher is better); FP: human-evaluated Factual
tion, we evaluated triple quality using two auto-            Precision (%).


                                                        20
   As shown in Table 8, SoKG achieves the high-               themselves (e.g., precision strikes, real-time surveil-
est Uniqueness Score (91.75%), confirming that                lance), failing to capture the causal mechanism the
5W1H-guided expansion surfaces distinct, non-                 question implies.
repetitive information even as extraction density
                                                              Sophisticated Non-Answers The answer re-
increases. The Granularity Score (82.59) substan-
                                                              states the document’s framing rather than ground-
tially exceeds those of Direct Extraction (69.09)
                                                              ing the response in specific evidence.
and GraphRAG (54.24), indicating that the QA
scaffold enforces atomic precision rather than pro-                Q: What is the essay exploring regarding military
ducing overly broad triples.                                       technology?
                                                                   A: This essay will explore the evolution of mili-
   For Factual Precision, SoKG achieves 84.38%—                    tary technology, highlighting key innovations.
comparable to Direct Extraction (88.00%) and
consolidation-based methods. Direct Extraction’s              The response remains at the level of the document’s
slightly higher precision reflects its tendency to            introduction, relying on general nouns (e.g., tech-
capture only surface-level, explicit facts that are           nology, innovation) rather than extracting concrete
inherently easier to verify. Notably, removing the            facts from the body of the text.
5W1H scaffold (SoKG w/o 5W1H) reduces preci-
                                                              Context Loss The answer is factually accurate
sion to 74.55%, suggesting that systematic interrog-
                                                              but omits the specific contextual details that would
ative guidance is critical for maintaining grounded
                                                              produce discriminative triples.
construction as extraction volume increases.
   These results collectively demonstrate that                     Q: What risk is associated with gene manipulation
                                                                   regarding individual rights?
SoKG’s expanded knowledge volume reflects gen-                     A: There is a risk of infringing on individuals’
uine informational gain rather than redundant or                   rights to make informed decisions about their own
hallucinated content.                                              genetic information.


F   Failure Analysis of QA Mediation                          While correct, the answer misses the specific exam-
                                                              ples provided in the source text—such as discrim-
To provide a transparent account of the frame-                ination based on genetic traits—producing triples
work’s limitations, we analyzed failure patterns              with low discriminative value for graph-based rea-
in the QA mediation stage—the primary contri-                 soning. These failure modes highlight a funda-
bution of this work—rather than triple extraction,            mental limitation: the quality of QA mediation is
which modern LLMs handle with relative profi-                 bounded by the LLM’s ability to formulate deep,
ciency. Our analysis revealed minimal factual hal-            specific queries. The concentration of failures
lucinations during QA generation; the generated               in “What” questions suggests that this category,
pairs were largely grounded in the source text. The           while the most frequently generated, is also the
primary failure mode was not a lack of truth but              most prone to shallow formulation. More targeted
a lack of semantic utility: the QA pairs failed to            prompt refinement or adaptive questioning strate-
surface sufficiently specific or discriminative infor-        gies that dynamically adjust question depth based
mation for downstream triple extraction.                      on document complexity represent promising di-
   These failures stem almost exclusively from uni-           rections for future work.
dimensional “What” queries that exhibit the follow-
ing recurring patterns:

Lack of Specificity The question targets a broad
topic rather than a specific mechanism, yielding
answers that list entities without clarifying their
functional roles.

     Q: What new challenges have emerged with re-
     cent military technology advancements?
     A: The rise of UAVs, drones, and AI has intro-
     duced new challenges and opportunities.

The answer enumerates technological entities
rather than addressing the nature of the challenges

                                                         21
