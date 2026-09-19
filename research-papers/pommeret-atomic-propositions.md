                                                  LLM-based Atomic Propositions Help Weak Extractors:
                                                    Evaluation of a Propositioner for Triplet Extraction

                                                             Luc Pommeret1 , Thomas Gerald1 , Patrick Paroubek1 ,
                                                             Sahar Ghannay1 , Christophe Servan2 , Sophie Rosset1
                                                                1
                                                                    Université Paris-Saclay, CNRS, LISN, 91400, Orsay, France
                                                                      2
                                                                        AMIAD, Pôle Recherche, 91120, Palaiseau, France
                                                                                      surname.name@lisn.fr
                                                                                             Abstract
                                        Knowledge Graph construction from natural language requires extracting structured triplets from complex,
                                        information-dense sentences. In this paper, we investigate if the decomposition of text into atomic propositions
                                        (minimal, semantically autonomous units of information) can improve the triplet extraction. We introduce
                                        MPropositionneur-V2, a small multilingual model covering six European languages trained by knowledge
                                        distillation from Qwen3-32B into a Qwen3-0.6B architecture, and we evaluate its integration into two extraction




arXiv:2604.02866v1 [cs.CL] 3 Apr 2026
                                        paradigms: entity-centric (GLiREL) and generative (Qwen3). Experiments on SMiLER, FewRel, DocRED and CaRB
                                        show that atomic propositions benefit weaker extractors (GLiREL, CoreNLP, 0.6B models), improving relation recall
                                        and, in the multilingual setting, overall accuracy. For stronger LLMs, a fallback combination strategy recovers entity
                                        recall losses while preserving the gains in relation extraction. These results show that atomic propositions are an
                                        interpretable intermediate data structure that complements extractors without replacing them.

                                        Keywords: atomic propositions, knowledge graph, triplet extraction, OpenIE, propositioner, multilingual
                                        NLP

                                                        1.    Introduction                          have a disjunction of two atomic propositions each
                                                                                                    one holding more semantic information than the
                                        The interpretability of Natural Language Process-           original text. More details are available in section 3.
                                        ing (NLP) models is a requirement for applications             For the implementation of the sentence de-
                                        like fact-checking and automated construction of            composition process into NL atomic proposi-
                                        Knowledge Graphs (KGs). While neural models                 tions, we rely in this first experiment on a
                                        have achieved state-of-the-art results, their internal      limited-depth recursive prompting of a multilingual
                                        mechanisms remain opaque. Most current explain-             LLM, an atomic proposition being defined as a
                                        ability methods are post hoc, seeking to explain            prompting fixed-point. To perform the atomisa-
                                        decisions after the training.                               tion, we propose a small multilingual model, the
                                           In this paper, we argue for a shift towards inter-       MPropositionneur-V2, that can perform coref-
                                        pretability by design, where the data structure itself      erence resolution to produce atomic propositions
                                        is traceable. Our approach is based on natural lan-         from text. For more details, please see section 4.
                                        guage (NL) atomic propositions, a minimal, seman-
                                                                                                       We aim to show that in Open Information Ex-
                                        tically autonomous unit of information. For defining
                                                                                                    traction (OpenIE), using NL atomic propositions
                                        and producing NL atomic propositions, we rely on
                                                                                                    can improve performance while preserving inter-
                                        the formalism of Semantic Information Theory(Bar-
                                                                                                    pretability. Here, the interpretability is the auditabil-
                                        Hillel and Carnap, 1953), which defines the de-
                                                                                                    ity of the data structure, i.e. the NL atomic proposi-
                                        composition of non-atomic propositions into atomic
                                                                                                    tions. These propositions serve as an intermediary
                                        ones 1 . For instance, the sentence "The cat and
                                                                                                    representation in the process of mapping a natu-
                                        the dog are in the kitchen" atomizes into two NL
                                                                                                    ral language sentence S to a set of formal triplets
                                        atomic propositions: "The cat is in the kitchen" and
                                                                                                    {(s, r, o)}, where s is the subject entity, o the ob-
                                        "The dog is in the kitchen". Neither of these two
                                                                                                    ject entity, and r the relation. In our experiment,
                                        propositions can be decomposed further, because
                                                                                                    we evaluate whether splitting textual information
                                        otherwise we would add to the original semantic
                                                                                                    into atomic propositions could positively impact
                                        content hallucinations (spurious information). If the
                                                                                                    triplet extraction. We hypothesize that extracting
                                        original proposition was not a conjunction but a
                                                                                                    entities and relations could be made easier from
                                        disjunction ("The cat or the dog”) decomposing it
                                                                                                    less information-dense text chunks, especially NL
                                        would introduce new information since we would
                                                                                                    atomic propositions.
                                           1
                                            In logic, atomic propositions are closed formulae          Our contributions consist of a new multilin-
                                        that cannot be further decomposed into several smaller      gual propositioner and a pipeline leveraging atomic
                                        propositions without loss of semantic integrity or intro-   propositions to enhance entity-relation extraction
                                        duction of "bad" informational cuts (see Section 3)         based on LLM.
Figure 1: The upper schema depicts the pipeline’s stages: at stage 1, we extract atomic propositions
from the source text; at stage 2, we extract triplets using either a dependency parser or generative LLMs;
and finally, we build the knowledge graph from the entities and relations retrieved. For evaluation, we limit
to the triplet entity-relation benchmark.


   We report the evaluation of the multilingual         (NER), and then applies entity linking and rela-
propositioner trained through distillation of a large   tion extraction approaches to build a knowledge
language model (LLM), compared to a baseline            graph. For example, in the sentence ’The Eiffel
composed of a rule-based method leveraging a            Tower is in Paris’, the entities must be linked to the
dependency parser. To assess the effectiveness of       knowledge graph (Paris is entry Q90 in Wikidata)
our model and the extraction pipeline, we evaluate      and the relation must be linked to a vocabulary of
the method on different entity-relation benchmarks      relations (here, hasLocation, property P131 in Wiki-
for relation extraction and triplet validation. These   data) (Hogan et al., 2021). However, this approach
benchmarks cover a variety of tasks, such as open       lacks flexibility when dealing with the intrinsic com-
and closed information extraction, sentence- and        plexity of natural language.
document-based extraction, and extraction based            The second kind of approach is called Open
solely on relations and triplets.                       Information Extraction (OpenIE), and it aims to
   The paper is organised as follows: First, in Sec-    extract triplets (s, r, o)2 , without relying on prede-
tion 2, we review related works, describing previous    fined sets of entities and/or relations (Etzioni et al.,
triplet extraction approaches and methods based         2008).
on propositioners. Section 3 presents the formal-          To extract triplets from sentences, modern ap-
ism of the atomic proposition. Section 4 describes      proaches use LLMs, especially transformer en-
our pipeline in depth. We then present the exper-       coder or decoder models. Models like mREBEL
imental protocol to answer the research question        (Huguet Cabot et al., 2023) generate triplets di-
in Section 5. Section 6 discusses the different re-     rectly from complex input sentences. Such genera-
sults, and Section 7 concludes with suggestions         tive models often struggle with long-range depen-
for future work.                                        dencies, nested clauses, and coordination, lead-
                                                        ing to lower recall on complex structures. More
                                                        recently, GLiREL (Boylan et al., 2025) performs
             2.    Related Works
                                                        zero-shot relation classification by jointly encod-
For inference and/or information retrieval (Xiang       ing entity pairs and candidate relation labels. It is
et al., 2026), it is common to represent information    more robust but can be sensitive to the quality of
as a Knowledge Graph (KG). While a wide range           the initial entity recognition. One way to address
of KGs automatically extracted from many different      this duality is through the recourse of sentence
sources is available, for instance, using wikidata      simplification for relation extraction.
taxonomy and entities (Waagmeester et al., 2020;           Simplifying text before extracting relations has
Hassanzadeh, 2021), extracting KGs from natural         been explored, and, for instance, (Miwa et al.,
language source of information remains a signif-        2010) proposed an entity-focused sentence sim-
icant challenge (Waagmeester et al., 2020). We          plification method to improve relation extraction.
can split the wide range of approaches for extract-     More recently, (Niklaus et al., 2016) introduced
ing automatically KG into two categories.
                                                           2
   The first one performs named entity recognition             (subject, relation, object)
a rule-based sentence simplification system that         worlds3 satisfying ϕ. A cut of ϕ into ψ is safe if
rewrites complex sentences into simpler sentences        ψ is a sub-formula of ϕ and I(ϕ) > I(ψ). It is bad
for Open Information Extraction. However, these          if I(ϕ) ≤ I(ψ).
approaches rely on syntactic rules or dependency
parsing, are limited to a single language, and lack      3.2.   The CNF Condition
coreference resolution.
    Simplification by structural atomisation of infor-   We prove (in Annex B) that a proposition is atomic
mation is of interest for many tasks: in Retrieval       if and only if it is a clause in a Conjunctive Normal
Augmented Generation (RAG), the indexing of              Form (CNF).
atomic propositions reduces the noise in dense
                                                           • Conjunctions (A ∧ B): Splitting into A or B is
retrieval (Chen et al., 2024); in Natural Language
                                                             safe (I(A ∧ B) > I(A)).
Inference (NLI), context augmentation by atomic
propositions increases performance (Stacey et al.,         • Disjunctions (A ∨ B): Splitting into A is bad
2023); in fact-checking, it is possible to decompose         (I(A) > I(A ∨ B)), as it "hallucinates" speci-
the text and verify each atomic proposition recur-           ficity.
sively (Min et al., 2023); and finally, in Summary
Evaluation, this method allows one to approach           Thus, our propositioner is designed to split conjunc-
human judgement (Herserant and Guigue, 2025).            tions while preserving the integrity of disjunctive
    Recently, (Min et al., 2023) proposed the use of     facts.
LLM-based atomic propositions in NLP. The atomic
propositions approach combines cutting the infor-        3.3.   BCP Paradox
mation into small sentence pieces (the smallest we
can, without loss of information, see Section 3) and     The BCP paradox states that a contradiction (like
coreference resolution. One advantage of atomic          A ∧ ¬A) has infinite information in the extended
propositions is the structure. This approach gives       reals R = R ∪ {−∞, +∞}:
a fixed structure to the information we want to ver-
ify, which is more easily parseable and has been                       |Cont(A ∧ ¬A)|
used for Information Retrieval (Chen et al., 2024)          − log2 (                  ) = − log2 (0) = +∞
                                                                            |W |
and Summary Evaluation (Herserant and Guigue,
2025).                                                      However, for clauses of a CNF, this paradox can-
    Based on these recent works, we propose using        not arise because they cannot be contradictions
atomic propositions to construct knowledge graphs.       (Cori and Lascar, 2003). Therefore, atomic propo-
We propose using a propositioner based on a dis-         sitions cannot have infinite information.
tilled multilingual language model to perform the           In the current implementation, we expect the
atomic propositions extraction. This model handles       model to follow the previous formal framework.
simplification and coreference resolution across six     However, we have only performed a manual and
languages and is grounded in a formal framework.         partial evaluation. If this is positive, we cannot
Once the propositions are extracted, the text input      guarantee that this will be the case for all extracted
is flattened. Then, we experiment with different         propositions. Additional experiments will be con-
methods to extract entity-relation triplets. For each    ducted outside the scope of this work.
triplet, we produce a graph that links the entities
with the relations produced by the model.
                                                                   4.     Proposed Approach
           3.   Formal Framework                         The aim of this paper is to show that triplet extrac-
                                                         tion could benefit from atomic propositions. We
The abstraction underlying our objectives is the         define different stages to extract triplets, with a full
theoretical atomic proposition. To enlighten the         pipeline depicted in Figure 1. The global pipeline
atomization process, we use the formalism of Se-         consists of three stages:
mantic Information Theory (Bar-Hillel and Carnap,
1953). This formalism provides a strong under-            1. Atomization: The complex text is processed
standing of information in terms of signification and        by MPropositionneur-V2. This model, dis-
gives a criterion for cutting a proposition in a way         tilled from Qwen3-32B into a Qwen3-0.6B ar-
that preserves information.                                  chitecture, recursively splits the text until each
                                                             proposition is stable and autonomous by using
                                                             the prompt proposed in Figure 2.
3.1.   Information Content
Let ϕ be a formula.          We define I(ϕ) =               3
                                                              A world W is a determined assignment of truth val-
− log2 ( |Cont(ϕ)|
           |W |    ), where Cont(ϕ) is the set of        ues for each atomic subformula of ϕ.
  2. LLM prompting: We use the Qwen3-4B                     Extract all factual
     model to generate triplets directly from an at-        (subject, predicate, object)
                                                            triples from the sentence.
     omized chunk using a dedicated prompt (Fig-
                                                            One triple per line in the format:
     ure 3)                                                 subject | predicate | object
                                                            No explanations. If no triple can be
  3. KG Building: Extracted triplets are aggre-             extracted, write nothing.
     gated into a Knowledge Graph, where nodes
     represent entities and edges represent rela-           Sentence: text
     tions.
                                                         Figure 3: Prompt Template used for the stage 2.
   As a baseline, we replace stage 2 with two sub-
stages by using the Parsing and Triplet Extrac-             Input: "Marie Curie, a Polish-born
tion as follows:                                            physicist, won the Nobel Prize in
                                                            Physics."
2.1. Parsing: Each atomic proposition is parsed
                                                            Atomic Props: ["Marie Curie is a
     (here using SpaCy or Stanza) to extract part-          physicist.", "Marie Curie was born in
     of-speech (POS) tags, named entity tags, and           Poland.", "Marie Curie won the Nobel
     dependency trees.                                      Prize in Physics."]

2.2. Triplet Extraction: A large language model             Parsed Triplets: (Marie Curie,
                                                            occupation, physicist), (Marie Curie,
     or a neural based pattern matcher (for exam-
                                                            birthplace, Poland), (Marie Curie,
     ple GLiREL) extracts triplet candidates (s, r, o)      award, Nobel Prize in Physics).
     from the simplified, parsed atoms.
                                                         Figure 4: Example of a sentence input processed
   You are an expert in disambiguation                   through the whole pipeline, the atomic output, and
   and information extraction.                           the triplets extracted.
   You must decompose the text into
   atomic propositions (single facts)                    2015), using Qwen3-32B as the teacher model
   that are FULLY AUTONOMOUS.
   ABSOLUTE RULES:                                       and Qwen3-0.6B as the student. The training
   1. ZERO PRONOUNS: "He", "She", "They",                data consist of chunks of Wikipedia articles in six
   "His", "Her", "Its", "This one"                       European languages: English, French, Spanish,
   ARE FORBIDDEN.                                        Italian, German, and Portuguese (Foundation). We
   ALWAYS replace them with the                          trained the models for 2 epochs, on an A6000
   full name of the entity.
   2. CONTEXT: Each sentence must be                     NVIDIA GPU.
   readable alone without knowing
   its source.                                           5.2.   Nat. Lang. Recursive Propositioner
   3. REPETITION: Repeat the subject
   in EACH sentence.                                     In Algorithm 1, we describe the recursive propo-
   OUTPUT FORMAT: Only a JSON array of                   sitioner method. This algorithm is designed to re-
   strings.                                              cursively apply the MPropositionneur-V2 to all
   Title: title                                          natural language propositions generated (Pi ) until
   Content: content                                      either all have been proved to be a propositioner
   Output:                                               fixed-point or the recursion depth has reached an
                                                         empirical threshold value (here N = 5). In the
Figure 2: Prompt Template used for the distillation      latter case, we return only the subset of proved
of the propositioner used in stage 1.                    atomic propositions (Ai ).

  Figure 4 illustrates the input and the output of       Algorithm 1 propositioner
the entire pipeline.
                                                         Require: N = 5, t is a text, i ∈ N, M is the propo-
                                                             sitioner,
        5.    Experimental Protocol                       1: i ← 0; P0 ← M(t);
                                                          2: A0 ← {p ∈ P0 , {p} = M(p)}
5.1.   Propositioner                                      3: while (Pi ̸=SAi ) ∧ (i < N ) do
                                                          4:      Pi+1 ← x∈Pi M(x)
We train a propositioner4 , i.e. a model that trans-
                                                          5:      Ai+1 ← {x ∈ Pi+1 , {x} = M(x)}
forms a text input into a list of atomic propo-
                                                          6:      i←i+1
sitions, via knowledge distillation (Hinton et al.,
                                                          7: end while
  4
                                                          8: return Ai
    Available here : https://huggingface.co/
Zual/MPropositionneur-V2
5.3.   Datasets and Benchmark                               The baseline approach is a triplet extractor us-
                                                         ing GLiREL. Then we compare the baseline with
We evaluate our pipeline on the SMiLER dataset
                                                         prompted approaches using LLMs, where mod-
(Seganti et al., 2021) (a multi-domain relation ex-
                                                         els are queried to generate the triplets from either
traction benchmark covering 14 languages and var-
                                                         the source text (direct) or from propositions (prop).
ious relation types) where the entities are already
                                                         For prompt models, we selected Qwen3-0.6B and
known and the task is to find the relations between
                                                         Qwen3-4B instruct models; these models are com-
entity pairs, on FewRel dataset (Han et al., 2018),
                                                         parable in terms of size (number of weights) to
on DocRED (Yao et al., 2019) (a Document-based
                                                         the size of the propositioner model. It should be
triplet extraction dataset) and on CaRB (Bhardwaj
                                                         noted that these models were not fine-tuned for the
et al., 2019) (an openIE dataset for triplet extrac-
                                                         triplet extraction task. Rather, a zero-shot instruc-
tion).
                                                         tion approach was employed (see the prompt used
                                                         in Figure 3).
5.4.   Metrics
To evaluate triplet extraction performances we will               6.    Results and Analysis
consider the following metrics:
                                                         In this section we report and discuss the results
   • Precision (P), Recall (R), and F1-Score. Area       of the designed experiments. We evaluate quan-
     Under the Curve (AUC) is also used for the          titatively the propositioner for the triplet extraction
     CaRB benchmark natively.                            method. By flattening the text, relations are made
   • Entity Recall: The percentage of gold entities      explicit, allowing the extractors to capture facts that
     found in the output (exact match with substring     would otherwise be missed in complex sentences.
     and macrostring accepted).
                                                         6.1.   Evaluation on Multilingual Triplet
   • Relation Recall: The accuracy of extracted                 Extraction
     triplets vs. gold standards, by mapping to the
     finite vocabulary of GLiREL using a semantic        We report in Table 1 results in accuracy (number of
     mapper (cosine similarity with BERT, threshold      triplet correctly extracted), entity-recall and relation-
     optimised on the dev set).                          recall on both SMiLER and FewRel benchmarks
                                                         compared to GLiREL approach.
                                                            We also report results for the different configu-
5.5.   Baselines & Configurations                        rations: "Direct", where models try to extract the
                                                         triplet directly from the raw text; "Prop", where the
We compare the performance of direct pipeline
                                                         models are only considering the set of propositions
(GLiREL or Qwen3) vs. MPropositionneur-V2
                                                         to extract relations; "Comb", which is the combina-
+ direct pipeline. Our hypothesis is that atomization
                                                         tion of raw text and the list of atomic propositions.
reduces the syntactic noise that typically hinders
                                                            First, looking at the Macro-avg, we can observe
relation extraction on complex sentences.
                                                         that in almost all cases, the number of correctly
   For the evaluation, we analyse three different
                                                         extracted relations benefits from the atomic propo-
configurations:
                                                         sitions. While direct pipelines achieve better ac-
   • Direct: Triplets are directly extracted from        curacy and entity recall, a combination of raw text
     source text.                                        and atomic propositions yields better performance
                                                         for small models, demonstrating that both methods
   • Prop: Triplets are extracted only from the          support the retrieval of different entities or rela-
     atoms produced by the propositioner.                tions. This combination of the two approaches has
                                                         a positive impact on the proposition in the triplet
   • Comb: For SMiLER and FewRel benchmarks,             extraction pipelines.
     since the entity oracle is known, if entities are      However, within this pipeline we consider that
     found, the associated triples are saved; oth-       we know a relation exists between two entities, and
     erwise, the triples are extracted from atomic       thus it corresponds to relations classification set-
     propositions. Comb is therefore a fallback          ting rather than a triplet extraction setting. From
     method.                                             a model perspective, we show that a larger LLM-
   • Union: The triplets are extracted from              based approach is more powerful across all met-
     atomic propositions and from the raw para-          rics (Qwen3-4B), although it comes at a higher
     graph/document independently. Triplets from         computational and memory cost.
     both the direct and prop pipelines are merged.
     The objective is to evaluate the contribution of    Evaluation on triplet extraction on documents.
     atomic propositions to the direct pipeline.         Table 2 presents results that favour proposition
                                                  GLiREL                 Qwen3-0.6B              Qwen3-4B
   Benchmark        Pipeline              Acc      e-rec   r-rec   Acc     e-rec   r-rec   Acc     e-rec    r-rec
                    English – Direct      49.8     99.0    50.3    35.7    100.0   35.7    71.4    100.0    71.4
                    English – Prop        43.7     86.6    50.4    35.1     86.6   40.5    59.6     86.6    68.8
                    English – Comb        51.5†    99.1    51.9     –        –      –       –        –       –
                    French – Direct       65.2     99.8    65.3    32.2    100.0   32.2    81.8    100.0    81.8
                    French – Prop         48.9     76.7    63.8    39.7     76.7   51.7    64.6     76.7    84.3
                    French – Comb         66.3     99.8    66.4     –        –      –       –        –       –

   SMiLER
                    German – Direct       59.2     99.6    59.4    22.9    100.0   22.9    76.5    100.0    76.5
                    German – Prop         49.2     85.1    57.8    46.3     85.1   54.5    68.7     85.1    80.8
                    German – Comb         59.4     99.7    59.5     –        –      –       –        –       –
                    Spanish – Direct      46.5     99.6    46.7    24.8    100.0   24.8    65.0    100.0    65.0
                    Spanish – Prop        38.5     79.2    48.6    27.4    79.2    34.6    58.9    79.2     74.3
                    Spanish – Comb        46.5     99.6    46.7     –        –      –       –        –       –
                    Portuguese – Direct   59.3     99.4    59.7    32.8    100.0   32.8    77.6    100.0    77.6
                    Portuguese – Prop     56.4     87.7    64.3    43.2     87.7   49.2    71.9     87.7    82.0
                    Portuguese – Comb     63.2†    99.4    63.5     –        –      –       –        –       –
                    Italian – Direct      63.9     99.4    64.3    36.2    100.0   36.2    81.8    100.0    81.8
                    Italian – Prop        54.1     82.2    65.8    41.8     82.2   50.8    69.5     82.2    84.5
                    Italian – Comb        67.7†    99.6    68.0     –        –      –       –        –       –
                    Macro-avg – Direct    57.3     99.5    57.6    30.8    100.0   30.8    75.7    100.0    75.7
                    Macro-avg – Prop      48.5     82.9    58.5    38.9     82.9   46.9    65.5    82.9     79.1
                    Macro-avg – Comb      59.1†    99.5    59.3     –        –      –       –        –       –
                    Direct                48.7     100.0   48.7    42.2    100.0   42.2    67.7    100.0    67.7
   FewRel
                    Prop                  40.2      75.8   53.1    40.3    75.8    53.2    51.5    75.8     68.0
                    Comb                  50.0†    100.0   50.0     –        –      –       –        –       –

Table 1: Results on the SMiLER and FewRel benchmarks. † indicates a statistically significant improve-
ment of Comb over Direct (bootstrap test, p < 0.05). Atomization significantly improves relation recall.
But entity recall decreases. The accuracy increases for the small LLM (Qwen3-0.6B). Bold indicates the
best result per extractor column for each benchmark/language.


granularity for GLiREL, which achieves a higher             present in the input text are absent from the knowl-
F1 score when using the propositioner. However              edge graph built from the triplets.
larger models (Qwen3-4B) reach higher scores                   Due to the atomic splitting of information during
(precision, recall and F1). Thus, for larger docu-          intermediary representation translation, such rela-
ments, larger models are preferable to the proposi-         tions may become latent. However, it is important
tioner. We hypothesize that this difference in score        to note that these propositions can nevertheless
could be reduced by considering the constructed             be recovered through the use of inference, which
graph and leveraging relation transitivity. Deeper          is supported by the formal properties of the atomic
experiments should be conducted to verify this as-          propositions.
sumption.                                                      For instance, when the following sentence is
                                                            entered:
Evaluation on triplet extraction on openIE con-                "Šafov is a village and
text. As shown in Table 3, we observe that us-              municipality (obec) in Znojmo
ing the propositioner upstream of CoreNLP (Prop             District in the South Moravian
method) improves recall and AUC, but not preci-             Region of the Czech Republic.",
sion. This drop in precision can be explained by
                                                               the propositioner’s output is as follows:
the increased number of triplets extracted by the
atomic propositions method. We observe that in                 [..., "Šafov is located in Znojmo
OpenIE context, the union of propositions and origi-        District.", "Znojmo District is
nal text is optimal for recall and AUC, having higher       located in the South Moravian
Recall and AUC for all pipelines.                           Region.", "The South Moravian Region
                                                            is located in the Czech Republic.",
                                                            ...]
6.2.        Qualitative Results
Even with no errors in the pipeline components,                 The corresponding graph, displayed in Figure 5,
it remains possible that some transitive relations          illustrates the aforementioned transitive relation.
                                               GLiREL                       Qwen3-0.6B                     Qwen3-4B
               Pipeline                  P       R          F1          P          R      F1           P         R     F1
               Sentence                  3.7    9.4        5.3        22.6     8.3       12.1      36.6      13.3     19.5
               Document                  1.4    14.8       2.5        20.3     15.7      17.7      32.0      24.4     27.7
               Proposition               5.9     6.9       6.3        25.9      6.7      10.6      36.5       9.4     14.9
               Sentence+Proposition      3.8    12.3       5.8         –         –        –         –          –       –
               Document+Proposition      1.6    18.1       2.9         –         –        –         –          –       –

Table 2: Results on the DocRED benchmark. Atomization significantly improves precision and F1 when
using a weaker extractor. However, when using LLM extraction, performance decreases.

                            CoreNLP                              Qwen3-0.6B                                 Qwen3-4B
      Pipeline      P       R     F1     AUC           P          R          F1        AUC         P         R       F1      AUC
      Direct       17.6    24.3   20.4   0.144       29.4         7.9       12.4       0.051      46.2      33.3     38.7    0.244
      Prop         16.4    31.3   21.5   0.182       24.0        11.1       15.2       0.069      31.6      35.0     33.2    0.230
      Union         –       –      –       –         23.7        14.7       18.1       0.091      30.8      43.6     36.1    0.285

Table 3: Results on the CaRB benchmark. Atomization significantly improves recall and F1 when a
weaker extractor is used. However, when using LLM extraction, performance decreases.


Precisely, while triplet (“Šafov”, “hasLocation”,                     ation of the propositioner, to ensure that text units
“Czech Republic”) is not directly extracted from                      are indeed all atomic according to the definition
propositions, we could deduce it by taking into                       (see Section 3).
account transitivity.                                                    Additionally, we think that such triplet extraction
                                                                      methods and the constructed graph could be used
                                                                      in applications such as information retrieval sys-
                                                                      tems.
                                                                         We think that the interpretability of such text units
                                                                      (atoms) could provide a statistical basis for creating
                                                                      or deducing a logical rule, which could benefit the
                                                                      KG traversal algorithm. These research leads are
                                                                      left for further work.

                                                                                             8.     Limitations
Figure 5: The Knowledge Graph built with triplets
extracted by GLiREL on the atomic propositions                        One limitation of this study is the relevance of the
for the above sentence.                                               recursive propositioner. To date, we have not com-
                                                                      pared propositions obtained with and without re-
                                                                      cursive refinement. The decision to use such an
                  7.      Conclusion                                  algorithm to refine atoms recursively was based
                                                                      on preliminary experiments in which we observed
In this work, we empirically demonstrate the ben-                     that some propositions were not atomic. In future
efits of the atomic proposition for triplet entity re-                works, we plan to compare the recursive proposi-
lation extraction. In particular, we show that de-                    tioner to the non-recursive one.
composing documents or paragraphs into atoms                             Another limitation is the absence of evaluation
helps retrieve the relation efficiently, improving per-               using larger LLMs than Qwen3-4B for fair compar-
formance on both the FewRel and SMiLER bench-                         ison and computational efficiency. Nonetheless,
marks.                                                                it is worth noticing that a larger model could be
   In addition, we observe that the combination of                    compared in future studies.
original input text with atomised text yields bet-
ter performance across all metrics. The results
obtained on both the CaRB and DocRED bench-
marks validate the previous hypothesis, showing
better recall performance.
   However, future studies should be conducted to
strengthen the methods, especially a human evalu-
     9.   Bibliographical References                    Oktie Hassanzadeh. 2021. Building a knowledge
                                                         graph of events and consequences using wiki-
                                                         data. Wikidata@ ISWC, 2982.

Yehoshua Bar-Hillel and Rudolf Carnap. 1953. Se-        Tanguy Herserant and Vincent Guigue. 2025.
  mantic information. The British Journal for the         Seval-ex : Un paradigme basé sur les phrases
  Philosophy of Science, 4(14):147–157.                   atomiques pour une évaluation explicable de la
                                                          qualité des résumés. In Actes de la 20e Con-
Sangnie Bhardwaj, Samarth Aggarwal, and                   férence en Recherche d’Information et Applica-
  Mausam. 2019. CaRB: A crowdsourced bench-               tions, CORIA 2025, Marseille, France, Juin 30 -
  mark for open IE. In Proceedings of the 2019            Juillet 4, 2025, pages 217–229. ATALA&ARIA.
  Conference on Empirical Methods in Natural
  Language Processing and the 9th International         Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean.
  Joint Conference on Natural Language Process-          2015. Distilling the knowledge in a neural net-
  ing (EMNLP-IJCNLP), pages 6262–6267, Hong              work. ArXiv, abs/1503.02531.
  Kong, China. Association for Computational Lin-
                                                        Aidan Hogan, Eva Blomqvist, Michael Cochez,
  guistics.
                                                          Claudia D’amato, Gerard De Melo, Claudio
                                                          Gutierrez, Sabrina Kirrane, José Emilio Labra
Jack     Boylan,      Chris     Hokamp,         and
                                                          Gayo, Roberto Navigli, Sebastian Neumaier,
  Demian Gholipour Ghalandari. 2025. GLiREL -
                                                          Axel-Cyrille Ngonga Ngomo, Axel Polleres, Sab-
  generalist model for zero-shot relation extraction.
                                                          bir M. Rashid, Anisa Rula, Lukas Schmelzeisen,
  In Proceedings of the 2025 Conference of
                                                          Juan Sequeda, Steffen Staab, and Antoine Zim-
  the Nations of the Americas Chapter of the
                                                          mermann. 2021. Knowledge graphs. ACM Com-
  Association for Computational Linguistics:
                                                          put. Surv., 54(4).
  Human Language Technologies (Volume 1:
  Long Papers), pages 8230–8245, Albuquerque,           Pere-Lluís Huguet Cabot, Simone Tedeschi, Axel-
  New Mexico. Association for Computational               Cyrille Ngonga Ngomo, and Roberto Navigli.
  Linguistics.                                            2023. Redfm : a filtered and multilingual relation
                                                          extraction dataset. In Proc. of the 61st Annual
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao               Meeting of the Association for Computational
  Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang,             Linguistics: ACL 2023, Toronto, Canada. Associ-
  and Dong Yu. 2024. Dense X retrieval: What              ation for Computational Linguistics.
  retrieval granularity should we use? In Pro-
  ceedings of the 2024 Conference on Empirical          Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike
  Methods in Natural Language Processing, pages           Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke
  15159–15177, Miami, Florida, USA. Association           Zettlemoyer, and Hannaneh Hajishirzi. 2023.
  for Computational Linguistics.                          FActScore: Fine-grained atomic evaluation of
                                                          factual precision in long form text generation. In
René Cori and Daniel Lascar. 2003. Logique math-          Proceedings of the 2023 Conference on Empir-
  ématique : cours et exercices corrigés, Tome 1          ical Methods in Natural Language Processing,
 - Calcul propositionnel, algèbres de Boole, cal-         pages 12076–12100, Singapore. Association for
  cul des prédicats. Sciences Sup. Dunod, Paris.          Computational Linguistics.
 Avec la collaboration de Jean-Louis Krivine.
                                                        Makoto Miwa, Rune Sætre, Yusuke Miyao, and
Oren Etzioni, Michele Banko, Stephen Soder-              Jun’ichi Tsujii. 2010. Entity-focused sentence
  land, and Daniel S. Weld. 2008. Open infor-            simplification for relation extraction. In Proceed-
  mation extraction from the web. Commun. ACM,           ings of the 23rd International Conference on
  51(12):68–74.                                          Computational Linguistics (Coling 2010), pages
                                                         788–796, Beijing, China. Coling 2010 Organiz-
Wikimedia Foundation. Wikimedia downloads.               ing Committee.

Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan           Christina Niklaus,      Bernhard Bermeitinger,
  Yao, Zhiyuan Liu, and Maosong Sun. 2018.               Siegfried Handschuh, and André Freitas. 2016.
  FewRel: A large-scale supervised few-shot re-          A sentence simplification system for improving
  lation classification dataset with state-of-the-art    relation extraction. In Proceedings of COLING
  evaluation. In Proceedings of the 2018 Confer-         2016, the 26th International Conference on
  ence on Empirical Methods in Natural Language          Computational Linguistics: System Demon-
  Processing, pages 4803–4809, Brussels, Bel-            strations, pages 170–174, Osaka, Japan. The
  gium. Association for Computational Linguistics.       COLING 2016 Organizing Committee.
Alessandro Seganti, Klaudia Firl ˛   ag, Helena          Open IE (CaRB). The following prompt is used
  Skowronska, Michał Satława, and Piotr An-              for open-domain triplet extraction:
  druszkiewicz. 2021. Multilingual entity and rela-      Extract all factual
  tion extraction dataset and model. In Proceed-         (subject, predicate, object)
  ings of the 16th Conference of the European            triples from the sentence.
  Chapter of the Association for Computational           One triple per line in the format:
  Linguistics: Main Volume, pages 1946–1955,             subject | predicate | object
  Online. Association for Computational Linguis-         No explanations. If no triple can be
  tics.                                                  extracted, write nothing.

Joe Stacey, Pasquale Minervini, Haim Du-                 Sentence: {text}
  bossarsky, Oana-Maria Camburu, and Marek
  Rei. 2023. Atomic inference for nli with gener-
                                                         A.2.   Propositioner Training Prompt
  ated facts as atoms. In Conference on Empirical
  Methods in Natural Language Processing.                The following prompt is used during the knowledge
                                                         distillation training of MPropositionneur-V2.
Andra Waagmeester, Gregory Stupp, Sebastian              The teacher model (Qwen3-32B) is prompted
  Burgstaller-Muehlbacher, Benjamin M Good,              to decompose a Wikipedia passage into fully
  Malachi Griffith, Obi L Griffith, Kristina Hanspers,   autonomous atomic propositions, which are
  Henning Hermjakob, Toby S Hudson, Kevin Hy-            then used as targets for the student model
  biske, et al. 2020. Wikidata as a knowledge            (Qwen3-0.6B).
  graph for the life sciences. Elife, 9:e52614.          You are an expert in disambiguation
                                                         and information extraction.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang,
                                                         You must decompose the text into
  Shengyuan Chen, Zijin Hong, Xiao Huang, and
                                                         atomic propositions (single facts)
  Jinsong Su. 2026. When to use graphs in rag:
                                                         that are FULLY AUTONOMOUS.
  A comprehensive analysis for graph retrieval-
  augmented generation.
                                                         ABSOLUTE RULES:
                                                         1. ZERO PRONOUNS: "He", "She", "They",
Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai
                                                            "His", "Her", "Its", "This one"
  Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang,
                                                            ARE FORBIDDEN.
  Jie Zhou, and Maosong Sun. 2019. DocRED:
                                                            ALWAYS replace them with the
  A large-scale document-level relation extraction
                                                            full name of the entity.
  dataset. In Proceedings of the 57th Annual Meet-
                                                         2. CONTEXT: Each sentence must be
  ing of the Association for Computational Linguis-
                                                            readable alone without knowing
  tics, pages 764–777, Florence, Italy. Association
                                                            its source.
  for Computational Linguistics.
                                                         3. REPETITION: Repeat the subject
                                                            in EACH sentence.

                 A.    Prompts                           OUTPUT FORMAT: Only a JSON array of
                                                         strings.
A.1.   Evaluation Prompts
                                                         Title: {title}
Closed IE (SMiLER, FewRel, DocRED). The                  Content: {content}
following prompt is used to classify the relation        Output:
between two identified entities:
Given the text, identify the relation                     B.    Proofs for the Formal Grounding
between the two entities.
                                                         Definition 1 (Safe Cut). A formula’s cut ϕ in a
Text: {text}                                             formula ψ is safe if ψ is a sub-formula of ϕ and if
Entity 1: {e1}                                           I(ϕ) > I(ψ)
Entity 2: {e2}                                           Definition 2 (Bad cut). A cut of a formula ϕ in a
                                                         formula ψ is bad if ψ is a sub-formula of ϕ and if
Choose exactly one relation from                         I(ϕ) ≤ I(ψ)
this list: {labels}                                      Lemma 1 (Divisibility of Conjunction — Lemma ).
                                                         Let ϕ = A ∧ B where A and B are logically inde-
Answer with just the relation name,                      pendent. Extracting the component A is a strictly
nothing else.                                            safe operation.
Proof. By definition of conjunction, Cont(A ∧ B) =
Cont(A) ∩ Cont(B). Because A and B are inde-
pendent, Cont(A ∧ B) ⊊ Cont(A). By monotonicity
of µ, we have µ(Cont(A ∧ B)) < µ(Cont(A)). The
function − log2 is strictly decreasing, so:

                 I(A ∧ B) > I(A).

The information content of A is strictly less than
that of A ∧ B : the operation is strictly safe.
Lemma 2 (Indivisibility of Disjunction — Lemma ).
Let ϕ = A ∨ B where A and B are logically in-
dependent. Extracting the component A is a bad
operation.
Proof. By definition of disjunction, Cont(A ∨ B) =
Cont(A) ∪ Cont(B). We have the strict inclusion
Cont(A) ⊊ Cont(A ∨ B), hence µ(Cont(A)) <
µ(Cont(A ∨ B)), which yields:

                 I(A) > I(A ∨ B).

The information content of A is strictly greater than
that of A ∨ B: decomposing a disjunction halluci-
nates a more specific fact than what was originally
stated.

Case of implication . Cutting A → B into A is
bad, because A → B ≡ ¬A ∨ B, and Lemma 2
applies directly by substitution.
Theorem 1 (Structural characterisation — Theo-
rem). A formula ϕ is atomic if and only if it is
logically equivalent to a clause (finite disjunction
of literals).
Proof. Let ϕ be in Conjunctive Normal Form (CNF)
:
             ϕ ≡ C1 ∧ C2 ∧ · · · ∧ Cn ,
where each Ci is a clause (disjunction of literals).
  (⇒) Let ϕ be atomic. Ad absurdum, let n ≥ 2.
By Lemma 1, the operation ϕ 7→ C1 is strictly safe,
which contradicts atomicity. Therefore, n = 1: ϕ is
a single clause.
  (⇐) Let ϕ = L1 ∨ · · · ∨ Lk . By Lemma 2 (gen-
eralised by induction on k), all strict sub-formulae
have greater information content than ϕ. Therefore,
every cut is bad, and ϕ atomic.
