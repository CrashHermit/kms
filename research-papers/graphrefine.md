                        LLMs as Knowledge Graph Refiners:
        Mitigating Factual Inconsistencies in Generative Knowledge Extraction

   Donghyun Kim1              Hyeongjun Yang1 Seokju Hwang1 Kyong-Ho Lee1 *                                 Chanhee Lee2
                               1
                                 Department of Computer Science, Yonsei University
                                               2
                                                 Samsung Securities
                             {dhkim92,edbm95,hsjtjrwn,khlee89}@yonsei.ac.kr
                                         chanhee88.lee@samsung.com


                            Abstract                                Among these LLM-driven approaches, a represen-
                                                                    tative paradigm is generative knowledge extraction
         Knowledge graphs (KGs) provide a structured
         representation of real-world facts as triples con-
                                                                    (GKE), which constructs a KG by directly generat-
         sisting of entities and their relationships. With          ing structured triples from natural language docu-
         the rapid progress of large language models                ments, rather than progressively identifying entities
         (LLMs), recent studies increasingly explore                and relations through a multi-stage pipeline (Zhang
         LLMs for end-to-end KG construction from                   et al., 2025). This paradigm offers strong flexibility
         text. In particular, generative knowledge extrac-          and scalability, substantially reducing the cost of
         tion (GKE) builds KGs by directly generating               KG construction across diverse scenarios.
         structured triples from documents. However,
                                                                       However, errors during generation are inevitable,
         generation errors are inevitable, and the result-
         ing KGs often contain triples that do not align            and consequently, the resulting KGs often contain
         with the facts expressed in the source text. To            triples that do not align with the real-world facts
         address these issues, we propose GraphRefine,              expressed in the source documents (Wang et al.,
         a framework that performs triple-level refine-             2021; Xue and Zou, 2022). In this work, we refer
         ment on KGs constructed via GKE. We first an-              to such cases as factual inconsistencies in KGs.
         alyze factual inconsistencies that arise in GKE            Existing studies on KG refinement and validation
         and categorize their types based on a human
                                                                    have largely focused on identifying and filtering in-
         evaluation. We then construct training data re-
         flecting these types and fine-tune an LLM as a
                                                                    correct triples using rule-based heuristics, schema
         KG refiner. Given a draft KG, the fine-tuned               constraints, or KG embedding methods (Paulheim,
         refiner selects a refinement operation for each            2016; Huaman et al., 2020). However, KGs con-
         triple and, if needed, deletes, edits, or rewrites it      structed via GKE exhibit a broader and often am-
         to reduce factual inconsistencies. Extensive ex-           plified range of errors, stemming from the genera-
         periments demonstrate that GraphRefine goes                tive nature of LLM-based extraction (Zhang et al.,
         beyond deletion-only approaches and improves               2024; Kamoi et al., 2024). These errors include
         KG quality from diverse perspectives.
                                                                    entity recognition failures, relational semantic dis-
    1    Introduction                                               tortions, unsupported hallucinations, and represen-
                                                                    tation inconsistencies that reduce the clarity and
    Knowledge graphs (KGs) represent real-world en-                 usability of extracted triples. Together, these error
    tities and their relations in a structured form, serv-          patterns differ fundamentally from the assumptions
    ing as a key resource for knowledge-driven down-                of prior refinement methods, making them difficult
    stream tasks such as question answering, recom-                 to be fully addressed in practice. Accordingly, mit-
    mendation, and fact-checking (Huang et al., 2019;               igating these inconsistencies requires careful anal-
    Yang et al., 2022; Kim et al., 2023). Traditionally,            ysis of their characteristics and underlying causes
    KGs have been constructed through information                   to facilitate effective refinement for GKE.
    extraction pipelines (Hogan et al., 2021). More re-                In this context, to improve the correctness of
    cently, advances in large language models (LLMs)                LLM-based KG construction, GraphJudge (Huang
    have spurred increasing efforts to automate end-                et al., 2025) employs an LLM as a judge to as-
    to-end KG construction using LLMs (Han et al.,                  sess the validity of triples and delete incorrect ones.
    2024; Chen et al., 2024; Zhang and Soh, 2024; Niu               While this approach can eliminate some factual in-
    et al., 2025; Lu and Wang, 2025; Mo et al., 2025).              consistencies in GKE, such as unsupported triples,
        * Corresponding Author                                      refinement by deletion has fundamental limitations.
                                                                 29358
Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 29358–29378
                                 July 2-7, 2026 ©2026 Association for Computational Linguistics
First, deleting triples may discard partially correct      2   Background & Related Work
facts that could otherwise be corrected through tar-
geted edits, thereby unnecessarily reducing KG             KG construction involves identifying entities and
coverage. Second, such inconsistencies in GKE              relations from raw text and representing them as
are not always captured by a binary judgment of            structured triples. Traditional approaches have typ-
whether a triple is valid or invalid. In particular, be-   ically relied on modular pipelines, which decom-
yond correctness-level errors, GKE also produces           pose this process into subtasks such as named entity
representation-level inconsistencies, such as impre-       recognition, relation extraction, and event extrac-
cise entity spans, that degrade triple clarity and us-     tion (Hogan et al., 2021). Recent progress in LLMs
ability without necessarily making them incorrect.         has introduced a generative paradigm that replaces
Therefore, correctness-only filtering is insufficient      such procedural pipelines with end-to-end gener-
to improve overall KG quality, motivating refine-          ation (Wan et al., 2023; Wang et al., 2025). In
ment methods that edit and normalize triples rather        particular, generative knowledge extraction (GKE)
than simply removing them.                                 leverages the contextual reasoning and language
   In this paper, we propose GraphRefine, a post-          generation capabilities of LLMs to directly gener-
hoc refinement framework for KGs constructed via           ate structured knowledge from natural language,
GKE. Given a draft KG and its source document,             enabling a more holistic and flexible approach to
GraphRefine selects a document-grounded refine-            KG construction that addresses the limitations of
ment operation for each triple and edits or rewrites       conventional methods (Zhang et al., 2025).
it when needed to mitigate factual inconsistencies.           Building on this generative paradigm, a grow-
To this end, we first analyze and systematically           ing body of work has proposed advanced frame-
define factual inconsistency types that frequently         works for KG construction. SAC-KG (Chen et al.,
arise in GKE settings, and then train an LLM as a          2024) adopts an LLM-driven framework for auto-
KG refiner to perform triple-level refinement op-          mated domain-specific KG construction. Its gen-
erations. In particular, we fine-tune the refiner on       erator retrieves relevant information from domain
training data constructed to cover diverse incon-          corpora and DBpedia to compose model inputs,
sistency types, enabling it to learn a principled re-      while the verifier performs rule-based correction
finement strategy that goes beyond simple filtering        and the pruner removes irrelevant entities using a
and supports deletion, editing, and rewriting. We          classification model. EDC (Zhang and Soh, 2024)
also evaluate KG quality beyond correctness alone          proposes a framework that decomposes KG con-
using multi-faceted metrics such as GenRES (Jiang          struction into extraction, definition, and canoni-
et al., 2024), measuring not only factual accuracy         calization to reduce schema dependence and re-
but also representation quality. Our experiments           dundancy. This approach utilizes LLM-generated
show that GraphRefine improves KG quality across           definitions and similarity-based canonicalization to
different dimensions by reducing factual inconsis-         construct structured knowledge without predefined
tencies and enhancing triple clarity and consistency,      ontologies. Tree-KG (Niu et al., 2025) constructs
rather than merely removing incorrect triples.             an explicit tree-structured KG by leveraging the
   Our main contributions are as follows:                  hierarchical organization of knowledge-intensive
                                                           documents. It expands this structure using LLM-
   • We analyze and systematically define factual          based operators, enabling efficient integration and
     inconsistency types that arise in GKE, thereby        expansion of knowledge while preserving struc-
     formalizing the refinement problem for LLM-           tural consistency. KGGen (Mo et al., 2025) utilizes
     based KG construction.                                an LLM-driven pipeline that performs entity and
                                                           relation extraction, integration, and deduplication
   • We propose GraphRefine, a post-hoc refine-
                                                           to construct a KG from plain text. It combines em-
     ment method for KGs constructed via GKE,
                                                           bedding clustering with LLM-based normalization
     which fine-tunes an LLM as a KG refiner to
                                                           to minimize semantic redundancy and enhance the
     perform triple-level refinement operations.
                                                           density and coherence of the resulting KG.
   • We conduct multi-faceted evaluations under               However, GKE remains prone to errors, includ-
     diverse metrics and demonstrate GraphRe-              ing incorrect entity recognition, relation distortion,
     fine’s effectiveness through extensive experi-        and hallucination. These errors often lead to fac-
     ments across datasets and base extractors.            tually inconsistent triples that degrade the quality
                                                      29359
of the constructed KG while also being difficult                                            5QWTEG&QEWOGPV
to fix during extraction, making post-hoc valida-             ę7KH 6SDFH 0LUURU 0HPRULDO ZKLFK IRUPV SDUW RI WKH ODUJHU $VWURQDXWV 0HPRULDO
                                                              LV D 1DWLRQDO 0HPRULDO RQ WKH JURXQGV RI WKH -RKQ ) .HQQHG\ 6SDFH &HQWHU
                                                              9LVLWRU &RPSOH[ RQ 0HUULWW ,VODQG )ORULGD ,W LV OCKPVCKPGF D[ WKH $VWURQDXWV
tion and refinement essential. GraphJudge (Huang              0HPRULDO )RXQGDWLRQ ZKRVH RIILFHV DUH NQECVGF KP WKH 1$6$ &HQWHU IRU 6SDFH
                                                              (GXFDWLRQ QH[W GRRU WR WKH 9LVLWRU &RPSOH[Ě
et al., 2025) filters out noisy or hallucinated triples
by employing an LLM as a graph judge to assess                                   %QTTGEVPGUUNGXGN+PEQPUKUVGPE[
the correctness of extracted triples. Although such                                 >ě6SDFH0LUURU0HPRULDOĜěEXLOWLQĜěĜ@
                                                                     ȥ)DFWQRWVXSSRUWHGE\WKHVRXUFHGRFXPHQW=*CNNWEKPCVKQP?
methods can improve accuracy, deletion-only filter-               >ě6SDFH0LUURU0HPRULDOĜěOCKPVCKPGFD[Ĝě1$6$&HQWHUIRU6SDFH(GXFDWLRQĜ@

ing may ultimately reduce KG coverage. Moreover,                           ȥ2EMHFWLVLQFRUUHFWO\VHOHFWHG=/KUKFGPVKHKECVKQP?

factual inconsistencies may go beyond correctness-                             4GRTGUGPVCVKQPNGXGN+PEQPUKUVGPE[
level errors, limiting the effectiveness of filtering-                      >ěRIILFHVĜěNQECVGFKPĜě1$6$&HQWHUIRU6SDFH(GXFDWLRQĜ@

based approaches. To address these limitations, we                    ȥ6XEMHFWLVQRWQRUPDOL]HGWRDVSHFLILFQRXQ=#ODKIWKV[?
                                                                      >ě$VWURQDXWV0HPRULDO)RXQGDWLRQĜěNQECVKQPĜ
employ an LLM as a KG refiner to mitigate diverse                     ě1$6$&HQWHUIRU6SDFH(GXFDWLRQQH[WGRRUWRWKH9LVLWRU&RPSOH[Ĝ@

factual inconsistencies arising in GKE.                                   ȥ2EMHFWVSDQLVUHFRJQL]HGWRREURDGO\=8GTDQUKV[?



3   Problem Definition                                    Figure 1: Representative examples of factual inconsis-
                                                          tencies observed in generative knowledge extraction.
We first describe generative knowledge extraction
(GKE) for LLM-based KG construction and then
introduce the KG refinement task.                         Here, Fθ takes the source document D and the ini-
                                                          tial graph GD as inputs and enhances the factual
3.1 Generative Knowledge Extraction                       consistency of the KG by identifying and correcting
                                                          errors. This refinement mitigates factual inconsis-
We define GKE as the task of extracting a KG GD
                                                          tencies introduced during GKE, yielding a KG that
from a source document D with an LLM, where
                                                          more faithfully reflects the original content of the
the KG consists of structured triples τ = (h, r, t).
                                                          source document.
The constructed GD is defined as:
                                                          4       Method
        GD = {(h, r, t) | h, t ∈ E, r ∈ R},        (1)
                                                          In this section, we propose GraphRefine for LLM-
where E denotes the set of entities and R denotes         based KG refinement. We first categorize factual
the set of relations, respectively. During the extrac-    inconsistencies in GKE, and then describe a strat-
tion process, an LLM generates an output sequence         egy for fine-tuning an LLM as a KG refiner.
conditioned on the source document D. This pro-
cess can be modeled autoregressively as:                  4.1        Factual Inconsistencies in GKE
                                                          By analyzing KGs constructed by GKE, we iden-
             P (xi | x1 , x2 , ..., x<i , D),      (2)    tify two prominent types of factual inconsisten-
                                                          cies: (i) correctness-level and (ii) representation-
where xi represents the i-th token of the output se-      level. Moreover, these inconsistencies are observed
quence. From the generated output, a set of triples       across datasets and extraction models.1 In the fol-
[τ1 , τ2 , ...] is derived, yielding GD . The objective   lowing, we describe each inconsistency type and
of this task is to extract a KG that covers as much       discuss how it can be addressed in the context of
relevant factual information as possible while re-        KG refinement.
maining consistent with the facts contained in the
source document.                                          Correctness-level Inconsistencies. Correctness-
                                                          level inconsistencies occur when individual triples
3.2 Knowledge Graph Refinement                            are factually incorrect. As illustrated in Figure 1,
The KG GD constructed through GKE may contain             this category includes hallucinated triples that are
factual inconsistencies. The goal of KG refinement        not supported by the source document, as well as
is to transform GD into a refined graph ĜD that is       cases where the meanings of entities or relations are
more faithful to the underlying document D. To            distorted, leading to incorrect factual statements.
formalize this process, we introduce an LLM-based         Such inconsistencies directly undermine the fac-
refinement operator Fθ as follows:                        tual accuracy of the constructed KG and mislead
                                                              1
                                                               Detailed statistics and analyses of factual inconsistencies
                 ĜD = Fθ (D, GD ).                (3)    are provided in Appendix D.

                                                     29360
                      (KPGVWPKPI../CU-)4GHKPGT
                                                                                                                                   6TCKPCDNG
                          *LYHQDVRXUFHGRFXPHQWDQGDFDQGLGDWHWULSOH                                                          (TQ\GP
                          H[WUDFWHGIURPLWGHFLGHKRZWRUHILQHWKHWULSOH
                          EDVHGRQO\RQWKHGRFXPHQW
                                                                                                                                   %QTTGEV'PVKV[4GNCVKQP
                          &KRRVHH[DFWO\RQHRSHUDWLRQ                                                                           +PEQTTGEV'PVKV[4GNCVKQP
                          -''2&'.'6'(+:RU4'94+6'
                          'RFXPHQW^WH[W`



                                                                                           -)4GHKPGT
                          7ULSOH^WULSOH`
       Ĕ
                              >6SDFH0LUURU0HPRULDOORFDWHGLQ)ORULGD@                                  -''2    >6SDFH0LUURU0HPRULDOORFDWHGLQ)ORULGD@

 Ground-Truth KG           >6SDFH0LUURU0HPRULDOHQWPFGFD['NQP/WUM@                                 &'.'6'                >18//18//18//@
        #
        !                    >6SDFH0LUURU0HPRULDOORFDWHGLQ%CNKHQTPKC@                                (+:     >6SDFH0LUURU0HPRULDOORFDWHGLQ(NQTKFC@

                                       >/KTTQTORFDWHGLQ)ORULGD@                                      4'94+6'   >5RCEG/KTTQT/GOQTKCNORFDWHGLQ)ORULGD@




                      )TCRJ4GHKPGOGPV
                                                                                &'.'6'



                                                               -)4GHKPGT
                                                                                                                    #IITGICVKQP
                                                                                  (+:
 Source Document
        !                            Ĕ                                                                  Ĕ                                            Ĕ
                                                                                4'94+6'

                              Draft KG      !                                                                                               Refined KG "!



Figure 2: Overview of the proposed GraphRefine framework. We synthesize operation-specific supervision to
model factual inconsistency types arising in GKE and fine-tune an LLM as a KG refiner. The refiner then refines a
draft KG at the triple level, and the outputs are aggregated to construct the final refined KG. This design enables
GraphRefine to be model-agnostic and applied post-hoc to draft KGs produced by diverse GKE systems.


downstream reasoning and applications. They are                                     Specifically, the KG refiner is designed to select
often removed, but this can reduce the coverage of                                  one of the following operations for each triple:
the resulting KG. When supporting evidence exists                                         • K EEP: If the triple is factually consistent with
in the source document, rewriting a triple to correct                                       the source document and its representation is
its semantics is often more effective than simply                                           appropriate, it is retained as is.
removing it, as it preserves information coverage
while improving factual accuracy.                                                         • D ELETE: If the triple is factually incorrect or
                                                                                            not supported by the source document, it is
Representation-level Inconsistencies. This type                                             removed to reduce noise.
of inconsistency encompasses cases where triples
are not strictly incorrect but nonetheless degrade                                        • F IX: If supporting evidence exists but the
KG quality due to inconsistent or imprecise rep-                                            meaning of the triple is incorrect, its semantics
resentations. Figure 1 provides representative ex-                                          are corrected to match the evidence.
amples, such as ambiguous span boundaries and
                                                                                          • R EWRITE: If the meaning is preserved but
unnecessarily verbose expressions. As a result, the
                                                                                            the representation is inconsistent or subopti-
KG may fail to express the facts in the source doc-
                                                                                            mal, the triple is rewritten.
ument precisely and unambiguously, introducing
factual inconsistencies at the representation level                                4.2.1 Dataset Generation
even when the underlying facts are correct. There-                                 We construct synthetic supervision for fine-tuning
fore, we mitigate these representation-level incon-                                the KG refiner by defining operation-specific sets
sistencies via meaning-preserving rewriting, reduc-                                of triples. Let SGtrain denote the set of ground-truth
ing ambiguity and surface-form variation while                                     KGs in the training split, where each G ∈ SGtrain is
enhancing clarity and consistency.                                                 a set of triples τ = (h, r, t) paired with a source
                                                                                   document D, and let E(D) denote the set of entities
4.2 Fine-tuning LLM as KG Refiner                                                  mentioned in D, with E and R denoting the global
To effectively mitigate the factual inconsistencies                                sets of entities and relations, respectively.
defined above, we fine-tune an LLM as a KG re-
                                                                                    K EEP.       We retain ground-truth triples as is:
finer. The KG refiner takes the source document                                                     [ n                            o
and an extracted KG triple as input, and applies                                          TKEEP =           (h, r, t) (h, r, t) ∈ G . (4)
an appropriate refinement operation to each triple.                                                     G∈SGtrain

                                                                            29361
D ELETE. We generate triples intended for re-             by the corruption process, and an instruction I
moval by jointly corrupting the relation and tail,        that asks the model to refine the triple. The target
encouraging non-recoverable hallucinations. Each          response is defined to contain an operation label
triple is further required to satisfy τDEL ∈ / G.         y ∈ {KEEP, DELETE, FIX, REWRITE} along with a
                   [    n                                 target triple τ ∗ that corresponds to the ground-truth
    TDELETE =             τDEL = (h, r̃, t̃)              refinement result.2 We represent each instance as
               G∈SGtrain                            (5)   token sequences, denoting the prompt as Xin and
                                           o
                 r̃ ∈ R \ {r}, t̃ ∈ E \ {t} .             the target response as Xout . Then, we optimize the
                                                          model parameters with teacher-forced supervised
F IX. We generate erroneous triples by substitut-         fine-tuning, maximizing pθ (Xout | Xin ). The train-
ing the tail entity with a different entity that does     ing objective is the standard negative log-likelihood
not appear in the source document. We further re-         over the output tokens:
quire that each generated triple satisfies τFIX ∈ / G.                       L
                                                                             X                             
The same corruption is applied to the head entity                LSFT = −           log pθ xout        out
                                                                                            i | Xin , x<i ,         (10)
in an analogous manner.                                                       i=1
             [ n                                 o        where θ denotes the learnable parameters and L
  TFIX =             τFIX = (h, r, t̃) t̃ ∈
                                          / E(D) .
                                                          is the output length. With this instruction-tuning
          G∈SGtrain
                                                          objective, the model is jointly trained to select an
                                                    (6)
                                                          appropriate refinement operation and generate the
R EWRITE. We generate representation-level in-            corresponding refined triple.
consistencies by perturbing the entity mention span
in the source document while preserving its under-        4.3      Graph Refinement
lying semantics. Let span(e, D) = (p, q) denote           Given a draft KG GD from GKE and a source doc-
the word-level span of entity e in D, where p and         ument D, a KG refiner assigns an operation label
q are the start and end positions, respectively. We       to each triple τ ∈ GD and generates a refined triple
then obtain a perturbed span (p′ , q ′ ) as follows:      τ̂ when applicable. Formally, the refiner defines a
                                                          mapping as follows:
   Expansion: (p′ , q ′ ) = (p − δl , q + δr ),
                                                    (7)                        (y, τ̂ ) = fθ (D, τ ),               (11)
    Shrinkage: (p′ , q ′ ) = (p + δl , q − δr ),
where δl and δr are integers controlling the left and     where fθ denotes the fine-tuned refiner. It then con-
right boundary shifts, respectively. We then define       structs the refined KG ĜD by accumulating triples
the perturbed surface form as ẽ = D[p′ : q ′ ]. In the   according to the predicted operation:
                                                                                [
following, we describe the tail-entity perturbation;                      ĜD =     g(τ ; y, τ̂ ).
the same procedure is applied to the head entity                                                           (12)
                                                                                      τ ∈GD
analogously.
                                                          The aggregation function g(τ ; y, τ̂ ) is defined as:
                     [ n
       TREWRITE =           τREW = (h, r, t̃)                              
                      G∈SGtrain                                            {τ }, if y = KEEP,
                                                                           
                                                    (8)
                                               o            g(τ ; y, τ̂ ) = ∅,      if y = DELETE,
                        t̃ = Di [p′t : qt′ ]    .                          
                                                                           
                                                                             {τ̂ }, if y ∈ {FIX, REWRITE}.
                                                                                                            (13)
4.2.2 Training Objective
                                                          Finally, the refined graph ĜD is obtained, yielding a
The synthetic training pool is defined as the union       KG that is more faithful to the source document D
of the operation-specific triple sets:                    by reducing factual inconsistencies. GraphRefine is
   Ttrain = TKEEP ∪ TDELETE ∪ TFIX ∪ TREWRITE .           model-agnostic to the underlying GKE system and
                                                (9)       can thus refine KGs produced by diverse extractors.
We convert Ttrain into instruction-following super-       This property allows it to be seamlessly integrated
vision and fine-tune the KG refiner. As illustrated       into existing LLM-based KG construction pipelines
in Figure 2, each training instance is formatted as       as a general refinement module, without modifying
a single prompt by concatenating the source doc-          the upstream extractor.
ument D, an erroneous input triple τ̃ generated              2
                                                                 Detailed instruction prompts are provided in Appendix F.

                                                     29362
5   Experimental Setup                                   5.4    Implementation Details
5.1 Datasets                                             Knowledge Extraction. GraphRefine is model-
                                                         agnostic and can be applied post-hoc to draft KGs
In this study, we utilize KG–text aligned datasets
                                                         from diverse GKE systems. For fair comparison,
that provide paired textual contexts and their corre-
                                                         we generate draft KGs using GraphJudge’s Entity-
sponding structured knowledge. We employ Gen-
                                                         Centric Text Denoising module (Huang et al., 2025)
Wiki (Jin et al., 2020) and DocRED (Yao et al.,
                                                         with GPT-4o-mini as the backbone.
2019) as general-domain datasets, and SciERC
(Luan et al., 2018) and CDR (Li et al., 2016) as         Dataset Generation. Given the ground-truth KG
domain-specific datasets, to comprehensively eval-       and the source document, we synthesize one exam-
uate the methods across both broad and specialized       ple per inconsistency type for each instance. For
domains. Detailed statistics and descriptions of the     TFIX , we corrupt the target triple by randomly se-
datasets are provided in Appendix A.                     lecting either the head or tail entity. For TREWRITE ,
                                                         we perturb the mention span of a randomly chosen
5.2 Evaluation Metrics                                   head or tail entity in the source document by ex-
Conventional evaluation metrics rely on rule-based       pansion or shrinkage, where the word-level span
exact string matching. However, such approaches          offset δ is sampled from 5 to 10, and only if the
fail to account for semantic variation, limiting their   entity mention appears in the document text.
ability to reliably assess KGs produced by GKE.
                                                         LLM Fine-tuning. We utilize Llama-2-7B (Tou-
Moreover, surface-level matching alone cannot cap-
                                                         vron et al., 2023) as the backbone of the KG refiner
ture broader quality dimensions of extracted KG
                                                         and fine-tune it with LoRA (Hu et al., 2021) for
triples, underscoring the need for a more compre-
                                                         parameter-efficient training. Detailed hyperparam-
hensive evaluation. Therefore, we adopt soft match-
                                                         eter settings are provided in Appendix C.
ing and semantic-level metrics such as G-BLEU
(BL), G-ROUGE (RO), and G-BERTScore (BS)
                                                         6     Experimental Results
(Huang et al., 2025), and report their precision, re-
call, and macro F1-score. In addition, we employ         In this section, we systematically evaluate the ef-
the multi-dimensional evaluation suite GenRES            fectiveness of GraphRefine by focusing on the fol-
(Jiang et al., 2024), which measures Topical Simi-       lowing research questions. RQ1. How effectively
larity (TS), Uniqueness (US), Factualness (FS),          does GraphRefine mitigate factual inconsistencies
Granularity (GS), and Completeness (CS) of the           arising in GKE and improve overall KG quality?
extracted KG triples. Detailed descriptions of these     RQ2. How broadly can GraphRefine be applied as
evaluation metrics are provided in Appendix B.           a post-hoc refiner to KGs from diverse GKE mod-
                                                         els? RQ3. How well does GraphRefine generalize
5.3 Baselines                                            when applied across different datasets?
In our experiments, we consider seven baselines,
including four vanilla LLMs and three LLM-based          6.1    Overall Performance Comparison (RQ1)
KG construction methods. Llama-2-7B, Llama-              Table 1 demonstrates that GraphRefine mitigates
2-13B (Touvron et al., 2023): These open-source          factual inconsistencies arising in GKE while im-
LLMs are used as baselines to analyze the impact of      proving KG quality from multiple perspectives. In
model scale on performance. GPT-4o-mini, GPT-            terms of the G-Score (BL/RO/BS) measured by F1,
4o (Hurst et al., 2024): These closed-source LLMs        GraphRefine consistently outperforms existing KG
represent efficiency-oriented and high-performance       construction methods and GraphJudge, a filtering-
models, respectively. The prompts used for these         based post-processing baseline, across all datasets.
LLMs are provided in Appendix F. EDC (Zhang              Directly compared with GraphJudge, GraphRefine
and Soh, 2024), KGGen (Mo et al., 2025): We fol-         improves precision while preserving recall, demon-
low the default settings of their official implemen-     strating the advantage of operation-based refine-
tations, using GPT-4o-mini as the backbone LLM.          ment over deletion-only filtering. Furthermore, un-
GraphJudge (Huang et al., 2025): We follow the           der the multi-faceted GenRES metrics, GraphRe-
default settings of GraphJudge, using GPT-4o-mini        fine maintains high factualness (FS) while sub-
for KG construction and Llama-2-7B as the judge          stantially enhancing granularity (GS) and unique-
model to evaluate extracted triples.                     ness (US). This suggests that refinement not only
                                                    29363
 Dataset     Method      BLP     BLR     BLF1    ROP     ROR      ROF1        BSP             BSR            BSF1           TS            US             FS              GS           CS
           Llama-2-7B    60.57   37.61   44.46   52.76   32.90    38.83      83.96           51.94          61.41        49.01           66.82         86.59          83.53          60.51
           Llama-2-13B   63.91   56.95   58.31   55.46   49.56    50.66      88.06           78.48          80.41        54.63           64.74         92.83          92.59          74.97
           GPT-4o-mini   64.39   62.22   62.02   55.57   53.91    53.62      87.65           84.68          84.43        71.76           67.91         98.74          95.22          77.36
           GPT-4o        64.78   63.51   62.80   56.00   54.97    54.30      87.04           85.32          84.38        68.56           66.60         98.40          95.03          80.65
 GenWiki
           EDC           63.62   64.61   62.92   56.80   57.96    56.30      85.76           87.04          84.79        64.00           62.16         94.75          94.58          81.70
           KGGen         57.63   56.79   55.88   47.81   47.39    46.48      85.37           83.67          82.56        42.32           49.25         78.29          98.58          60.34
           GraphJudge    68.69   62.61   63.40   59.41   54.27    54.86      88.26           80.61          81.61        33.81           65.97         97.92          98.04          75.66
           GraphRefine   69.14   65.38   65.23   60.73   56.60    56.41      89.64           86.05          85.32        65.24           72.68         98.36          98.67          85.53
           Llama-2-7B    49.28   23.32   26.83   37.92   17.50    20.33      80.82           38.35          44.03        13.43           74.41         85.77          71.84          22.58
           Llama-2-13B   45.52   41.49   37.41   33.27   30.01    27.05      73.27           66.28          60.18        19.76           76.10         85.45          80.45          33.04
           GPT-4o-mini   34.72   58.33   39.98   25.19   43.24    29.20      54.55           89.48          62.35        46.70           88.34         96.25          82.76          45.72
           GPT-4o        35.24   59.34   40.71   25.87   44.41    30.06      54.44           89.74          62.48        43.95           86.91         96.63          84.57          48.79
 DocRED
           EDC           34.03   59.62   39.88   25.06   45.02    29.61      52.82           90.24          61.37        37.75           87.47         90.61          81.78          50.78
           KGGen         41.13   49.84   41.07   27.62   33.79    27.62      67.75           81.58          67.62        19.99           71.86         69.66          92.24          32.42
           GraphJudge    47.87   56.20   47.57   36.09   42.57    35.86      70.46           82.41          70.06        18.83           80.25         95.24          96.02          54.74
           GraphRefine   50.41   60.26   52.94   37.28   44.60    38.29      71.23           85.34          73.84        36.72           89.46         96.38          97.10          59.63
           Llama-2-7B    64.77   26.77   34.65   58.65   24.71    31.61      89.79           38.98          49.24        58.29           50.88         78.77          73.60          37.10
           Llama-2-13B   64.36   28.79   36.37   55.92   24.62    31.24      91.07           41.94          52.14        63.69           52.18         75.95          76.08          29.75
           GPT-4o-mini   55.68   58.92   54.02   48.17   51.49    46.70      76.83           82.13          74.82        85.03           93.18         97.14          72.68          57.64
           GPT-4o        55.49   59.23   54.10   48.11   52.25    47.01      76.87           82.51          75.10        85.90           92.34         98.22          69.99          60.42
 SciERC
           EDC           54.36   62.91   55.08   47.59   55.47    48.22      73.83           85.77          74.91        75.20           91.96         93.46          69.80          64.64
           KGGen         53.92   56.62   52.33   46.22   48.84    44.82      77.32           82.08          75.24        80.93           90.14         94.05          81.93          44.71
           GraphJudge    59.26   53.69   52.91   52.43   47.11    46.51      82.13           74.84          73.47        74.38           87.28         97.47          83.98          49.74
           GraphRefine   57.85   58.13   55.43   51.15   50.32    47.76      80.64           77.34          76.29        79.61           94.42         97.31          86.33          56.75
           Llama-2-7B    19.88    8.05    9.02   15.17    6.01     6.77      43.28           17.58          20.06        33.73           89.39         42.86          72.75           6.61
           Llama-2-13B   34.26   23.07   22.01   25.39   16.95    16.12      69.59           45.40          44.34        59.86           72.55         81.78          60.14          18.42
           GPT-4o-mini   30.51   54.14   34.88   21.04   38.38    24.22      50.76           87.58          57.59        82.86           91.55         96.33          74.76          32.50
           GPT-4o        32.94   53.67   36.22   23.21   38.70    25.67      53.93           85.53          58.87        79.47           87.80         96.80          76.28          35.03
  CDR
           EDC           29.35   54.24   33.84   20.38   38.86    23.70      48.91           87.62          55.88        78.21           92.22         92.45          68.00          35.68
           KGGen         37.49   50.98   38.38   25.40   35.40    26.18      61.57           81.93          62.68        67.82           82.66         86.58          88.69          29.33
           GraphJudge    38.78   53.84   40.16   27.62   39.13    28.74      61.03           82.96          62.93        61.33           84.62         93.36          91.17          35.17
           GraphRefine   38.12   55.36   42.73   28.49   40.71    31.26      61.24           84.22          64.10        66.54           92.15         94.21          93.40          36.54


Table 1: Overall performance comparison on GenWiki, DocRED, SciERC, and CDR. We report BL/RO/BS (P/R/F1)
as KG quality metrics, and multi-faceted GenRES metrics (TS/US/FS/GS/CS) to assess improvement from multiple
perspectives. Best and second-best scores are highlighted in bold and underlined, respectively.

                                                                                              Draft KG (GPT-4o-mini)               + GraphJudge                 + GraphRefine
improves factualness but also restructures triples
                                                                      12
into more atomic and consistent forms, reducing                       10
                                                                                                                                    14
                                                                                                                                    12
redundancy and stabilizing noisy representations.                      8                                                            10
                                                                                                                                     8
GraphRefine also achieves strong factual cover-                        6
                                                                       4
                                                                                                                                     6
                                                                                                                                     4
age (CS). Together, these findings suggest that it                     2                                                             2

mitigates inconsistencies while preserving core in-                    0
                                                                           Hallucination Misidentification Ambiguity
                                                                                                  DocRED
                                                                                                                       Verbosity
                                                                                                                                     0
                                                                                                                                         Hallucination Misidentification Ambiguity
                                                                                                                                                                SciERC
                                                                                                                                                                                     Verbosity


formation, alleviating the trade-off between noise
removal and coverage.                                                Figure 3: Distribution of factual inconsistency types in
   For a more detailed analysis, we conducted a                      human-annotated KGs for DocRED and SciERC, where
                                                                     “+” indicates refinement applied to the drafts.
human evaluation on sampled KGs from DocRED
and SciERC, two domain-distinct datasets.3 Fig-
ure 3 presents the distribution of factual inconsis-                 Meanwhile, GraphRefine substantially reduces the
tency types in our annotations. GraphJudge alle-                     proportions of both inconsistency types, underscor-
viates correctness-level inconsistencies (e.g., hal-                 ing that operation-based refinement captures errors
lucination and misidentification), but its impact                    that deletion-only filtering may miss.
on representation-level inconsistencies (e.g., am-
biguity and verbosity) is limited. These findings                    6.2            Model-Agnostic Applicability (RQ2)
suggest that addressing representation-level issues                  Figure 4 illustrates a multi-aspect comparison on
often requires rewriting, rather than deletion alone.                KGs generated by two distinct base extractors, in-
   3
                                                                     cluding the unprocessed drafts and the results af-
     For each dataset, we sampled five generated KGs per
method, yielding 15 KGs per dataset (30 total across two             ter applying GraphJudge or GraphRefine. Overall,
datasets). Annotators followed the guidelines in Appendix D.         GraphRefine consistently improves performance
                                                                 29364
           Draft KG (Base Extractor)             + GraphJudge          + GraphRefine                  Src → Tgt    BLF1 (∆)       ROF1 (∆)       BSF1 (∆)
                   BS-F1                                             BS-F1
                                                                                                      Doc → Gen   64.60(↓0.63)   55.62(↓0.79)   84.19(↓1.13)
     TS                          RO-F1                  TS                       RO-F1
                                                                                                      Gen → Doc   52.12(↓0.82)   37.54(↓0.75)   72.59(↓1.25)
                                                                                                   Doc → CDR      41.01(↓1.72)   29.45(↓1.81)   62.92(↓1.18)
US                                       BL-F1     US                                     BL-F1
                                                                                                   CDR → Doc      52.41(↓0.53)   37.91(↓0.38)   72.85(↓0.99)
                                                                                                      Gen → Sci   56.46(↑1.03)   48.45(↑0.69)   77.87(↑1.58)
                                                                                                      Sci → Gen   58.12(↓7.11)   51.40(↓5.01)   76.17(↓9.15)
     FS                            CS                   FS                           CS

                     GS                                               GS                          Table 2: Results of cross-dataset generalization under
      Llama-2-13B on DocRED                               GPT-4o-mini on DocRED                   train → test transfer across domain pairs, where ∆ de-
                   BS-F1                                             BS-F1
                                                                                                  notes change from in-domain training.
     TS                          RO-F1                  TS                       RO-F1


                                                                                                  over, even in cross-domain transfers (Doc → CDR
US                                       BL-F1     US                                     BL-F1
                                                                                                  and CDR → Doc), GraphRefine exhibits some
                                                                                                  degradation but largely maintains consistent per-
     FS                            CS                   FS                           CS           formance across metrics. These findings suggest
                     GS                                               GS                          that GraphRefine captures document-grounded re-
          Llama-2-13B on SciERC                              GPT-4o-mini on SciERC
                                                                                                  finement behaviors that generalize beyond dataset-
Figure 4: Multi-aspect comparison of draft KGs and                                                specific surface patterns. Notably, when trained on
their post-hoc refinement results. For visual clarity, each                                       GenWiki and evaluated on SciERC (Gen → Sci),
metric is min-max normalized and rescaled to [0.7, 0.9].                                          GraphRefine outperforms the SciERC in-domain
                                                                                                  model, implying that refinement behaviors learned
                                                                                                  from a more diverse training distribution improve
across base extractors, boosting not only accuracy-                                               transferability. In contrast, the reverse setting (Sci
oriented metrics (BL/RO/BS-F1) but also the multi-                                                → Gen) leads to a substantially larger drop, sug-
faceted GenRES measures. Across both datasets,                                                    gesting that transfer becomes more challenging
it maintains or improves factualness (FS) while en-                                               under larger domain shifts. This result motivates
hancing granularity (GS) and uniqueness (US), and                                                 future work on mixed-domain training or domain
it often improves coverage (CS) without sacrific-                                                 adaptation to mitigate such domain gaps.
ing quality. By comparison, GraphJudge reduces
correctness-level errors but tends to lower over-                                                 6.4    Further Analysis
all KG quality due to triple removal. This effect is
more pronounced for Llama-2-13B, where the draft                                                  We further analyze how GraphRefine improves per-
KGs are smaller and additional removals more di-                                                  formance by examining the behavior of its refine-
rectly reduce coverage and representation quality.                                                ment operations and their contributions to each
Notably, GraphRefine improves factualness while                                                   evaluation metric.
also increasing GS and US even for smaller models,
                                                                                                  Operation-Level Confusion Analysis. We eval-
supporting its ability to correct and rewrite triples
                                                                                                  uate GraphRefine’s operation prediction on syn-
with document grounding. In summary, GraphRe-
                                                                                                  thetically generated test samples derived from Do-
fine provides consistent gains for KGs produced
                                                                                                  cRED. The confusion matrix in Figure 5 shows
by both open-source and proprietary LLM-based
                                                                                                  that GraphRefine separates non-edit vs. edit deci-
extractors, supporting its use as a model-agnostic,
                                                                                                  sions reasonably well, while exhibiting a tendency
post-hoc KG refinement module. Extended results
                                                                                                  toward R EWRITE. For gold K EEP and D ELETE,
are provided in Appendix E.1.
                                                                                                  correct non-edit predictions dominate (78.1% and
                                                                                                  71.3%), yet R EWRITE is assigned to a non-trivial
6.3 Cross-Dataset Generalization (RQ3)
                                                                                                  portion (11.1% and 18.0%). Among edit oper-
Table 2 presents the cross-dataset generalization                                                 ations, F IX is the most challenging: 64.5% are
performance of GraphRefine. First, GraphRefine                                                    correctly predicted, while 28.0% are mapped to
achieves performance comparable to in-domain                                                      R EWRITE, reflecting overlap between meaning cor-
training under within-domain transfers (Doc →                                                     rection and surface-form normalization under docu-
Gen and Gen → Doc), indicating stable trans-                                                      ment evidence. Meanwhile, R EWRITE is relatively
fer across datasets with similar domains. More-                                                   consistent (72.6%), suggesting a coherent rewrit-
                                                                                              29365
                        9583      582           751       1359
                                                                          Factualness (FS). Although GraphRefine does
                KEEP   (78.1%)   (4.7%)        (6.1%)    (11.1%)   8000   not always achieve the best FS, it consistently main-
                                                                          tains robust factualness. By performing document-
                        394       8755          919       2207
              DELETE   (3.2%)    (71.3%)       (7.5%)    (18.0%)   6000   grounded refinement, it suppresses unsupported
Gold Label
                                                                          content while preserving document-supported state-
                 FIX    583       343           7914      3435     4000
                                                                          ments. In particular, D ELETE and F IX remove or
                       (4.7%)    (2.8%)        (64.5%)   (28.0%)
                                                                          correct clear factual inconsistencies, thereby sus-
                                                                   2000
             REWRITE    523
                       (5.6%)
                                  472
                                 (5.0%)
                                                1572
                                               (16.8%)
                                                          6794
                                                         (72.6%)
                                                                          taining high FS in practice.
                        KEEP     DELETE          FIX
                                    Predicted Label
                                                         REWRITE          Granularity (GS). LLM-based GKE often pro-
                                                                          duces overly descriptive phrases or merges multiple
 Figure 5: Operation-level confusion matrix of GraphRe-                   attributes into a single triple. GraphRefine yields
 fine on the synthetically labeled DocRED test set.                       large gains in GS because R EWRITE refactors such
                                                                          outputs into more atomic and clearer triples by
 ing policy. Overall, GraphRefine favors R EWRITE-                        removing unnecessary modifiers and making the
 based refinement, which may improve representa-                          core relation explicit, while staying faithful to the
 tion quality, while leaving room to better calibrate                     source document. This improves triple granularity,
 rewriting to avoid unnecessary edits.                                    leading to higher GS.

 BLEU/ROUGE/BERTScore. Since BL/RO/BS                                     Completeness (CS). CS remains high because
 capture surface- and semantic-level similarity to                        GraphRefine edits triples beyond deletion to pre-
 the ground-truth triples, all operations can improve                     serve supported information. F IX revises partially
 them. D ELETE removes unsupported triples, F IX                          incorrect but correctable triples into document-
 corrects entities/relations, and R EWRITE prunes                         supported statements instead of discarding them.
 redundant modifiers to better match the references.                      R EWRITE standardizes surface forms by resolving
 Overall, BL/RO/BS gains reflect both noise reduc-                        unclear mention spans and trimming verbose phras-
 tion and representational alignment.                                     ing, making outputs closer to the ground truth. To-
                                                                          gether, these operations reduce errors while main-
 Topical Similarity (TS). TS appears relatively                           taining coverage, leading to strong CS.
 low in Table 1 because it is influenced more by
 the base extractor’s output characteristics than by                      7   Conclusion
 GraphRefine itself. Since TS reflects how well
 extracted triples align with the document topic, un-                     In this paper, we proposed GraphRefine, a post-hoc
 stable topical coherence or surface forms at ex-                         refinement method to effectively mitigate factual
 traction leave limited room for refinement to im-                        inconsistencies in KGs constructed via LLM-based
 prove TS. This trend is also reflected in Figure 9,                      GKE. By analyzing the KGs generated across di-
 where TS can increase when GraphRefine is ap-                            verse GKE models and datasets, we systematically
 plied to high-TS extractors (e.g., GPT-4o-mini and                       defined fine-grained factual inconsistency types. To
 GPT-4o). Therefore, GraphRefine does not inher-                          this end, we fine-tuned an LLM as a KG refiner
 ently reduce TS; observed decreases are largely                          with operation-specific synthetic supervision to pre-
 attributable to the underlying extractor.                                dict a refinement label for each triple and generate
                                                                          an edited triple. Extensive experiments using a di-
 Uniqueness (US). GraphRefine improves US by                              verse set of evaluation metrics show that GraphRe-
 reducing redundancy among triples in the extracted                       fine consistently improves overall KG quality while
 KG. R EWRITE makes triples more atomic and stan-                         reducing inconsistencies and preserving core infor-
 dardized, so the same fact is less likely to appear                      mation. Beyond correctness-level issues, it also
 in multiple surface forms. F IX can provide a mod-                       improves representation-level quality, particularly
 est benefit by correcting spurious triples into valid                    in terms of granularity and uniqueness, leading to
 facts, reducing noisy variants and better separating                     more atomic and consistent triples. Moreover, con-
 distinct facts. As a result, GraphRefine mainly re-                      sistent gains across different base extractors and
 duces repeated paraphrases and can also alleviate                        datasets further support its model-agnostic applica-
 cases where distinct meanings collapse into overly                       bility and cross-dataset generalization, making it
 similar forms, leading to higher US.                                     broadly applicable in real-world settings.
                                                                      29366
Limitations                                             References
LLM Stability and Reliability. Since GraphRe-           Hanzhu Chen, Xu Shen, Qitan Lv, Jie Wang, Xiaoqi
                                                          Ni, and Jieping Ye. 2024. Sac-kg: Exploiting large
fine relies on an LLM to choose refinement opera-         language models as skilled automatic constructors for
tions and perform edits, its quality depends on the       domain knowledge graph. In Proceedings of the 62nd
model’s reasoning stability and output reliability.       Annual Meeting of the Association for Computational
When the model is uncertain or inconsistent, it may       Linguistics (Volume 1: Long Papers), pages 4345–
                                                          4360.
still edit decisively, causing unnecessary changes
and over-correction. Still, document-grounded re-       Jiuzhou Han, Nigel Collier, Wray Buntine, and Ehsan
finement helps suppress unsupported speculation            Shareghi. 2024. Pive: Prompting with iterative verifi-
and align edits with the input document. In future         cation improving graph-based generative capability
                                                           of llms. In Findings of the Association for Computa-
work, reliability could be further improved by lever-      tional Linguistics: ACL 2024, pages 6702–6718.
aging approaches such as post-edit verification or
evidence-constrained decoding.                          Aidan Hogan, Eva Blomqvist, Michael Cochez, Clau-
                                                          dia d’Amato, Gerard De Melo, Claudio Gutierrez,
Synthetic–Real Gap. GraphRefine fine-tunes on             Sabrina Kirrane, José Emilio Labra Gayo, Roberto
operation-specific synthetic noise, which may not         Navigli, Sebastian Neumaier, and 1 others. 2021.
                                                          Knowledge graphs. ACM Computing Surveys (Csur),
fully match the complex errors produced by real           54(4):1–37.
GKE systems. However, this supervision scales
without large manual labeling and yielded consis-       Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
                                                          Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
tent gains across datasets and extractors. To reduce      and Weizhu Chen. 2021. Lora: Low-rank adap-
the gap, future work may calibrate synthesis rules        tation of large language models. arXiv preprint
with a small amount of human-verified data, mine          arXiv:2106.09685.
failure patterns from real outputs to update the syn-
                                                        Elwin Huaman, Elias Kärle, and Dieter Fensel.
thetic distribution, and iteratively add hard cases       2020. Knowledge graph validation. arXiv preprint
via self-training or active learning.                     arXiv:2005.01389.

Cost and Scalability. Because GraphRefine runs          Haoyu Huang, Chong Chen, Zeang Sheng, Yang Li, and
LLM inference at the triple level, cost and latency       Wentao Zhang. 2025. Can llms be good graph judge
                                                          for knowledge graph construction? In Proceedings
can grow for large KGs or long documents. Prac-
                                                          of the 2025 Conference on Empirical Methods in
tical deployment may require system optimiza-             Natural Language Processing, pages 10940–10959.
tions for throughput, cost, and response time, es-
pecially to avoid redundant document–triple calls.      Xiao Huang, Jingyuan Zhang, Dingcheng Li, and Ping
                                                          Li. 2019. Knowledge graph embedding based ques-
These issues are common to LLM-based meth-                tion answering. In Proceedings of the twelfth ACM
ods, and we observed improvements even with               international conference on web search and data min-
lightweight backbones, suggesting applicability un-       ing, pages 105–113.
der constrained budgets.
                                                        Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
                                                          Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Ethical Considerations                                    Akila Welihinda, Alan Hayes, Alec Radford, and 1
                                                          others. 2024. Gpt-4o system card. arXiv preprint
All authors of this paper acknowledge and adhere          arXiv:2410.21276.
to the ACL Code of Ethics. Annotators were com-
pensated in accordance with institutional policies,     Pengcheng Jiang, Jiacheng Lin, Zifeng Wang, Jimeng
                                                          Sun, and Jiawei Han. 2024. Genres: Rethinking
and the study was designed to avoid undue burden
                                                          evaluation for generative relation extraction in the era
or risk. We used existing datasets derived from pub-      of large language models. In Proceedings of the 2024
licly available and open-licensed resources. Our          Conference of the North American Chapter of the
use complies with the terms of use and licensing          Association for Computational Linguistics: Human
policies of the corresponding datasets and plat-          Language Technologies (Volume 1: Long Papers),
                                                          pages 2820–2837.
forms. While we did not observe any sensitive or
potentially harmful content in our study, such con-     Zhijing Jin, Qipeng Guo, Xipeng Qiu, and Zheng Zhang.
tent may still be present in the source corpora. We       2020. Genwiki: A dataset of 1.3 million content-
                                                          sharing text and graphs for unsupervised graph-to-
also leveraged AI assistants for editorial support,       text generation. In Proceedings of the 28th Inter-
such as language polishing and grammar checking,          national Conference on Computational Linguistics,
to improve the writing quality of this paper.             pages 2398–2409.
                                                   29367
Ryo Kamoi, Sarkar Snigdha Sarathi Das, Renze Lou,            Heiko Paulheim. 2016. Knowledge graph refinement:
  Jihyun Janice Ahn, Yilun Zhao, Xiaoxin Lu, Nan               A survey of approaches and evaluation methods. Se-
  Zhang, Yusen Zhang, Ranran Haoran Zhang, Su-                 mantic web, 8(3):489–508.
  jeeth Reddy Vummanthala, and 1 others. 2024. Eval-
  uating llms at detecting errors in llm responses. arXiv    Swarnadeep Saha, Prateek Yadav, Lisa Bauer, and Mo-
  preprint arXiv:2404.03602.                                   hit Bansal. 2021. Explagraphs: An explanation graph
                                                               generation task for structured commonsense reason-
Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James           ing. In Proceedings of the 2021 Conference on Em-
   Thorne, and Yoonjae Choi. 2023. Factkg: Fact veri-          pirical Methods in Natural Language Processing,
   fication via reasoning on knowledge graphs. In 61st         pages 7716–7740.
   Annual Meeting of the Association for Computational       Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
   Linguistics, ACL 2023, pages 16190–16206. Associ-           bert, Amjad Almahairi, Yasmine Babaei, Nikolay
   ation for Computational Linguistics (ACL).                  Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
                                                               Bhosale, and 1 others. 2023. Llama 2: Open foun-
Diederik P Kingma. 2014. Adam: A method for stochas-           dation and fine-tuned chat models. arXiv preprint
  tic optimization. arXiv preprint arXiv:1412.6980.            arXiv:2307.09288.
Jiao Li, Yueping Sun, Robin J Johnson, Daniela Sci-          Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu,
   aky, Chih-Hsuan Wei, Robert Leaman, Allan Peter             Haiyue Song, Jiwei Li, and Sadao Kurohashi. 2023.
   Davis, Carolyn J Mattingly, Thomas C Wiegers, and           Gpt-re: In-context learning for relation extraction
   Zhiyong Lu. 2016. Biocreative v cdr task corpus:            using large language models. In Proceedings of the
   a resource for chemical disease relation extraction.        2023 Conference on Empirical Methods in Natural
   Database, 2016.                                             Language Processing, pages 3534–3547.

Chin-Yew Lin. 2004. Rouge: A package for automatic           Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang,
  evaluation of summaries. In Text summarization               Fei Wu, Tianwei Zhang, Jiwei Li, Guoyin Wang, and
  branches out, pages 74–81.                                   Chen Guo. 2025. Gpt-ner: Named entity recognition
                                                               via large language models. In Findings of the asso-
Yuxing Lu and Jinzhuo Wang. 2025. Karma: Leverag-              ciation for computational linguistics: NAACL 2025,
  ing multi-agent llms for automated knowledge graph           pages 4257–4275.
  enrichment. arXiv preprint arXiv:2502.06472.
                                                             Xiangyu Wang, Lyuzhou Chen, Taiyu Ban, Muhammad
                                                               Usman, Yifeng Guan, Shikang Liu, Tianhao Wu, and
Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh
                                                               Huanhuan Chen. 2021. Knowledge graph quality
  Hajishirzi. 2018. Multi-task identification of entities,
                                                               control: A survey. Fundamental Research, 1(5):607–
  relations, and coreference for scientific knowledge
                                                               626.
  graph construction. In Proceedings of the 2018 Con-
  ference on Empirical Methods in Natural Language           Bingcong Xue and Lei Zou. 2022. Knowledge graph
  Processing, pages 3219–3232.                                 quality management: A comprehensive survey. IEEE
                                                               Transactions on Knowledge and Data Engineering,
Belinda Mo, Kyssen Yu, Joshua Kazdan, Proud Mpala,             35(5):4969–4988.
  Lisa Yu, Chris Cundy, Charilaos Kanatsoulis, and
  Sanmi Koyejo. 2025. Kggen: Extracting knowledge            Yuhao Yang, Chao Huang, Lianghao Xia, and Chenliang
  graphs from plain text with language models. arXiv           Li. 2022. Knowledge graph contrastive learning for
  preprint arXiv:2502.09956.                                   recommendation. In Proceedings of the 45th inter-
                                                               national ACM SIGIR conference on research and
Christina Niklaus, Matthias Cetto, André Freitas, and          development in information retrieval, pages 1434–
  Siegfried Handschuh. 2018. A survey on open infor-           1443.
  mation extraction. In Proceedings of the 27th Inter-
  national Conference on Computational Linguistics,          Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin,
  pages 3866–3878.                                             Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou,
                                                               and Maosong Sun. 2019. Docred: A large-scale
Songjie Niu, Kaisen Yang, Rui Zhao, Yichao Liu,                document-level relation extraction dataset. In Pro-
  Zonglin Li, Hongning Wang, and Wenguang Chen.                ceedings of the 57th Annual Meeting of the Associa-
  2025. Tree-kg: An expandable knowledge graph                 tion for Computational Linguistics, pages 764–777.
  construction framework for knowledge-intensive do-         Bowen Zhang and Harold Soh. 2024. Extract, define,
  mains. In Proceedings of the 63rd Annual Meeting of          canonicalize: An llm-based framework for knowl-
  the Association for Computational Linguistics (Vol-          edge graph construction. In Proceedings of the 2024
  ume 1: Long Papers), pages 18516–18529.                      Conference on Empirical Methods in Natural Lan-
                                                               guage Processing, pages 9820–9836.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
  Jing Zhu. 2002. Bleu: a method for automatic evalu-        Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q
  ation of machine translation. In Proceedings of the          Weinberger, and Yoav Artzi. 2019. Bertscore: Eval-
  40th annual meeting of the Association for Computa-          uating text generation with bert. arXiv preprint
  tional Linguistics, pages 311–318.                           arXiv:1904.09675.
                                                        29368
Weiyan Zhang, Wanpeng Lu, Jiacheng Wang, Yating                                            General-Domain                         Domain-Specific
                                                                 Dataset
 Wang, Lihan Chen, Haiyun Jiang, Jingping Liu, and                                       GenWiki                      DocRED      SciERC        CDR
 Tong Ruan. 2024. Unexpected phenomenon: Llms’
 spurious associations in information extraction. In       Train KGs                      69,788                       3,027         395        1,000
 Findings of the Association for Computational Lin-        Test KGs                        1,000                        985          100         500
 guistics ACL 2024, pages 9176–9190.                       Train Triples                 295,380                      38,180        3,674       10,327
                                                           Test Triples                   4,235                       12,275         974         5,204
Zikang Zhang, Wangjie You, Tianci Wu, Xinrui Wang,
  Juntao Li, and Min Zhang. 2025. A survey of gen-         Avg. Triples                        4.23                   12.58         9.39         10.35
  erative information extraction. In Proceedings of        Avg. Length                         24.22                  198.33       130.99       231.66
  the 31st International Conference on Computational
  Linguistics, pages 4840–4870.                          Table 3: Statistics of the datasets. Avg. Triples and Avg.
                                                         Length denote the average numbers of triples per KG
A   Datasets                                             and words per document, respectively.

In this appendix, we provide additional details of                                                                                               GenWiki
                                                                                                                                                 DocRED
the datasets used in our experiments. Table 3 sum-           0.20                                                                                SciERC
marizes the dataset statistics, and Figure 6 illus-                                                                                              CDR
                                                             0.15
trates the distributions of triple counts and docu-
ment lengths for each dataset.                          Ratio0.10


GenWiki. GenWiki (Jin et al., 2020) is a large-              0.05

scale non-parallel graph-to-text dataset built from          0.00
Wikipedia text and DBpedia graphs, consisting of                        1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 >20
                                                                                                            Number of triples
1.3 million text–graph pairs. For experimental ef-
                                                                                                                                                 GenWiki
ficiency, we employ the GenWiki-fine subset for                 0.5                                                                              DocRED
both training and evaluation, removing extremely                                                                                                 SciERC
                                                                0.4                                                                              CDR
short documents and triples with formatting errors,
                                                                0.3
following Huang et al. (2025).                           Ratio
                                                                0.2
DocRED. DocRED (Yao et al., 2019) is a dataset
                                                                0.1
constructed for document-level relation extraction,
                                                                0.0
containing manual annotations of entities, relations,
                                                                          5         45         75         105         5     5     5     5     5     5   0
and coreference information for 5,053 Wikipedia                        0-1    30-        60-        90-            -13 50-16 80-19 10-22 40-25 70-28 >30
                                                                                                                120     1     1     2     2     2
documents aligned with Wikidata. Compared to                                                                    Number of words
GenWiki, DocRED contains longer documents and            Figure 6: Distributional characteristics of the datasets.
a larger number of annotated triples, making it par-     The top plot shows the average number of triples per
ticularly suitable for evaluating model performance      document for each dataset, and the bottom plot shows
in more challenging and realistic scenarios. In this     the distribution of document lengths.
study, we use the human-annotated train and dev
subsets while excluding documents with insuffi-
                                                         chemical entities, disease entities, and their inter-
cient annotation quality.
                                                         actions from 1,500 PubMed abstracts. Owing to
SciERC. SciERC (Luan et al., 2018) is a domain-          its relatively large number of triples and longer
specific dataset built from 500 scientific abstracts,    documents compared to SciERC, CDR serves as a
annotated with scientific entity types such as Task,     challenging benchmark for evaluating knowledge
Method, and Metric, as well as diverse semantic          extraction in specialized domains. We also merge
relations and coreference links. Due to the limited      the train and dev subsets into a unified training set
dataset size, we merge the train and dev subsets         due to the limited volume of data.
into a unified training set. In addition, following
Huang et al. (2025), we exclude samples that con-        B            Evaluation Metrics
tain empty graphs.                                       B.1           G-Score
CDR. CDR (Li et al., 2016) is a biomedical-              G-Score (Huang et al., 2025) is a family of auto-
domain dataset designed for chemical–disease re-         matic evaluation metrics that extends widely used
lation extraction, where expert annotators curated       text generation metrics to the task of KG evalua-
                                                   29369
tion. It linearizes the predicted and gold KGs into        Hyperparameter/Model                     Setting
triple sentences and measures their structural and                           GenWiki                 200
semantic-level similarity.                               LDA latent topics
                                                                             DocRED                  100
                                                                             SciERC                  50
G-BLEU. G-BLEU adapts BLEU (Papineni                                         CDR                     50
et al., 2002) for KG evaluation by measuring struc-           Triple threshold ϕ                     0.95
tural similarity based on n-gram overlap between                    LLM                     gpt-3.5-turbo-0125
the linearized predicted and ground-truth triple sen-         Embedding model              text-embedding-ada-002
tences. This metric focuses on the fidelity of gener-
ated triples to the phrasing and structural patterns    Table 4: Detailed hyperparameters and model configu-
of the gold triples. In our experiments, Precision,     rations used to compute the GenRES metrics.
Recall, and Macro F1-score based on G-BLEU are
reported as BLP , BLR , and BLF1 , respectively. Fol-
                                                        where θD and θGD denote the topic distributions
lowing Huang et al. (2025), the n-gram order is set
                                                        of D and GD , respectively. A higher TS indicates
to n = 4.
                                                        stronger topical consistency of the extracted triples
G-ROUGE. G-ROUGE applies ROUGE (Lin,                    with the source document.
2004) to KG evaluation by measuring information
                                                        Uniqueness Score. Uniqueness Score (US) quan-
coverage through n-gram co-occurrence statistics
                                                        tifies the diversity of the extracted knowledge by
between the linearized predicted and ground-truth
                                                        penalizing overly similar or duplicated triples. Let
triple sentences. This metric evaluates how well
                                                        vi denote the embedding of triple τi in GD . US is
the generated triples capture the key information
                                                        computed as:
contained in the gold triples. In our experiments,
Precision, Recall, and Macro F1-score based on                       1     XX         n    n

G-ROUGE are reported as ROP , ROR , and ROF1 ,          US(GD ) =             1[cos(vi , vj ) < ϕ],
                                                                  n(n − 1)
respectively. Following Huang et al. (2025), the                                     i=1 j=1
                                                                                         j̸=i
n-gram order is set to n = 2.                                                                           (15)
G-BERTScore. G-BERTScore (Saha et al.,                  where n = |GD | and ϕ is a cosine similarity thresh-
2021) extends BERTScore (Zhang et al., 2019) to         old. A higher US indicates greater diversity of the
the KG evaluation setting by treating each triple as    extracted triples in GD .
a sentence and computing the semantic similarity        Factualness Score. Factualness Score (FS) mea-
between the predicted and ground-truth triples. In      sures whether each triple τ ∈ GD is supported by
our experiments, Precision, Recall, and Macro F1-       the source document D:
score based on G-BERTScore are reported as BSP ,
BSR , and BSF1 , respectively.                                           1 X
                                                         FS(D, GD ) =             1[τ is supported by D].
                                                                       |GD |
                                                                                   τ ∈GD
B.2   GenRES                                                                                            (16)
GenRES (Jiang et al., 2024) is a suite of multi-        For each triple, the support decision is obtained by
aspect automatic metrics for evaluating generative      prompting an LLM-based fact checker. A higher
triple extraction results. We adopt these metrics to    FS indicates stronger factual correctness of the ex-
assess KG quality. The detailed settings used for       tracted triples in GD .
computing each metric are provided in Table 4. For
                                                        Granularity Score. Granularity Score (GS) eval-
any aspects not specified in our paper, we follow
                                                        uates whether extracted triples are overly com-
the original implementation.
                                                        plex or properly decomposed into atomic relational
Topical Similarity Score. Topical Similarity            units. Let nτ denote the number of atomic triples
Score (TS) measures how well the extracted triples      derived from τ . To obtain nτ , an LLM is prompted
in GD align with the main topics of the source doc-     to determine whether a given triple can be further
ument D. TS is computed by deriving LDA-based           decomposed into finer-grained triples. GS is com-
topic distributions for D and GD and evaluating         puted as:
their divergence as:
                                                                                       1 X 1
                                                                     GS(GD ) =                  .             (17)
          TS(D, GD ) = e−KL(θD ∥θGD ) ,         (14)                                 |GD | e nτ
                                                                                            τ ∈GD
                                                   29370
A higher GS indicates better decomposition and                Task
relational granularity.                                       Goal:
                                                              Given a source document and an extracted triple [head, relation, tail],
                                                              determine whether the triple is factually consistent with the document by
Completeness Score. Completeness Score (CS)                   assigning a binary label: CONSISTENT or INCONSISTENT.
                                                              Provided Materials:
measures how well the extracted KG GD covers                  - Source document
                                                              - Model-extracted triple
the gold relational information in the ground-truth           - Ground-truth KG (for reference)

KG GD ∗ . Let v and v ∗ denote the embedding                  Notes
                 τ       τ                                    -   The ground-truth KG may be non-exhaustive. Do not mark a triple as an
vectors of an extracted triple τ and a gold triple τ ∗ ,      -
                                                                  error solely because it does not appear in the ground-truth KG.
                                                                  Following an OpenIE-style, recall-maximizing criterion, label a triple as
respectively. CS is computed as:                                  CONSISTENT if it can be reasonably inferred from the document and is
                                                                  semantically and structurally valid, even if it is absent from the ground-
                                                                  truth KG.
         ∗          1 X                                       -   Allowed inference is limited to document-grounded reasoning (e.g., clear
CS(GD , GD )=        ∗|     1[max cos(vτ , vτ ∗ ) ≥ ϕ],           coreference, explicitly stated appositions/aliases, and cross-sentence
                  |GD   ∗ ∗
                              τ ∈GD                               entailment directly supported by the document).
                       τ ∈GD                                  -   Disallowed inference includes relying on external/world knowledge or
                                                                  making speculative assumptions beyond textual evidence.
                                              (18)            -   Surface form matters: even if the meaning matches the document, mark
                                                                  a triple INCONSISTENT if the entity mentions are unclear or verbose.
where ϕ is a cosine similarity threshold. A higher
                                                              Decision Criteria
CS indicates stronger coverage of relational infor-           CONSISTENT (all must hold)
                                                              - Document support: the document entails the relation stated in the triple.
mation in the ground-truth KG.                                - Semantic alignment: the relation meaning matches the document
                                                                (no contradiction or polarity reversal).
                                                              - Correct directionality: head–tail roles align with the document
C    Implementation Details                                     (no subject–object swap).
                                                              - Entity grounding: head and tail refer to identifiable mentions in the
                                                                document (or unambiguous coreference).
We fine-tune an LLM as a KG refiner in a super-               - Appropriate surface form: entity mentions are clear and concise.
                                                              INCONSISTENT (any applies)
vised manner to judge and refine triples from a draft         - The triple is unsupported by the document or contradicts it.
                                                              - The meaning is distorted due to incorrect entity/relation interpretation.
KG based on evidence in the source document. The              - The triple has incorrect directionality (subject–object reversal).
                                                              - The entities are not grounded in the document
model is optimized with Adam (Kingma, 2014) us-                 (hallucinated or unresolvable mentions).
                                                              - Inappropriate surface form: entity mentions are unclear or overly verbose.
ing 100 warmup steps, a learning rate of 3 × 10−4 ,
and 500 training steps. We train with a global batch
                                                           Figure 7: Guidelines for triple consistency annotation.
size of 128 by using a micro-batch size of 8 and ac-
cumulating gradients for 16 steps. We apply LoRA
to the q_proj and v_proj projection layers, with           evaluation was performed by three graduate student
r = 8, α = 16, and dropout = 0.05. For valida-             annotators with sufficient expertise in KG construc-
tion, we sample 2,000 instances from the training          tion. During evaluation, annotators were provided
set for GenWiki and DocRED, and use a 0.2 split            with the source document, the ground-truth KG,
for SciERC and CDR. All results are reported as the        and the model-extracted KG for triple-level consis-
average performance across three independent runs,         tency assessment. We also adopted an evaluation
each using a different random seed for generating          criterion that maximizes recall from an OpenIE per-
the dataset used for fine-tuning. All experiments          spective (Niklaus et al., 2018). Specifically, we did
are conducted on two NVIDIA V100 32GB GPUs.                not count a triple as an error if it does not explicitly
                                                           appear in the ground-truth KG, as long as it can
D    Factual Inconsistencies in GKE                        be reasonably inferred from the document and is
In this section, we describe the evaluation protocol       semantically and structurally valid.
used to analyze the factual inconsistency types and        Stage 1 (Binary Consistency). The annotators
report detailed quantitative statistics.                   independently and blindly judged whether each
D.1 Sampling and Evaluation Protocol                       triple is factually consistent with the source docu-
                                                           ment at the triple level. We used majority voting (at
To comprehensively examine factual inconsisten-
                                                           least 2 out of 3 annotators) to determine the final
cies arising in GKE, we conducted a human evalu-
                                                           binary label, and a total of 283 triples were labeled
ation by sampling KGs generated by multiple mod-
                                                           as inconsistent and forwarded to Stage 2. The an-
els across four datasets (GenWiki, DocRED, Sci-
                                                           notators followed a written guideline (Figure 7),
ERC, and CDR). To cover a diverse range of LLM
                                                           and we report Fleiss’ κ to measure inter-annotator
families and model sizes, we considered Llama-2-
                                                           agreement on this binary decision (κ = 0.67).
7B, Llama-2-13B, GPT-4o-mini, and GPT-4o. For
each dataset, we sampled 20 KGs (4 models × 5              Stage 2 (Inconsistency Typing). For triples la-
KGs each), resulting in 80 KGs evaluated at the            beled as inconsistent, the annotators jointly per-
triple level with a total of 658 triples. This human       formed a consolidation step to assign a salient in-
                                                      29371
consistency type. During this stage, we excluded                           Correctness-level      Representation-level              N/A

minor surface variations that do not affect factual      UH         5.6%
meaning or were too sparse and idiosyncratic to          MI         5.3%
form a meaningful category. As a result, among the       EA              10.6%
                                                         EV               12.0%
283 inconsistent triples, we assigned types to 221
                                                        N/A                                                                          66.4%
triples using four categories, {UH, MI, EA, EV}.          0%         10%        20%        30%    40%       50%          60%          70%
Each triple received a single primary type, and final
labels were determined by consensus.                    Figure 8: Proportions of factual inconsistencies in sam-
                                                        pled KG triples. N/A includes valid triples and minor
D.2 Types of Factual Inconsistencies                    inconsistencies not covered by the four types.

We now provide a detailed description of the factual                                                        Corr.-level        Repr.-level
                                                         Domain        Dataset          Model       NT
inconsistencies observed in our human evaluation.                                                           UH      MI         EA         EV
                                                                                    Llama-2-7B      2.6      -       -          1         2
Correctness-level Inconsistencies. This type                          GenWiki
                                                                                    Llama-2-13B     3.2      2       1          2         1
                                                                                    GPT-4o-mini     3.4      1       -          3         2
captures cases where a triple is factually incorrect                                GPT-4o          3.2      1       -          1         -
                                                         General
with respect to the source document, thus should be                                 Llama-2-7B      2.6      1       -          3          4
                                                                                    Llama-2-13B     7.8      4       2          5          6
removed or corrected. Such errors undermine the                       DocRED
                                                                                    GPT-4o-mini     19       3       4         10         10
factual correctness of the KG and can substantially                                 GPT-4o          19       2       2          6         12
degrade its overall reliability. We categorize such                                 Llama-2-7B      3.6      1       2          2         2
                                                                                    Llama-2-13B     5.6      2       3          3         3
cases into the following subtypes:                                     SciERC
                                                                                    GPT-4o-mini     9.8      4       4          6         7
                                                                                    GPT-4o          11.4     1       1          5         6
                                                         Specific
   • Unsupported Hallucination (UH): Triples                                        Llama-2-7B      1.8      1       -          2          1
                                                                                    Llama-2-13B      6       4       3          4          5
     generated without document support, includ-                        CDR
                                                                                    GPT-4o-mini     17.2     6       8         10         11
     ing cases where the fact is absent from or                                     GPT-4o          15.4     3       5          7          8

     contradicts the source document.
                                                        Table 5: Counts of factual inconsistency types by dataset
                                                        and model. NT denote the average number of triples in
   • Misidentification (MI): Factually incorrect        the extracted KGs.
     triples caused by misidentified entities or re-
     lations, which can often be corrected using
     evidence from the source document.                 D.3     Quantitative Analysis
                                                        Figure 8 shows the distribution of factual inconsis-
Representation-level Inconsistencies. This type         tencies. Among all sampled triples, Correctness-
captures cases where a triple is not necessarily in-    level inconsistencies account for 10.9% (UH+MI),
correct, but its surface form is inconsistent or sub-   while Representation-level inconsistencies account
optimal, and thus can be improved via normaliza-        for 22.6% (EA+EV), indicating that representa-
tion or rewriting. Due to such representational is-     tion issues are more frequent than strictly factual
sues, the resulting KG is not sufficiently consistent   errors. This observation suggests that, beyond fil-
with the facts stated in the source document. We        tering clearly incorrect triples, improving extracted
categorize such cases into the following subtypes:      KGs often requires normalization and rewriting. In
                                                        Table 5, the Corr.-level and Repr.-level columns
   • Entity Ambiguity (EA): Entity spans are            report the counts of inconsistencies observed for
     overly narrow, yielding incomplete mentions        each model on each dataset. Smaller models yield
     (e.g., missing modifiers) that make the entity     fewer inconsistencies in absolute terms because
     reference ambiguous or unstable.                   they produce smaller KGs, but they may exhibit
                                                        a higher inconsistency density. Larger models re-
   • Entity Verbosity (EV): Entity spans are            duce inconsistency rates, but non-trivial inconsis-
     overly broad, including unnecessary surround-      tencies remain across all datasets. Notably, while
     ing words or clauses (e.g., descriptive phrases)   LLMs tend to perform better on general-domain
     that introduce redundancy and reduce repre-        extraction, inconsistency issues persist across both
     sentational consistency.                           general- and domain-specific datasets, indicating
                                                        that refinement is needed regardless of domain.
                                                   29372
                                     Number of Triples                Dataset    Method   BLF1    ROF1    BSF1     TS      US      FS      GS      CS
 Dataset      Model
                                                                                 D+F+R    52.94   38.29   75.83   36.72   89.46   96.38   97.10   59.63
                          Draft KG    GraphJudge    GraphRefine
                                                                                 D+F      50.10   36.91   72.69   25.43   83.59   96.17   96.35   57.49
                                                                      DocRED
            Llama-2-7B      2.35          2.08           2.16                    D+R      49.84   37.23   71.08   27.29   88.13   95.32   97.21   55.15
                                                                                 D        47.32   35.80   68.64   18.31   78.49   96.10   95.82   52.96
            Llama-2-13B     3.75          3.27           3.34
 GenWiki                                                                         D+F+R    55.43   48.07   76.29   79.61   94.42   97.31   86.33   56.75
            GPT-4o-mini     4.14          3.79           3.95
                                                                                 D+F      53.26   46.74   74.85   74.58   88.25   97.20   83.76   53.82
            GPT-4o          4.22          3.97           4.08         SciERC
                                                                                 D+R      52.84   46.38   73.18   76.14   92.61   96.45   85.61   50.37
            Llama-2-7B     4.84           3.79           4.11                    D        51.76   46.12   72.80   72.69   86.03   97.35   83.39   48.13
            Llama-2-13B    11.20          8.01           9.42
 DocRED
            GPT-4o-mini    20.48         15.82           18.13       Table 7: Ablation results of GraphRefine on DocRED
            GPT-4o         20.51         16.17           19.40
                                                                     and SciERC. D, F, and R denote DELETE, FIX, and
            Llama-2-7B      3.32          2.74           2.86
            Llama-2-13B     3.77          3.33           3.38
                                                                     REWRITE, respectively.
 SciERC
            GPT-4o-mini     9.68          7.20           8.01
            GPT-4o          9.83          7.49           8.54
            Llama-2-7B     1.47           1.14           1.25        Results by Base Extractor. For Llama-2-7B and
            Llama-2-13B    4.63           3.67           3.81
    CDR
            GPT-4o-mini    15.88         10.72           12.47
                                                                     Llama-2-13B, GraphRefine exhibits larger gains
            GPT-4o         14.46         10.94           12.11       across all datasets, with repeated improvements in
                                                                     GS and US. By contrast, GraphJudge may improve
Table 6: Average number of triples per KG after apply-               some accuracy metrics but is often accompanied by
ing GraphJudge or GraphRefine to draft KGs generated
                                                                     decreases in CS and representation-level measures,
by different base extractors across datasets.
                                                                     highlighting the limitation of deletion-based refine-
                                                                     ment. For GPT-4o-mini and GPT-4o, GraphRefine
E     Additional Experimental Results                                provides consistent additional gains even on high-
                                                                     quality draft KGs, with stable improvements par-
In this appendix, we provide additional results to                   ticularly on representation quality (GS/US). This
complement the main experiments.                                     indicates that GraphRefine does not depend heavily
                                                                     on model-specific error patterns. Instead, it lever-
E.1       GraphJudge vs. GraphRefine
                                                                     ages document-grounded correction and rewriting
Figure 9 presents a multi-metric comparison of ap-                   to function as a post-hoc module across diverse
plying GraphJudge and GraphRefine to draft KGs                       base extractors.
generated across all base extractors and datasets.
                                                                     Summary across Datasets. Across GenWiki,
Overall Results. GraphRefine forms the largest                       DocRED, SciERC, and CDR, GraphRefine con-
polygon in most settings, showing consistent im-                     sistently improves BL/RO/BS-F1 while preserving
provements not only on accuracy-oriented metrics                     factualness (FS) and enhancing granularity (GS)
(BL/RO/BS-F1) but also on the multi-faceted Gen-                     and uniqueness (US). Coverage (CS) is also often
RES measures. In particular, it repeatedly improves                  maintained or improved, indicating that the gains
granularity (GS) and uniqueness (US) while main-                     come from document-aligned correction and rewrit-
taining or improving factualness (FS). Coverage                      ing rather than simple removal. Overall, these re-
(CS) is also largely preserved or improved, suggest-                 sults suggest that GraphRefine provides reliable
ing that GraphRefine enhances KG quality through                     post-hoc improvements across domains.
document-grounded correction and rewriting rather
than simple deletion. In contrast, GraphJudge tends                  E.2        Ablation Study
to reduce correctness-level errors, but its deletion-                Table 7 reports an ablation analysis of the refine-
based filtering often decreases CS and is frequently                 ment operations used in GraphRefine. D+F+R de-
accompanied by drops in representation quality                       notes the full model trained with D ELETE, F IX,
(GS/US). This trade-off is more pronounced for                       and R EWRITE. D+F and D+R are trained with-
the Llama-2 family (7B and 13B), which produces                      out R EWRITE and without F IX, respectively. D is
relatively smaller draft KGs, where additional re-                   trained with D ELETE only.
movals more directly translate into coverage loss.
Table 6 compares the average number of triples per                   Overall Results. The D+F+R demonstrates the
KG for the draft KGs and the outputs after applying                  strongest overall performance across both datasets,
GraphJudge or GraphRefine. The results confirm                       suggesting that a richer set of refinement operations
that GraphJudge substantially reduces the absolute                   improves KG quality more effectively than dele-
number of triples, whereas GraphRefine preserves                     tion alone. In particular, compared to D, D+F+R
the triple count to a much greater extent.                           yields the largest gains on accuracy-oriented met-
                                                                  29373
rics (BL/RO/BS) and also attains the best scores on     this filtering suppresses noise and reduces factual
representation-related measures, including TS and       errors, it also discards several document-grounded
US. Overall, these results indicate that GraphRe-       details (e.g., multiple beneficiary groups and top-
fine is most effective when it combines document-       ical descriptors), leading to lower coverage and a
grounded correction with rewriting, rather than re-     sparser description of the entity. Moreover, dele-
lying solely on deletion.                               tion does not resolve reference fragmentation: the
                                                        output still distributes information across UNESCO
Limitations of Deletion-Only Refinement. The            Confucius Prize for Literacy, Confucius Prize, and
deletion-only model variant (D) performs the worst      Prize, so the graph remains partially disconnected
on BL/RO/BS across both datasets and shows par-         even when remaining triples are correct.
ticularly large drops in TS and US, indicating de-
graded representation quality. It also achieves the     GraphRefine. By contrast, GraphRefine edits
lowest coverage (CS), suggesting that deletion-         triples via F IX/R EWRITE operations rather than re-
based refinement can reduce errors but often does       moval. A representative change is rewriting an am-
so at the expense of information retention and struc-   biguous subject such as Prize into UNESCO Con-
tural quality. By contrast, D+F+R improves both         fucius Prize for Literacy, which consolidates previ-
correctness- and representation-related measures        ously scattered attributes and improves connectiv-
while maintaining higher CS, supporting the view        ity. Moreover, R EWRITE makes triples more con-
that effective refinement requires editing operations   cise and atomic by splitting conflated phrases (e.g.,
beyond mere removal.                                    separating rural adults and out-of-school youth)
                                                        and removing unnecessary modifiers while staying
Contributions of F IX and R EWRITE. Both                faithful to the document. Overall, GraphRefine pre-
D+F and D+R outperform D, indicating comple-            serves most information from the draft KG while
mentary contributions from F IX and R EWRITE.           presenting it in a clearer, more compact form with-
D+F consistently improves BL/RO/BS and also             out substantially reducing the triple count.
increases FS and CS, suggesting that F IX corrects
erroneous entities or relations using document evi-     F   Prompts
dence and strengthens factual correctness while pre-
                                                        For KG refinement, we input the source document
serving coverage. Meanwhile, D+R yields stronger
                                                        together with a candidate triple and ask the model
gains on representation-oriented measures, espe-
                                                        to select an operation and produce the correspond-
cially US and GS, indicating that R EWRITE re-
                                                        ing output triple grounded in document evidence.
duces redundancy and rewrites triples into more
                                                        Figure 10 shows the refinement prompt, which in-
atomic and consistent surface forms. Nevertheless,
                                                        structs the model to choose an appropriate opera-
D+R shows lower CS in some settings, implying
                                                        tion and return either the original triple, a deletion
that rewriting alone may not fully preserve sup-
                                                        decision, or a corrected/rewritten triple.
ported information and that combining it with F IX
                                                           For knowledge extraction, we prompt an LLM to
leads to the most stable improvements.
                                                        extract relations between entities in a document as
E.3   Case Study                                        [head, relation, tail] triples, using dataset-specific
                                                        templates across domains following the prompt
Table 8 compares the outputs on the same source
                                                        design of Jiang et al. (2024). Figure 11 presents the
document from (i) a draft KG generated by GPT-
                                                        extraction prompt for the general-domain datasets
4o-mini, (ii) GraphJudge, and (iii) GraphRefine.
                                                        GenWiki and DocRED, which restricts the output
While the draft KG captures many core facts, it is
                                                        to a list of triples and includes two-shot examples.
fragmented by inconsistent entity mentions. For
                                                        Figures 12 and 13 provide analogous prompts for
example, the same target is referred to as UNESCO
                                                        the scientific dataset SciERC and the biomedical
Confucius Prize for Literacy, Confucius Prize, and
                                                        dataset CDR, respectively, tailored to each domain
Prize, which splits information across multiple sub-
                                                        while preserving the same output constraints and
ject nodes and weakens graph connectivity.
                                                        two-shot examples.
GraphJudge. GraphJudge improves KG quality
primarily through deletion. As shown in Table 8,
it removes many triples that are likely uncertain or
weakly supported, resulting in a smaller KG. While
                                                   29374
                                     Draft KG (Llama-2-7B)              + GraphJudge           + GraphRefine
           BS-F1                                 BS-F1                                 BS-F1                               BS-F1
     TS             RO-F1            TS                       RO-F1            TS                   RO-F1             TS           RO-F1



US                        BL-F1 US                                  BL-F1 US                               BL-F1 US                      BL-F1



     FS              CS              FS                        CS              FS                     CS              FS            CS
            GS                                    GS                                    GS                                  GS
          GenWiki                              DocRED                                  SciERC                              CDR
                                     Draft KG (Llama-2-13B)             + GraphJudge            + GraphRefine
           BS-F1                                 BS-F1                                 BS-F1                               BS-F1
     TS             RO-F1            TS                       RO-F1            TS                   RO-F1             TS           RO-F1



US                        BL-F1 US                                  BL-F1 US                               BL-F1 US                      BL-F1



     FS              CS              FS                        CS              FS                     CS              FS            CS
            GS                                    GS                                    GS                                  GS
          GenWiki                              DocRED                                  SciERC                              CDR
                                     Draft KG (GPT-4o-mini)             + GraphJudge           + GraphRefine
           BS-F1                                 BS-F1                                 BS-F1                               BS-F1
     TS             RO-F1            TS                       RO-F1            TS                   RO-F1             TS           RO-F1



US                        BL-F1 US                                  BL-F1 US                               BL-F1 US                      BL-F1



     FS              CS              FS                        CS              FS                     CS              FS            CS
            GS                                    GS                                    GS                                  GS
          GenWiki                              DocRED                                  SciERC                              CDR
                                          Draft KG (GPT-4o)           + GraphJudge        + GraphRefine
           BS-F1                                 BS-F1                                 BS-F1                               BS-F1
     TS             RO-F1            TS                       RO-F1            TS                   RO-F1             TS           RO-F1



US                        BL-F1 US                                  BL-F1 US                               BL-F1 US                      BL-F1



     FS              CS              FS                        CS              FS                     CS              FS            CS
            GS                                    GS                                    GS                                  GS
          GenWiki                              DocRED                                  SciERC                              CDR

Figure 9: Multi-aspect comparisons between draft KGs and their post-hoc refinement results using GraphJudge
or GraphRefine across all datasets (GenWiki, DocRED, SciERC, and CDR) and base extractors (Llama-2-7B,
Llama-2-13B, GPT-4o-mini, and GPT-4o). For visual clarity, all metrics are min–max normalized and rescaled to
[0.7, 0.9].




                                                                      29375
Original Document:
The UNESCO Confucius Prize for Literacy recognizes the activities of outstanding individuals, governments or governmental
agencies and non-governmental organizations (NGOs) working in literacy serving rural adults and out-of-school youth, particu-
larly women and girls. The Prize was established in 2005 through the support of the Government of the People’s Republic of
China in honour of the great Chinese scholar Confucius. It is part of the International Literacy Prizes, which UNESCO awards
every year in recognition of excellence and inspiring experiences in the field of literacy throughout the world. The Confucius
Prize offers two awards of US$ 20,000 each, a medal and a diploma, as well as a study visit to literacy project sites in China.
The Prize is open to institutions, organizations or individuals displaying outstanding merit in literacy, achieving particularly
effective results and promoting innovative approaches. The selection of prizewinners is made by an International Jury appointed
by UNESCO’s Director-General, which meets in Paris once a year. The Prize is awarded at an official ceremony held for that
purpose at UNESCO Headquarters in Paris on the occasion of International Literacy Day (8 September).
Draft KG (GPT-4o-mini):
                                             [ [’UNESCO Confucius Prize for Literacy’, ’recognizes’, ’activities of outstanding
                                             individuals’], [’UNESCO Confucius Prize for Literacy’, ’recognizes’, ’governments
                                             or governmental agencies’], [’UNESCO Confucius Prize for Literacy’, ’recognizes’,
                                             ’non-governmental organizations (NGOs)’], [’UNESCO Confucius Prize for Literacy’,
                                             ’serving’, ’rural adults and out-of-school youth’], [’UNESCO Confucius Prize for
                                             Literacy’, ’serving’, ’particularly women and girls’], [’Prize’, ’established
                                             in’, ’2005’], [’Prize’, ’through the support of’, "Government of the People’s
                                             Republic of China"], [’Prize’, ’in honour of’, ’great Chinese scholar Confucius’],
                                             [’Prize’, ’part of’, ’International Literacy Prizes’], [’International Literacy
                                             Prizes’, ’awarded by’, ’UNESCO’], [’UNESCO’, ’awards’, ’every year’], [’UNESCO’,
                                             ’recognition of’, ’excellence and inspiring experiences in the field of literacy’],
                                             [’Confucius Prize’, ’offers’, ’two awards of US$ 20,000 each’], [’Confucius Prize’,
                                             ’offers’, ’a medal’], [’Confucius Prize’, ’offers’, ’a diploma’], [’Confucius
                                             Prize’, ’offers’, ’study visit to literacy project sites in China’], [’Prize’, ’open
                                             to’, ’institutions’], [’Prize’, ’open to’, ’organizations’], [’Prize’, ’open to’,
                                             ’individuals’], [’Prize’, ’displaying’, ’outstanding merit in literacy’], [’Prize’,
                                             ’achieving’, ’particularly effective results’], [’Prize’, ’promoting’, ’innovative
                                             approaches’], [’Prize’, ’awarded at’, ’official ceremony’], [’official ceremony’,
                                             ’held at’, ’UNESCO Headquarters in Paris’], [’official ceremony’, ’on the occasion
                                             of’, ’International Literacy Day (8 September)’] ]



GraphJudge:
                                             [ [’UNESCO Confucius Prize for Literacy’, ’recognizes’, ’activities of outstanding
                                             individuals’] [’Prize’, ’established in’, ’2005’] [’Prize’, ’through the support of’,
                                             "Government of the People’s Republic of China"] [’Prize’, ’in honour of’, ’great
                                             Chinese scholar Confucius’] [’Prize’, ’part of’, ’International Literacy Prizes’]
                                             [’International Literacy Prizes’, ’awarded by’, ’UNESCO’] [’Confucius Prize’,
                                             ’offers’, ’two awards of US$ 20,000 each’] [’Confucius Prize’, ’offers’, ’a medal’]
                                             [’Confucius Prize’, ’offers’, ’a diploma’] [’Confucius Prize’, ’offers’, ’study visit
                                             to literacy project sites in China’] [’Prize’, ’open to’, ’institutions’] [’Prize’,
                                             ’open to’, ’organizations’] [’Prize’, ’open to’, ’individuals’] [’Prize’, ’awarded
                                             at’, ’official ceremony’] [’official ceremony’, ’held at’, ’UNESCO Headquarters in
                                             Paris’] [’official ceremony’, ’on the occasion of’, ’International Literacy Day (8
                                             September)’] ]



GraphRefine:
                                             [ [’UNESCO Confucius Prize for Literacy’, ’recognizes’, ’outstanding individuals’],
                                             [’UNESCO Confucius Prize for Literacy’, ’recognizes’, ’governmental agencies’],
                                             [’UNESCO Confucius Prize for Literacy’, ’recognizes’, ’non-governmental organizations
                                             (NGOs)’], [’UNESCO Confucius Prize for Literacy’, ’serves’, ’rural adults’], [’UNESCO
                                             Confucius Prize for Literacy’, ’serves’, ’out-of-school youth’], [’UNESCO Confucius
                                             Prize for Literacy’, ’serves’, ’women and girls’], [’UNESCO Confucius Prize
                                             for Literacy’, ’was established in’, ’2005’], [’UNESCO Confucius Prize for
                                             Literacy’, ’supported by’, "Government of the People’s Republic of China"], [’UNESCO
                                             Confucius Prize for Literacy’, ’honors’, ’Confucius’], [’UNESCO Confucius Prize for
                                             Literacy’, ’is part of’, ’International Literacy Prizes’], [’International Literacy
                                             Prizes’, ’awarded by’, ’UNESCO’], [’UNESCO’, ’awards’, ’International Literacy
                                             Prizes annually’], [’International Literacy Prizes’, ’recognize’, ’excellence in
                                             literacy’], [’UNESCO Confucius Prize for Literacy’, ’offers’, ’two awards of US$
                                             20,000’], [’UNESCO Confucius Prize for Literacy’, ’offers’, ’medal’], [’UNESCO
                                             Confucius Prize for Literacy’, ’offers’, ’diploma’], [’UNESCO Confucius Prize
                                             for Literacy’, ’offers’, ’study visit to China’], [’UNESCO Confucius Prize for
                                             Literacy’, ’is open to’, ’institutions’], [’UNESCO Confucius Prize for Literacy’,
                                             ’is open to’, ’organizations’], [’UNESCO Confucius Prize for Literacy’, ’is open
                                             to’, ’individuals’], [’UNESCO Confucius Prize for Literacy’, ’is awarded at’,
                                             ’official ceremony’], [’official ceremony’, ’held at’, ’UNESCO Headquarters, Paris’],
                                             [’official ceremony’, ’held on’, ’International Literacy Day (8 September)’] ]




  Table 8: Qualitative comparison of knowledge extraction results (Draft KG vs. GraphJudge vs. GraphRefine).



                                                            29376
Knowledge Graph Refinement
Given a source document and a candidate triple extracted from it , decide how to refine
the triple based only on the document .

Choose exactly one operation : KEEP , DELETE , FIX , or REWRITE .
- KEEP : The triple is supported by the document and its representation is appropriate ;
keep it unchanged .
- DELETE : The triple is not supported by the document or is factually wrong ; remove it .
- FIX : The document supports a similar fact , but the candidate triple 's meaning is
incorrect (e . g. , wrong entity or relation ) ; correct it to match the document .
- REWRITE : The fact is supported , but the triple 's representation is ambiguous ,
imprecise , or verbose ; rewrite it to a clearer and more consistent form without changing
the meaning .

Return ONLY one item in the exact format :
[' OPERATION ', [' ENTITY 1 ', ' RELATIONSHIP ', ' ENTITY 2 ']]
If OPERATION is DELETE , return :
[' DELETE ' , [' NULL ', 'NULL ' , 'NULL ']]

Do not output any additional text . Do not explain your decision .

document : $TEXT$
triple : $TRIPLE$
output :



                             Figure 10: Prompt template for KG refinement.




General-Domain Knowledge Extraction
Given a prompt , identify and list the relationships between entities within the text .
Extract relationships both within a single sentence ( intra - sentence ) and across multiple
sentences ( inter - sentence ) .
Provide a list of triplets in the format [' ENTITY 1 ' , ' RELATIONSHIP ', ' ENTITY 2 ']. The
relationship is directed , so the order of entities in each triplet matters .
The output should only be a list of triplets ([[ ' ENTITY 1 ', ' RELATIONSHIP ', ' ENTITY 2 '] ,
...]) without any additional information . Do not explain how you extract them .

Example 1:
prompt : In 2020 , the Nobel Peace Prize was awarded to the World Food Programme for its
efforts to combat hunger . The organization has been operational since 1961.
relations :
[[ ' Nobel Peace Prize ', ' awarded in ' , '2020 '] , [ ' Nobel Peace Prize ', ' awarded to ', ' World
Food Programme '] , [ ' World Food Programme ', ' efforts to ', ' combat hunger '] , [' World Food
Programme ', ' operational since ', '1961 ']]

Example 2:
prompt : The Great Barrier Reef , located off the coast of Australia , is the world 's
largest coral reef system . It has been severely affected by climate change , leading to
coral bleaching .
relations :
[[ ' Great Barrier Reef ', ' located at ', ' coast of Australia '] , [ ' Great Barrier Reef ', 'is ',
' world 's largest coral reef system '] , [ ' Great Barrier Reef ', ' affected by ', ' climate
change '] , [ ' Climate change ', ' leads to ', ' coral bleaching ']]

prompt : $TEXT$
relations :



     Figure 11: Prompt template for general-domain knowledge extraction (GenWiki and DocRED).




                                                 29377
Scientific-Domain Knowledge Extraction
Given a prompt , identify and list the relationships between entities within the text .
Extract relationships both within a single sentence ( intra - sentence ) and across multiple
sentences ( inter - sentence ) .
Provide a list of triplets in the format [' ENTITY 1 ' , ' RELATIONSHIP ', ' ENTITY 2 ']. The
relationship is directed , so the order of entities in each triplet matters .
The output should only be a list of triplets ([[ ' ENTITY 1 ', ' RELATIONSHIP ', ' ENTITY 2 '] ,
...]) without any additional information . Do not explain how you extract them .

Example 1:
prompt : Sources of training data suitable for language modeling of conversational speech
are limited . In this paper , we show how training data can be supplemented with text
from the web filtered to match the style and / or topic of the target recognition task ,
but also that it is possible to get bigger performance gains from the data by using
class - dependent interpolation of N - grams .
relations :
[[ ' conversational speech ', ' used for ', ' language modeling '] , [ ' class - dependent
interpolation of N - grams ', ' used for ', ' recognition task ']]

Example 2:
prompt : We propose a draft scheme of the model formalizing the structure of communicative
context in dialogue interaction . The relationships between the interacting partners are
considered as system of three automata representing the partners of the dialogue and
environment .
relations :
[[ ' model ' , ' used for ', ' structure of communicative context '] , [ ' dialogue interaction ',
' feature of ', ' structure of communicative context ']]

prompt : $TEXT$
relations :



           Figure 12: Prompt template for scientific-domain knowledge extraction (SciERC).




Biomedical-Domain Knowledge Extraction
Given a prompt , identify and list the relationships between entities within the text .
Extract relationships both within a single sentence ( intra - sentence ) and across multiple
sentences ( inter - sentence ) .
Provide a list of triplets in the format [' ENTITY 1 ' , ' RELATIONSHIP ', ' ENTITY 2 ']. The
relationship is directed , so the order of entities in each triplet matters .
The output should only be a list of triplets ([[ ' ENTITY 1 ', ' RELATIONSHIP ', ' ENTITY 2 '] ,
...]) without any additional information . Do not explain how you extract them .

Example 1:
prompt : Penicillin is an antibiotic that treats bacterial infections . It was discovered
by Alexander Fleming .
relations :
[[ ' Penicillin ', 'is a type of ', ' antibiotic '] , [ ' Penicillin ', ' treats ' , ' bacterial
infections '] , [' Penicillin ', ' discovered by ', ' Alexander Fleming ']]

Example 2:
prompt : Metformin is commonly prescribed for managing type 2 diabetes . It helps by
lowering glucose production in the liver and increasing the body 's sensitivity to insulin .
relations :
[[ ' Metformin ', 'is prescribed for ', ' managing type 2 diabetes '] , [' Metformin ', ' helps
by ', ' lowering glucose production in the liver '] , [' Metformin ', ' increases ', ' body \'s
sensitivity to insulin ']]

prompt : $TEXT$
relations :



            Figure 13: Prompt template for biomedical-domain knowledge extraction (CDR).



                                               29378
