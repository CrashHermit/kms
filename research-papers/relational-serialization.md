 How to Talk to Language Models: Serialization Strategies for Structured
                          Entity Matching
                                                    Haoteng Yin*
                                                   Purdue University
                                                  yinht@purdue.edu

         Jinha Kim and Prashant Mathur and Krishanu Sarker and Vidit Bansal
                                      Amazon
                   {jinhak,pramathu,kssarker,bansalv}@amazon.com

                       Abstract                               Classic EM tasks (Getoor and Machanavajjhala,
                                                              2012) focus on structured data in relational tables
     Entity matching (EM), which identifies whether
                                                              with homogeneous schemas, while more recent
     two data records refer to the same real-world
     entity, is crucial for knowledge base construc-          studies (Köpcke et al., 2010; Sun et al., 2020; Wang
     tion and enhancing data-driven AI systems. Re-           et al., 2021) have expanded to semi-structured (e.g.,
     cent advances in language models (LMs) have              JSON, XML), unstructured (e.g., text) and com-
     shown great potential in resolving entities with         plex relational data (e.g., KGs), as well as entities
     rich textual attributes. However, their perfor-          with missing and/or noisy attributes (see Fig. 1).
     mance heavily depends on how structured en-              The complexity of EM lies in identifying and link-
     tities are "talked" through serialized text. The
                                                              ing inconsistent, incomplete, denormalized records
     impact of this serialization process remains un-
     derexplored, particularly for entities with com-         across multiple data sources. Typically, EM in-
     plex relations in knowledge graphs (KGs). In             volves two steps: blocking, which eliminates clear
     this work, we systematically study entity serial-        non-matches to reduce the number of pairwise
     ization by benchmarking the effect of common             comparisons, and matching, which identifies true
     schemes with LMs of different sizes on diverse           matches from the filtered candidate set. While
     tabular matching datasets. We apply our find-            rule-based (Fan et al., 2009; Singh et al., 2017)
     ings to propose a novel serialization scheme             and traditional learning-based (Konda et al., 2016)
     for KG entities based on random walks and uti-
                                                              matchers work well for tabular data, they strug-
     lize LLMs to encode sampled semantic walks
     for matching. Using this lightweight approach            gle with tasks that involve noisy or unstructured
     with open-source LLMs, we achieve a leading              data (Mudgal et al., 2018). Neural network-based
     performance on EM in canonical and highly                methods, particularly those that utilize word em-
     heterogeneous KGs, demonstrating significant             beddings and sequence models (Manning, 2017),
     throughput increases and superior robustness             have demonstrated their effectiveness in matching
     compared to GPT-4-based methods. Our study               entities with rich textual attributes (Parikh et al.,
     on serialization provides valuable insights for
                                                              2016; Ebraheem et al., 2018).
     the deployment of LMs in real-world EM tasks.
                                                                 Recent research (Brunner and Stockinger, 2020;
1    Introduction                                             Li et al., 2020; Peeters and Bizer, 2021; Paganelli
                                                              et al., 2022; Akbarian Rastaghi et al., 2022; Wang
Entity matching (EM) aims to identify whether                 et al., 2022; Zeakis et al., 2023; Peeters and Bizer,
two data records refer to the same real-world en-             2025; Wang et al., 2025; Wadhwa et al., 2024;
tity, even if their descriptions differ (Herzog et al.,       Huang and Zhao, 2024) has explored using pre-
2007). As a core challenge in data cleaning and               trained encoder models and large language mod-
integration, EM has a wide range of applications,             els (LLMs) to resolve and match entities based
from knowledge base construction to empowering                on their textual attributes. Modern language mod-
data-driven AI systems (Diefenbach et al., 2018;              els can generate highly contextualized embeddings
Guu et al., 2020; Frey et al., 2023). The flourishing         that capture semantic meaning across the entire
of knowledge-based AI applications (e.g., recom-              input, greatly alleviating the word ambiguity prob-
mendation, question answering, and information re-            lem when comparing entity attributes. Particularly,
trieval) has particularly driven the need to integrate        with more grounded knowledge and generalization
entities from different knowledge graphs (KGs).               capability, LLMs show better performance than
    * Work done during an internship at Amazon.               classic pretrained encoder models on challenging
                                                         7851
                                Findings of the Association for Computational Linguistics:
                                              NAACL 2025, pages 7851–7865
                         April 29 - May 4, 2025 ©2025 Association for Computational Linguistics
EM tasks (Mudgal et al., 2018; Wang et al., 2021;           Semi-structured:                          Graph-structured:          Funded: August
                                                                                                                                     10, 1793

Jiang et al., 2024b). However, existing LLM-based           { "title": "chinois on main",
                                                              "phone": "(310)392-9025",                       is a     is located         is a
                                                                                                                            in
matchers use chat-based interactions, which are               "address": { "street": "2709 main st.",
                                                                             "city": "santa monica",
                                                                             "zipcode": "90405”},     Museum      Louvre           Paris       Place
severely limited by high latency, computational               "category": "french" }
                                                                                                      Walk-based Serialization
                                                                                                                                  Sampled Fact Path
                                                                                                                                      Step L=2    Wm
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             <latexit sha1_base64="HUO7GKc9Q0j7moHchFnHmmjfqLI=">AAAB6nicbVBNSwMxEJ3Ur1q/qh69BIvgqexKUQ8eCl48VrQf0C4lm2bb0CS7JFmhLP0JXjwo4tVf5M1/Y9ruQVsfDDzem2FmXpgIbqznfaPC2vrG5lZxu7Szu7d/UD48apk41ZQ1aSxi3QmJYYIr1rTcCtZJNCMyFKwdjm9nfvuJacNj9WgnCQskGSoecUqskx7afdkvV7yqNwdeJX5OKpCj0S9/9QYxTSVTlgpiTNf3EhtkRFtOBZuWeqlhCaFjMmRdRxWRzATZ/NQpPnPKAEexdqUsnqu/JzIijZnI0HVKYkdm2ZuJ/3nd1EbXQcZVklqm6GJRlApsYzz7Gw+4ZtSKiSOEau5uxXRENKHWpVNyIfjLL6+S1kXVv6zW7muV+k0eRxFO4BTOwYcrqMMdNKAJFIbwDK/whgR6Qe/oY9FaQPnMMfwB+vwBNXqNvg==</latexit>




complexity, and API call cost. As a result, previ-          Tabular:                                                     wm,1
                                                                                                                         <latexit sha1_base64="Xy5XYT++U7jswaBw8eK6tFMTWsM=">AAAB7nicbVBNSwMxEJ31s9avqkcvwSJ4kLIrRT14KHjxWMF+QLuUbJptQ5NsSLJKWfojvHhQxKu/x5v/xrTdg7Y+GHi8N8PMvEhxZqzvf3srq2vrG5uFreL2zu7efungsGmSVBPaIAlPdDvChnImacMyy2lbaYpFxGkrGt1O/dYj1YYl8sGOFQ0FHkgWM4Ktk1pPvUycB5NeqexX/BnQMglyUoYc9V7pq9tPSCqotIRjYzqBr2yYYW0Z4XRS7KaGKkxGeEA7jkosqAmz2bkTdOqUPooT7UpaNFN/T2RYGDMWkesU2A7NojcV//M6qY2vw4xJlVoqyXxRnHJkEzT9HfWZpsTysSOYaOZuRWSINSbWJVR0IQSLLy+T5kUluKxU76vl2k0eRwGO4QTOIIArqMEd1KEBBEbwDK/w5invxXv3PuatK14+cwR/4H3+AAX7j1s=</latexit>




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (“Louvre”, “is located in”, “Paris”)
                                                               title   address                 phone         category    wm,2
                                                                                                                         <latexit sha1_base64="4AZbpgaFyZdv6Cmm8m1frglb8vc=">AAAB7nicbVA9SwNBEJ2LXzF+RS1tFoNgIeEuBLWwCNhYRjAfkBxhb7OXLNndO3b3lHDkR9hYKGLr77Hz37iXXKGJDwYe780wMy+IOdPGdb+dwtr6xuZWcbu0s7u3f1A+PGrrKFGEtkjEI9UNsKacSdoyzHDajRXFIuC0E0xuM7/zSJVmkXww05j6Ao8kCxnBxkqdp0EqLmqzQbniVt050CrxclKBHM1B+as/jEgiqDSEY617nhsbP8XKMMLprNRPNI0xmeAR7VkqsaDaT+fnztCZVYYojJQtadBc/T2RYqH1VAS2U2Az1steJv7n9RITXvspk3FiqCSLRWHCkYlQ9jsaMkWJ4VNLMFHM3orIGCtMjE2oZEPwll9eJe1a1bus1u/rlcZNHkcRTuAUzsGDK2jAHTShBQQm8Ayv8ObEzovz7nwsWgtOPnMMf+B8/gAHgI9c</latexit>




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (“Paris”, “is a”, “Place”)
ous studies are unable to fully evaluate the target            21 club 21 w. 52nd st. new york 212/582 -7200 american                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            <latexit sha1_base64="Y2Qyu3F9tY1X7UH5srf59CWDfTo=">AAAB73icbVA9SwNBEJ2LXzF+RS1tFoMQm3AnQS0sAjaWEcwHJEfY2+wlS3b3zt09IRz5EzYWitj6d+z8N+4lV2jig4HHezPMzAtizrRx3W+nsLa+sblV3C7t7O7tH5QPj9o6ShShLRLxSHUDrClnkrYMM5x2Y0WxCDjtBJPbzO88UaVZJB/MNKa+wCPJQkawsVJX4LjaGYjzQbni1tw50CrxclKBHM1B+as/jEgiqDSEY617nhsbP8XKMMLprNRPNI0xmeAR7VkqsaDaT+f3ztCZVYYojJQtadBc/T2RYqH1VAS2U2Az1steJv7n9RITXvspk3FiqCSLRWHCkYlQ9jwaMkWJ4VNLMFHM3orIGCtMjI2oZEPwll9eJe2LmndZq9/XK42bPI4inMApVMGDK2jAHTShBQQ4PMMrvDmPzovz7nwsWgtOPnMMf+B8/gBUM49/</latexit>




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      map(Wm )
LLM without downsampling the test set (Peeters                 addr          city         phone1       type             class
                                                               2709 main st. santa monica 310-392-9025 pacific new wave NULL
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 “Louvre is located in Paris,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Paris is a Place.”

and Bizer, 2025; Li et al., 2024).
   To use language models (LMs) for matching,               Figure 1: Real-world matching tasks involve various
entities must first be serialized into sequences, a         types of structured entities, posing great challenges in
process known as entity serialization. For instance,        their serialization for applying language models.
a restaurant record comes with attributes of "title",
"address", and "phone". Its serialization can be            of zero-shot and with fine-tuning; RQ2 how dif-
done by sequentially concatenating all attributes,          ferent ways of handling missing or dirty values
which imposes an artificial order, placing "title"          affect the model; and RQ3 are special tokens in
ahead of "address" and "phone". This practice               serialization helpful for matching structured enti-
breaks the permutation invariance of the attributes,        ties. Based on our findings, we propose a novel
which has unknown effects on model utility. Nested          serialization scheme for LLMs based on random
attributes (e.g., "address" consists of "street", "city",   walks: it efficiently captures both semantic and
and "zipcode") in semi-structured data are often            structural contexts from KG entities, leading to ro-
flattened, which loses hierarchical information and         bust entity embeddings for matching. Our results
dilutes key signals. These issues further deteriorate       show that open-source LLMs with the proposed
for graph-structured data, where serialization often        walk-based strategy achieve state-of-the-art (SoTA)
results in either significant loss of structural infor-     performance with superior throughput for matching
mation or increased overhead to preserve relations          entities in four canonical and highly heterogeneous
between entities in sequences.                              KG datasets (Sun et al., 2020; Jiang et al., 2024b).
   So far, no comprehensive study has been per-                Our contributions are summarized as follows: (1)
formed to analyze the serialization effect on LMs           We are the first to systematically study the effect of
for structured entity matching. Existing schemes            entity serialization on LMs for matching across tab-
to serialize tabular and semi-structured data are ad        ular, semi-structured, and graph-structured data. (2)
hoc (Brunner and Stockinger, 2020; Li et al., 2020;         We empirically evaluate and compare how serial-
Wang et al., 2021; Peeters and Bizer, 2025), forcing        ization schemes affect LMs with different sizes and
attributes into predefined orders that may distort          backbones. (3) We propose a scalable paradigm for
their inherent non-sequential and/or hierarchical re-       applying open-source LLMs with walk-based seri-
lationships as shown in the above examples. Such            alization to generate robust embeddings for match-
practices potentially hamper the model’s under-             ing KG entities, outperforming previous SoTA
standing of the entity, thereby harming its utility.        GPT-4-based systems by 2995× speedup.
While structured entities with complex relations
are commonly serialized through verbal descrip-             2         Preliminaries and Prior Work
tions (Agarwal et al., 2021; Fatemi et al., 2023;
Wang et al., 2023; Wadhwa et al., 2024), which              2.1          Notations & Problem Formulation
greatly limits the model’s ability to generate rich         Entity matching (EM) takes two collections D
and concise representations for matching, partic-           and D′ of data records as input and outputs a set
ularly for knowledge graph entities with hyper-             M ⊆ D × D′ of entity pairs, where each pair
relations (Galkin et al., 2020).                            (e, e′ ) ∈ M is identified as the same real-world
   In this study, we systemically benchmark seven           entity. An entity e is a set of key-value pairs
common serialization schemes (see Table 1) for              e = {(attri , vali )}ki=1 , where attri denotes the at-
LMs on tabular and semi-structured data and ex-             tribute name and vali is the corresponding value
plore new strategies for LLMs to efficiently per-           in base types of number, string or list. Classic
form matching on entities with complex relations.           EM (Getoor and Machanavajjhala, 2012; Christen,
There are three key questions to be answered: RQ1           2012) assumes that entities in relational tables have
whether attribute order matters under the settings          a homogeneous schema, where e and e′ share the
                                                        7852
same set of attributes {attri }ki=1 ; while generalized       Index
                                                                S1
                                                                      Scheme
                                                                      Fixed Order (Li et al., 2020)
                                                                                                                    Attribute Order
                                                                                                                         Fixed
                                                                                                                                         Special Tokens
                                                                                                                                         [COL], [VAL]
                                                                                                                                                             NULL
                                                                                                                                                              ✓

EM (Wang et al., 2021) removes this assumption,                 S2
                                                                S3
                                                                      Random Order
                                                                      Pairwise Order (Naeim abadi et al., 2023)
                                                                                                                       Random
                                                                                                                       Pairwise
                                                                                                                                         [COL], [VAL]
                                                                                                                                         [COL], [VAL]
                                                                                                                                                              ✓
                                                                                                                                                              ✓
                                                                S4    Valid Value (Wang et al., 2021)                    Fixed           [COL], [VAL]          x
allowing entities e to have a heterogeneous schema,             S5    Plain Format (Brunner and Stockinger, 2020)        Fixed                  x              x
                                                                S6    Span Typing (Li et al., 2020)                      Fixed        [COL], [VAL], [LAST]    ✓
with nested or unstructured attributes.                         S7    JSON Format (Sisaengsuwanchai et al., 2023)   Fixed / Nested              x             x

   In addition to tables and semi-structured data,
another classic type of structured entity is stored        Table 1: Summary & Comparison of Common Serial-
                                                           ization Schemes for Structured Entities.
in graphs (see Fig. 1). In particular, knowledge
graph (KGs) G = (E, R, F) is an organized repre-
sentation of real-world entities E and their relation-     ied how to cost-effectively deploy LLMs for match-
ship R through triplets of facts (eh , r, et ) ∈ F,        ing but mainly focused on improving throughput by
where eh , et ∈ E and r ∈ R. The integra-                  prompt engineering or blocking techniques. How-
tion of real-world entities from KGs is known              ever, the bottleneck of text generation ultimately
as entity alignment: Given G = (E, R, F) and               limits their applicability.
G ′ = (E ′ , R′ , F ′ ), the task is to find the identi-
cal entity set {(e, e′ )|e ∈ E, e′ ∈ E ′ }, where each     2.3         Prior work on Serialization
pair (e, e′ ) represents the same real-world entity        The serialization process converts an entity e from a
but exists in different KGs.                               set of key-value pairs {(attri , vali )}ki=1 into a mean-
2.2 Language Models for Entity Matching                    ingful sequence that can be ingested by LMs. Exist-
                                                           ing serialization schemes can be divided into three
Pretrained encoder models, including BERT (De-             categories based on entity structure types.
vlin et al., 2019) and RoBERTa (Liu, 2019), show
great abilities in natural language understanding for      Serialization for Tabular Data. For pretrained
various downstream tasks. For EM tasks, the model          BERT models, the standard scheme for serializ-
takes a pair of entities serialized by the function        ing entities in relational tables is introduced by Li
S(·) as input and is fine-tuned with objectives in         et al. (2020): S(e) → "[COL] attr1 [VAL] val1 . . .
two common settings: (1) cross-encoder: the joint          [COL] attrk [VAL] valk ", where [COL] and [VAL]
of serialized entities e, e′ is fed into model fLM si-     are special tokens indicating the start of attribute
multaneously, followed by a classifier g predicting        names and values, respectively. This scheme is
"match" or "no match" (Li et al., 2020; Brunner            flexible in serializing tabular data with homoge-
and Stockinger, 2020; Wang et al., 2021; Peeters           neous and heterogeneous schema. To serialize
and Bizer, 2021) as g(fLM (S(e, e′ ))) → {0, 1};           an entity pair (e, e′ ), let S(e, e′ ) → "[CLS] S(e)
(2) bi-encoder: serialized entities e, e′ are fed to       [SEP] S(e′ ) [SEP]", where [SEP] is the token that
model fLM independently and their similarity is            separates two sequences and [CLS] is the token
computed by a scoring function d (or neural net-           used in BERT models to encode serialized entity
works) over their embeddings (Paganelli et al.,            pairs into a joint representation for classification.
2022; Zeakis et al., 2023) as d(he , he′ ) ∈ [0, 1],           Multiple schemes are developed with variations
where he = fLM (S(e)).                                     in (1) Attribute order: S1 Fixed order concate-
   LLMs, including GPT (Radford et al., 2019) and          nates attributes sequentially following the given
Llama (Touvron et al., 2023), are designed pri-            order of the table schema (Wang et al., 2021; Miao
marily for language generation. They often show            et al., 2021; Hegselmann et al., 2023). S2 Ran-
greater generalizability and zero-shot performance         dom order permutes entity attributes during serial-
than pretrained encoder models on new tasks,               ization, leading to misaligned attributes between
which is particularly useful for EM tasks with lim-        pairs of serialized entities. S3 Pairwise order puts
ited samples for training or out-of-distribution sam-      the values of common attributes together in seri-
ples for inference. Several studies have utilized          alized entity pairs (Naeim abadi et al., 2023) and
LLMs to solve EM problems through chat-based               fills the mismatched attributes with the value NULL
interactions, but suffer from high latency and in-         as Sp (e, e′ ) → "[CLS] [COL] attr1 [VAL] vale1 ,
                                                                ′
ference cost (Peeters and Bizer, 2025; Li et al.,          vale1 . . . [COL] attrk [VAL] valek , NULL [SEP]." (2)
2024). To save prompting token budgets, serial-            Special tokens: S4 Valid value removes attributes
ization schemes adopted in these works are all             with missing values from the output (Wang et al.,
plain text. Fan et al. (2024); Wang et al. (2025);         2021). S5 Plain format omits [COL] and [VAL]
Huang and Zhao (2024); Jiang et al. (2024a) stud-          tokens (Brunner and Stockinger, 2020) and sim-
                                                       7853
ply concatenates the names and values of all at-         For example, the KG entity "Louvre" in Fig. 1
tributes as S(e) → "attr1 val1 . . . attrk valk ." Due   is described as "Louvre is a museum located in
to its simplicity, this scheme is widely adopted to      Paris." Madaan et al. (2022); Jiang et al. (2024a)
pair with matching query prompts for decoder-only        exploit code format to aid LLMs processing
models (Peeters and Bizer, 2025; Sisaengsuwan-           graphs by providing a structure abstraction defined
chai et al., 2023; Wang et al., 2024). S6 Span typing    in Python-style classes. For instance, the KG
is introduced by Li et al. (2020) to inject domain       entity class is defined as "class Entity(): def
knowledge into serialized entities by annotating         __init__(self,name,id,tuples=[]):. . . def
the type of a span of their attribute values. Spe-       get_neighbors(self):. . . " These methods
cial tokens are inserted to reflect the type of the      primarily focus on flattening graphs, while the
recognized span {(si , ti , typei )}i≥1 from attribute   new serialization strategy we proposed exploits
values, where si , ti are the start/end positions of     both semantic relationship and structural context
the span with the annotated type. For instance,          between structured entities in KGs through random
the phone number "(123)456-7890" can be re-              walks, detailed in Sec. 3.2.
placed with "( 123 ) 456 - [LAST] 7890 [/LAST]",
where [LAST], [/LAST] are added to indicate the          3     Serialization for Structured Entities
start/end of the last 4 digits of phone number that
might help the model compare between entities.           In this section, we investigate how different serial-
                                                         izations of structured entities affect the utility and
Serialization for Semi-Structured Data. An               efficiency of language models for matching tasks.
attribute attri of semi-structured entities can be       Building on our findings, we explore new strategies
either values in base types or an entity itself, such    for LLMs to efficiently encode structured entities
as JSON format with nested attributes. Its serializa-    with complex relations, especially with application
tion can follow a similar fashion as tabular data, but   to entity alignment in KGs.
differs in: the [COL] and [VAL] tokens are recur-
sively added along with attribute names and values       3.1    Effects of Entity Serialization on Tabular
nested at each level i (underlined) as S(e) →                   & Semi-structured Data
"[CLS] [COL] attr1 [VAL] val1 . . . [COL] attrk
                                                         To study the impact of entity serialization on
[VAL]. . . [COL] attrki [VAL] valki . . . [SEP]."
                                                         LMs, we decompose the problem into serialization
For list-type attribute values, all elements in the
                                                         scheme and model size. The serialization scheme
list are merged into a single string, separated by
                                                         determines how structured entities are converted
commas. Alternatively, one can skip all the special
                                                         into sequences, which affects the model’s ability to
tokens and directly use S7 JSON format or XML-
                                                         interpret the input. Specifically, we benchmark the
style parentheses structures (Sisaengsuwanchai
                                                         effect of serialization schemes S(·) listed in Table
et al., 2023), which preserves the hierarchical
                                                         1 for pretrained encoder models and open-source
information in some sense. All seven serialization
                                                         LLMs on EM tasks under zero-shot and fine-tuning
variants are summarized and compared in Table 1.
                                                         settings. We quantify and analyze the impact of
Serialization for Graph Data. Unlike tabular             attribute order (e.g., fixed, random, or pairwise),
or semi-structured data, the inherent relationships      special tokens (e.g. COL, VAL, span typing), and
between entities in a graph make serialization an        sequence formats (e.g., plain, JSON) in entity seri-
open challenge. Plain graphs are often serialized        alization on the model performance.
as a flat list of nodes and edges. To better align          The model size determines the capacity of a
with the text corpora that LMs were pretrained           model to represent entities, which affects the util-
on, graphs often get contextualized through a            ity of learned representations used for matching.
mapping function (Wang et al., 2023; Fatemi              Pretrained encoder models are cost-effective for
et al., 2023), such as using TV character names          EM due to their compact sizes and low inference
and friendships "G describes a friendship graph          cost. Previous work (Pham et al., 2021; Sugawara
among Ned, Cat, Daenerys, ... In this friendship         et al., 2020; Albilali et al., 2021) has shown that
graph: Ned and Cat are friends..." For entities          input shuffle can lead to significant performance
with complex relations, Agarwal et al. (2021)            degradation for BERT models in natural language
formulates it as a data-to-text generation task and      understanding, which may be amplified through
uses Seq2seq models to verbalize KG entities.            serializing structured entities for matching. LLMs
                                                     7854
are known to accept various input formats, and they     et al., 2024; Jiang et al., 2024a) have begun to lever-
have strong generalizations and capabilities of con-    age the reasoning power of LLMs to align entities
text understanding in new domains. Preliminary          across KGs, but suffer from high latency and infer-
studies (Peeters and Bizer, 2025; Sisaengsuwan-         ence cost. The fundamental challenge lies in how
chai et al., 2023; Wang et al., 2024) directly pair     to effectively encode KG entities and efficiently
entities serialized as plain text with prompts to       use LLMs to extract their semantic and structural
query LLMs and collect their responses in natural       information for matching.
language. This QA-based matching is greatly lim-           To tackle this challenge, we apply our findings in
ited by the high latency and inference cost. Instead,   Sec. 3.1 to propose a lightweight serialization Sw
we propose to utilize a bi-encoder framework to         for graph-structured entities by sampling seman-
obtain entity embeddings from the target LLM for        tically meaningful paths on KGs through random
matching and study the impact of serialization. For     walks (see Fig. 1). Specifically, given a target entity
open-source LLMs, such conversion can be done           ei ∈ E, the walk-based serialization:
by LLM2Vec (BehnamGhader et al., 2024).
   Our empirical study seeks to understand the opti-    • Samples M -many L-step walks starting from the
mal form of serialization to maximize the matching        root entity ei on graph G, and obtains a collec-
performance of different LMs on tabular and semi-         tion of sampled facts {Wm }M    m=1 , where Wm =
structured data. We briefly summarize the most            (wm,1 , . . . , wm,L ) and wm,1 = (ei , r, ej ).
exciting results here and defer the detailed analysis
to Sec. 4.2:
                                                        • Applies a function map(·) on each path Wm that
• With fine-tuning, pretrained encoder models are         maps the entities and their relations from all L
  more sensitive to how entities are serialized than      sampled facts into a sentence, separated by com-
  LLMs on structured EM tasks;                            mas. For example, the path Wm that contains two
                                                          facts wm,1 =("Louvre", "located in", "Paris")
• Both encoder models and LLMs have a prefer-
                                                          and wm,2 =("Paris", "is a", "Place") can be pro-
  ence for serialization schemes that are close to
                                                          cessed by map(Wm ) = "Louvre located in Paris,
  the corpus they were pretrained on, especially
                                                          Paris is a Place."
  plain text without special tokens.
• Injecting randomness into attribute orders during     • Concatenates sentences of all M paths into a
  serialization can be beneficial for models under        paragraph as Sw (ei ; G) = ∪M
                                                                                      m=1 map(Wm ).
  supervised fine-tuning, especially on noisy data.
   Our benchmark results show that: (1) plain text      If entities contain other attributes, previous serial-
is sufficient for LMs to capture subtle differences     ization schemes can be used to obtain an entity-
between entities with rich attributes. (2) the pre-     level description and then call the map(·) function.
defined attribute order is not optimal for matching,       This walk-based serialization Sw naturally en-
while the injected randomness can improve the           codes the local structures of KG entities in a se-
model’s robustness to input perturbations. These        quential form, in which relative positions and se-
two key observations will play an important role in     mantic relationships between entities are embed-
designing new strategies for serializing structured     ded. By adjusting the number M of path sam-
entities with complex relations.                        pling and the step size L, the output sequence
                                                        can be controlled according to the model’s con-
3.2 Serialization Strategies for Knowledge              text window and desired neighborhood coverage,
    Graph Entities                                      without being affected by irregular sizes of entity-
KG entities not only contain attributes but are also    induced subgraphs. We further extend the bi-
connected to each other through complex relations.      encoder framework of LLMs to KG entities by
The knowledge stored in the form of fact triples        pairing it with the proposed strategy Sw , i.e., he =
poses unique challenges for applying LM to their        fLLM (Sw (e; G)). The proposed Walk-LLM pro-
matching. Classical methods rely on measuring           vides a scalable paradigm that can efficiently utilize
the similarity of entity embeddings derived from        the extensive parametric knowledge from off-the-
knowledge representation learning (KRL) tech-           shelf LLMs through sampled semantic-rich paths
niques (Zhang et al., 2022). Recent studies (Yang       to match KG entities at scale.
                                                    7855
    Index    Dataset               Format            Domains         |D|     |D′ |    #Attr k     #Pos   Index   Dataset         Format     #Entities |E|   #Relations |R|           #Facts |F|   #Anchors
     D1      Rel-HETER          Hetero. Table          POI           534      332        6; 7      946    D7     DBP15K(EN-FR)    KG              15,000         193; 166       96,318; 80,112      15,000
     D2      Semi-Rel           Table; JSON           Movies      29,180   32,823    8; 13.81    2,183    D8     DBP-WIKI         KG            100,000          413; 261      293,990; 251,708    100,000
     D3      iTunes-Amazon    Homo. Table (Dirty)     Music        6,907   55,923           8      539    D9     ICEWS-WIKI       KG      11,047; 15,896         272; 226    3,527,881; 198,257      5,058
     D4      Walmart-Amazon   Homo. Table (Dirty)   Electronics    2,554   22,074           5   10,242    D10    ICEWS-YAGO       KG      26,863; 22,734         272; 41     4,192,555; 107,118     18,824
     D5      IMDb-TVDB          Hetero. Table       TV Shows       5,118    7,810       30; 9    1,072     /
     D6      IMDb-DBpedia       Hetero. Table         Movies      27,615   23,182        4; 7   22,863     /



    Table 2: Dataset Summary & Statistics (D1-6 Tabular and Semi-structured Data; D7-10 Knowledge Graphs).

                         D1 (MRR)                      D2 (MRR)                         D3 (MRR)                        D4 (MRR)                    D5 (MRR)                         D6 (MRR)
    Scheme
                  RoBERTa Mistral      Llama    RoBERTa Mistral        Llama     RoBERTa Mistral         Llama   RoBERTa Mistral    Llama    RoBERTa Mistral        Llama     RoBERTa Mistral       Llama
    S1 Fixed       1.000     0.995     1.000     0.888     0.826       0.914      0.634     0.787        0.771    0.958     0.972   0.962     0.929     0.933       0.929      0.944     0.923      0.942
    S2 Random      0.991     1.000     1.000     0.868     0.828       0.824      0.672     0.770        0.753    0.960     0.970   0.976     0.940     0.946       0.944      0.929     0.925      0.939
    S3 Pairwise    0.995       /          /      0.906       /            /       0.672       /             /     0.929       /        /      0.928       /            /       0.931       /           /
    S4 Valid       1.000     1.000     1.000     0.854     0.942       0.964      0.546     0.790        0.789    0.965     0.964   0.968     0.929     0.929       0.931      0.930     0.924      0.943
    S5 Plain       1.000     1.000     1.000     0.837     0.891       0.926      0.736     0.795        0.764    0.973     0.953   0.970     0.929     0.931       0.930      0.924     0.920      0.942
    S6 Span        1.000     1.000     1.000     0.860     0.920       0.913      0.663     0.778        0.803    0.965     0.963   0.971     0.610     0.932       0.932      0.934     0.923      0.936
    S7 JSON        0.994     0.987     1.000     0.884     0.833       0.877      0.667     0.782        0.759    0.970     0.965   0.970     0.920     0.932       0.930      0.936     0.926      0.942



Table 3: Language Models (Fine-tuned) for Structured EM with Different Serialization (1st bold, 2nd underline).


4       Experiments                                                                                         shown high utility in practice. Two popular LLMs
                                                                                                            of Llama3-8B (AI@Meta, 2024) and Mistral-7B
4.1 Experiment Settings                                                                                     (Jiang et al., 2023) are also evaluated. We utilize
Dataset. Six diverse and challenging datasets                                                               LLM2Vec (BehnamGhader et al., 2024) to trans-
(Das et al.; Wang et al., 2021; Obraczka et al., 2021;                                                      form these open-source LLMs into text encoders,
Papadakis et al., 2011) are selected to quantify the                                                        where no additional prompts are required other than
impact of serialization schemes on LMs for struc-                                                           serialized entities. To our knowledge, this is the
tured EM. D1 Rel-HETER, D5 IMDb-TVDB, and                                                                   first exploration of using Llama3-8B and Mistral-
D6 IMDb-DBPedia are heterogeneous tabular data                                                              7B as text encoders for EM.
of entities from restaurants, TV shows, and movies,
                                                                                                            Baselines. Three types of methods are selected
respectively. D2 Semi-Rel contains book entities
                                                                                                            for matching KG entities, which cover differ-
with nested attributes in JSON format. We also
                                                                                                            ent input features of KGs and KRL techniques:
adopted a dirty version of the D3 iTunes-Amazon
                                                                                                            translation-based method MTransE (Chen et al.,
and D4 Walmart-Amazon datasets by randomly
                                                                                                            2017), GNN-based methods of RDGCN (Wu et al.,
shifting attribute values to different fields. These
                                                                                                            2019) and Dual-AMN (Mao et al., 2021), LM-
two datasets are used to measure the model’s ro-
                                                                                                            based methods of BERT (Devlin et al., 2019),
bustness against noisy attribute value pairs.
                                                                                                            BERT-INT (Tang et al., 2020), and the previous
   To address the alignment of structured entities
                                                                                                            SoTA methods of Simple-HHEA (Jiang et al.,
in KGs, we pick four evaluation datasets. D7
                                                                                                            2024b) and two-stage GPT-4-based ChatEA (Jiang
DBP15K(EN-FR) is a classical dataset for aligning
                                                                                                            et al., 2024a).
bilingual entity pairs of DBpedia. D8 DBP-WIKI
is used for entity alignment across Wikipedia and                                                           Metrics. In line with widely adopted evaluation
DBpedia. Both datasets (Sun et al., 2020) have                                                              methods for EM tasks, we use two ranking met-
an equal number of entities in KGs with similar                                                             rics: Hits@K, measuring the percentage of correct
structural features, such as the number of facts                                                            predicted entity pairs among the top-K matches,
and density. D9 ICEWS-WIKI and D10 ICEWS-                                                                   and mean reciprocal rank (MRR), calculating the
YAGO (Jiang et al., 2024b) are two new datasets of                                                          average inverse ranking of correct predicted match
highly heterogeneous KGs (HHKG) with different                                                              pairs. For datasets D1-6, we use the BM25 algo-
numbers of entities and distinct structures.                                                                rithm (Robertson et al., 2009) to pair each matched
   Table 2 summarizes the statistics of selected                                                            pair with K = 10 hard non-match entities to ob-
datasets for evaluation. The attribute names of                                                             tain the ranking. For datasets D7-10, we follow the
datasets D1-D6 are listed in Table 7, Appx. B.1.                                                            same evaluation setting as in Jiang et al. (2024a).
The license information and source for all datasets
can be found in Table 8, Appx. B.1.                                                                         Implementation Details. For encoder models,
                                                                                                            we incorporate their base models with the Ditto
Backbone Models. BERT (Devlin et al., 2019)                                                                 framework (Li et al., 2020) as cross-encoder for
and RoBERTa (Liu, 2019) are the most commonly                                                               matching, which is one of the first dedicated EM
used encoder models on EM tasks, which have                                                                 systems using pretrained LMs (see Fig. 4 in Appx.
                                                                                                    7856
                                  DBP15K(EN-FR)                DBP-WIKI                   ICEWS-WIKI               ICEWS-YAGO
      Models
                             Hits@1 Hits@10 MRR       Hits@1    Hits@10   MRR      Hits@1 Hits@10 MRR        Hits@1 Hits@10 MRR
      MTransE †               0.247    0.577  0.360    0.281     0.520    0.363     0.021     0.158  0.068    0.012    0.084  0.040
      RDGCN†                  0.873    0.950  0.901    0.974     0.994    0.980     0.064     0.202  0.096    0.029    0.097  0.042
      Dual-AMN†               0.954    0.994  0.970    0.983     0.996    0.991     0.083     0.281  0.145    0.031    0.144  0.068
      BERT                    0.811    0.859  0.829    0.800     0.852    0.818     0.609     0.739  0.655    0.794    0.871  0.822
      BERT-INT †              0.990    0.997  0.993    0.996     0.997    0.996     0.561     0.700  0.607    0.756    0.859  0.793
      Simple-HHEA†            0.959    0.995  0.972    0.975     0.991    0.988     0.720     0.872  0.754    0.847    0.915  0.870
      ChatEA (GPT-4)†         0.990    1.000  0.995    0.995     1.000    0.998     0.880     0.945  0.912    0.935    0.955  0.944
      Walk-Mistral-7BM =10    0.980    0.999  0.988    0.980     0.998    0.988     0.986     0.999  0.992    0.980    0.994  0.986
      Walk-Llama3-8BM =10     0.983    0.999  0.990    0.974     0.997    0.984     0.988     1.000  0.993    0.974    0.995  0.983


 Table 4: Result Comparison on Matching Entities from Canonical KGs and HHKGs (1st bold, 2nd underline).


A.1). The probability of a pair match predicted                       improved, with S2 and S4 achieving the largest
by the model is used for ranking. For open-source                     gains and the best performance in general. This
LLMs, we use them as bi-encoders, which are fine-                     observation suggests that structural properties of
tuned by contrastive-based objectives (see Fig. 6                     entity attributes and data quality play a crucial role
in Appx. A.2). The scheme S3 pairwise order is                        in fine-tuning LLMs for matching.
incompatible with bi-encoders and thus skipped.                          RQ3 S5 Plain format leads four of six datasets,
The source code is available at https://github.                       and comparable results can be observed from S7
com/amazon-science/serializeEM.                                       JSON format. This indicates that using special to-
                                                                      kens to retain the structure of the input by S1,2,4
4.2 Experiments on Tabular &                                          or inject domain knowledge by S6 Span typing is
    Semi-structured Entities                                          not helpful in improving encoder models and po-
Table 3 shows how serialization schemes shape                         tentially causes more overhead. For LLMs, the
language models of different sizes and backbones                      model without fine-tuning shows a strong prefer-
on datasets D1-6 with fine-tuning. The results for                    ence for S5 and S7, which do not contain special
zero-shot are reported in Table 9, Appx. B.3.                         tokens and are mostly close to the text corpus on
   RQ1-2 Randomly shuffling attribute orders does                     which the model was pretrained. Surprisingly, with
not necessarily hurt the matching performance; in                     fine-tuning, LLMs were able to capture the domain
fact, the randomness injected by S2 Random order                      knowledge injected by S6 on D3. However, it is not
can improve the robustness of RoBERTa, especially                     as effective on other datasets, which are similarly
on dirty datasets D3-4. Instead, S1 Fixed order, the                  observed on pretrained encoder models.
common by default choice, does not show an ad-
vantage over other schemes, especially falling far                    4.3         Experiments on KG Entities
behind on D3-4. This result suggests that break-
ing the permutation invariance of attributes can                      Table 4 compares different baseline models on
negatively affect the model, especially when data                     matching entities in canonical and highly heteroge-
quality is not ideal. S3 Pairwise order shows its                     neous KGs. Results marked with † are from Jiang
strength in matching semi-structured entities on                      et al. (2024a). Pairing walk-based serialization with
D2, suggesting that aligning common attributes                        open-source LLMs achieves performance compa-
between entity pairs can potentially enhance the                      rable to BERT-INT and GPT-4-based ChatEA on
signal and mitigate the side effects of flattening                    canonical KG datasets. On two HHKG datasets,
nested attributes. In addition, removing attributes                   our proposed method Walk-LLM obtains MRR
with missing values through S4 Valid value is gen-                    scores of 0.993 (+8.88%) and 0.986 (+4.45%), re-
erally not beneficial to pretrained encoder models.                   spectively, which is a significant improvement over
   As for LLMs, they perform reasonably well on                       ChatEA. These results highlight the uniform effec-
the noisy datasets D3-4 for zero-shot. Randomly                       tiveness and consistent superiority of the proposed
permuting the order of attributes without supervis-                   Walk-LLM on various types of KG datasets. It also
ing signals usually hurts the model, but the perfor-                  shows that the power of LLMs can be effectively
mance drop is not substantial. These two obser-                       exploited through entity embeddings in combina-
vations show the overall robustness of LLMs to                        tion with rich contextual input produced by the
input perturbations. After fine-tuning, the results                   walk-based strategy but does not suffer from high
of LLMs with all types of serialization are greatly                   interactive inference costs like ChatEA.
                                                                 7857
                                                                                ICEWS-WIKI (non-trivial)               ICEWS-YAGO (non-trivial)                               ICEWS-WIKI (dirty)            ICEWS-YAGO (dirty)
                                                    Models
                                                                           Hits@1 Hits@10 MRR                      Hits@1 Hits@10 MRR                                      Hits@1 Hits@10 MRR            Hits@1 Hits@10 MRR
                                                    BERT                    0.359   0.587    0.438 (↓33.1%)         0.566   0.726  0.624 (↓24.1%)                           0.211   0.235    0.220        0.225    0.250   0.234
                                                    Simple-HHEA             0.490   0.777    0.579 (↓23.2%)         0.614   0.788  0.675 (↓22.4%)                           0.298   0.522    0.374        0.309    0.497   0.373
                                                    Llama3-8B (MNTP)        0.653   0.895    0.737                  0.571   0.790  0.647                                    0.226   0.358    0.296        0.234    0.297   0.258
                                                    Mistral-7B (MNTP)       0.821   0.948    0.868                  0.822   0.924  0.860                                    0.385   0.553    0.444        0.339    0.467   0.384
                                                    Walk-Mistral-7BM =10    0.981   0.998    0.989 (↓0.3%)          0.942   0.985  0.959 (↓2.7%)                            0.834   0.975    0.888        0.786    0.944   0.844
                                                    Walk-Llama3-8BM =10     0.980   1.000    0.989 (↓0.4%)          0.937   0.977  0.954 (↓3.0%)                            0.745   0.907    0.804        0.673    0.840   0.732


Table 5: Result Comparison on Matching Non-trivial and Dirty Entities from HHKGs (1st bold, 2nd underline).

                                                      Efficiency Analysis of LLM-based Methods on Matching KG Entities                                               Effect of Walk Numbers M on Structured Entity Matching
                                                          ICEWS-WIKI                                       131.8
                                                                                                                   90.8                                                                 Mistral-7B - M=3           Llama3-8B - M=3




 avg. inf. time (s) per entity [log scale]
                                             102          ICEWS-YAGO                                                                                                                    Mistral-7B - M=5           Llama3-8B - M=5
                                                                                          23.7 18.9                                                            1.0                      Mistral-7B - Full (S2)     Llama3-8B - Full (S2)




                                                                                                                                     Model Performance (MRR)
                                             101

                                                                                                                                                               0.9
                                             100

                                                                                                                                                               0.8
                                             10 1
                                                       0.044             0.037
                                                            0.027
                                                                              0.016                                                                            0.7
                                             10 2
                                                     Walk-Mistral-7B   Walk-Llama3-8B   ChatEA-GPT-3.5    ChatEA-GPT-4
                                                                                                                                                               0.6
                                                                                                                                                                         D1       D2         D3           D4       D5          D6
Figure 2: Per-query Runtime Comparison of LLM-                                                                                                                                                 Datasets
based Methods on Matching Entities across HHKGs.
                                                                                                                                  Figure 3: Effect of Walk Numbers M on LLMs with
                                                                                                                                  Fine-tuning for Matching Tabular and Semi-structured
4.3.1 Ablation Study                                                                                                              Entities. M represents the number of entity attributes
                                                                                                                                  sampled in the serialized input, and increasing M even-
Hard Examples. Table 5 (Left, non-trivial) re-                                                                                    tually leads to the input sequence equivalent to S2 Ran-
ports results on entity pairs with non-identical                                                                                  dom order without downsampling.
names in the source and target KGs (23% over-
all) from the HHKG datasets. Here, we focus on
LM-based methods because they rely primarily on                                                                                   igate the impact of input noise on model utility.
name attributes to differentiate entities. BERT and                                                                               A direct comparison with ChatEA cannot be done
Simple-HHEA suffer more than 20% performance                                                                                      due to its limited accessibility. Based on results
degradation on these non-trivial entity pairs, while                                                                              over noisy embeddings (see Fig. 3 of Jiang et al.
the proposed Walk-LLM barely drops. We pro-                                                                                       (2024a)), both Walk-LLM and ChatEA achieve
vide the results by comparing embeddings of en-                                                                                   better robustness than Simple-HHEA. Note that
tity names directly generated from LLMs for refer-                                                                                ChatEA has 2 stages, where the first stage relies on
ence. These models are trained through LLM2Vec                                                                                    embeddings generated by Simple-HHEA (or other
with the objective of masked next token prediction                                                                                encoders) and thus is still sensitive to input noise.
(MNTP), where the structure in KG is excluded                                                                                     4.3.2                              Efficiency Analysis
from the input. The model performance gap be-
                                                                                                                                  Fig. 2 compares the averaged inference time per
tween MNTP and Walk-LLM further emphasizes
                                                                                                                                  entity between LLM-based methods to match
that the proposed walk-based strategy introduces a
                                                                                                                                  KG entities. Note that, the runtime results of
strong ability to distinguish hard examples.
                                                                                                                                  GPT-based methods are obtained from Table 5 of
Model Robustness. Table 5 (Right, dirty) com-                                                                                     ChatEA (Jiang et al., 2024a). Walk-LLM (M =
pares different LM-based methods on the noisy                                                                                     10) achieves a speedup of more than 538 × and
version of HHKG datasets, where either half of                                                                                    2995 × than ChatEA with the GPT-3.5 and GPT-4
the "name" attribute is masked or gets mixed with                                                                                 backbones, respectively. The reduction in inference
its URL uniformly at random. This simulates one                                                                                   cost comes from two main aspects: (1) Replacing
kind of dirty data commonly seen in practice while                                                                                chat-based queries with a bi-encoder framework
keeping the modification simple. Both BERT and                                                                                    that matches entities based on comparing their em-
Simple-HHEA suffer severe performance degra-                                                                                      beddings generated from context-rich features sam-
dation due to their over-reliance on entity names,                                                                                pled by random walks. (2) Running open-source
while Walk-LLM is robust to such perturbations.                                                                                   LLMs with fewer parameters locally is cheaper
Semantic paths sampled by random walks can mit-                                                                                   than the API calls of online GPT models.
                                                                                                                            7858
 Models
                               ICEWS-WIKI
                        Hits@1 Hits@10 MRR
                                                        ICEWS-YAGO
                                                  Hits@1 Hits@10 MRR
                                                                           scalably generate robust embeddings for matching
 Walk-Mistral-7BM =5     0.977     0.999  0.987    0.969    0.994  0.980   KG entities. The proposed LLM-Walk achieves
 Walk-Mistral-7BM =10    0.986     0.999  0.992    0.980    0.994  0.986
 Walk-Mistral-7BSubg     0.982     1.000  0.990    0.974    0.995  0.983   SoTA performance and is three orders of magni-
 Walk-Llama3-8BM =5     0.983     1.000   0.991   0.972    0.994   0.981   tude faster than GPT-4-based methods.
 Walk-Llama3-8BM =10    0.988     1.000   0.993   0.974    0.995   0.983
 Walk-Llama3-8BSubg     0.987     1.000   0.993   0.973    0.994   0.982
                                                                           6   Limitations
Table 6: Effect of Walk Numbers M on LLMs with
Fine-tuning for Matching Entities from HHKGs. subg                         While our study provides a comprehensive analy-
indicates the use of all neighboring entities in KGs as                    sis of serialization strategies for entity matching,
the input without downsampling.                                            several limitations remain. First, the scope of the
                                                                           benchmark datasets is mainly tabular and knowl-
                                                                           edge graph entities, which may not generalize to
4.4 Hyperparameter Study                                                   other types of structured data, such as complex
We start with the effect of the number of walks on                         nested or temporal entities. Second, although our
model performance using walk-based serialization                           proposed random walk-based serialization method
over tabular and semi-structured datasets. Fig. 3                          demonstrates strong performance in knowledge
shows that the walk-based serialization of sampling                        graphs, it may struggle with entities that lack well-
M entity attributes (M = 3, 5) provides reason-                            defined or sufficient graph structure, potentially
able results compared to S2 Random order without                           impacting the method’s efficacy in domains with
downsampling. The number of entity attributes in                           sparse or incomplete relations. Third, our explo-
D1-6 is mostly between 4 and 9. The performance                            ration in this work is limited to open-source LLMs
of the walk-based strategy improves with the in-                           like Llama and Mistral. We have not evaluated our
crease of M and eventually converges to the results                        methods over proprietary or closed-source models
of S2. For the KG dataset, Table 6 summarizes                              like GPT-4o (and inter alia). Lastly, the implica-
the comparison of model performance when sam-                              tions of entity serialization on downstream tasks
pling different numbers of paths M for each entity.                        that leverage structured data and LLMs, e.g., in
Similarly, increasing M can improve the matching                           question answering with knowledge graphs, are not
performance of the model, and it is sufficient to                          explored in this work.
set the step L to 1. However, when all neighbor-                              Overall, these limitations suggest that while the
ing entities in the induced subgraph are used as                           findings offer valuable insights, further work is
input without downsampling, the performance is                             needed to fully understand the generalizability and
not substantially boosted. Meanwhile, using the                            practical applicability of the studied serialization
full subgraph increases the training and inference                         approaches in real-world scenarios.
time by 4 times compared to M = 10. Considering
the trade-off between complexity and utility, the                          7   Ethics Statement
choice of M should generally follow the density of
                                                                           This work focuses on studying how the serialization
the input graph.
                                                                           of structured entities affects language models on
                                                                           matching tasks and is conducted under the guide-
5    Conclusion
                                                                           lines of the ACL Ethics Policy. The datasets used
In this work, we systematically study the effect of                        for experiments are publicly available, widely rec-
entity serialization on language models with differ-                       ognized by the research community, and, to the
ent sizes and backbones for matching structured                            best of our knowledge, contain no personally iden-
entities. We empirically find that BERT-based en-                          tifiable information or content that is harmful, of-
coder models are more sensitive to serialization                           fensive, or biased.
than LLMs, especially in terms of attribute order
and data quality. Both types of language models
prefer serialization schemes that are close to the                         References
text corpus on which they are pretrained. When                             Oshin Agarwal, Heming Ge, Siamak Shakeri, and Rami
entity attributes are noisy, injecting randomness                            Al-Rfou. 2021. Knowledge graph based synthetic
                                                                             corpus generation for knowledge-enhanced language
into the input can benefit the model’s performance.                          model pre-training. In Proceedings of the 2021 Con-
By applying these findings, we propose a random                              ference of the North American Chapter of the Asso-
walk-based serialization for open-source LLMs to                             ciation for Computational Linguistics: Human Lan-
                                                                       7859
  guage Technologies, pages 3554–3565, Online. As-         Muhammad Ebraheem, Saravanan Thirumuruganathan,
  sociation for Computational Linguistics.                  Shafiq Joty, Mourad Ouzzani, and Nan Tang. 2018.
                                                            Distributed representations of tuples for entity reso-
AI@Meta. 2024. Llama 3 model card. https://www.             lution. Proc. VLDB Endow., 11(11):1454–1467.
  llama.com/.
                                                           Meihao Fan, Xiaoyue Han, Ju Fan, Chengliang Chai,
Mehdi Akbarian Rastaghi, Ehsan Kamalloo, and
                                                            Nan Tang, Guoliang Li, and Xiaoyong Du. 2024.
 Davood Rafiei. 2022. Probing the robustness of pre-
                                                            Cost-effective in-context learning for entity resolu-
 trained language models for entity matching. In Pro-
                                                            tion: A design space exploration. In 2024 IEEE
 ceedings of the 31st ACM International Conference
                                                            40th International Conference on Data Engineering
 on Information and Knowledge Management, pages
                                                            (ICDE), pages 3696–3709. IEEE.
 3786–3790.
Eman Albilali, Nora Altwairesh, and Manar Hosny.           Wenfei Fan, Xibei Jia, Jianzhong Li, and Shuai Ma.
  2021. What does BERT learn from Arabic machine            2009. Reasoning about record matching rules. Proc.
  reading comprehension datasets? In Proceedings            VLDB Endow., 2(1):407–418.
  of the Sixth Arabic Natural Language Processing
 Workshop, pages 32–41, Kyiv, Ukraine (Virtual). As-       Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi.
  sociation for Computational Linguistics.                   2023. Talk like a graph: Encoding graphs for large
                                                             language models. arXiv preprint arXiv:2310.04560.
Parishad BehnamGhader, Vaibhav Adlakha, Marius
  Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and         Johannes Frey, Lars-Peter Meyer, Natanael Arndt, Felix
  Siva Reddy. 2024. LLM2Vec: Large language mod-             Brei, and Kirill Bulert. 2023. Benchmarking the
  els are secretly powerful text encoders. In First Con-     abilities of large language models for rdf knowledge
  ference on Language Modeling.                              graph creation and comprehension: How well do llms
                                                             speak turtle? arXiv preprint arXiv:2309.17122.
Misha Bilenko. 2019. Duplicate detection, record link-
  age, and identity uncertainty: Datasets. https:          Mikhail Galkin, Priyansh Trivedi, Gaurav Maheshwari,
  //www.cs.utexas.edu/~ml/riddle/data.html.                  Ricardo Usbeck, and Jens Lehmann. 2020. Message
Ursin Brunner and Kurt Stockinger. 2020. Entity match-       passing for hyper-relational knowledge graphs. In
  ing with transformer architectures-a step forward in       Proceedings of the 2020 Conference on Empirical
  data integration. In Proceedings of the 23rd Interna-      Methods in Natural Language Processing (EMNLP),
  tional Conference on Extending Database Technol-           pages 7346–7359, Online. Association for Computa-
  ogy, pages 463–473. OpenProceedings.                       tional Linguistics.

Muhao Chen, Yingtao Tian, Mohan Yang, and Carlo            Lise Getoor and Ashwin Machanavajjhala. 2012. Entity
 Zaniolo. 2017. Multilingual knowledge graph embed-           resolution: theory, practice & open challenges. Proc.
 dings for cross-lingual knowledge alignment. In Pro-        VLDB Endow., 5(12):2018–2019.
 ceedings of the Twenty-Sixth International Joint Con-
 ference on Artificial Intelligence, IJCAI-17, pages       Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
 1511–1517.                                                  pat, and Mingwei Chang. 2020. Retrieval augmented
                                                             language model pre-training. In International Con-
Peter Christen. 2012. The Data Matching Process,             ference on Machine Learning, pages 3929–3938.
  pages 23–35. Springer Berlin Heidelberg, Berlin,           PMLR.
  Heidelberg.
                                                           Stefan Hegselmann, Alejandro Buendia, Hunter Lang,
Sanjib Das, AnHai Doan, Paul Suganthan G. C.,                 Monica Agrawal, Xiaoyi Jiang, and David Sontag.
  Chaitanya Gokhale, Pradap Konda, Yash Govind,               2023. Tabllm: Few-shot classification of tabular
  and Derek Paulsen. The magellan data repos-                 data with large language models. In Proceedings
  itory.      https://sites.google.com/site/                  of the 26th International Conference on Artificial
  anhaidgroup/projects/data.                                  Intelligence and Statistics, pages 5549–5581. PMLR.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
   Kristina Toutanova. 2019. BERT: Pre-training of         Thomas N. Herzog, Fritz J. Scheuren, and William E.
   deep bidirectional transformers for language under-       Winkler. 2007. Data Quality and Record Linkage
   standing. In Proceedings of the 2019 Conference of        Techniques, 1st edition. Springer Publishing Com-
   the North American Chapter of the Association for         pany, Incorporated.
  Computational Linguistics: Human Language Tech-
   nologies, Volume 1 (Long and Short Papers), pages       Qianyu Huang and Tongfang Zhao. 2024. Leveraging
  4171–4186, Minneapolis, Minnesota. Association for         large language models for entity matching. arXiv
   Computational Linguistics.                                preprint arXiv:2405.20624.

Dennis Diefenbach, Vanessa Lopez, Kamal Singh, and         AQ Jiang, A Sablayrolles, A Mensch, C Bamford,
  Pierre Maret. 2018. Core techniques of question            DS Chaplot, D de las Casas, F Bressand, G Lengyel,
  answering systems over knowledge bases: a survey.          G Lample, L Saulnier, et al. 2023. Mistral 7b (2023).
  Knowledge and Information Systems, 55:529–569.             arXiv preprint arXiv:2310.06825.
                                                       7860
Xuhui Jiang, Yinghan Shen, Zhichao Shi, Chengjin        Zhengjie Miao, Yuliang Li, and Xiaolan Wang. 2021.
  Xu, Wei Li, Zixuan Li, Jian Guo, Huawei Shen, and       Rotom: A meta-learned data augmentation frame-
  Yuanzhuo Wang. 2024a. Unlocking the power of            work for entity matching, data cleaning, text classifi-
  large language models for entity alignment. In Pro-     cation, and beyond. In Proceedings of the 2021 Inter-
  ceedings of the 62nd Annual Meeting of the Associa-     national Conference on Management of Data, SIG-
  tion for Computational Linguistics (Volume 1: Long      MOD ’21, page 1303–1316, New York, NY, USA.
  Papers), pages 7566–7583, Bangkok, Thailand. As-        Association for Computing Machinery.
  sociation for Computational Linguistics.
                                                        Sidharth Mudgal, Han Li, Theodoros Rekatsinas, An-
Xuhui Jiang, Chengjin Xu, Yinghan Shen, Yuanzhuo          Hai Doan, Youngchoon Park, Ganesh Krishnan, Ro-
  Wang, Fenglong Su, Zhichao Shi, Fei Sun, Zixuan Li,     hit Deep, Esteban Arcaute, and Vijay Raghavendra.
  Jian Guo, and Huawei Shen. 2024b. Toward practical      2018. Deep learning for entity matching: A design
  entity alignment method design: Insights from new       space exploration. In Proceedings of the 2018 Inter-
  highly heterogeneous knowledge graph datasets. In       national Conference on Management of Data, SIG-
  Proceedings of the ACM on Web Conference 2024,          MOD ’18, page 19–34, New York, NY, USA. Asso-
  pages 2325–2336.                                        ciation for Computing Machinery.

Pradap Konda, Sanjib Das, Paul Suganthan G. C., An-     Ali Naeim abadi, Mir Tafseer Nayeem, and Davood
  Hai Doan, Adel Ardalan, Jeffrey R. Ballard, Han         Rafiei. 2023. Product entity matching via tabular
  Li, Fatemah Panahi, Haojun Zhang, Jeff Naughton,        data. In Proceedings of the 32nd ACM International
  Shishir Prasad, Ganesh Krishnan, Rohit Deep, and        Conference on Information and Knowledge Manage-
  Vijay Raghavendra. 2016. Magellan: toward building      ment, pages 4215–4219.
  entity matching management systems. Proc. VLDB
  Endow., 9(12):1197–1208.                              Daniel Obraczka, Jonathan Schuchart, and Erhard
                                                          Rahm. 2021. Eager: embedding-assisted entity
Hanna Köpcke, Andreas Thor, and Erhard Rahm.              resolution for knowledge graphs. arXiv preprint
  2010. Evaluation of entity resolution approaches        arXiv:2101.06126.
  on real-world match problems. Proc. VLDB Endow.,
  3(1–2):484–493.                                       Matteo Paganelli, Francesco Del Buono, Andrea
                                                         Baraldi, and Francesco Guerra. 2022. Analyzing
Huahang Li, Longyu Feng, Shuangyin Li, Fei Hao,          how bert performs entity matching. Proc. VLDB
  Chen Jason Zhang, Yuanfeng Song, and Lei Chen.         Endow., 15(8):1726–1738.
  2024.    On leveraging large language models
  for enhancing entity resolution. arXiv preprint       George Papadakis, Ekaterini Ioannou, Claudia Niederée,
  arXiv:2401.03426.                                       and Peter Fankhauser. 2011. Efficient entity resolu-
                                                          tion for large heterogeneous information spaces. In
                                                          Proceedings of the 4th ACM International Confer-
Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan,     ence on Web Search and Data Mining, pages 535–
  and Wang-Chiew Tan. 2020. Deep entity matching          544.
  with pre-trained language models. Proc. VLDB En-
  dow., 14(1):50–60.                                    Ankur Parikh, Oscar Täckström, Dipanjan Das, and
                                                          Jakob Uszkoreit. 2016. A decomposable attention
Yinhan Liu. 2019.      Roberta: A robustly opti-          model for natural language inference. In Proceedings
  mized bert pretraining approach. arXiv preprint         of the 2016 Conference on Empirical Methods in Nat-
  arXiv:1907.11692.                                       ural Language Processing, pages 2249–2255, Austin,
                                                          Texas. Association for Computational Linguistics.
Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang,
 and Graham Neubig. 2022. Language models of code       Ralph Peeters and Christian Bizer. 2021. Dual-objective
 are few-shot commonsense learners. In Proceedings        fine-tuning of bert for entity matching. Proc. VLDB
 of the 2022 Conference on Empirical Methods in Nat-      Endow., 14(10):1913–1921.
 ural Language Processing, pages 1384–1403, Abu
 Dhabi, United Arab Emirates. Association for Com-      Ralph Peeters and Christian Bizer. 2025. Entity match-
 putational Linguistics.                                  ing using large language models. In Proceedings
                                                          of the 28th International Conference on Extending
Christopher Manning. 2017. Representations for lan-       Database Technology, pages 529–541. OpenProceed-
  guage: From word embeddings to sentence meanings.       ings.
  Simons Institute for the Theory of Computing, UC
  Berkeley.                                             Thang Pham, Trung Bui, Long Mai, and Anh Nguyen.
                                                          2021. Out of order: How important is the sequen-
Xin Mao, Wenting Wang, Yuanbin Wu, and Man Lan.           tial order of words in a sentence in natural language
  2021. Boosting the speed of entity alignment 10×:       understanding tasks? In Findings of the Association
  Dual attention matching network with normalized         for Computational Linguistics: ACL-IJCNLP 2021,
  hard sample mining. In Proceedings of the Web Con-      pages 1145–1160, Online. Association for Computa-
  ference 2021, pages 821–832.                            tional Linguistics.
                                                    7861
Alec Radford, Jeffrey Wu, Rewon Child, David Luan,         Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan
  Dario Amodei, Ilya Sutskever, et al. 2019. Language        Tan, Xiaochuang Han, and Yulia Tsvetkov. 2023.
  models are unsupervised multitask learners. OpenAI         Can language models solve graph problems in nat-
  blog, 1(8):9.                                              ural language? In Advances in Neural Information
                                                             Processing Systems, volume 36.
Stephen Robertson, Hugo Zaragoza, et al. 2009. The
   probabilistic relevance framework: Bm25 and be-         Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu,
  yond. Foundations and Trends® in Information Re-            Alan Wee-Chung Liew, Shirui Pan, and Baocai Yin.
   trieval, 3(4):333–389.                                     2024. Large language models-guided dynamic adap-
                                                              tation for temporal knowledge graph reasoning. In
Rohit Singh, Vamsi Meduri, Ahmed Elmagarmid,                  Advances in Neural Information Processing Systems,
  Samuel Madden, Paolo Papotti, Jorge-Arnulfo                 volume 37.
  Quiané-Ruiz, Armando Solar-Lezama, and Nan Tang.
  2017. Generating concise entity matching rules. In       Jin Wang, Yuliang Li, and Wataru Hirota. 2021.
  Proceedings of the 2017 ACM International Confer-           Machamp: A generalized entity matching benchmark.
  ence on Management of Data, SIGMOD ’17, page                In Proceedings of the 30th ACM International Con-
  1635–1638, New York, NY, USA. Association for               ference on Information and Knowledge Management,
  Computing Machinery.                                        pages 4633–4642.
                                                           Pengfei Wang, Xiaocan Zeng, Lu Chen, Fan Ye, Yuren
Khanin Sisaengsuwanchai, Navapat Nananukul, and              Mao, Junhao Zhu, and Yunjun Gao. 2022. Promptem:
  Mayank Kejriwal. 2023. How does prompt engi-               prompt-tuning for low-resource generalized entity
  neering affect chatgpt performance on unsupervised         matching. Proc. VLDB Endow., 16(2):369–378.
  entity resolution? arXiv preprint arXiv:2310.06174.
                                                           Tianshu Wang, Xiaoyang Chen, Hongyu Lin, Xuanang
Saku Sugawara, Pontus Stenetorp, Kentaro Inui, and            Chen, Xianpei Han, Le Sun, Hao Wang, and Zhenyu
  Akiko Aizawa. 2020. Assessing the benchmarking              Zeng. 2025. Match, compare, or select? an investi-
  capacity of machine reading comprehension datasets.         gation of large language models for entity matching.
  In Proceedings of the AAAI Conference on Artificial         In Proceedings of the 31st International Conference
  Intelligence, volume 34, pages 8918–8927.                   on Computational Linguistics, pages 96–109, Abu
                                                              Dhabi, UAE. Association for Computational Linguis-
Zequn Sun, Qingheng Zhang, Wei Hu, Chengming                  tics.
  Wang, Muhao Chen, Farahnaz Akrami, and Chengkai
  Li. 2020. A benchmarking study of embedding-based        Cameron R Wolfe. 2024.     Decoder-only trans-
  entity alignment for knowledge graphs. Proc. VLDB          formers: The workhorse of generative llms.
  Endow., 13(12):2326–2340.                                  https://cameronrwolfe.substack.com/p/
                                                             decoder-only-transformers-the-workhorse.
Xiaobin Tang, Jing Zhang, Bo Chen, Yang Yang, Hong
  Chen, and Cuiping Li. 2020. Bert-int:a bert-based        Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang, Rui
  interaction model for knowledge graph alignment.           Yan, and Dongyan Zhao. 2019. Relation-aware en-
  In Proceedings of the Twenty-Ninth International           tity alignment for heterogeneous knowledge graphs.
  Joint Conference on Artificial Intelligence, IJCAI-20,     In Proceedings of the Twenty-Eighth International
  pages 3174–3180. International Joint Conferences on        Joint Conference on Artificial Intelligence, IJCAI-19,
  Artificial Intelligence Organization.                      pages 5278–5284. International Joint Conferences on
                                                             Artificial Intelligence Organization.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
  Martinet, Marie-Anne Lachaux, Timothée Lacroix,          Linyao Yang, Hongyang Chen, Xiao Wang, Jing Yang,
  Baptiste Rozière, Naman Goyal, Eric Hambro,                Fei-Yue Wang, and Han Liu. 2024. Two heads are
  Faisal Azhar, et al. 2023. Llama: Open and effi-           better than one: Integrating knowledge from knowl-
  cient foundation language models. arXiv preprint           edge graphs and large language models for entity
  arXiv:2302.13971.                                          alignment. arXiv preprint arXiv:2401.16960.
                                                           Alexandros Zeakis, George Papadakis, Dimitrios Sk-
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
                                                             outas, and Manolis Koubarakis. 2023. Pre-trained
  Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz
                                                             embeddings for entity resolution: An experimental
  Kaiser, and Illia Polosukhin. 2017. Attention is all
                                                             analysis. Proc. VLDB Endow., 16(9):2225–2238.
  you need. In Advances in Neural Information Pro-
  cessing Systems, volume 30.                              Rui Zhang, Bayu Distiawan Trisedya, Miao Li, Yong
                                                             Jiang, and Jianzhong Qi. 2022. A benchmark and
Somin Wadhwa, Adit Krishnan, Runhui Wang, Byron C            comprehensive survey on knowledge graph entity
  Wallace, and Luyang Kong. 2024. Learning from              alignment via representation learning. The VLDB
  natural language explanations for generalizable entity     Journal, 31(5):1143–1168.
  matching. In Proceedings of the 2024 Conference on
  Empirical Methods in Natural Language Processing,
  pages 6114–6129, Miami, Florida, USA. Association
  for Computational Linguistics.
                                                       7862
A       Architecture of Entity Matching
        Systems Using Language Models
A.1 Pretrained Encoder Models
Entity matching requires rich contextualized repre-
sentations, where pretrained encoder models such
as BERT can produce entity embeddings with high
throughput and low latency but have limited ability
of reasoning and generalization. The most common
                                                            Figure 5: Architecture of Decoder-only Transformer
entity matching (EM) system using pretrained lan-           Model (Vaswani et al., 2017). Image source (Wolfe,
guage models (LMs) is based on the cross-encoder            2024).
framework 1 proposed by Li et al. (2020), where
the model takes serialized entities in pairs as input
and formulates their matching as a classification
task, as shown in Fig. 4.


                                                            Figure 6: 3 Steps of LLM2vec to convert decoder-only
                                                            models as universal text embedding models: enabling
                                                            bidirectional attention, fine-tuning with masked next
                                                            token prediction and unsupervised contrastive learning
                                                            (BehnamGhader et al., 2024).


                                                                schema. The source is the Fodors-Zagats task
                                                                from the Deep Matcher datasets (Bilenko, 2019).

                                                            • D2 Semi-Rel: This task is for matching between
                                                              semi-structured and structured tables. The source
    Figure 4: Architecture of Ditto (Li et al., 2020).        is the set of 5 Movie tasks collected by the Mag-
                                                              ellan project (Das et al.).
A.2 Large Language Models (Decoder-only)
                                                            • D3 iTunes-Amazon (Dirty) contains music data
Compared to encoder models, LLMs with billions                from iTunes and Amazon. This was created by
of parameters contain more knowledge from per-                students in the CS 784 data science class at UW-
taining, but their decoding part is very expensive            Madison. The dirty version was obtained by
and time-consuming. LLM2vec 2 (BehnamGhader                   modifying the structured iTunes-Amazon dataset
et al., 2024) can transform any pretrained decoder-           to simulate dirty data. Specifically, for each at-
only LLM into a (universal) text encoder, where               tribute other than "title", they randomly moved
the model can be used in the bi-encoder framework             each value to the attribute "title" in the same tuple
for matching that breaks the bottleneck of text gen-          with 50% probability.
eration in chat-based matching (See Figs. 5, 6).
                                                            • D4 Walmart-Amazon (Dirty) contains product
B       More Experimental Details                             data from Walmart and Amazon. The procedure
                                                              for generating the dirty version of this dataset is
B.1       Benchmark Datasets
                                                              the same as that for D3.
The following provides a detailed description of
the 10 datasets selected for entity matching and            • D5 IMDb-TVDB consists of two individual
alignment tasks. Table 7 lists the attribute names            data sources, which comprise movie descriptions
for tabular and semi-structured datasets.                     from imdb.com (IMDb) and TV shows from
                                                              TheTVDB.com (TVDB) (Obraczka et al., 2021).
• D1 Rel-HETER: This task is for matching
  between structured tables with heterogeneous              • D6 IMDb-DBpedia matches movies from IMDb
    1
        https://github.com/megagonlabs/ditto                  and DBpedia (Papadakis et al., 2011), but has no
    2
        https://github.com/McGill-NLP/llm2vec                 overlap with the IMDb data source of D5.
                                                         7863
 Dataset             D attribute names                                                                   D′ attribute names
 D1 Rel-HETER        name, address, phone, category                                                      addr, city, phone, type, class
 D2 Semi-Rel         id, title, authors, venue, year                                                     Movie1: id, name, year_range, relase_date, director, creator, cast, du-
                                                                                                         ration, rating:[rating_value, content_rating], genre, url, description;
                                                                                                         Movie2: id, name, year, director, writers(list), actor(list); Movie3: id,
                                                                                                         title, year, director, creators, cast, genre, duration, rating: [rating, con-
                                                                                                         tent_rating], summary; Movie4: id, title, time, director, year, stars(list),
                                                                                                         rating:[rotten_tomatoes, audience_rating, review(list)]; Movie5: id, mo-
                                                                                                         vide_name, year, directors, actors, movide_rating, genre, duration.
 D3 iTunes-Amazon    id, Song_Name, Artist_Name, Album_Name, Genre, Price, CopyRight Time, Released same as D
 D4 Walmart-Amazon   id, title, category, brand, modelno, price                                          same as D
 D5 IMDb-TVDB        title, name, episodeNumber, seasonNumber, deathYear, birthYear, endYear, startYear, title, name, abstract, episodeNumber, seasonNumber, releaseDate, job
                     genre_list, primaryProfessions, runtimeMinutes
 D6 IMDb-DBpedia     id, title, starring, writer, editor, aggregate value                                id, title, actor name, director name, year, genre, aggregate value


                          Table 7: The schema (attribute names) of tabular and semi-structured data.

 Backbone Model                             #param                 License                Model Card
 BERT.base                                  110M                   Apache License 2.0     https://huggingface.co/google-bert/bert-base-uncased
 RoBERTa.base                               125M                   MIT License            https://huggingface.co/FacebookAI/roberta-base
 LLM2Vec-Mistral-7B-Instruct-v2-mntp        7B                     MIT License            https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp
 LLM2Vec-Meta-Llama-3-8B-Instruct-mntp      8B                     MIT License            https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
 Dataset                                    Type                   License                Data Source
 D1 Rel-HETER, D2 Semi-Rel                  Clean, Hetero.         BSD-3-Clause           https://github.com/megagonlabs/machamp
 D3 iTunes-Amazon, D4 Walmart-Amazon        Dirty, Homo.           GPL-3.0 license        https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
 D5 IMDb-TVDB, D6 IMDb-DBpedia              Clean, Homo.           CC BY 4.0              https://zenodo.org/records/6950980
 D7 DBP15K(EN-FR), D8 DBP-WIKI              KG, Canonical          GPL-3.0 license        https://github.com/nju-websoft/OpenEA
 D9 ICEWS-WIKI, D10 ICEWS-YAGO              KG, Highly Hetero.     CC BY 4.0              https://github.com/IDEA-FinAI/Simple-HHEA/tree/main/data


                              Table 8: Data license and model card of pretrained language models.


• D7 DBP15K(EN-FR) and D8 DBP-WIKI                                                          ble 9. We further fine-tune each model on the train
  are two canonical entity alignment datasets.                                              set of all six datasets at once for each serialization
  DBP15K(EN-FR) is for bilingual entity align-                                              and report their performance in Table 3.
  ment on DBpedia, and DBP-WIKI is for aligning                                                For all baselines compared for matching KG
  entities obtained from Wikipedia and DBpedia.                                             entities, we followed the original hyperparameter
                                                                                            settings reported in their papers with the same 3:7
• D9 ICEWS-WIKI and D10 ICEWS-YAGO come                                                     splitting ratio in the training/testing set. All base-
  from Highly Heterogeneous Knowledge Graph                                                 lines use the same preprocessing procedure to ob-
  Datasets (Jiang et al., 2024b). Both datasets are                                         tain the initial features from four KG datasets. We
  for KG entity alignment, integrating the event                                            fine-tuned the base model of open-source LLMs un-
  knowledge graph derived from the Integrated Cri-                                          der supervised fine-tuning through LLM2vec with
  sis Early Warning System (ICEWS) and general                                              recommended parameters and entity pairs serial-
  KGs (i.e., WIKIDATA, YAGO).                                                               ized by the walk-based strategy (M = 10, L = 1)
B.2    Experimental Settings                                                                as input. Datasets D7-8 and D9-10 have common
                                                                                            domains with DBpedia and ICEWS, respectively.
To benchmark the effect of entity serialization                                             Thus, we fine-tune one model for datasets with a
schemes on pretrained encoder models, we use                                                sharing domain. The reported results are the aver-
the default setting of the Ditto framework (Li et al.,                                      age of 5 runs.
2020), which only modifies how entity serialization                                            The code base is developed based on Pytorch
is performed, to fine-tune the encoder model for                                            2.3.1 with Transformer 4.40.2, and llm2vec
each scheme S listed in Table 1 on datasets D1-6,                                           0.2.2. The experiments are performed on a server
and report the results in Table 3. For LLMs, we first                                       of Ubuntu 24.04LTS OS with 3.0 GHz 2nd Gen
transform the base model (listed in Table 8) using                                          Intel Xeon processors (96 cores) and 8 NVIDIA
LLM2vec and then obtain the entity embeddings                                               A100 Tensor Core GPUs (40G).
directly. For matching tasks, the default instruc-
tion used by LLM2Vec is replaced by "Given a
                                                                                            B.3        More Experimental Results
description of a real-world entity, retrieve the most
relevant entities that match or align with the given                                        Zero-shot EM with LLMs To study how at-
entity." The matching score for each entity tuple                                           tribute order, special tokens, or missing/dirty at-
(one positive versus k negative pairs) in datasets                                          tributes affect LLMs for entity matching under the
D1-6 is computed under each serialization scheme,                                           zero-shot setting, we compare the entity embed-
and the zero-shot performance is summarized in Ta-                                          dings generated by off-the-shelf LLMs in one run
                                                                                    7864
                    D1 (MRR)        D2 (MRR)        D3 (MRR)        D4 (MRR)        D5 (MRR)        D6 (MRR)
      Scheme
                  Mistral Llama   Mistral Llama   Mistral Llama   Mistral Llama   Mistral Llama   Mistral Llama
      S1 Fixed    0.903   0.901   0.843   0.813   0.739   0.612   0.840   0.780   0.713   0.544   0.695   0.685
      S2 Random   0.883   0.892   0.839   0.812   0.696   0.626   0.834   0.769   0.707   0.499   0.670   0.663
      S4 Valid    0.903   0.901   0.852   0.830   0.703   0.548   0.815   0.743   0.718   0.579   0.712   0.719
      S5 Plain    0.917   0.964   0.836   0.839   0.698   0.636   0.856   0.827   0.702   0.595   0.702   0.733
      S7 JSON     0.939   0.929   0.881   0.869   0.708   0.539   0.839   0.774   0.721   0.583   0.709   0.745

      Table 9: LLMs for Zero-shot Structured EM with Different Serialization (1st bold, 2nd underline)


under the bi-encoder framework and report the re-
sults in Table 9. S3 Pairwise order is skipped as
the bi-encoder framework encodes the serialized
entities separately. S6 Span typing is designed to
be paired with supervised training, which is also
skipped under the zero-shot setting.

C   Use of Generative AI
In this work, generative AI services are used to
study several GPT-based baselines, test corner
cases in entity matching, and polish writing at the
sentence level.




                                                       7865
