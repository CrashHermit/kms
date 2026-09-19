        Entity Pair-guided Relation Summarization and Retrieval in LLMs for
                         Document-level Relation Extraction

                      Fu Zhang† , Hongsen Yu† , Jingwei Cheng∗ , Huangming Xu
               School of Computer Science and Engineering, Northeastern University, China
                {zhangfu,chengjingwei}@mail.neu.edu.cn; yuhongsen163@163.com



                           Abstract                                                  Original document
                                                                     Skai TV is a Greek free-to-air television network
                                                                     based in Piraeus. It is part of the Skai Group, one of
         Document-level relation extraction (DocRE)                  the largest media groups in the country…

         aims to extract relations between entities in a                            Traditional approach
                                                                                Document-level candidate relations
         document. While previous research has primar-
                                                                                            Relation_a
         ily focused on traditional small models, recent                                    Relation_b
                                                                                                                 Relation
                                                                                                                Extraction
                                                                                                …
         studies have extended the scope to large lan-                    Finetuned

         guage models (LLMs). Current LLM-based                                 Our approach                         Entity-pair-level candidate relations
                                                                                                                                 Relation_a
         methods typically focus on filtering all poten-                    Relation summarizations                              Relation_b
                                                                                                                                                      Relation
                                                                                 for entity pairs                                                    Extraction
                                                                                                                                     …
         tial relations (candidate relations) within a doc-                                                     Retrieve

         ument at one time and then performing triplet
         fact extraction. However, most approaches
                                                                  Figure 1: Differences between traditional approach and
         for candidate relation filtering are based on
                                                                  our approach in LLM-based document-level relation
         the document level, which results in insuffi-
                                                                  extraction. Additionally, we provide a preliminary com-
         cient correlation between candidate relations
                                                                  parison of the F1 scores between the document-level and
         and entity pairs. In addition, the data imbal-
                                                                  entity-pair-level candidate relation filtering methods.
         ance problem caused by a large amount of
         no-relation data (NA problem) is another im-
         portant reason for the suboptimal performance            challenge of document-level approaches is that a
         of LLM-based methods. To address these is-
                                                                  large number of triplet facts need to be obtained
         sues, we propose an entity pair-guided relation
         summarization and retrieval model (EP-RSR)               through joint reasoning of multiple sentences (Yao
         for DocRE, which introduces an innovative                et al., 2019), which places higher demands on the
         LLM-based document-level relation extraction             reasoning capabilities of the model.
         paradigm, EPRF (Entity Pair-Relation-Fact),                 Traditional approaches for DocRE primarily re-
         along with an entity pair-level candidate rela-          lies on graph neural networks (GNNs) and pre-
         tion filtering method. Our approach first selects        trained language models (PLMs). GNN-based
         entity pairs that potentially contain relations
                                                                  models mainly perform explicit reasoning on
         and uses them to guide relation summarization
         and retrieval for extracting relation facts. This
                                                                  graphs constructed from entities and sentences in a
         enhances the relevance between candidate rela-           document, e.g., EoG (Christopoulou et al., 2019),
         tions and entity pairs while alleviating the issue       GAIN (Zeng et al., 2020), and SIRE (Zeng et al.,
         of imbalanced NA data. Benchmark testing on              2021). Approaches based on PLMs take the word
         three datasets demonstrates that our approach            sequence of a document as input and leverage the
         achieves state-of-the-art (SOTA) performance             transformer (Vaswani, 2017) to implicitly capture
         for LLM-based models1 .                                  the long-range contextual dependencies between
1        Introduction                                             entities, e.g., ATLOP (Zhou et al., 2021), EIDER
                                                                  (Xie et al., 2022), and DREEAM (Ma et al., 2023).
Document-level relation extraction (DocRE) is a                      With the notable success of large language mod-
crucial task in natural language processing (NLP),                els (LLMs) like GPT in the field of NLP (Brown
aimed at identifying and extracting semantic re-                  et al., 2020; Touvron et al., 2023), approaches for
lations between entities within a given document.                 LLM-based DocRE have gradually gained trac-
Compared to sentence-level relation extraction, the               tion. Unlike traditional approaches, LLM-based
1
    Our code: https://github.com/LookingYu/EP-RSR.                DocRE approaches primarily treat LLMs as black
    †
    Equal contribution. ∗ Corresponding author.                   boxes, focusing on relation extraction paradigms
                                                              4022
                                    Findings of the Association for Computational Linguistics:
                                                  NAACL 2025, pages 4022–4037
                             April 29 - May 4, 2025 ©2025 Association for Computational Linguistics
and prompt engineering (Gao et al., 2023; Li et al.,      finally performs triplet fact extraction. Based on
2023). Moreover, leveraging the strong generaliza-        this paradigm, our approach first selects entity pairs
tion capabilities of LLMs, specific fine-tuned on         that potentially contain relations from a given docu-
DocRE tasks has become an essential process for           ment, thereby alleviating the imbalance issue of NA
enhancing relation extraction performance (Xue            data. Subsequently, we introduce entity-pair-level
et al., 2024). However, existing research indicates       relation retrieval, which retrieves candidate rela-
that the performance of LLM-based DocRE still             tions for an entity pair from our constructed train-
lags behind that of traditional small models at their     ing data based on relation summarization of the
state-of-the-art (SOTA) levels, underscoring the          entity pair. Finally, utilizing the candidate relations
necessity of further advancing research in this area.     and their descriptions, we judge whether the entity
   Given the wide variety of relation types between       pair has the candidate relations, thus achieving the
entities in DocRE (e.g., there are 96 types of rela-      goal of triplet fact extraction. Our contributions are
tions in the Re-DocRED (Tan et al., 2022) dataset),       summarized as follows:
the filtering of candidate relations between entities
has become one of the key factors influencing the             • We propose an entity pair-guided relation
performance of current LLM-based DocRE meth-                    summarization and retrieval model EP-RSR
ods. Current relation extraction paradigms in the               based on a novel paradigm EPRF for LLM-
sentence-level field generally favor the inclusion              based document-level relation extraction. Our
of candidate relations in prompts to enhance the                method effectively alleviates the issue of NA
effectiveness of triplet fact extraction, which often           data imbalance and enhances the model’s ca-
treat the entire set of relation types as the candidate         pacity to extract triplet facts.
relation set (Wang et al., 2023b). However, this
                                                              • We propose a candidate relation filtering
approach is not suitable for DocRE with a large
                                                                method based on entity-pair-level relation re-
number of relations. Recently, Xue et al. (2024)
                                                                trieval, which retrieves candidate relations at
propose a document-based candidate relation fil-
                                                                the entity-pair level based on relation summa-
tering method, which first selects all potentially
                                                                rizations of entity pairs, thereby enhancing
existing relations within a given document at one
                                                                the relevance between candidate relations and
time and then performs triplet fact extraction. How-
                                                                entity pairs.
ever, this method relies heavily on the document
itself for candidate relation selection and lacks suf-        • We implemented this method on the DocRED,
ficient correlation with the head and tail entities to          Re-DocRED and DWIE datasets. Results
be extracted, as illustrated in Figure 1.                       demonstrate that our approach achieves sig-
   Additionally, the no-relation (NA label) data                nificant performance improvements over com-
imbalance problem (NA problem) in DocRE is an-                  petitive LLM-based baselines (+7.42 F1 on
other significant factor leading to suboptimal per-             DocRED test set, +4.97 F1 on Re-DocRED
formance of LLM-based models. For example, in                   dev set and +18.00 F1 on DWIE test set).
the DocRED (Yao et al., 2019) dataset, entity pairs
without relation account for 97.17% of the dataset.       2     Related Work
The excessive amount of NA data poses a risk of
false positives in relation prediction (Wan et al.,       2.1    Traditional DocRE
2023). Gao et al. (2023) addresses this by enhanc-        Document-level relation extraction is a pivotal task
ing prediction capabilities through data program-         in NLP, wherein over 40.7% of relations are depen-
ming combined with multiple weak supervision              dent on cross-sentence joint reasoning (Yao et al.,
sources. However, the effectiveness of this method        2019). Early studies predominantly employ deep
is often constrained in the absence of high quality       learning models such as convolutional neural net-
sources of weak supervision.                              works (CNN) and long short-term memory net-
   To address the aforementioned issues, we pro-          works (LSTM) for semantic representation learn-
pose an entity pair-guided relation summarization         ing (Zheng et al., 2018; Yao et al., 2019; Tang
and retrieval model EP-RSR based on a novel               et al., 2020). As research progressed, the con-
LLM-based DocRE paradigm EPRF (Entity Pair-               cept of graphs is increasingly integrated, exem-
Relation-Fact). This paradigm EPRF first selects          plified by graph convolutional networks enhanced
entity pairs, then filters candidate relations, and       with global contextual information (Sahu et al.,
                                                      4023
2019), edge-oriented graph extraction techniques        we propose an entity pair-guided relation summa-
(Christopoulou et al., 2019), and graph aggregation-    rization and retrieval model EP-RSR based on a
and-inference network which features a double           novel LLM-based DocRE paradigm EPRF (Entity
graph design (Zeng et al., 2020). Moreover, the         Pair-Relation-Fact), which focuses on entity-pair-
application of pre-trained language models BERT         level candidate relations, thereby enhancing the
(Devlin et al., 2019) and RoBERTa (Liu et al.,          relevance between candidate relations and entity
2019) in this domain has proliferated, encompass-       pairs to achieve better triplet fact extraction.
ing approaches like adaptive thresholding with lo-
cal context pooling (Zhou et al., 2021), knowledge      3     Methodology
distillation strategies (Ma et al., 2023), and train-   3.1    Problem Definition
able memory module (Gao et al., 2024).
                                                        Given a document D = {si }ni=1
                                                                                    s
                                                                                       , where each sen-
2.2 LLM-based DocRE                                     tence si = {wj }j=1
                                                                            ni
                                                                             w
                                                                                  contains niw words, and an
With the immense potential of LLMs increasingly         entity set V = {ei }ni=1e
                                                                                   . The DocRE task is to pre-
evident across various domains (Wang et al., 2023a;     dict the relation r ∈ R ∪ {N A} between entity
Zhou et al., 2023; Xu et al., 2023), document-level     pair (eh , et ), where h, t ∈ {1, · · · , ne } and h ̸= t.
relation extraction methods centered around LLMs        The set R = {ri }ni=1 r
                                                                                  represents a predefined col-
have emerged, encompassing both fine-tuned and          lection of relations, while N A signifies the absence
non-fine-tuned approaches.                              of relation between the entity pairs.
   For non-fine-tuned approaches, Gao et al. (2023)
enhance model’s relation extraction capabilities by     3.2    Overview
combining prompting techniques with data pro-           Based on our new paradigm EPRF (Entity Pair-
gramming. Li et al. (2023) enrich the DocRE             Relation-Fact), which first selects entity pairs, then
dataset by integrating LLMs with a natural lan-         filters candidate relations, and finally performs
guage inference module. Additionally, Ozyurt et al.     triplet fact extraction, our model EP-RSR com-
(2023) propose the REPLM model, which is de-            prises three key components: (1) Entity informa-
signed for few-shot relation extraction within a        tion enhanced relation summarization module,
contextual framework using LLMs.                        which first selects entity pairs that potentially con-
   In terms of fine-tuned approaches, Li et al.         tain relations from a given document. Then, it gen-
(2024) propose a new fine-tuned LLM-based               erates a relation summarization for each entity pair,
DocRE method, which adds relation sets and en-          which includes entity information and relation in-
tity pairs to prompts for document-level relation       formation related to the entity pair. (2) Entity-pair-
extraction. Xue et al. (2024) propose a new             level relation retrieval module, which retrieves
RE paradigm model AutoRE. Although the au-              candidate relations that exhibit a higher relevance
thor initially adopted the calculation method and       to the entity pair based on the relation summariza-
goal of document-level relation triplet extraction      tion. (3) Triplet fact judgement module, which
(DocRTE), this method has achieved exceptional          achieves the goal of triplet fact extraction by judg-
performance in LLM-based DocRE. Due to the              ing whether the entity pair has the candidate rela-
large number of relations involved in DocRE, cur-       tions. An illustration of the overall framework of
rent LLM-based approaches generally avoid incor-        our approach is shown in Figure 2.
porating the entire relation list into the prompt
template. To achieve this, AutoRE proposes a            3.3    Entity Information enhanced Relation
paradigm RHF (Relation-Head-Fact) that enhances                Summarization
relation extraction performance by filtering the re-    A significant challenge in DocRE is long contexts.
lation list to obtain candidate relations and sub-      Existing LLM-based methods typically identify all
sequently extracting triplet facts. However, this       candidate relations from a given document without
method primarily filters candidate relations based      adequately considering the relevance of the con-
on the given document, which often has low corre-       textual information carried by these candidate rela-
lation with individual entity pairs. Since the core     tions to the target head and tail entities. To obtain
objective of DocRE is to acquire triplets of entity     context information that is more pertinent to the
pairs, candidate relations should be more strongly      entity pairs, we propose a novel entity information
associated with specific entity pairs. Therefore,       enhanced relation summarization approach.
                                                    4024
          Entity Information enhanced Relation Summarization
                                                                                                         Selected entity pairs
                      Original document                                                             (Skai TV , Skai Group )
     Skai TV is a Greek free-to-air television network                                              (Skai TV , Piraeus)
     based in Piraeus. It is part of the Skai Group, one of                                         (Skai TV , Greek)
     the largest media groups in the country…                                                       …



                                                 Relation summarization for an entity pair
         Skai TV is a Greek free-to-air television network, Skai Group is one of the largest media groups in Greece, Skai TV is a
         part of the Skai Group.


         Entity-pair-level Relation Retrieval                                Triplet Fact Judgement

                                    Relation summarization
                                                                                               Instruction
                                       for the entity pair
                                                                      Based on the text and the description of the relation "owned
   retrieve training data
                                                                      by", give an answer about whether the head and tail entity
                                                                      pairs (head entity and tail entity) satisfy the "owned by"
           Coarse filtering of candidate relations                    relation..
    owned by -- P127
    country -- P17                                                                              Entity pair
    located in the administrative territorial entity -- P131          Entity_h: Skai TV         Entity_t: Skai Group

            Fine filtering of candidate relations                                              Entity pair text
    **Entity pair text**                                              Skai TV is a Greek free-to-air television network based in
    Skai TV is a Greek free-to-air television network                 Piraeus. It is part of the Skai Group …
    based in Piraeus. It is part of the Skai Group …
                                                                                         Relation description
    **Options**
                                                                      'owned by' designates the owner (object) of the subject.
    A. Skai Group is one of the owners of Skai TV
    B. Skai TV is located in country Skai Group                      Is there a 'owned by’ relation between Skai Group and Skai TV?
    C. Skai TV is located in the administrative territorial
    Skai Group
    D. Skai TV is not related to Skai Group                                          YES (Skai TV, Skai Group, owned by)



Figure 2: The overview of our model EP-RSR. It contains three key parts: (1) Select potential entity pairs in a
document and obtain enhanced relation summarization for each entity pair. (2) Retrieve entity-pair-level candidate
relations using the relation summarizations based on double filtering mechanisms. (3) Judge and extract triplet facts
based on the candidate relations and their descriptions.


   Specifically, we first select entity pairs that po-                takes as input an instruction I, a document D, and
tentially contain relations from a given document.                    a set of entities V = {ei }ni=1
                                                                                                   e
                                                                                                      . It employs LLMs to
Although, considering that the huge number of en-                     perform k random sampling, resulting in an entity
tity pairs in DocRE poses a significant challenge,                    pair set Pi in one sampling iteration.
for instance, the DocRED dataset’s dev set includes
26,141 entities, leading to 362,313 entity pairs.                                         Pi = LLM (I, D, V ).                        (1)
Getting enhanced relation summaries for all entity
pairs would incur significant computational costs.                       Selecting entity pairs that potentially contain
In addition, the large number of entity pairs causes                  relations. During each sampling iteration, entity
NA problem, which can lead to false positives in                      pairs are filtered based on their cosine similarity to
relation prediction (Gao et al., 2023). To address                    the true entities in the dataset. This filtering process
these issues, we propose using LLMs to conduct                        uses a threshold T to obtain the set of valid entity
multi-sampling selection of entity pairs within the                   pairs Pi′ :
document, prior to relation summarization. This
                                                                              Pi′ = {(eh , et ) ∈ Pi | sim(eh , e′h ) > T,
approach identifies the entity pairs that potentially                                                                                 (2)
contain relations, thereby reducing costs and ef-                                                       sim(et , e′t ) > T },
fectively alleviating NA problem. In detail, we
                                                                      where e′h , e′t ∈ V , sim(eh , e′h ) denotes the cosine
implement the following steps:
                                                                      similarity between the entity eh and the entity e′h
  Multi-sampling. The multi-sampling process                          (similar to sim(et , e′t )).
                                                                 4025
   We finally merge the sets of valid entity pairs          H(eTh , eTt ) using Sentence-transformers (Reimers
obtained from all k sampling iterations to obtain the       and Gurevych, 2019):
set of entity pairs potentially containing relations,
denoted as Pe :                                                     H(eTh , eTt ) = Encode(Rs′ (eTh , eTt )).    (8)

                            k
                            [                               The vector representations of the entity pairs are
                     Pe =         Pi′ .              (3)    stored as keys, with their true relation labels as
                            i=1                             values in the pre-defined training data TD :
   Further, by utilizing the summarization capabili-               TD = {H(eTh , eTt ) : Label(eTh , eTt )}.     (9)
ties of LLMs, we can deduce the relation summa-
rization Rs between an entity pair (eh , et ) in Pe .          Coarse filtering of candidate relations: Ini-
As an example, for the entity pair ("Skai TV", "Skai        tially, the relation summarization Rs′ for entity pair
Group"), LLMs gets the relations summarization              (eh , et ) in Pe is computed by Eq. (8) to obtain H.
"Skai TV is a component of the Skai Group" based               Next, the cosine similarity between encoded rep-
on the document D.                                          resentation H and each encoded representation H T
   Additionally, considering that relation descrip-         in training data TD is computed using the formula:
tions often impose implicit constraints on entity
types, for instance, the "country" relation typically                                       H · HT
                                                                        sim(H, H T ) =               .          (10)
assumes that the head entity is non-human and the                                          ∥H∥∥H T ∥
tail entity is a nation. Therefore, we integrate en-
                                                            All cosine similarity results are sorted, and top
tity information E(eh ) and E(et ), such as "Skai
                                                            k similar data are selected. The corresponding
TV is a Greek free-to-air television network" and
                                                            relation labels form the candidate relations set RC .
"Skai Group is one of the largest media groups in
                                                                Fine filtering of candidate relations: By lever-
Greece" with the relation summarization Rs to con-
                                                            aging predefined relation templates RT , the coarse-
struct a new enhanced relation summarization Rs′ .
                                                            filtered candidate relations RC for entity pair
The formula is defined as follows:
                                                            (eh , et ) are transformed into natural language. This
              E(eh ) = LLM (eh , D),                 (4)    transformation is followed by a fine filtering using
                                                            multiple-choice QA and prior knowledge of the en-
               E(et ) = LLM (et , D),                (5)    tity pairs to yield the final set of candidate relations
                                                            RF . The steps are outlined as follows:
             Rs(h,t) = LLM (eh , et , D),            (6)        First, the candidate relations RC are converted
                                                            into a set of natural language options Q using the
   Rs′ (eh , et ) = E(eh ) + E(et ) + Rs(h,t) ,      (7)
                                                            predefined relation templates RT and prior knowl-
where + represents the concatenation of strings             edge P K of the entity pairs:
and the entity pair (eh , et ) ∈ Pe .
                                                               Q = Conversion(RC , RT, P K, eh , et ). (11)
3.4 Entity-pair-level Relation Retrieval
                                                            If the natural language option set Q is empty, it is
To obtain candidate relations, we propose an entity-        considered that there is no relation between the en-
pair-level relation retrieval method. The method            tity pair (eh , et ). Otherwise, the natural language
retrieves the top-k candidate relations by double           options set Q, along with the document D and
filtering from our pre-constructed training data TD         the instruction I, are combined to form a multiple-
based on the relation summarization corresponding           choice question. We add "no relation" as an addi-
to an entity pair.                                          tional option to the question. This question is pro-
    Pre-constructed training data TD : The train-           cessed by LLMs, resulting in an answer A(eh , et ):
ing data is constructed from the train set T of the
dataset. Specifically, for each entity pair (eTh , eTt )          A(eh , et ) = LLM (I, D, Q, eh , et ),        (12)
with relations in the train set of the dataset, a rela-
tion summarization Rs′ (eTh , eTt ) for it will be gener-   where A(eh , et ) may contain multiple options.
ated according to the approach in Section 3.3.                Finally, the answer A(eh , et ) is converted into
    The relation summarization Rs′ (eTh , eTt ) is          corresponding relation labels, yielding final set of
then converted into a vector representation                 candidate relations RF for the entity pair (eh , et ).
                                                        4026
3.5 Triplet Fact Judgement                                  NVIDIA 3090 GPUs for fine-tuned, with inference
For each candidate relation r ∈ RF , we construct           operations carried out on a single 4090 GPU.
a prompt and feed it into LLMs. Each prompt                    We only fine-tune the LLM for entity pair selec-
consists of the following components:                       tion, multiple-choice QA in Section 3.4, and triple
   Instruction Ir : A tailored instruction is pro-          fact judgment stages. Fine-tuning parameters and
vided for each candidate relation r, detailing the          inference parameter settings at each stage can be
DocRE task and specifying the output requirement.           found in the Appendix B. The prompt templates
The model is required to output "YES" or "NO". If           for each stage are shown in the Appendix C.
the entity pair contains the candidate relation, the           In addition, the experimental analysis of the k in
model outputs "YES"; otherwise, it outputs "NO".            top-k candidate relations retrieval and the k-times
   Relation Description RDr : To enhance the                multi-sampling in entity pair selection are detailed
model’s understanding of the specific meaning of            in Appendix D.
the candidate relation r, we include a detailed rela-          In order to better test the versatility of our ap-
tion description in the prompt. This improves the           proach, we also conducted experiments on the
model’s ability to judge candidate relations.               DocRTE task. The details and results of the ex-
   Test Input xtest : The document D, along with            periments are shown in Appendix E.
the head entity eh and the tail entity et , is provided
                                                            Baseline Models In this study, the primary mod-
to the model as input. The model generates the
                                                            els for comparison include both non-fine-tuned and
corresponding answer.
                                                            fine-tuned LLM models for DocRE. The non-fine-
   The entire process is as follows:
                                                            tuned models consist of PromptRE (Gao et al.,
           ytest = LLM (Ir , RDr , xtest ),         (13)    2023), DocGNRE (Li et al., 2023), and few-shot
                                                            on ChatGPT (Han et al., 2023). For the fine-tuned
where ytest ∈ {Y ES, N O}. If the output is                 model, we select LMRC (Li et al., 2024) and Au-
"YES", we infer that the given entity pair (eh , et )       toRE (Xue et al., 2024) as a comparison baseline
has the relation r, and the model will output triplet       for DocRE. Additionally, we also incorporate the
(eh , et , r). If the output is "NO", it is inferred that   relation extraction paradigms of D-F (Document-
there is no relation between the entity pair (eh , et ),    facts) and D-R-F (Document-relation-facts) men-
and the model does not output anything.                     tioned in AutoRE (Xue et al., 2024) as our compar-
                                                            ison baselines.
4   Experiments
                                                               In addition, to further demonstrate the perfor-
4.1 Experimental Setup and Baselines                        mance of LLM-based and traditional models, we
Dataset We conducted experiments on three                   also select some traditional models introduced in
DocRE datasets: DocRED (Yao et al., 2019), its              Section 2.1 as baselines, including CNN (Yao et al.,
revised version Re-DocRED (Tan et al., 2022), and           2019), LSTM (Yao et al., 2019), BiLSTM (Yao
a new gold-annotated dataset DWIE (Zaporojets               et al., 2019), ATLOP (Zhou et al., 2021), DREEAM
et al., 2021). The detailed statistics of the datasets      (Ma et al., 2023), and TTM-RE (Gao et al., 2024).
can be found in Appendix A.
                                                            4.2   Main Results
Evaluation Metrics We employ F1 and Ign F1
                                                            All experimental results are presented in Table 1
as main evaluation metrics following (Yao et al.,
                                                            and Table 2. Our model consistently outperforms
2019). Ign F1 is employed to assess the F1 score
                                                            all LLM-based baselines on three datasets. More-
while excluding relations shared between the train-
                                                            over, we draw several interesting conclusions:
ing and test sets.
                                                               Our approach significantly outperforms current
Implementation Settings We utilized Llama3-                 fine-tuned baseline AutoRE (Xue et al., 2024),
8B (Touvron et al., 2023) as primary model                  achieving SOTA performance in the LLM-based
for our experimental framework, and employed                DocRE methods. Specifically, compared to Au-
the Sentence-Transformer model all-mpnet-base               toRE, our model achieves an enhancement of 7.42
(Reimers and Gurevych, 2019). We use Llama-                 in F1 score on DocRED test set and an improve-
Factory (Zheng et al., 2024) framework to fine-tune         ment of 18.00 in F1 score on DWIE test set. These
LLM based on LoRA (Hu et al., 2022). Our ex-                significant improvements demonstrate the effective-
periments were predominantly conducted on three             ness of the proposed method.
                                                        4027
                                                           DocRED                                 Re-DocRED
     Model                                         Dev                    Test              Dev                Test
                                              F1        Ign F1       F1     Ign F1     F1     Ign F1      F1     Ign F1
     Traditional models
     CNN (Yao et al., 2019)                  43.45      41.58       42.26    40.33      -        -        -         -
     LSTM (Yao et al., 2019)                 50.68      48.44       50.07    47.71      -        -        -         -
     BiLSTM (Yao et al., 2019)               50.94      48.87       51.06    48.78      -        -        -         -
     ATLOP (Zhou et al., 2021)               61.09      59.22       61.30    59.31    77.63    76.88    77.73     76.94
     DREEAM (Ma et al., 2023)                61.42      59.60       61.13    59.12      -        -      77.94     77.34
     TTM-RE (Gao et al., 2024)                 -          -           -        -      78.13    78.05    79.95     78.20
     LLM-based models
     PromptRE (Gao et al., 2023)               -          -           -        -      10.55     9.03      -         -
     DocGNRE (Li et al., 2023)*              13.84      13.65       13.93    13.67    11.18    11.10    11.12     11.04
     ChatGPT (Han et al., 2023)              32.21        -           -        -      28.89       -       -         -
     LMRC (Li et al., 2024)                  39.25      38.62       38.66    38.09    52.56    52.29    52.45     52.15
     D-F (Xue et al., 2024)*                 46.38      44.77       47.08    45.30    54.22    53.48    53.33     52.50
     D-R-F (Xue et al., 2024)*               45.77      44.32       47.50    45.98    56.58    56.10    54.84     54.35
     AutoRE (Xue et al., 2024)*              47.17      45.58       47.15    45.45    60.17    59.25    59.29     58.33
     Ours (EP-RSR)                           53.77      51.25       54.57    51.77    65.14    63.93    64.24     63.03

Table 1: Experimental results on two public datasets for DocRE. Results with * are our reproduction using
Llama3-8B. Bold indicates the best results among the LLM-based methods.


                                             DWIE                     time, the experimental results indicate that these
Model                                                                 investments are worthwhile.
                                     Dev                Test
                                F1    Ign F1       F1     Ign F1         Additionally, our approach is better than some
ChatGPT (Han et al., 2023)   -     -   26.72   -
                                                                      earlier traditional small models (e.g., BiLSTM
DocGNRE (Li et al., 2023)* 11.85 10.55 13.12 10.73                    (Yao et al., 2019)), but is still inferior to the latest
AutoRE (Xue et al., 2024)* 56.53 52.74 56.38 49.31                    traditional small models (e.g., DREEAM (Ma et al.,
Ours (EP-RSR)                  70.23 66.32 74.38 69.56                2023) and TTM-RE (Gao et al., 2024)). Our LLM-
                                                                      based results further narrows the performance gap
Table 2: Performance on the DWIE dataset for DocRE.                   with the latest traditional small models, making it
Results with * are our reproduction using Llama3-8B.
                                                                      a promising approach for future DocRE.

 Model                                         F1        Ign F1       4.3   Ablation Study
 Our Method                                   53.77      51.25        To evaluate the contribution of each module to the
 w/o Entity pair selection                    21.83      19.12
 w/o Entity information                       51.51      49.25        model’s performance, we conducted an ablation
 w/o Entity-pair-level candidate relations    47.00      45.57        study on DocRED dev set, as illustrated in Table 3.
 w/o Triplet fact judgement                   47.65      44.49        Our observations include the following aspects:
        Table 3: Ablation study on the DocRED.                        Impact of entity pair selection Entity pair se-
                                                                      lection is very effective for LLM-based approach
                                                                      on DocRE task. Removing it leads to a sharp drop
  Moreover, compared to non-fine-tuned methods,                       in performance. F1 and Ign F1 drop by 31.94 and
our model demonstrates a significant performance                      32.13 respectively. It indicates that selecting en-
breakthrough, achieving an improvement of ap-                         tity pairs that potentially contain relations further
proximately 21.56 in F1 on DocRED dev set over                        enhances the ability of LLM triplet fact extraction.
the ChatGPT-based DocRE baseline (Han et al.,
2023), thereby exhibiting a greater degree of com-                    Impact of entity information for relation sum-
petitiveness. Although the fine-tuned process ne-                     marization Removing the entity information in
cessitates additional computational resources and                     the relation summarization, the F1 and Ign F1 of
                                                                   4028
  Original document: Roketsan is a major Turkish weapons manufacturer and defense contractor based in the central Anatolian province of
  Ankara . Incorporated in 1988 by Turkey 's Defense Industry Executive Committee ( SSİK ) in order to establish the nation 's industrial base
  on rocket technology , the company has quickly risen to become one of Turkey 's top 500 industrial establishments…
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Entity pair: (Ankara, Turkey)

   Ground truth relation label: country, located in the administrative territorial entity, contains administrative territorial entity


   Entity-pair-level candidate relation: country, located in the administrative territorial entity, contains administrative territorial entity


   Document-level candidate relation: inception, manufacturer, country, contains administrative territorial entity, applies to jurisdiction,
   located in the administrative territorial entity, headquarters location



                                             Figure 3: Case study for candidate relation selection.


our model decrease by 2.26 and 2 respectively, in-                                         Model            Number of entity pairs                F1      NA percentage
dicating that although the entity information is sim-                                      w Select                     15480                   53.77             51.85
ple, adding it is very helpful for making the relation                                     w/o Select                   362313                  21.83             97.17
summarization contain more relation information.
                                                                                          Table 4: Experimental analysis of the effectiveness of
                                                                                          alleviating the NA problem.
Impact of entity-pair-level candidate relations
Replacing the entit-pair-level candidate relations
                                                                                          4.5 Further Analysis of candidate relations at
with the document-level candidate relations, the
                                                                                              the entity-pair-level and document-level
performance of our model experiences a significant
drop, with F1 and Ign F1 dropping by 6.77 and                                             The preceding ablation study has demonstrated
5.68 respectively, which indicates that our entity-                                       the effectiveness of entity-pair-level candidate rela-
pair-level method is more effective for selecting                                         tions. To further investigate their significance, we
candidate relations.                                                                      replace the coarse filtering of candidate relations
                                                                                          in Section 3.4 with document-level candidate rela-
                                                                                          tions and only calculate the F1 of this part as shown
Impact of triplet fact judgement Removing the                                             in Table 5. When using document-level candidate
triplet fact judgement, F1 and Ign F1 decreased by                                        relations, the F1 score decreases by 8.28, accuracy
6.12 and 6.76. It indicates that our method enables                                       decreases by 5.65, and recall decreases by 12.46,
LLMs to better understand relations and facilitates                                       which indicates that the entity-pair-level candidate
more accurate relation extraction.                                                        relation method is more effective.
   In addition, we conduct the ablation analysis on
the multiple-choice QA and prior knowledge used                                                 Model                              P                R                F1
in the fine filtering of candidate relations of Section                                         Entity-pair-level               23.46             91.24            37.33
                                                                                                Document-level                  17.81             78.78            29.05
3.4, and the results are detailed in Appendix F.
                                                                                           Table 5: Results on different level candidate relations.
4.4 Assessment of the effectiveness in
    alleviating NA problem
                                                                                          4.6       Case Study
To further illustrate that our approach alleviates the                                    We present a test case in Figure 3. In this case,
NA problem, we perform data analysis based on                                             compared with our entity-pair-level candidate re-
the entity pair selection (denoted as Select) in Sec-                                     lation method, the candidate relations obtained by
tion 3.3. As shown in Table 4, compared with no                                           the document-level method contain some correct
selection, our method reduces the number of entity                                        answers, but the number of candidate relations is
pairs by about 95.72%, reduces the percentage of                                          too large, and the correlation between many candi-
NA data by about 45.32, and improves F1 by 31.94.                                         date relations and entity pairs is low. This case also
It indicates that our method effectively alleviates                                       illustrates the effectiveness of our entity-pair-level
NA problem in the LLM-based DocRE task.                                                   candidate relation method.
                                                                                   4029
4.7 Time Cost Analysis                                    Insufficient Training Data As shown in our
                                                          experiment and analysis in Appendix G, the
We perform an analysis of the time efficiency of our
                                                          long-tail problem in train set of dataset results in a
proposed model, EP-RSR, in terms of training and
                                                          skewed distribution of relation instances, leading
inference time, in comparison to the competitive
                                                          to a limited training effects for certain relations
baseline, AutoRE (Xue et al., 2024). Here, the
                                                          and subsequently deteriorating the prediction
reported training time corresponds to the duration
                                                          results for the corresponding relation triplets.
required for fine-tuning LLMs, while the inference
                                                          To effectively tackle this challenge, leveraging
time refers to the time taken for LLMs to perform
                                                          external knowledge sources or generating relevant
predictions.
                                                          data using LLMs could help mitigate the long-tail
   The results, presented in Table 6, show that our
                                                          issue.
approach reduces training time by 19,285 seconds
and inference time by 2,814 seconds. Combining
                                                          Acknowledgments. The authors sincerely thank
the results from Table 1 and Table 2, our model
                                                          the reviewers for their valuable comments, which
shows a significant improvement over AutoRE on
                                                          improved the paper. The work is supported by
the DocRE task. This further demonstrates that our
                                                          the National Natural Science Foundation of China
model not only achieves superior performance but
                                                          (62276057).
also maintains a relatively low time cost.

    Model          Training time     Inference time       References
    AutoRE             68672s            14877s           Tom Brown, Benjamin Mann, Nick Ryder, Melanie
    Our Method         49387s            12063s
                                                            Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
                                                            Neelakantan, Pranav Shyam, Girish Sastry, Amanda
      Table 6: Time cost on the DocRED dataset.             Askell, et al. 2020. Language models are few-shot
                                                            learners. Advances in neural information processing
                                                            systems, 33:1877–1901.
5    Conclusion                                           Fenia Christopoulou, Makoto Miwa, and Sophia Ana-
                                                            niadou. 2019. Connecting the dots: Document-level
In this paper, we introduce a novel LLM-based               neural relation extraction with edge-oriented graphs.
DocRE framework based on our proposed entity-               In Proceedings of the 2019 Conference on Empirical
pair-level candidate relations and a new LLM-               Methods in Natural Language Processing and the 9th
                                                            International Joint Conference on Natural Language
based DocRE paradigm EPRF. Our model achieves               Processing (EMNLP-IJCNLP), pages 4925–4936.
SOTA results in LLM-based methods on DocRED,
Re-DocRED and DWIE datasets, while also alle-             Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
viating the NA problem in DocRE. Although our                Kristina Toutanova. 2019. Bert: Pre-training of deep
                                                             bidirectional transformers for language understand-
model is still inferior to the latest traditional small      ing. In Proceedings of NAACL, pages 4171–4186.
models, our LLM-based results further narrows the
performance gap with the latest small models, mak-        Chufan Gao, Xulin Fan, Jimeng Sun, and Xuan Wang.
                                                            2023. Promptre: Weakly-supervised document-level
ing it a promising approach for future LLM-based            relation extraction via prompting-based data program-
DocRE.                                                      ming. arXiv preprint arXiv:2310.09265.

Limitations                                               Chufan Gao, Xuan Wang, and Jimeng Sun. 2024. TTM-
                                                            RE: Memory-augmented document-level relation ex-
                                                            traction. In Proceedings of the 62nd Annual Meet-
Error Propagation Issue The entire model is                 ing of the Association for Computational Linguistics
constrained by the initial entity pairs filtering step,     (Volume 1: Long Papers), pages 443–458, Bangkok,
which, while eliminating many irrelevant entity             Thailand. Association for Computational Linguistics.
pairs, may also discard some genuinely relation-
                                                          Ridong Han, Tao Peng, Chaohao Yang, Benyou Wang,
containing pairs. This can adversely affect the             Lu Liu, and Xiang Wan. 2023. Is information extrac-
model’s overall performance ceiling. To better ad-          tion solved by chatgpt? an analysis of performance,
dress this issue, it may be beneficial to consider          evaluation criteria, robustness and errors. arXiv
multiple entity pairs filtering sources or to employ        preprint arXiv:2305.14450.
alternative methods that alleviate the false positive     Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-
problem commonly associated with LLMs.                      Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
                                                      4030
  Chen. 2022. Lora: Low-rank adaptation of large             Faisal Azhar, et al. 2023. Llama: Open and effi-
  language models. In International Conference on            cient foundation language models. arXiv preprint
  Learning Representations.                                  arXiv:2302.13971.

Junpeng Li, Zixia Jia, and Zilong Zheng. 2023. Semi-      A Vaswani. 2017. Attention is all you need. Advances
  automatic data enhancement for document-level re-         in Neural Information Processing Systems.
  lation extraction with distant supervision from large
  language models. In Proceedings of the 2023 Con-        Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu,
  ference on Empirical Methods in Natural Language          Haiyue Song, Jiwei Li, and Sadao Kurohashi. 2023.
  Processing, pages 5495–5505.                              Gpt-re: In-context learning for relation extraction
                                                            using large language models. In Proceedings of the
Xingzuo Li, Kehai Chen, Yunfei Long, and Min                2023 Conference on Empirical Methods in Natural
  Zhang. 2024. Llm with relation classifier for             Language Processing, pages 3534–3547.
  document-level relation extraction. arXiv preprint
  arXiv:2408.13889.                                       Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang,
                                                            Fei Wu, Tianwei Zhang, Jiwei Li, and Guoyin Wang.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-         2023a. Gpt-ner: Named entity recognition via large
  dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,             language models. arXiv preprint arXiv:2304.10428.
  Luke Zettlemoyer, and Veselin Stoyanov. 2019.
  Roberta: A robustly optimized bert pretraining ap-      Xiao Wang, Wei Zhou, Can Zu, Han Xia, Tianze Chen,
  proach. arXiv preprint arXiv:1907.11692.                  Yuan Zhang, Rui Zheng, Junjie Ye, Qi Zhang, Tao
                                                            Gui, Jihua Kang, J. Yang, Siyuan Li, and Chun-
Youmi Ma, An Wang, and Naoaki Okazaki. 2023.                sai Du. 2023b. Instructuie: Multi-task instruction
  Dreeam: Guiding attention with evidence for improv-       tuning for unified information extraction. ArXiv,
  ing document-level relation extraction. In Proceed-       abs/2304.08085.
  ings of the 17th Conference of the European Chap-
  ter of the Association for Computational Linguistics,   Yiqing Xie, Jiaming Shen, Sha Li, Yuning Mao, and Ji-
  pages 1971–1983.                                          awei Han. 2022. Eider: Empowering document-level
                                                            relation extraction with efficient evidence extraction
Yilmazcan Ozyurt, Stefan Feuerriegel, and Ce Zhang.
                                                            and inference-stage fusion. In Findings of the As-
  2023. Document-level in-context few-shot relation
                                                            sociation for Computational Linguistics: ACL 2022,
  extraction via pre-trained language models. arXiv
                                                            pages 257–268.
  preprint arXiv:2310.11085.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:     Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong
  Sentence embeddings using siamese bert-networks.          Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, and
  In Proceedings of the 2019 Conference on Empirical        Enhong Chen. 2023. Large language models for
  Methods in Natural Language Processing. Associa-          generative information extraction: A survey. arXiv
  tion for Computational Linguistics.                       preprint arXiv:2312.17617.

Sunil Kumar Sahu, Fenia Christopoulou, Makoto Miwa,       Lilong Xue, Dan Zhang, Yuxiao Dong, and Jie Tang.
  and Sophia Ananiadou. 2019. Inter-sentence relation        2024. Autore: Document-level relation extraction
  extraction with document-level graph convolutional         with large language models. In Proceedings of the
  neural network. In Proceedings of the 57th Annual          62nd Annual Meeting of the Association for Compu-
  Meeting of the Association for Computational Lin-          tational Linguistics (Volume 3: System Demonstra-
  guistics, pages 4309–4316.                                 tions), pages 211–220.

Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and          Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin,
  Sharifah Mahani Aljunied. 2022. Revisiting docred-        Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou,
  addressing the false negative problem in relation ex-     and Maosong Sun. 2019. Docred: A large-scale
  traction. In Proceedings of the 2022 Conference on        document-level relation extraction dataset. In Pro-
  Empirical Methods in Natural Language Processing,         ceedings of the 57th Annual Meeting of the Associa-
  pages 8472–8487.                                          tion for Computational Linguistics, pages 764–777.

Hengzhu Tang, Yanan Cao, Zhenyu Zhang, Jiangxia           Klim Zaporojets, Johannes Deleu, Chris Develder, and
  Cao, Fang Fang, Shi Wang, and Pengfei Yin. 2020.          Thomas Demeester. 2021. Dwie: An entity-centric
  Hin: Hierarchical inference network for document-         dataset for multi-task document-level information
  level relation extraction. In Advances in Knowledge       extraction. Information Processing Management,
  Discovery and Data Mining: 24th Pacific-Asia Con-         58(4):102563.
  ference, PAKDD 2020, Singapore, May 11–14, 2020,
  Proceedings, Part I 24, pages 197–209. Springer.        Shuang Zeng, Yuting Wu, and Baobao Chang. 2021.
                                                            Sire: Separate intra-and inter-sentential reasoning for
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier       document-level relation extraction. In Findings of
  Martinet, Marie-Anne Lachaux, Timothée Lacroix,           the Association for Computational Linguistics: ACL-
  Baptiste Rozière, Naman Goyal, Eric Hambro,               IJCNLP 2021, pages 524–534.
                                                      4031
Shuang Zeng, Runxin Xu, Baobao Chang, and Lei Li.         from the German broadcaster Deutsche Welle be-
  2020. Double graph based reasoning for document-        tween 2002 and 2018. It is annotated at the doc-
  level relation extraction. In Proceedings of the 2020
                                                          ument level for NER, coreference resolution, RE,
  Conference on Empirical Methods in Natural Lan-
  guage Processing (EMNLP), pages 1630–1640.              and entity linking. It comprises 602 train docu-
                                                          ments, 98 dev documents, and 99 test documents,
Wei Zheng, Hongfei Lin, Zhiheng Li, Xiaoxia Liu,          encompassing 65 relation types.
 Zhengguang Li, Bo Xu, Yijia Zhang, Zhihao Yang,
 and Jian Wang. 2018. An effective neural model ex-          Dataset      Split    #Doc.   #Rel.   #Ent.    #Facts.
 tracting document level chemical-induced disease re-
 lations from biomedical literature. Journal of biomed-                    train   602             16,494   14,403
 ical informatics, 83:1–9.                                      DWIE        dev    98        65    2,785    2,624
                                                                            test    99             2,623    2,495
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan                          train   3,053           59,493   38,180
  Ye, and Zheyan Luo. 2024. Llamafactory: Unified             DocRED        dev     998      96    19,578   12,323
  efficient fine-tuning of 100+ language models. arXiv                      test   1,000           19,539      -
  preprint arXiv:2403.13372.                                               train   3,053           59,359   85,932
                                                             Re-DocRED      dev     500      96    9,684    17,284
Huixue Zhou, Mingchen Li, Yongkang Xiao, Han Yang,                          test    500            9,779    17,448
  and Rui Zhang. 2023. Llm instruction-example adap-
  tive prompting (leap) framework for clinical relation   Table 7: Statistics on datasets, where Doc. (resp. Rel or
  extraction. medRxiv, pages 2023–12.                     Ent) abbreviates documents (resp. relations or entities).

Wenxuan Zhou, Kevin Huang, Tengyu Ma, and Jing
 Huang. 2021. Document-level relation extraction
 with adaptive thresholding and localized context pool-
                                                          B     Parameter Settings
 ing. In Proceedings of the AAAI conference on artifi-    B.1    Fine-tuned Parameters
 cial intelligence, volume 35, pages 14612–14620.
                                                          For the entity pair selection, multiple-choice QA,
A    Details and Statistics of Datasets                   and triple fact judgment stages, the parameters of
                                                          LLM fine-tuning are shown in the Table 12.
We conducted our experiments on the DocRED
(Yao et al., 2019), Re-DocRED (Tan et al., 2022),         B.2    Inference Parameters
and DWIE (Zaporojets et al., 2021) datasets. Statis-      The parameter settings for LLMs inference at each
tics for three datasets are reported in Table 7.          stage are shown in the Table 8.
   DocRED is a dataset designed for the task of re-
lation extraction from multi-paragraph documents,             Stage                        temperature      top_p
comprising 132,375 entities, 1,829,756 entity pairs           Entity information               0.9           0.9
and 56,354 relation facts annotated across 5,053              Relation summarization           0.9           0.9
                                                              Entity pair selection            0.9           0.9
Wikipedia articles. It comprises 3,053 train doc-             Multiple-choice QA             0.0001          0.9
uments, 998 dev documents, and 1,000 test docu-               Triplet fact judgment            0.1           0.9
ments, encompassing 96 relation types. In contrast
to previous relation extraction tasks that primarily            Table 8: Inference parameters at each stage.
focused on single sentences or short text corpora,
DocRED incorporates more complex contextual
                                                          C     Prompt Templates
information and inter-paragraph relations, necessi-
tating models to possess enhanced understanding           We use different prompts for LLM in different ex-
and reasoning capabilities.                               perimental stages. For LLMs in the entity infor-
   Re-DocRED dataset serves as an improvement             mation and relation summarization stage, we adopt
over DocRED, analyzing the causes and impacts             the non-fine-tuned approach. The specific prompts
of the false negative issues present in the original      are shown in the Table 13. For LLMs in the entity
dataset, and re-annotating a total of 4,053 docu-         pair selection, multiple-choice QA, and triplet fact
ments from DocRED to address these concerns. It           judgment stages, we use fine-tuned. The specific in-
comprises 3,053 train documents, 500 dev docu-            struct tuning template is shown in the Table 14. In
ments, and 500 test documents.                            the multiple-choice QA stage, the relation template
   DWIE consists of 802 general news articles in          that converts candidate relations into natural lan-
English, randomly selected from a corpus collected        guage sentences is shown in Table 16. The relation
                                                      4032
description of the candidate relations involved in                            Model                          Dev        Test
the triplet fact judgment stage is shown in Table 17.                         AutoRE(Mistral-7B)*           53.01      51.91
                                                                              AutoRE(Llama3-8B)*            57.79      56.74
D    Experimental Analysis of Parameters k                                    Our Method(Mistral-7B)        54.90      53.41
                                                                              Our Method(Llama3-8B)         58.77      56.81
D.1 Impact of the parameter k on the top-k
    retrieved candidate relations                                         Table 10: Experimental F1 results on Re-DocRED for
                                                                          the DocRTE Task. Results marked with * are our repro-
As illustrated in the Table 9, an increase in the pa-                     ductions.
rameter k corresponds to a rise in the number of
true relation labels within the retrieved candidate
relations. However, this augmentation is accom-                           E Scalability Analysis of EP-RSR on
panied by a decline in the overall accuracy of the                          DocRTE
experimental results.
                                                                          To validate the performance of our model on the
     Parameter               P            R             F1                DocRTE, we make a modification to the structure
     k=80                   52.76        54.81         53.77              of our model EP-RSR:
     k=40                   53.04        54.57         53.79                 (1) The prompt in the entity pair selection phase
     k=20                   53.36        54.19         53.77              has been changed, and the input of the entity list
     k=10                   53.73        53.40         53.57
     k=5                    54.29        52.26         53.26              has been deleted. The specific prompt is shown
     k=1                    56.51        37.41         45.02              in Table 15. The subsequent selecting entity pairs
                                                                          that potentially contain relations part, which selects
Table 9: The impact of the parameter k on the top-k                       appropriate entities based on cosine similarity, has
retrieved candidate relations on the DocRE.
                                                                          been deleted.
                                                                             (2) Because our method uses prior knowledge
D.2 Analysis of the parameter k on                                        related to entities, the specific type of the entity is
    multi-sampling                                                        required. Therefore, we added an additional step of
The choice of parameter k on multi-sampling in                            predicting the entity type in the entire experiment
entity pair selection stage is crucial. Increased sam-                    to obtain the entity type of a given entity based
pling improves estimation precision by reducing                           on LLMs. This step only increases the additional
random errors. Larger k values provide better data                        reasoning time, and the training time remains un-
distribution insights and enhance model robustness                        changed. The specific additional reasoning time is
to outliers. However, higher sampling iterations                          467s. The reasoning time of the entire experimental
also increase computational costs. The results, as                        large model is 12530s, which is still lower than the
shown in Figure 4, indicate that increased sampling                       reasoning time of the comparison model AutoRE
improves recall but reduces accuracy, emphasizing                         of 14877s.
the need for an optimal choice of k.                                         We compare it with AutoRE, using the same
                                                                          evaluation metrics as those used in AutoRE for
                                                           F1 Score       DocRTE. The comparison results in Table 10 in-
                                                           Precision
60
                                                           Recall
                                                                          dicate that EP-RSR demonstrates strong competi-
                                                                          tiveness and generalization in DocRTE. Compared
                                                                          with AutoRE, our method shows an improvement
                                                                          of 1.89 in F1 score on the Re-DocRED dev set for
55
                                                                          the DocRTE task. Considering that our model is
                                                                          primarily designed for DocRE, the performance
                                                                          improvement in the DocRE task demonstrates that
                                                                          our method is clearly more advantageous in the
50
                                                                          DocRE task than in DocRTE.

     1      2    3      4        5   6     7       8   9        10
                                                                          F    Analysis of Multiple-choice QA and
                     Sampling iteration number k                               Prior Knowledge
Figure 4: Analysis of parameter k on multi-sampling.                      we conduct the ablation analysis on the multiple-
                                                                          choice QA and prior knowledge mentioned in Sec-
                                                                       4033
tion 3.4, as illustrated in Table 11. Our observations
include the following aspects:
Impact of multiple-choice QA Removing
multiple-choice QA from the fine filtering of can-
didate relations causes the F1 and Ign F1 scores to
drop by 13.07 and 11.71, which further illustrates
that after the initial coarse filtering of candidate
relations, further fine filtering is still required.
Impact of prior knowledge Removing prior
knowledge in mentioned in Section 3.4, F1 and
Ign F1 decreased 3.46 and 3.56. It indicates that
prior knowledge is helpful in removing more in-
significant candidate relations.

    Model                        F1        Ign F1
    Our Method                  53.77       51.25
    w/o Multiple-choice QA      40.70       39.54
    w/o Prior knowledge         50.31       47.69

Table 11: Analysis of Multiple-choice QA and Prior
Knowledge.


G Analysis of Long-tail Problem in Train
  Set
There exists a long-tail problem in DocRE dataset,
where many relations have a small number of as-
sociated labels. To further assess the impact of
relation labels number in the train set on the model,
we calculated the F1 score for each relation in the
DocRED dev set. The results are shown in Figure 5.
As the number of relation labels in the train set de-
creases, the overall F1 score for the relation also
exhibits a downward trend. There is a clear posi-
tive correlation between the frequency of relations
in the train set and their relation extraction perfor-
mance. For example, the high-frequency relation
P569 represents the “date of birth” relation with an
F1 score of 88.82, while the low-frequency relation
P39 represents the “position held” relation with an
F1 score of 0. It indicates that the presence of the
long-tail problem will degrade the triplet prediction
results for some relations. To better address this is-
sue, one might consider leveraging external knowl-
edge sources or LLMs to generate corresponding
data, thereby alleviating the long-tail problem.




                                                     4034
                                     Number of relation labels                                                              F1 of relations

                             8000                                                                                                             80



                                                                                                                                              60




 Number of relation labels
                             6000



                             4000
                                                                                                                                              40  F1 of relations



                                                                                                                                              20
                             2000


                                                                                                                                              0
                               0
                                    PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
                                    112151555132415415433564211621511161212171111216514811373115293151437321281521123475861151317181
                                    7375776726669976040030906400257773 2000515162797844047507638431183054607767773756848470353933901
                                     1 0759071145 132 7 0 407 9 804 7 86605863644 19009 6 26017 257359 562 26971 4802065613 67079
                                                        3    11      4               2                           6        66 6 5 8
                                                       Figure 5: Analysis of long-tail problem in train set.




Stage                                                            Parameter
Entity pair selection                                            lora_rank: 64, lora_alpha: 128, lora_dropout: 0.05, cutoff_len: 2048, learn-
                                                                 ing_rate: 1.0e-4, num_train_epochs: 5.0, warmup_ratio: 0.1, bf16: true
Multiple-choice QA                                               lora_rank: 64, lora_alpha: 128, lora_dropout: 0.05, cutoff_len: 2048, learn-
                                                                 ing_rate: 1.0e-4, num_train_epochs: 1.0, warmup_ratio: 0.1, bf16: true
Triplet fact judgment                                            lora_rank: 64, lora_alpha: 128, lora_dropout: 0.05, cutoff_len: 2048, learn-
                                                                 ing_rate: 1.0e-4, num_train_epochs: 1.0, warmup_ratio: 0.1, bf16: true

                                                                     Table 12: Fine-tuned parameters.




Stage                                                            Prompt
Entity information                                               The text is as follows:
                                                                 {title}
                                                                 {doc}
                                                                 What is "{entity}"? (For example, US is a country and 12 is a number) Answer
                                                                 in one sentence. Only output answers without outputting anything else.
                                                                 The answer is:
Relation summarization                                           The text is as follows:
                                                                 {title}
                                                                 {doc}
                                                                 What is the relationship between {entity_h} and {entity_t}? Answer in one
                                                                 sentence. Only output answers without outputting anything else.

                                      Table 13: Entity information prompt and Relation summarization prompt.



                                                                                    4035
Stage                     Instruct Tuning Template
Entity pair selection     Given a text and an entity list as input, list the entity pairs that can be identified
                          as possibly containing a relation.
                          ## Text:
                          {doc_text}

                          ## Entity list:
                          {entity_list}
Multiple-choice QA        Determine which option can be inferred from the given text.
                          ## Text:
                          {doc_text}

                          ## Options:
                          {options}
Triplet fact judgment     Based on the text and the description of the relation "{rel}", give an answer
                          about whether the head and tail entity pairs (head entity and tail entity) satisfy
                          the "{rel}" relation.
                          ## Relation description:
                          {rel_description}

                          ## The text to be extracted:
                          {doc_text}

                          ## Entity pair to be extracted:
                          {extract_entity_pair}

                        Table 14: Instruct Tuning Template for EPRF.



Stage                     Pormpt
Entity pair selection     Given a text as input, list the entity pairs that can be identified as possibly
                          containing a relation.
                          ## Text:
                          {doc_text}

                                Table 15: Pormpt for DocRTE.



Relation                   Relation Template
P6                        <tail> is the head of government of <head>
P17                       <head> is located in country <tail>
P19                       <tail> is the place of birth of <head>
P20                       <tail> is the place of death of <head>
P22                       <tail> is the father of <head>
P25                       <tail> is the mother of <head>
P26                       <tail> is the spouse of <head>
P27                       <head> is a citizen of <tail>
P30                       <head> is on the continent of <tail>
P31                       <head> is the instance of <tail>
P35                       <tail> is the head of state <head>
P36                       <tail> is the capital of <head>
P37                       <tail> is the official language of <head>
...                       ...

                                 Table 16: Relation Template.


                                               4036
Relation                      Relation Description
country                      For the ’country’ relation, the subject pertains to a non-human
                             entity, such as an organization, place, or event. The object signifies
                             the sovereign state where the subject is based or occurs. Example:
                             (Amazon Inc, country, United States).
country of citizenship       The ’country of citizenship’ relation denotes that the subject, an
                             individual, is recognized as a citizen by the object, a country.
                             Example: (Elon Musk, country of citizenship, United States).
contains administrative ter- The relation ’contains administrative territorial entity’ involves
ritorial entity              a subject, an administrative territory, encompassing the object,
                             a subdivision or part of this administrative territory. Example:
                             (California, contains administrative territorial entity, Los Angeles).
has part                     The ’has part’ relation reflects that the subject, an entity or whole,
                             comprises the object, a part or component of the subject. Example:
                             (A car, has part, engine).
date of birth                In the ’date of birth’ relation, the subject, a person, was born on
                             the object, the specified date. Example: (John Doe, date of birth,
                             January 1, 1990).
part of                      In the ’part of’ relation, the subject, a component or section, be-
                             longs to the object, a larger whole or aggregate. Example: (Engine,
                             part of, a car).
...                          ...

                                  Table 17: Relation Description.




                                               4037
