Automated Knowledge Graph Construction using Large Language Models
                and Sentence Complexity Modelling
       Sydney Anuyah1 Mehedi Mahmud Kaushik1 Krishna Dwarampudi2
           Rakesh Shiradkar3 Arjan Durresi1 Sunandan Chakraborty1
            Luddy School of Informatics, Indiana University, Indianapolis, IN1
                 School of Science Purdue University, Indianapolis, IN2
Department of Biomedical Engineering and Informatics, Indiana University, Indianapolis, IN3
     {sanuyah, mekaush, rshirad, adurresi, sunchak}@iu.edu; sdwaramp@purdue.edu


                     Abstract                              large models (Pan et al., 2024). One of such ad-
                                                           vantages is in the creation of domain-specific on-
    We introduce CoDe-KG, an open-source, end-             tologies (Karim et al., 2023; Chandak et al., 2023)
    to-end pipeline for extracting sentence-level          closely associated with creating new reasoning and
    knowledge graphs by combining robust coref-
    erence resolution with syntactic sentence de-
                                                           inference methods (Kau et al., 2024; Zhang et al.,
    composition. Using our model, we contribute            2024).
    a dataset of over 150 000 knowledge triples,               Previous research has built the foundational con-
    which is open source. We also contribute a             cepts of KGs, which include the models used
    training corpus of 7248 rows for sentence com-         in creating these graphs and their representation
    plexity, 190 rows of gold human annotations            (Hogan et al., 2021). Automated KG construc-
    for co-reference resolution using open source          tions (Zhong et al., 2023) and representation learn-
    lung-cancer abstracts from PubMed, 900 rows
                                                           ing (Ji et al., 2021) have defined major stages in
    of gold human annotations for sentence con-
    version policies, and 398 triples of gold hu-          building a KG: from knowledge acquisition and se-
    man annotations. We systematically select              mantic table interpretation (Liu et al., 2023) to en-
    optimal prompt-model pairs across five com-            tity extraction–covering Named Entity Recognition
    plexity categories, showing that hybrid chain-         (NER), Named Entity Disambiguation (NED), and
    of-thought and few-shot prompting yields up            Named Entity Linking (NEL) (Al-Moslmi et al.,
    to 99.8% exact-match accuracy on sentence              2020). These studies and many more provide a
    simplification. On relation extraction (RE),           system in which unstructured text can be trans-
    our pipeline achieves 65.8% macro-F1 on
    REBEL, an 8-point gain over the prior state
                                                           formed into an organized corpus of interlinked en-
    of the art, and 75.7% micro-F1 on WebNLG2,             tities. Secondly, domain techniques such as graph
    while matching or exceeding performance on             knowledge distillation (Tian et al., 2023) and em-
    Wiki-NRE and CaRB. Ablation studies demon-             bedding schemes (Cao et al., 2024) have helped
    strate that integrating coreference and decom-         reinforce the ability to compress, optimize, and rep-
    position increases recall on rare relations by         resent KGs which are then utilized in downstream
    over 20%. Code and dataset are available               applications. The goal of event KGs (Guan et al.,
    at https://github.com/KaushikMahmud/CoDe-
                                                           2022) and explainable artificial intelligence (AI)
    KG_EMNLP_2025.
                                                           on KGs (Schramm et al., 2023) is to empathically
1   Introduction and Background                            ensure that models not only have to be efficient but
                                                           also interpretable (Kaur et al., 2022).
One way to represent data is through knowledge                 The challenges we are tackling is two fold: (1)
graphs (KGs) (Hogan et al., 2021). KGs have trans-         We have a large volume of unstructured text data,
formed the way data is organized and by leverag-           and because it varies in structure, writing style, and
ing complex network chains, we have been able to           vocabulary across different domains, it has become
explore complex fields like causality in different         harder to parse, and (2) Many automated pipelines
domains (Friedman et al., 2022; MacLean, 2021;             for KG creation are not really automated, as some
Naser, 2022).                                              are heavily prompt reliant on the end user (Buehler,
  With the recent advancements in Natural Lan-             2024), others have issues of handling noisy datasets
guage Processing (NLP) using Large Language                (Zhang et al., 2023). This is why we are researching
Models (LLMs), KGs have become instrumental,               a one-size-fits-all open-source model framework
both as knowledge bases and in finetuning these            that could help in knowledge extraction irrespec-
                                                      15515
     Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 15515–15539
                          November 4-9, 2025 ©2025 Association for Computational Linguistics
tive of our data. Through the typical structure of               3. A 900-sample sentence transformation
the English Language, it is possible to extract re-                 dataset, consisting of 300 annotated ex-
lationships through verb usage and clauses. This                    amples each for converting complex,
leads us to the first research question. RQ1: Can                   compound, and compound-complex sen-
sentence modelling be used to effectively create                    tences into simple, extractable forms.
KGs that rival other methods? We also compare                    4. A machine-generated KG corpus of over
our method to the popular closed-source AI model:                   150,000 structured triples, created using
GPT 4 series which is renowned for parsing aca-                     our full end-to-end pipeline.
demic literature, and we design evaluation prompts
to benchmark their performance against ours. This        2     Background
can be summarized as RQ2: can an open-source,
LLM model using the sentence semantics approach          2.1    Sentence Semantic Modelling for
reliably construct KGs from raw texts? Our contri-              Knowledge Extraction
butions are as follows:                                  Sentence semantic modelling involves organizing
                                                         sentences into various types, which structure how
   • We introduce a novel sentence-semantic              ideas can relate to one another. Let us define a
     framework for relation extraction (RE) and          grammar structure as G = (N, Σ, P, S) where N
     KG construction, borrowing from linguistic          is a finite set of non-terminal symbols, Σ is a fi-
     theory and semantic parsing. This idea though       nite set of terminal symbols (the actual words or
     common, to the best of our knowledge has            tokens in the language), P is a finite set of pro-
     been under-explored in mainstream NLP in-           duction rules that describe how non-terminals can
     formation extraction pipelines. The novelty         be expanded into sequences of non-terminals and
     of our work lies in the integration of multi-       terminals and S ∈ N is the start symbol, which
     ple frameworks rather than just one task. Our       we conventionally call Sentence. Appendix A
     method explicitly models semantic sentence          discusses the interplay of sentence and clauses in
     types (e.g., complex (CX), compound (CD),           more detail.
     and compound-complex (CC) forms) as the                To understand the interplay of clauses and how
     foundation for extracting knowledge triples.        they make up a sentence, we need to consider the
     Each triples is a simple, three-part structure      types of sentences in English Language (Das et al.,
     (entity1 , relationship, entity2 ) used to repre-   2018), which are:
     sent a single fact in a KG.
                                                              • Simple Sentences: Having only one indepen-
   • We also explore diverse prompting strate-                  dent clause and no dependent clause
     gies across our pipeline, including Chain-              Ssimple = { (N P, V P ) | N P ∈ N , V P ∈ V}
     of-Thought (CoT) reasoning, Few-Shot In-
     Context Learning (FICL), and Zero-Shot Gen-              • Complex Sentences: Having one independent
     eral Instruction prompting (GIP), and empiri-              clause and at least one dependent clause
     cally demonstrate their varying contributions
     to structural decomposition. To support this                         Scomplex = Smain ∪ DC
     architecture, we release a suite of open-source
     resources:                                               • Compound Sentences: Having two or more
                                                                independent clauses joined by a conjunction
       1. A 7248-row dataset that categorizes and               and no dependent clause
          maps diverse sentence semantics aligned
          to our model’s decomposition strat-                   Scompound = S1 ⊕S2 where ⊕ is a conjunction
          egy (complex, compound, compound-
          complex, simple and incomplete sen-                 • Compound-Complex Sentences: Having two
          tence).                                               or more independent clauses joined by a con-
       2. A gold-standard co-reference resolution               junction and have at least one dependent
          corpus comprising 190 PubMed lung-                    clause
          cancer abstracts annotated by four do-
          main experts.                                               Scomp-comp = (S1 ⊕ S2 ) ∪ DC
                                                    15516
 Method                               Sentence Decomp.   Coref Res.      Open-Source   Domain-Agnostic   Eval Scripts
 GraphRAG (Han et al., 2024)                 ×                ×              ×               ✓               ×
 EDC (Zhang and Soh, 2024)                   ×                ×              ✓               ✓               ✓
 GKG-LLM (Zhang et al., 2025)                ×                ✓              ✓               ✓               ✓
 Neo4j LLM-KG (Bharti et al., 2024)          ×                ×              ×               ✓               ×
 KGGen (2025) (Mo et al., 2025)              ×                ×              ✓               ✓               ✓
 Our Pipeline                                ✓                ✓              ✓               ✓               ✓

                            Table 1: Current LLM-induced KG Methods Comparison


The core motivation behind this work stems from           and broad generalizability (Ouyang et al., 2022; Ko-
the assumption that LLMs emulate human reason-            jima et al., 2022). FICL incorporates a small set of
ing (Wu et al., 2024). Additionally, as shown by          in-context examples to guide the model, improving
Nurmalan (Hendrawati, 2018), human comprehen-             structure-sensitive tasks like sentence decomposi-
sion of sentence structure is far from uniform. Un-       tion (Brown et al., 2020; Li et al., 2023). COT
dergraduate students fail to accurately interpret CC      prompting, which encourages step-by-step reason-
sentences in 44.54% of cases, followed by CD              ing, has proven especially effective in multi-step
(23.2%) and CX (22.13%) sentences. In contrast,           reasoning and relation-rich generation (Wei et al.,
error rates drop significantly to 10.13% for simple       2022; Li et al., 2025). Our hybrid CoT + FICL
sentences. Notably, academic and scientific writing       strategy combines the benefits of example-guided
rarely employs simple sentences, favouring more           prompting with intermediate reasoning steps, sig-
elaborate constructions aligned with formal and           nificantly improving accuracy in sentence decom-
jargon-heavy discourse. We posit that modelling           position and RE. We benchmark each strategy
and converting these complex sentence types into          across multiple subtasks in our pipeline and find
simpler forms enables more effective interpretation       that hybrid prompting consistently yields the most
by LLMs, particularly for structured information          precise and coherent results, particularly in com-
extraction.                                               plex biomedical sentences, which is in tune with
   A common misconception is that a simple sen-           recent advances in prompt engineering that empha-
tence means a simplified or short sentence. How-          size structure-aware and compositional prompting
ever, as Phil (Atteberry, 2016) illustrates, even syn-    for complex NLP tasks (Kojima et al., 2022).
tactically rich sentences, such as “Being an English
teacher with a penchant for syntactical complex-
ity, I love simple sentences upon getting up and          3       Data
before going to bed”—qualify as simple if they
contain only one independent clause. Despite struc-
tural simplicity, such sentences may encode mul-          3.1      PubMed Lung Cancer Abstracts
tiple relationships, contradicting the assumption
that simple sentences yield only one extractable          PubMed is an open biomedical literature reposi-
relation. Importantly, a simple sentence can fea-         tory. From PubMed, we randomly parsed 7,500
ture compound subjects (“John and Mary run. . . ”),       abstracts related to the lung cancer keyword, pub-
compound predicates (“runs and jumps. . . ”), or          lished between 2020 and 2025, to create our pri-
compound objects (“an apple and a banana. . . ”).         mary evaluation corpus. This dataset supports our
Our framework explicitly models these variations,         co-reference resolution, sentence decomposition,
ensuring RE remains robust across all syntactic           and triple extraction tasks. The inclusion principle
permutations of the simple sentence form.                 was any abstract that mentioned lung cancer, was
                                                          open source and free to use, and we did not particu-
                                                          larly exclude any research apart from those that fell
2.2 Prompting Strategies
                                                          outside the random sample. The data was sampled
We explore four prompting strategies within our           on March 18, 2025. The motivation behind using
pipeline to evaluate their effectiveness in sentence      the lung cancer abstract was linked to a 20-year
restructuring and RE: GIP, FICL, CoT, and Hybrid          international study in Radiology (Henschke et al.,
CoT + FICL. GIP relies on general instructions            2023). We believed that the dataset is well-versed
without examples, offering baseline performance           and well-researched.
                                                     15517
                                                                                         Sentence Classification




                       Coreference Resolution
                                                                                                        Model Selection and



                                                                                                                                                                                                Relationship Extraction
                                                                     Extract One                                                                                    Converting Sentences



                                                                 Sentence per abstract                 Sentence Classification                                           to Simple




      P 1, P 2,   Pi                            M 1, M 2,   Mk                            P 1, P 2,   Pi                    M 1, M 2,   Mk     P 1, P 2,   Pi     M 1, M 2,                Mk




Figure 1: Overview of CoDe-KG, the automated KG creation pipeline. First, the input set of abstracts is given
to the Coreference Resolution stage. In this phase, a team of annotators , a collection of prompt strategies ,
and models are jointly applied to produce the coreference-resolved abstract set , which is given as input in
the Sentence Classification stage. With the help of verifiers , prompting strategies and models , a list of
correctly classified sentences with labels is generated in this stage. Then, in the Converting Sentences to
Simple stage, S̃comx, comp, comx_comp , prompt strategies , and models are given as input and converted into
simple sentences S̃simp . In Relationship Extraction stage, S̃simp , Sinit and best model–prompt pair (P ∗ , M ∗ )
from previous stage are given as input and relationships (entity1 , relationship , entity2 ) are extracted for
constructing KG.

3.2 REBEL (Cabot and Navigli, 2021)                                                                                              4      Methodology
We adopt the same 1,000-sample subset used in the                                                                                In this research, we propose an automated KG cre-
EDC model (Zhang and Soh, 2024) for evaluation,                                                                                  ation pipeline, CoDe-KG, for creating a KG from
originally drawn from the REBEL test partition of                                                                                abstracts. Our approach, as shown in Figure 1,
105,516 entries, also published in EMNLP.                                                                                        consists of four key stages: doing coreference reso-
                                                                                                                                 lution, sentence classification, sentence conversion,
3.3 WebNLG+2020 (Ferreira et al., 2020)                                                                                          and RE. In this section, we give a detailed overview
                                                                                                                                 of our pipeline implementation.
WebNLG+2020 (v3.0) is a semantic parsing bench-
mark containing text-triple pairs. We use its full
                                                                                                                                 4.1     Problem Setup
test split of 1,165 samples covering 159 unique
relation types.                                                                                                                  Let the set of input abstracts be denoted by

3.4 Wiki-NRE (Distiawan et al., 2019)                                                                                                              A = { a1 , a2 , . . . , an },                                          (1)
Wiki-NRE is a distant supervision dataset for RE.                                                                                   And let the set of valid relation triples extracted
We also used the same sample of 1,000 pairs used                                                                                 from A be denoted by
in the EDC model (Zhang and Soh, 2024). The
dataset contains 29,619 entries encompassing 45
                                                                                                                                            
distinct relation types .                                                                                                                R = (e1 , r, e2 ) | e1 , e2 ∈ E,
                                                                                                                                                                              r ∈ R,                                      (2)
3.5 CaRB (Bhardwaj et al., 2019)
                                                                                                                                                                              (e1 , r, e2 ) is valid .
The CaRB dataset is a benchmark for Open In-
formation Extraction (OpenIE), created by re-                                                                                      Where E is the set of all unique entities appear-
annotating the original OIE2016 dataset with im-                                                                                 ing in those triples and R is the relation vocabulary.
proved human judgments. The exact number of the                                                                                  Let the resulting KG be denoted by
final dataset is not known; what was reported in
the paper was the devset from Amazon Mechanical                                                                                                                 G = (E, R),                                               (3)
Turk, which was 1,282 sentences. However, on
the GitHub page, we found 50 unique sentences                                                                                      Our goal is to construct G so that it faithfully
spanning through 172 lines.                                                                                                      represents all extracted factual relations across A.
                                                                                                       15518
4.2 Coreference Resolution                               4.3    Sentence Classification
In our proposed pipeline, the coreference-               The first step of the sentence classification stage
resolution stage (as shown in Appendix: Algo-            (as shown in Appendix: Algorithm 2) processes
rithm 1) proceeds by creating a gold-standard            the resolved abstracts Â by sampling per category,
through expert annotation, and then selecting the        extracting one representative sentence from each
optimal prompt-model combination for creating            sampled abstract, and keeping only those sentences
coreference-resolved abstracts. First, we draw a         on which two expert verifiers agree. For the five
random subset of size s:                                 complexity categories
          A′ = UniformSample(A, s).                      C = {simp, comx, comp, comx_comp, incomp},

For four expert annotators–two with biological ex-       we draw a random subset
pertise and two with linguistic expertise–working              Ac = UniformSample(Â, pc ),             c ∈ C,
in pairs to resolve coreference on each a ∈ A′ ,
                                                         so that |Ac | = pc . From each a ∈ Ac we then
producing
                                                         extract exactly one representative sentence:
                                                                (
        Rj (a) = fann (hj , a),    j = 1, 2.                      fcreate (annotator, a), c∈{simp,incomp},
                                                           s=
Here, fann apply annotator hj ’s annotation proce-                fchoose (a),            otherwise.
dure to abstract a, yielding the resolution Rj (a).      We aggregate all candidates into
  We then define the gold set of abstracts A, which                               [
can be unanimously annotated:                                              Sall =     Sc .
                                                                                       c∈C
    G = a ∈ A′ Rj (a) = Rk (a) ∀ j, k .                  Next, two expert verifiers v1 , v2 independently re-
And extract the corresponding annotated gold stan-       view every s ∈ Sall , and we keep only those
dard                                                     sentence-category pairs on which they agree:
                   
             G′ = Rj (a) a ∈ G ,                                      (                                     )
where any Rj (a) may be used since all agree.                                   c ∈ C, s ∈ Sc ,
                                                               Ŝ =    (s, c)                                   .
  Next, we exhaustively evaluate each prompt-                                   fver (v1 , s) = fver (v2 , s)
model pair (P, M ) ∈ P × M by generating pre-
                                                            The output Ŝ is thus a high-agreement, category-
dicted annotations on the gold inputs
                                                         labeled sentence set, ready to serve as the input for
                                      
          R̂P,M = fprompt P, M, G ,                      Step 2.
                                                            The second step of the sentence classification
and computing a score (e.g., F1 ) against the gold-      stage (as shown in Appendix: Algorithm 3) takes
standard pairs:                                          the verified sentence-category set Ŝ along with the
                                     
           SP,M = score R̂P,M , G′ .                     set of prompting strategies P, and models M as
                                                         input to produce a fully labeled sentence corpus S̃.
We then select the best pair by                          First, we formed the training dataset
      (P ∗ , M ∗ ) = argmaxP ∈P, M ∈M SP,M .                          D = {(si , yi ) | (si , yi ) ∈ Ŝ}.
Finally, this optimal configuration is applied to the    For each m ∈ M, train on Dtrain and we computed
full collection:                                         its validation score
                                                                     scorem = Evaluate(m, Dval ).
             Â = fprompt P ∗ , M ∗ , A ,
                                                         Then we selected the best model
yielding fully resolved co-reference annotations Â.
                                                                       m∗ = argmaxm∈M scorem .
   This design ensures that (i) human expertise de-      Finally, we applied m∗ to every sentence in the
fines a robust gold standard through unanimous           fully coreference-resolved abstract set Â:
agreement, (ii) prompt-model selection is system-              
                                                          S̃ = (s, ℓs ) | s ∈ Sentences(Â), ℓs = m∗ (s) .
atic and exhaustive, and (iii) large-scale annota-
tion inherits the reliability established in the gold-   The resulting S̃ is the complete collection of
standard phase.                                          sentence-label pairs for downstream tasks.
                                                    15519
4.4 Converting Sentences to Simple                         assemble the extracted triples into our knowledge
The approach of this stage (as shown in Appendix:          graph. Let
Algorithm 4) consists of prompt-model selection                      [                             [
                                                           E =             {e1 , e2 }, R =              {r},
for each category, and large-scale sentence simpli-                (e1 ,r,e2 )∈R                 (e1 ,r,e2 )∈R
fication using the selected configurations.
   For each category c, we hold out the set Sc of          and define the graph as
complex (comx), compound (comp), and complex-
                                                                                   G = (E, R).
compound (comx_comp) sentences, and exhaus-
tively evaluate every prompt-model combination Each triple (e1 , r, e2 ) ∈ R becomes a directed,
(P, M ) ∈ P × M. We compute a performance      labeled edge from node e1 to node e2 . This knowl-
score via                                      edge graph G now encodes all valid factual rela-
                                               tions extracted across the corpus and can be used
scoreP,M (c) = EvaluatePromptModel(P, M, Sc ). for downstream querying and inference.

  and choose                                               5     Experiments

     (Pc∗ , Mc∗ ) = argmax(P,M ) scoreP,M (c).             5.1    Experiment 1: Results of Co-reference
                                                                  Resolution
  With (Pc∗ , Mc∗ ) fixed for each category, we pro-       Biomedical text is harder to understand and there-
cess every sentence s ∈ Sc by invoking:                    fore, RE is perceived to be harder (Johnson and
                                                          Bernstam, 2023). Therefore, we construct our
ŝ = fprompt Pc∗ , Mc∗ , s ,    Ssimp ← Ssimp ∪{ŝ}.       benchmark dataset of 190 coreference abstracts
                                                           in biomedical literature on Lung Cancer to eval-
Once all categories are processed, Ssimp consti-
                                                           uate the performance of LLM. Therefore, we
tutes the collection of simplified sentences from
                                                           crafted the SOTA prompts discussed in chap-
complex, compound, and complex-compound sen-
                                                           ter 2.2. The prompts we finally used were the
tences.
                                                           COT+FICL prompt after experimenting on the dif-
4.5 Relationship Extraction                                ferent prompts, which are in Appendix E.1. We
                                                           then randomly sampled 190 abstracts from the full
In this stage, we implement RE (as shown in Ap-            set of 7,500. Each abstract was independently anno-
pendix: Algorithm 5) through sentence consoli-             tated by two domain experts and two language ex-
dation and triple generation. First, we form the           perts. After the initial pass, the experts exchanged
working sentence set by combining:                         annotations and discussed any discrepancies. Full
                                                           annotation information available in Appendix E.2.
       Sinit = { s | (s, ℓ) ∈ S̃, ℓ = simp},
                                                              We evaluated several LLMs on our benchmark.
          S = Ssimp ∪ Sinit .                              Results were poor for most models. Predic-
                                                           tions were scored using MUC, B3 , CEAF4 , and
   Here, Ssimp is the set of all sentences produced
                                                           the CoNLL F1 aggregate (Pradhan et al., 2012).
by the simplification stage, while Sinit contains
                                                           Deepseek-distill-Qwen-7B, Qwen-Chat-7B, and
those initially classified as simple. Next, for each
                                                           Qwen-7B scored 0% F1 . Deepseek-7B, Deepseek-
s ∈ S, we extract a candidate relation triple via the
                                                           6.7B, and Deepseek-Prover-7B scored below 2%
frel (s) function, where the best prompt and model
                                                           F1 . Deepseek-distill-Llama-8B scored below 10%
combination (P ∗ , M ∗ ) from the previous stage
                                                           F1 in all categories. Table 2 lists F1 scores for
was used. Here,
                                                           the models doing co-reference resolution and was
         (e1 , r, e2 ) = frel (P ∗ , M ∗ , s).             benchmarked with ChatGPT o4-mini-high and
                                                           ChatGPT-4.5 responses as a baseline. ChatGPT o4-
We collect only non-empty outputs:                         mini-high did the best overall with an F1 of approx-
                                                           imately 63% using the FICL prompt. We bolded
  R ← R ∪ {(e1 , r, e2 )}      if (e1 , r, e2 ) ̸= ∅.      the best open-source model, which was comparable
                                                           to the closed source models for this task.
Upon completion,             R holds all valid                The evaluation of the co-reference was done us-
(entity1 , relation, entity2 ) triples. Finally, we        ing a cosine similarity score of 0.9, because we
                                                        15520
            Model                         Prompt        MUC (%)      B3 (%)      CEAF (%)         CoNLL (%)
            Mixtral-8x7B-Instruct-v0.1    FICL              32.42      70.61            70.61             57.88
            Llama-3.1-8B-Instruct         FICL              27.16      69.57            69.57             55.43
            Llama-3.2-3B-Instruct         COT_FICL          16.98       70.7             70.7             52.79
            Llama-3.3-70B-Instruct        FICL              31.25      70.94            70.94             57.71
            Mistral-7B-Instruct-v0.3      COT_FICL          18.58      70.74            70.74             53.35

                                   Table 2: Comparison of F1 Scores by Models.


noticed that 99% of all values that were marked             Table 4: Model performance on the conversion of Com-
at 0.9 correctly but not exact match were actually          pound to Simple Sentences
correct, just differing in preposition. For instance,
                                                             Model                          Macro Avg.   Exact-Match   RMSE
the gold standard says "a house", and the model
                                                             DeepSeek-LLM-67B               15.56%         14.67%      1.5891
says "house". At 0.8% the values were not signifi-           DeepSeek-LLM-7B                55.83%         50.00%      1.1506
                                                             DeepSeek-R1-Distill-Llama-8B   68.04%         65.00%      1.0571
cant to be considered, therefore, we stuck to using          DeepSeek-Prover-V1.5-7B        81.38%         76.33%      0.8591
a cosine similarity score of 0.9 for the co-reference        Llama-3.3-70B
                                                             Llama-3-8B
                                                                                            95.30%
                                                                                            99.78%
                                                                                                           81.00%
                                                                                                           98.00%
                                                                                                                       0.3213
                                                                                                                       0.1078
evaluation.                                                  Mistral-7B-Instruct-v0.3       96.64%         90.33%      0.2356
                                                             Mixtral-8x7B-Instruct-v0.1     96.14%         91.67%      0.2323
                                                             Qwen-7B                        56.17%         54.00%      1.2339
5.2 Experiment 2: Syntactic Sentence                         Qwen-7B-Chat                   1.33%           1.33%      1.7193
                                                             GPT 4 o                        91.68%         77.67%      0.4204
    Classification                                           GPT 4 o-3                      87.69%         68.67%      0.5924

We created a dataset for classification, and all de-
tails are given in Appendix E.3. We fine-tuned six
transformer-based models and two smaller LLMs               Table 5: Model performance on the conversion of Com-
on the training set and evaluated them on the test set.     plex to Simple Sentences
Table 3 reports test accuracy and macro-averaged
F1 for each model. We evaluated the entire test set          Model                          Macro Avg.   Exact-Match   RMSE

on a GPT-4o model.                                           DeepSeek-LLM-67B
                                                             DeepSeek-LLM-7B
                                                                                            26.23%
                                                                                            63.30%
                                                                                                           22.00%
                                                                                                           62.33%
                                                                                                                       1.7562
                                                                                                                       1.1866
                                                             DeepSeek-R1-Distill-Llama-8B   43.78%         33.00%      1.5540
Table 3: Sentence-type classification results (train set:    DeepSeek-Prover-V1.5-7B        91.33%         65.67%      0.8841
                                                             Llama-3.3-70B                  99.59%         98.67%      0.1364
2,00- sentences test set: 5,269 sentences)                   Llama-3-8B                     97.17%         92.67%      0.2866
                                                             Mistral-7B-Instruct-v0.3       93.73%         81.00%      0.3114
                                                             Mixtral-8x7B-Instruct-v0.1     98.48%         94.67%      0.1533
 Model                             Accuracy      F1macro     Qwen 7B                        64.61%         64.00%      1.1987
                                                             Qwen-7B-Chat                   9.24%           7.67%      1.8699
 BERT                              87.25%    86.14%          GPT 4 o
                                                             GPT 4 o-3
                                                                                            96.72%
                                                                                            99.05%
                                                                                                           91.33%
                                                                                                           97.00%
                                                                                                                       0.3050
                                                                                                                       0.1714
 BERT-Large                        87.68%    86.69%
 BioBERT                           87.19%    86.21%
 BioBERT-Large                     85.97%    85.16%
 ClinicalBERT                      86.92%    85.87%
                                                      COT, and COT+FICL prompts (see Appendix F.4).
 RoBERTa                           87.13%    85.83%
                                                      From the 65,175 sentences extracted from 7,500
 Gemma3 1-B                         9.24%    4.15 %
                                                      abstracts, our classifier labelled 42,282 as complex,
 LLama3.2 1-B                      17.71 %   0.27 %
                                                      4,942 as compound, 2,366 as compound-complex,
 GPT 4-0                           80.30 %   76.14%
                                                      13,465 as simple, and 2,120 as incomplete. We
 Random Guessing Lowest            8.83 %     3.24%
                                                      then randomly sampled 300 sentences each from
 Random Guessing Highest           32.61 %   9.84 %
                                                      the complex, compound, and compound-complex
                                                      classes and translated them into simple sentences.
                                                      These sample sizes at 300 achieve 95% confidence
5.3 Experiment 3: Evaluating the Prompting            for their respective populations with margins of
      Strategies and Semantic Conversion              error of ±5.62%, ±5.57%, and ±5.26%, respec-
We created a systematic structure (see Appendices     tively. We tested on the top performing models
F.1, F.2, and F.3 for the systemic conversion pro- and evaluated their performance on the different
cess) of evaluating how a model would convert a       prompting strategies. The results are shown in Ta-
cx, cd or cc sentence to a simple one, thereby, eval- ble 10, with the hybrid prompt performing the best
uating the four prompting strategies–GIP, FICL, in all cases.
                                                       15521
Table 6: Model performance on the conversion of                             igli, 2021), which tells us that our pipeline can
Compound-Complex to Simple Sentences                                        handle diverse, low-frequency relations reasonably
                                                                            well (macro), yet would not perform the best on the
    Model                           Macro Avg.   Exact-Match    RMSE
                                                                            most common triplets that dominate micro averag-
    DeepSeek-LLM-67B                28.42%         11.00%       1.6178
    DeepSeek-LLM-7B                 49.45%         32.33%       1.2848      ing. The very low exact-match rate (14.5%) are due
    DeepSeek-R1-Distill-Llama-8B    37.25%         21.67%       1.5183
    DeepSeek-Prover-V1.5-7B         71.12%         51.00%       0.8411      to the sentence decomposition. In Web-NLG2 and
    Llama-3.3-70B
    Llama-3-8B
                                    89.57%
                                    91.71%
                                                   72.67%
                                                   78.00%
                                                                0.4836
                                                                0.4544
                                                                            Wiki-NRE RE, we record a micro-F1 of 75.67%
    Mistral-7B-Instruct-v0.3        91.19%         80.00%       0.3273      and 58.84% respectively, which falls largely behind
    Mixtral-8x7B-Instruct-v0.1      92.16%         76.67%       0.2727
    Qwen 7B                         56.68%         40.33%       1.1977      the SOTA at 93.6% (Ferreira et al., 2020) and Ge-
    Qwen-7B-Chat                    15.81%          4.33%       1.7556
    GPT 4 o                         82.75%         68.33%       0.5354
                                                                            nIE’s 91.48% (Distiawan et al., 2019) respectively.
    GPT 4 o-3                       94.10%         81.67%       0.2320      However, knowing that this generative task without
                                                                            any fine-tuning shows opportunities for great im-
                                                                            provement. In the CaRB OpenIE benchmark, we
5.4 Extracting Relationship Pairs from
                                                                            achieve micro-F1 61.54% and macro-F1 62.61%,
    Simple Sentences
                                                                            compared to DetIE’s SOTA 67.7% (Bhardwaj et al.,
With the total number of simple sentences exceed-                           2019).
ing 177,000, we used a carefully crafted COT +                                 Though the results were not as high as we ex-
FICL prompt as it has shown from data to perform                            pected, they look promising for an unsupervised
the best. The annotators AB and CD studied 100                              approach. Future work would look into fine-tuning
sentences generated from each model that were                               LLMs for specific tasks like these. To justify the
tested below and came to an agreement that the                              need for our pipeline, we performed an ablation
Mixtral-8x7B-Instruct-v0.1 model performed well                             study on 23 articles already in our coreferenced set.
with a 99% accuracy in parsing relationships from                           Table 11, Appendix D, shows that removing coref-
simple sentences, which also involved capturing                             erence resolution or sentence-level decomposition
multiple relationships in text. It was a Boolean                            hurts our performance sharply, and without these
task. The table for the task is Table 9 located in                          decomposition, our recall falls below half, which
Appendix D.                                                                 explains to us that biomedical sentences have a lot
                                                                            of nuanced co-referent names that must be split
6      Discussion                                                           before extraction. Therefore, having this pipeline
We evaluated the results on a subset of 200 sam-                            is a reliable way to use LLMs for generated con-
ples each from Rebel, Web-NLG, and Wiki-NRE                                 tent. We have baselines like ChatGPT-4.5, which
obtained from the GitHub page of (Zhang and Soh,                            do have near-perfect precision. They often sum-
2024). We also evaluated the 50 unique sentences                            marize relationships and do not give the user the
from the CaRB dataset (Bhardwaj et al., 2019).                              ability to pick and choose, as it makes the decision
                                                                            on the user’s behalf. The DeepSeek-405B model
      Table 7: Evaluation metrics across benchmarks                         performs well too, as it has a higher recall, but in-
                                                                            troduces a lot of errors. It is interesting to see how
    Metrics           ReBEL        Web-NLG2      Wiki-NRE      CaRB
                                                                            our pipeline maintained high precision and recall,
    Exact-Match
    Prec Macro
                      14.50%
                      72.97%
                                    43.00%
                                    80.70%
                                                   7.00%
                                                  60.26%
                                                               43.14%
                                                               66.96%
                                                                            but extracted some relationships beyond human ref-
    Rec Macro         59.88%        71.64%        60.13%       68.43%       erence. The results were hand-graded by humans
    F1-Score Macro    65.78%        75.90%        60.20%       62.61%
    Prec Micro        66.34%        79.35%        59.02%       63.69%       using the gold standard.
    Rec Micro         59.88%        72.32%        58.67%       59.52%
    F1-Score Micro    62.94%        75.67%        58.84%       61.54%
    RMSE              0.8813        0.5785        0.9648       0.3633       6.1   Error Analysis
                                                                            From the analysis, we classified the errors into 3
   The results show that even with the extra aid,                           types, namely: "missing", "spurious", and "relation-
LLMs still have difficulty parsing relationships the                        mismatch". Missing error happens when a rela-
way humans can. Comparing to current SOTA                                   tionship type is not considered. Spurious is non-
techniques, for the ReBEL benchmark, our model                              existent but invented relationships and relation mis-
achieves a macro-F1 of 65.78% and a micro-F1                                match occurs when there is a switch between Entity
of 62.94%, surpassing the published macro-F1 of                             1 and Entity 2 or the relationship actually exists
51.0% (Cabot and Navigli, 2021) but falling short                           with another Entity (Entity3), but the model mis-
of the SOTA micro-F1 of 74.0% (Cabot and Nav-                               classified it. Furthermore, in the Rebel dataset,
                                                                         15522
the relation-extraction is consistent, as it only con-      ture work in information extraction and co refer-
tains four variables across the dataset. We noticed         ence resolution. The annotated datasets from hu-
that the pipeline parses some “less important” or           mans include 190 abstract texts, 7248 rows for
“missed relationships”, which we categorized as             sentence classification, and 900 rows for sentence
spurious in this evaluation. During the evaluation,         decomposition.
if a model outputs something not in the gold, but is
a valid relationship from the text, the model is not        Limitations
given a score (neither penalized nor awarded).              The limitations of our paper are as follows:

                                                               • Before, our pipeline would be of industry-
                                                                 standard, it is imperative that we find a so-
                                                                 lution to coreference resolution and its weak-
                                                                 nesses. Future work would look into how to
                                                                 improve coreference resolution in thick ab-
                                                                 stract texts.

                                                               • Using just open-source models, though com-
                                                                 parative with ChatGPT tends to fail at times,
                                                                 and thereby, average the RE score. Future
                                                                 iterations of the work would employ agent
Figure 2: Distribution of error bucketization categories.
                                                                 systems that can scan for missing or inconsis-
                                                                 tent data through a feedback loop and judge
   However, if it outputs something not in the gold,
                                                                 the quality to improve the output.
and cannot be inferred from the text, then we penal-
ize the model. We also noticed a few relation-type             • We only focused on prompting this time;
mismatches, where the model defaults to a more                   future iterations would look at fine-tuning
simplistic relationship type, like “is”, “was”, rather           strategies like Parameter-Efficient Fine-tuning
than focusing on the actual relationship. For exam-              (PEFT) that could help the LLM perform bet-
ple, the model used a generic “is” for party member-             ter in weak tasks.
ship instead of “member of political party”, in the
sentence “Anju Dhillon (born 1979) is a Canadian               • We had a lot of human annotations for quality.
Liberal politician, who was elected to represent the             Future work would engage more code scripts,
riding of Dorval-Lachine-LaSalle in the House of                 so we can evaluate a wider range of output.
Commons of Canada in the 2015 federal election.”.
                                                               • We acknowledge that our pipeline is costly
Missing and Spurious relationships amongst the
                                                                 computationally, and as such, it might not be
errors were profound, as the model typically high-
                                                                 possible to run with low-level resources, and
lights 3 out of 4 of the relationship types in REBEL
                                                                 though the CoT+FICL prompting did well, it
and typically omits one. A combination of spurious
                                                                 might incur a non-trivial inference time and
and missing relationships characterized 73% of the
                                                                 token consumption. However, since this is a
errors, while spurious relationships alone scored
                                                                 one-off knowledge graph, it is still deployable.
11%, and missing relationships 9% and relation-
type is 7%. Interestingly, more often than not, it             • Our evaluation is confined to scientific and
highlights newer relationships that we were not                  benchmark datasets. We have not tested the
considering.                                                     pipeline on domains such as newswire, legal
                                                                 text, or social media. Thus, its generalizability
7   Conclusion                                                   to noisier or less formal text remains unveri-
In this work, we created a pipeline and verified                 fied.
that using sentence decomposition on open-source
                                                            Acknowledgements
models actually helps the model think through each
problem uniquely. By releasing our implementa-              We thank the expert annotators for their work. The
tion, annotated datasets, and evaluation scripts, we        biologist: Thelma Emeji and Olaniyi Glory; and the
aim to promote reproducibility and accelerate fu-           linguist: Oluwafaratunmi Olakanmi, and Godman
                                                       15523
Adebayo. We also thank the rest of the other anno-        Jiahang Cao, Jinyuan Fang, Zaiqiao Meng, and Shang-
tators who did not want their names mentioned. All           song Liang. 2024. Knowledge graph embedding: A
                                                             survey from the perspective of representation spaces.
instructions given to the annotators were the same
                                                             ACM Computing Surveys, 56(6):1–42.
as the prompt. Each annotator was paid weekly at
a rate of 40,000 Nigerian Naira, over a period of 8       Payal Chandak, Kexin Huang, and Marinka Zitnik.
weeks, and this project was self-funded.                    2023. Building a knowledge graph to enable pre-
                                                            cision medicine. Scientific Data, 10(1):67.

                                                          Bidyut Das, Mukta Majumder, and Santanu Phadikar.
References                                                  2018. A novel system for generating simple sen-
Tareq Al-Moslmi, Marc Gallofré Ocaña, Andreas L             tences from complex and compound sentences. Inter-
  Opdahl, and Csaba Veres. 2020. Named entity ex-           national Journal of Modern Education and Computer
  traction for knowledge graphs: A literature overview.     Science, 11(1):57–64.
  IEEE Access, 8:32862–32881.
                                                          Bayu Distiawan, Gerhard Weikum, Jianzhong Qi, and
Phil. Atteberry. 2016. Sentence types: Simple, com-         Rui Zhang. 2019. Neural relation extraction for
  pound, complex, and compound-complex sentences.           knowledge base enrichment. In Proceedings of the
  https://sites.pitt.edu/~atteberr/                         57th Annual Meeting of the Association for Compu-
  comp/0150/grammar/sentencetypes.                          tational Linguistics, pages 229–240.
  html. Accessed: March 16, 2025.
                                                          Thiago Castro Ferreira, Claire Gardent, Nikolai Ilinykh,
Sangnie Bhardwaj, Samarth Aggarwal, and Mausam.             Chris Van Der Lee, Simon Mille, Diego Moussallem,
  2019. CaRB: A crowdsourced benchmark for open             and Anastasia Shimorina. 2020. The 2020 bilingual,
  IE. In Proceedings of the 2019 Conference on Empiri-      bi-directional webnlg+ shared task overview and eval-
  cal Methods in Natural Language Processing and the        uation results (webnlg+ 2020). In Proceedings of the
  9th International Joint Conference on Natural Lan-        3rd International Workshop on Natural Language
  guage Processing (EMNLP-IJCNLP), pages 6262–              Generation from the Semantic Web (WebNLG+).
  6267, Hong Kong, China. Association for Computa-
  tional Linguistics.                                     Scott Friedman, Ian Magnusson, Vasanth Sarathy, and
                                                            Sonja Schmer-Galunder. 2022. From unstructured
Suman Bharti, Dan Chia-Tien Lo, and Yong Shi. 2024.         text to causal knowledge graphs: A transformer-
  Enhancing contextual understanding in knowledge           based approach. arXiv preprint arXiv:2202.11768.
  graphs: Integration of quantum natural language pro-
  cessing with neo4j llm knowledge graph. In 2024         Saiping Guan, Xueqi Cheng, Long Bai, Fujun Zhang,
  IEEE International Conference on Big Data (Big-           Zixuan Li, Yutao Zeng, Xiaolong Jin, and Jiafeng
  Data), pages 8628–8630. IEEE.                             Guo. 2022. What is event knowledge graph: A sur-
                                                            vey. IEEE Transactions on Knowledge and Data
Stephen H Bradley, Nathaniel Luke Fielding Hatton,          Engineering, 35(7):7569–7589.
   Rehima Aslam, Bobby Bhartia, Matthew EJ Callister,
   Martyn PT Kennedy, Luke TA Mounce, Bethany
                                                          Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan
   Shinkins, William T Hamilton, and Richard D Neal.
                                                            Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A
   2021. Estimating lung cancer risk from chest x-ray
                                                            Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al.
   and symptoms: a prospective cohort study. British
                                                            2024. Retrieval-augmented generation with graphs
  Journal of General Practice, 71(705):e280–e286.
                                                            (graphrag). arXiv preprint arXiv:2501.00309.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
  Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind      Nurmala Hendrawati. 2018. An analysis on students’
  Neelakantan, Pranav Shyam, Girish Sastry, Amanda          errors in writing sentence patterns. Loquen: English
  Askell, et al. 2020. Language models are few-shot         Studies Journal, 11(1):63–85.
  learners. Advances in neural information processing
  systems, 33:1877–1901.                                  Claudia I Henschke, Rowena Yip, Dorith Shaham,
                                                            Steven Markowitz, José Cervera Deval, Javier J Zu-
Markus J Buehler. 2024. Accelerating scientific dis-        lueta, Luis M Seijo, Cheryl Aylesworth, Karl Klin-
 covery with generative knowledge extraction, graph-        gler, Shahriyour Andaz, et al. 2023. A 20-year
 based representation, and multimodal intelligent           follow-up of the international early lung cancer ac-
 graph reasoning. Machine Learning: Science and             tion program (i-elcap). Radiology, 309(2):e231988.
 Technology, 5(3):035083.
                                                          Aidan Hogan, Eva Blomqvist, Michael Cochez, Clau-
Pere-Lluís Huguet Cabot and Roberto Navigli. 2021.          dia d’Amato, Gerard De Melo, Claudio Gutierrez,
  Rebel: Relation extraction by end-to-end language         Sabrina Kirrane, José Emilio Labra Gayo, Roberto
  generation. In Findings of the Association for Com-       Navigli, Sebastian Neumaier, et al. 2021. Knowledge
  putational Linguistics: EMNLP 2021, pages 2370–           graphs. ACM Computing Surveys (Csur), 54(4):1–
  2381.                                                     37.
                                                     15524
Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Martti-         2022. Training language models to follow instruc-
  nen, and S Yu Philip. 2021. A survey on knowledge           tions with human feedback. Advances in neural in-
  graphs: Representation, acquisition, and applications.      formation processing systems, 35:27730–27744.
  IEEE transactions on neural networks and learning
  systems, 33(2):494–514.                                  Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-
                                                             apu Wang, and Xindong Wu. 2024. Unifying large
Todd R Johnson and Elmer V Bernstam. 2023. Why               language models and knowledge graphs: A roadmap.
  is biomedical informatics hard? a fundamental              IEEE Transactions on Knowledge and Data Engi-
  framework. Journal of Biomedical Informatics,              neering, 36(7):3580–3599.
  140:104327.

Md Rezaul Karim, Lina Molinas Comet, Md Shajalal,          Sameer Pradhan, Alessandro Moschitti, Nianwen Xue,
 Oya Deniz Beyan, Dietrich Rebholz-Schuhmann, and            Olga Uryupina, and Yuchen Zhang. 2012. Conll-
 Stefan Decker. 2023. From large language models to          2012 shared task: Modeling multilingual unrestricted
 knowledge graphs for biomarker discovery in cancer.         coreference in ontonotes. In Joint conference on
 arXiv preprint arXiv:2310.08365.                            EMNLP and CoNLL-shared task, pages 1–40.

Amanda Kau, Xuzeng He, Aishwarya Nambissan,                Simon Schramm, Christoph Wehner, and Ute Schmid.
 Aland Astudillo, Hui Yin, and Amir Aryani. 2024.            2023. Comprehensible artificial intelligence on
 Combining knowledge graphs and large language               knowledge graphs: A survey. Journal of Web Se-
 models. arXiv preprint arXiv:2407.06564.                    mantics, 79:100806.
Davinder Kaur, Suleyman Uslu, Kaley J Rittichier, and      Yijun Tian, Shichao Pei, Xiangliang Zhang, Chuxu
  Arjan Durresi. 2022. Trustworthy artificial intelli-       Zhang, and Nitesh Chawla. 2023. Knowledge distilla-
  gence: a review. ACM computing surveys (CSUR),             tion on graphs: A survey. ACM Computing Surveys.
  55(2):1–38.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yu-        Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
  taka Matsuo, and Yusuke Iwasawa. 2022. Large lan-           Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
  guage models are zero-shot reasoners. Advances in           et al. 2022. Chain-of-thought prompting elicits rea-
  neural information processing systems, 35:22199–            soning in large language models. Advances in neural
  22213.                                                      information processing systems, 35:24824–24837.

Jia Li, Ge Li, Yongmin Li, and Zhi Jin. 2025. Struc-       Tianhao Wu, Janice Lan, Weizhe Yuan, Jiantao Jiao, Ja-
   tured chain-of-thought prompting for code genera-          son Weston, and Sainbayar Sukhbaatar. 2024. Think-
   tion. ACM Transactions on Software Engineering             ing llms: General instruction following with thought
   and Methodology, 34(2):1–23.                               generation. arXiv preprint arXiv:2410.10630.
Tianle Li, Xueguang Ma, Alex Zhuang, Yu Gu, Yu Su,         Bowen Zhang and Harold Soh. 2024. Extract, de-
   and Wenhu Chen. 2023. Few-shot in-context learn-          fine, canonicalize: An llm-based framework for
   ing for knowledge base question answering. arXiv          knowledge graph construction. arXiv preprint
   preprint arXiv:2305.01750.                                arXiv:2404.03868.
Jixiong Liu, Yoan Chabot, Raphaël Troncy, Viet-Phi
   Huynh, Thomas Labbé, and Pierre Monnin. 2023.           Hanrong Zhang, Xinyue Wang, Jiabao Pan, and Hong-
   From tabular data to knowledge graphs: A survey           wei Wang. 2023. Saka: an intelligent platform for
   of semantic table interpretation tasks and methods.       semi-automated knowledge graph construction and
   Journal of Web Semantics, 76:100761.                      application. Service Oriented Computing and Appli-
                                                             cations, 17(3):201–212.
Finlay MacLean. 2021. Knowledge graphs and their
  applications in drug discovery. Expert opinion on        Jian Zhang, Bifan Wei, Shihao Qi, Jun Liu, Qika Lin,
  drug discovery, 16(9):1057–1069.                            et al. 2025. Gkg-llm: A unified framework for gener-
                                                              alized knowledge graph construction. arXiv preprint
Belinda Mo, Kyssen Yu, Joshua Kazdan, Proud Mpala,            arXiv:2503.11227.
  Lisa Yu, Chris Cundy, Charilaos Kanatsoulis, and
  Sanmi Koyejo. 2025. Kggen: Extracting knowledge
                                                           Wen Zhang, Jiaoyan Chen, Juan Li, Zezhong Xu, Jeff Z
  graphs from plain text with language models. arXiv
                                                             Pan, and Huajun Chen. 2024. Knowledge graph
  preprint arXiv:2502.09956.
                                                             reasoning with logics and embeddings: Survey and
MZ Naser. 2022. Causality, causal discovery, and causal      perspective. In 2024 IEEE International Conference
 inference in structural engineering. arXiv preprint         on Knowledge Graph (ICKG), pages 492–499. IEEE.
 arXiv:2204.01543.
                                                           Lingfeng Zhong, Jia Wu, Qian Li, Hao Peng, and Xin-
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,            dong Wu. 2023. A comprehensive survey on auto-
  Carroll Wainwright, Pamela Mishkin, Chong Zhang,           matic knowledge graph construction. ACM Comput-
  Sandhini Agarwal, Katarina Slama, Alex Ray, et al.         ing Surveys, 56(4):1–62.
                                                      15525
A   Sentence Steps
Sentences are made up of clauses, phrases, words,
and punctuation. Phrases are a group of words
that act as a single unit but do not have both a sub-
ject and a predicate. The common types are noun
phrase N P , verb phrase V P , and prepositional
phrases P P . A sentence in English is defined as:
Sentence → N P, V P . This means every sen-
tence must have a noun phrase, typically including
the subject and the verb. Not all sentences have
a predicate or an object; however, this is very un-
common in academic writing. Noun phrases are            Algorithm 1 Coreference Resolution
defined as: N P → |Det N | Det Adj N | Pronouns         Input: abstracts A, subset size s, annotators H,
|Proper Nouns|, while verb phrases are defined as:      prompt strategies P, models M
V P → |V |V N P |V N P P P |, where Det is a de-        Output: resolved abstracts Â
terminer: for example, "a", "the", "an", "those",        1: A′ ← UniformSample(A, s)
etc., Adj is an adjective, and V is a verb. Clauses         ▷ Select s abstracts at random
are typically of two types: independent clauses (IC)     2: for all a ∈ A′ do
and dependent clauses (DC). An IC can stand alone        3:     for all hj ∈ H do
as a complete sentence, while the DC relies on an        4:         Rj (a) ← fann (hj , a)
IC. Dependent clauses start with subordinating con-         ▷ Annotator hj resolves coreference on a
junctions like “because”, “although”, “when”, “if”,      5: G ← { a ∈ A′ | Rj (a) = Rk (a) ∀ j, k}
etc., and are then followed by an IC to make a              ▷ Gold set of unanimous abstracts
complete sentence. Clause → N P, V P . Math-             6: G′ ← {Rj (a) | a ∈ G}
ematically, it is represented by DC → Subconj               ▷ Gold-standard annotated abstracts
IC.                                                      7: (P ∗ , M ∗ ) ← (∅, ∅); bestScore ← −∞
                                                         8: for all P ∈ P do
B   Algorithms
                                                         9:     for all M ∈ M do
                                                        10:         R̂P,M ← fprompt (P, M, G)
                                                            ▷ Predict annotations on G
                                                        11:         SP,M ← score(R̂P,M , G′ )
                                                            ▷ Evaluate predictions against G′
                                                        12:         if SP,M > bestScore then
                                                        13:              (P ∗ , M ∗ ) ← (P, M )
                                                        14:              bestScore ← SP,M
                                                            ▷ Update best prompt-model pair
                                                        15: Â ← fprompt (P ∗ , M ∗ , A)
                                                            ▷ Resolve coreference on full collection
                                                        16: return Â




                                                   15526
Algorithm 2 Step 1: Sample Abstracts, Extract            Algorithm 4 Unified Sentence Simplification
One Sentence, and Verify                                  1: Input:                     sentence        sets
 1: Input:     resolved abstracts Â, sample sizes           {Sc }c∈{comx,comp,comx_comp} ,          prompt
    psimp , pcomx , pcomp , pcomx_comp , pincomp , ex-       strategies P, models M
    pert verifiers V = {v1 , v2 }                         2: Output: simplified sentences Ssimp
 2: Output: verified sentences with category la-          3: Ssimp ← ∅
    bels Ŝ                                                  ▷ Initialize output set
 3: for all category c in {simp, comx, comp,              4: for         all      category         c      ∈
    comx_comp, incomp} do                                    {comx, comp, comx_comp} do
 4:     Ac ← UniformSample(Â, pc ) ▷ Select                 ▷ Iterate over sentence categories
    pc abstracts for c                                    5:      bestScore ← −∞
 5:     Sc ← ∅                                               ▷ Reset best score
 6:     for all a ∈ Ac do                                 6:      for all P ∈ P do
 7:          if c ∈ {simp, incomp} then                      ▷ For each prompting strategy
 8:               s ← fcreate (annotator, a) ▷ Create     7:          for all M ∈ M do
    sentence for simple/incomplete                           ▷ For each model
 9:          else                                         8:               score                         ←
10:               s ← fchoose (a) ▷ Choose sentence          EvaluatePromptModel(P, M, Sc )
    for other categories                                     ▷ Evaluate on Sc
                                                          9:               if score > bestScore then
11:          Sc ← Sc ∪ {s} ▷ Collect one sentence
                                                         10:                    bestScore        ←    score;
    per abstract
               S                                             (P ∗ , M ∗ ) ← (P, M )
12: Ensure c Ac = Â                ▷ Coverage of all        ▷ Update best pair
    abstractsS
                                                         11:      for all s ∈ Sc do
13: Sall ← c Sc             ▷ All candidate sentences
                                                             ▷ Simplify each sentence in category c
14: Ŝ = {(s, c) | c ∈ C, s ∈ Sc , fver (v1 , s) =
                                                         12:          ŝ ← fprompt (P ∗ , M ∗ , s)
    fver (v2 , s)} ▷ Keep only unanimously veri-
                                                             ▷ Generate simplified sentence
    fied sentences with their category; where C =
                                                         13:          Ssimp ← Ssimp ∪ {ŝ}
    {simp, comx, comp, comx_comp, incomp}
                                                             ▷ Collect simplified sentence
15: return Ŝ
                                                         14: return Ssimp
                                                             ▷ Return all simplified sentences
Algorithm 3 Step 2: Model Selection and Full
Classification
                                                         Algorithm 5 Relationship Extraction from Simpli-
 1: Input: verified dataset D = {(si , yi ) | si ∈
                                                         fied Sentences
    Ŝ}, candidate models M, resolved abstracts Â
                                                           1: Input: simplified sentences Ssimp , classi-
 2: Output: full classification S̃ = {(s, ℓ) | s ∈
                                                              fied sentences S̃ = {(s, ℓ)} best prompting
    Sentences(Â)}
                                                              strategy-model pair (P ∗ , M ∗ )
 3: bestScore ← −∞,         m∗ ← ∅
                                                           2: Output: relation triples R = {(e1 , r, e2 )}
 4: for all m ∈ M do
 5:     Train m on training split of D                    3: Sinit ← { s | (s, ℓ) ∈ S̃, ℓ = simp}        ▷ Select
 6:     score ← Evaluate(m, val split of D)                  only initially classified simple sentences
 7:     if score > bestScore then                         4: S ← Ssimp ∪ Sinit                  ▷ Combine with
 8:          bestScore ← score                               previously simplified sentences
 9:          m∗ ← m                                       5: R ← ∅                       ▷ Initialize relation set
10: S̃ ← ∅                                                6: for all s ∈ S do
11: for all abstract a ∈ Â do                            7:     (e1 , r, e2 ) ← frel (P ∗ , M ∗ , s) ▷ Extract
12:     for all sentence s ∈ Sentences(a) do                 (entity1 , relationship, entity2 )
13:          ℓ ← m∗ .classify(s) ▷ Classify each          8:     if (e1 , r, e2 ) ̸= ∅ then
    sentence in the main abstracts                        9:          R ← R ∪ {(e1 , r, e2 )} ▷ Keep valid
14:          S̃ ← S̃ ∪ {(s, ℓ)}                              triples
15: return S̃                                            10: return R            ▷ All extracted relation triples

                                                    15527
C   Tables
C.1 Coreference Resolution

    Table 8: Coreference resolution performance using cosine similarity across models and prompting styles.

                        Model                        Prompt     MUC (%) B3 (%) CEAF (%) CoNLL (%)
                                                     GIP         13.34   70.96   70.96    51.75
                                                     COT         14.50   70.87   70.87    52.08
                        Mixtral-8x7B-Instruct-v0.1
                                                     FICL        32.42   70.61   70.61    57.88
                                                     COT+FICL    27.75   70.73   70.73    56.40
                                                     GIP          9.74   70.44   70.44    50.20
                                                     COT         10.19   70.47   70.47    50.37
                        Llama-3.1-8B-Instruct
                                                     FICL        27.16   69.57   69.57    55.43
                                                     COT+FICL    26.00   69.58   69.58    55.06
                                                     GIP          7.13   70.56   70.56    49.41
                                                     COT          6.14   71.32   71.32    49.60
                        Llama-3.2-3B-Instruct
                                                     FICL        16.07   70.21   70.21    52.16
                                                     COT+FICL    16.98   70.70   70.70    52.79
                                                     GIP          7.79   71.28   71.28    50.12
                                                     COT          8.75   71.34   71.34    50.48
                        Llama-3.3-70B-Instruct
                                                     FICL        31.25   70.94   70.94    57.71
                                                     COT+FICL    28.97   70.95   70.95    56.95
                                                     GIP          6.99   71.07   71.07    49.71
                                                     COT          7.26   70.87   70.87    49.67
                        Mistral-7B-Instruct-v0.3
                                                     FICL        16.96   70.99   70.99    52.98
                                                     COT+FICL    18.58   70.74   70.74    53.35




D   Extracting Relationship Pairs from Simple Sentences
We went back to our coreference annotators and asked them if they could look at a small sample of the
dataset and see if the model was successfully able to parse all relationships

             Table 9: Model Accuracy on Boolean Relationship Extraction from Simple Sentences

                Model Name                                      # Params (B)      Accuracy        F1 -Score
                LLaMA-3-8B                                            8            98.00%         98.00%
                Mistral-7B                                            7            87.00%         93.55%
                DeepSeek-Distilled-LLaMA-8B                          8             62.00%         77.02%
                Qwen-7B                                               7            52.00%         68.42%
                LLaMA-2-7B                                            7            35.00%         51.85%
                QwenChat-7B                                           7            23.00%         37.40%
                DeepSeek-7B                                           7            21.00%         34.71%
                DeepSeek-Prover-7B                                    7            20.00%         33.61%
                DeepSeek-Distilled-Qwen-7B                           7             11.00%         19.82%
                Mistral-MoE-8×7B                                    8×7            99.00%         99.50%
                DeepSeek-67B                                         67            43.00%         60.14%
                LLaMA-2-13B                                          13            13.00%         23.01%
                GPT-NeoX-20B                                         20             0.00%          0.00%
                LLaMA-2-70B                                          70            41.00%         58.57%




                                                              15528
            Table 10: Performance comparison across models and prompting styles

                Model            Prompting Style    Macro Average   Exact-Match   RMSE
                                 COT+FICL              99.78%         98.00%      0.1078
                                 COT                   82.86%         64.00%      0.4265
                LLAMA 3 8B
                                 FICL                  68.89%         28.33%      0.5376
                                 GIP                   45.81%         27.33%      0.9319
                                 COT+FICL              96.14%         91.67%      0.2323
                                 COT                   92.42%         83.00%      0.3146
                MISTRAL 8 BY 7
                                 FICL                  90.23%         78.00%      0.3585
                                 GIP                   82.26%         57.67%      0.4563
                                 COT+FICL              96.64%         90.33%      0.2356
                                 COT                   84.72%         64.33%      0.4280
                MISTRAL 7 B
                                 FICL                  85.78%         67.00%      0.4016
                                 GIP                   78.88%         53.00%      0.5055
                                 COT+FICL              95.30%         81.00%      0.3213
                                 COT                   86.39%         68.33%      0.4064
                LLAMA 3 70B
                                 FICL                  71.98%         45.33%      0.6064
                                 GIP                   62.01%         35.00%      0.7829




                 Table 11: Triple-extraction performance on the evaluation set

Configuration                                           Triples     Precision         Recall   F1 Score
Human Standard                                              398     100.00%       100.00%      100.00%
Full Model (Ours)                                           422      92.00%        92.90%       92.40%
  – Remove Coref Resolution                                 376      80.60%        74.60%       77.50%
  – Remove Sentence Decomposition                           208      74.60%        46.20%       57.20%
  – Remove Coref + Sentence Decomposition                   220      76.80%        42.70%       54.80%
DeepSeek R1                                                 323      93.50%        78.60%       85.40%
ChatGPT 4o                                                  215      98.10%        52.80%       68.50%
NotebookLM                                                   67     100.00%        16.83%       28.82%
ChatGPT 4.5                                                 238      99.58%        59.55%       74.53%




                                                   15529
Table 12: Co-reference Group AB and Group CD, where
each group’s “link” set is the intersection of its two         ("BACKGROUND:", 0), ("There", 1),
                                                               ("are", 2), ("few", 3), ("cases", 4),
annotators.                                                    ("of", 5), ("pulmonary", 6),
                                                               ("granulomatous", 7), ("changes", 8),
 Statistic                                   Value             ("secondary", 9), ("to", 10),
                                                               ("primary", 11), ("biliary", 12),
 Group definitions: A and B = 2,041; C and D = 1,929           ("cirrhosis", 13), ("(PBC).", 14),
                                                               ("No", 15), ("case", 16), ("of", 17),
 Number of “link” assignments (Group AB)     2,041             ("granulomatous", 18), ("lung", 19),
 Number of “link” assignments (Group CD)     1,929             ("disease", 20), ("secondary", 21),
 Intersection                                1,847             ("to", 22), ("PBC", 23),
 Union                                       2,123             ("misdiagnosed", 24), ("as", 25),
                                                               ("lung", 26), ("cancer", 27),
 Observed agreement Po                      ≈ 0.87             ("had", 28), ("been", 29),
 Expected agreement estimated Pe             0.50              ("reported.", 30), ("CASE", 31),
 Cohen’s κ                                   0.74              ("SUMMARY:", 32), ("A", 33),
                                                               ("middle-aged", 34), ("woman", 35),
                                                               ("presented", 36), ("with", 37),
                                                               ("lung", 38), ("nodules", 39),
                                                               ("and", 40), ("was", 41),
E      Prompting Strategy                                      ("misdiagnosed", 42), ("with", 43),
                                                               ("lung", 44), ("cancer", 45),
                                                               ("by", 46), ("positron", 47),
E.1     Prompt Templates for Co-reference                      ("emission", 48),
        Resolution                                             ("tomography/computed", 49),
                                                               ("tomography.", 50), ("She", 51),
      COT+FICL Co-reference Resolution                         ("underwent", 52), ("left", 53),
                                                               ("lobectomy,", 54), ("and", 55),
      Prompt                                                   ("the", 56), ("pathology", 57),
                                                               ("of", 58), ("the", 59),
    You are a coreference resolution agent. Be-                ("nodules", 60), ("showed", 61),
    low is a biomedical abstract presented as                  ("granulomatous", 62),
                                                               ("inflammation,", 63), ("which", 64),
    tokenized text with indices. Your task is                  ("was", 65), ("then", 66),
    to identify and annotate coreference expres-               ("treated", 67), ("with", 68),
                                                               ("antibiotics.", 69),
    sions within the text. For each co-referent                ("However,", 70), ("a", 71),
    expression:                                                ("new", 72), ("nodule", 73),
                                                               ("appeared.", 74), ("Further", 75),
         • Record the surface form under “Ex-                  ("investigation", 76), ("with", 77),
                                                               ("lung", 78), ("biopsy", 79),
           pression”.                                          ("and", 80), ("liver", 81),
                                                               ("serology", 82), ("led", 83),
         • Use the provided token indices as                   ("to", 84), ("the", 85),
           “StartToken” and “EndToken” (they are               ("diagnosis", 86), ("of", 87),
                                                               ("PBC,", 88), ("and", 89),
           the same for single-token expressions).             ("chest", 90), ("computed", 91),
                                                               ("tomography", 92),
         • Map each expression to its antecedent               ("indicated", 93),
           using “RefersTo” — either a noun                    ("significant", 94),
                                                               ("reduction", 95), ("in", 96),
           phrase or named entity from the text.               ("the", 97), ("pulmonary", 98),
                                                               ("nodule", 99), ("by", 100),
         • Only include pronouns or repeated                   ("treatment", 101), ("with", 102),
           noun phrases referring back to a prior              ("methylprednisolone", 103),
                                                               ("and", 104),
           concept or entity.                                  ("ursodeoxycholic", 105),
                                                               ("acid.", 106),
      Use this format:                                         ("CONCLUSION:", 107),
      {                                                        ("Diagnosis", 108), ("of", 109),
      "Expression": "string",                                  ("pulmonary", 110), ("nodules", 111),
      "StartToken": int,                                       ("requires", 112),
      "EndToken": int,                                         ("integrating", 113),
      "RefersTo": "string"                                     ("various", 114), ("clinical", 115),
      }                                                        ("data", 116), ("to", 117),
                                                               ("avoid", 118), ("unnecessary", 119),
      Example:                                                 ("pulmonary", 120),
                                                               ("lobectomy.", 121)
      Given this tokenized abstract:

                                                       15530
[                                                       "Expression": "string",
{                                                       "StartToken": int,
"Expression": "PBC",                                    "EndToken": int,
"StartToken": 14,                                       "RefersTo": "string"
"EndToken": 14,                                         }
"RefersTo": "Primary
biliary cirrhosis"                                      Now process this tokenized abstract:
},                                                      {tokenized_text}
{
"Expression": "PBC",
"StartToken": 23,                                       FICL Co-reference Resolution Prompt
"EndToken": 23,
"RefersTo": "Primary biliary
cirrhosis"                                              You are a coreference resolution agent. Be-
},                                                      low is a biomedical abstract presented as
{                                                       tokenized text with indices. Your task is
"Expression": "She",
"StartToken": 51,                                       to identify and annotate coreference expres-
"EndToken": 51,                                         sions within the text.
"RefersTo": "A middle-aged                              Use this format:
woman"
},
{                                                       {
"Expression": "PBC",                                    "Expression": "string",
"StartToken": 88,                                       "StartToken": int,
"EndToken": 88,                                         "EndToken": int,
"RefersTo": "Primary biliary                            "RefersTo": "string"
cirrhosis"                                              }
}
]                                                       Example:
                                                        Given this tokenized abstract:
Now process this tokenized abstract:                    ("BACKGROUND:", 0), ("There", 1),
{tokenized_text}                                        ("are", 2), ("few", 3), ("cases", 4),
                                                        ("of", 5), ("pulmonary", 6),
                                                        ("granulomatous", 7), ("changes", 8),
COT Co-reference Resolution Prompt                      ("secondary", 9), ("to", 10),
                                                        ("primary", 11), ("biliary", 12),
                                                        ("cirrhosis", 13), ("(PBC).", 14),
You are a coreference resolution agent. Be-             ("No", 15), ("case", 16), ("of", 17),
low is a biomedical abstract presented as               ("granulomatous", 18), ("lung", 19),
tokenized text with indices. Your task is               ("disease", 20), ("secondary", 21),
                                                        ("to", 22), ("PBC", 23),
to identify and annotate coreference expres-            ("misdiagnosed", 24), ("as", 25),
sions within the text. For each co-referent             ("lung", 26), ("cancer", 27),
expression:                                             ("had", 28), ("been", 29),
                                                        ("reported.", 30), ("CASE", 31),
                                                        ("SUMMARY:", 32), ("A", 33),
    • Record the surface form under “Ex-                ("middle-aged", 34), ("woman", 35),
      pression”.                                        ("presented", 36), ("with", 37),
                                                        ("lung", 38), ("nodules", 39),
    • Use the provided token indices as                 ("and", 40), ("was", 41),
                                                        ("misdiagnosed", 42), ("with", 43),
      “StartToken” and “EndToken” (they are             ("lung", 44), ("cancer", 45),
      the same for single-token expressions).           ("by", 46), ("positron", 47),
                                                        ("emission", 48),
    • Map each expression to its antecedent             ("tomography/computed", 49),
                                                        ("tomography.", 50), ("She", 51),
      using “RefersTo” — either a noun                  ("underwent", 52), ("left", 53),
      phrase or named entity from the text.             ("lobectomy,", 54), ("and", 55),
                                                        ("the", 56), ("pathology", 57),
    • Only include pronouns or repeated                 ("of", 58), ("the", 59),
                                                        ("nodules", 60), ("showed", 61),
      noun phrases referring back to a prior            ("granulomatous", 62),
      concept or entity.                                ("inflammation,", 63), ("which", 64),
                                                        ("was", 65), ("then", 66),
Use this format:                                        ("treated", 67), ("with", 68),
                                                        ("antibiotics.", 69),
{                                                       ("However,", 70), ("a", 71),


                                                15531
("new", 72), ("nodule", 73),
                                                GIP Co-reference Resolution Prompt
("appeared.", 74), ("Further", 75),
("investigation", 76), ("with", 77),           You are a coreference resolution agent. Be-
("lung", 78), ("biopsy", 79),                  low is a biomedical abstract presented as
("and", 80), ("liver", 81),
                                               tokenized text with indices. Your task is
("serology", 82), ("led", 83),
("to", 84), ("the", 85),                       to identify and annotate coreference expres-
("diagnosis", 86), ("of", 87),                 sions within the text. Use this format:
("PBC,", 88), ("and", 89),
("chest", 90), ("computed", 91),
                                               Use this format:
("tomography", 92),                             {
("indicated", 93),                              "Expression": "string",
("significant", 94),                            "StartToken": int,
("reduction", 95), ("in", 96),                  "EndToken": int,
("the", 97), ("pulmonary", 98),                 "RefersTo": "string"
("nodule", 99), ("by", 100),                    }
("treatment", 101), ("with", 102),
("methylprednisolone", 103),                   Now process this tokenized abstract:
("and", 104),
                                               {tokenized_text}
("ursodeoxycholic", 105),
("acid.", 106),
("CONCLUSION:", 107),
("Diagnosis", 108), ("of", 109),          E.2     Annotators Details
("pulmonary", 110), ("nodules", 111),
("requires", 112),
                                          Annotator A is working on a medical degree An-
("integrating", 113),                     notator B is a linguistic Annotator C is a biologist
("various", 114), ("clinical", 115),      Annotator D is a linguistic
("data", 116), ("to", 117),
("avoid", 118), ("unnecessary", 119),        Annotators A and B are grouped to work to-
("pulmonary", 120),                       gether and produce a perfect work and Annotators
("lobectomy.", 121)                       C and D are also grouped in a similar pattern.
[                                            Annotator E, F and G all were tested before given
{                                         the code and had a score of 93% in a specialized
"Expression": "PBC",
"StartToken": 14,
                                          test different from the normal tests before being
"EndToken": 14,                           accepted for review. Annotators H and I, are post-
"RefersTo": "Primary biliary              graduate students. All annotators are from Nigeria.
cirrhosis"
},                                           They were all recruited by a recruitment expert
{                                         Sophia Anuyah. They all spoke English and sub-
"Expression": "PBC",                      mitted their resumes and were invited for an online
"StartToken": 23,
"EndToken": 23,                           interview.
"RefersTo": "Primary biliary                 The annotators were paid in their local currency
cirrhosis"                                weekly, at an average of 40,000 NGN a week over
},
{                                         the period of 5 weeks. The project was self-funded
"Expression": "She",                      by the authors.
"StartToken": 51,
"EndToken": 51,
"RefersTo": "A middle-aged                E.3     Creating the Sentence Structure Data Set
woman"                                    Three initial annotators - Annotator E, B and F se-
},
{                                         lected 7,500 sentences spanning five syntactic cate-
"Expression": "PBC",                      gories (compound-complex, compound, complex,
"StartToken": 88,
"EndToken": 88,
                                          simple, incomplete). Then two senior experts - G
"RefersTo": "Primary biliary              and H selected from a cohort of 12 people who took
cirrhosis"                                a classification test and scored above 93% were se-
}
]                                         lected and then adjudicated these and reached con-
                                          sensus on 7,269 sentences out of 7,500 sentences
Now process this tokenized abstract:      making a 96.92% agreement rate. The authors
{tokenized_text}                          chose not to resolve the 231 disagreements due to
                                          the current size of the dataset. The final dataset
                                          comprises of 2,118 (29.1%) compound-complex,
                                          1,191 (16.4%) compound, 865 (11.9%) complex,
                                       15532
1,585 (21.8%) simple, and 1,510 (20.8%) incom-                where R → Rewrite
plete sentences. From this set, we drew a balanced
training sample of 2,000 sentences (400 per class)                             S = Smain ∪ DC                   (6)
and reserved the remaining 5,269 sentences for
                                                              we get:
testing. The dataset is available on github.

F     Prompt for Sentence Conversion                                        E(S) = {S1, S2, S3} .

F.1 Converting Complex Sentences to Simple                          S1 → A prospective cohort study was
    Sentences                                                       conducted in Leeds, UK. S2 → The
Given the sentence:                                                 study was based on routinely collected
                                                                    data from a service. S3 → The service
      “A prospective cohort study was con-                          allowed patients with symptoms of lung
      ducted in Leeds, UK, based on routinely                       cancer to request CXR
      collected data from a service that allowed
      patients with symptoms of lung cancer                   F.2   Converting Compound Sentences to
      to request CXR” (Bradley et al., 2021).                       Simple Sentences
                                                              Given the sentence:
The process in this conversion is to identify the sin-
gular independent clause and the other dependent                    “Lung cancer stands prominently among
clauses                                                             the foremost contributors to human mor-
                                                                    tality, distinguished by its elevated fa-
    • Independent Clause: “A prospective cohort
                                                                    tality rate and the second-highest inci-
      study was conducted in Leeds, UK."
                                                                    dence rate among malignancies, and the
    • Dependent Clauses and Modifiers: (a) “that                    metastatic dissemination of lung cancer
      allowed patients with symptoms of lung can-                   stands as a primary determinant of its
      cer to request CXR.” (b) “based on routinely                  elevated mortality and recurrence rates.”
      collected data from a service”
                                                              Our goal is to break down this compound sentence
In this example, there was one dependent clause,              into simpler, stand-alone statements.
and one modifier which in our context still depends
                                                                 • Independent Clause 1 (IC1): “Lung cancer
on the subject for RE, hence, they are looped to-
                                                                   stands prominently among the foremost con-
gether in our prompt. Once the LLM can cor-
                                                                   tributors to human mortality.”
rectly identify the independent clause, the next
stage would be parse each relationship separately                • Independent Clause 2 (IC2): “The metastatic
meaning we have three simple sentences i.e. (1) the                dissemination of lung cancer stands as a pri-
independent clause (2) the subject of the IC and the               mary determinant of its elevated mortality and
DC and (3) the subject of the IC and the modifier.                 recurrence rates.”
In this case:
                                                                 • Dependent Modifier (DM): “distinguished
                                                                   by its elevated fatality rate and the second-
S = (A prospective cohort study was conducted in Leeds, UK)
    |                          {z                         }        highest incidence rate among malignancies”
                                 Smain

     ∪ (that allowed patients . . . to request CXR) .         Here, IC1 and IC2 are connected by a coordinat-
       |                   {z                     }
                            DC                                ing conjunction (i.e., “and”), which is typical in
                                                        (4)
                                                              compound sentences. The phrase “distinguished
   We define an extraction operator E(·) that maps            by its elevated fatality rate ... among malignancies”
a complex sentence to a set of simple (independent)           modifies “Lung cancer” (from IC1).
sentences:                                                    To convert this compound sentence into simple
                                                              sentences, we isolate each clause, ensuring each
                                                             stands alone:
    E(Scomplex ) =       Smain , R(DC1 ), R(DC2 ), . . . .
                                                         (5)              S = (IC1 ∪ DM ∪ IC2)
                                                         15533
We define an extraction operator E(·) that maps       We see that DC modifies or sets a contrasting con-
a compound sentence Scompound to a set of simple      text for IC1, and IC1 is coordinated with IC2 via
(independent) sentences:                              “and.” To convert this into simple sentences, each
                                                      clause (or key part of a clause) should form its own
         E(Scompound ) = {S1 , S2 , S3 } .            standalone statement:
Applying it to our sentence:
                                                                    S = (DC ∪ IC1 ∪ IC2)
           S = (IC1 ∪ DM ∪ IC2)                       Using our extraction operator E(·):
we obtain three simple sentences:
                                                          E(Scompound-complex ) = {S1 , S2 , S3 , ..., Sn } ,
     S1 → Lung cancer stands prominently
     among the foremost contributors to hu-           we obtain:
     man mortality.                                          S1 → Lung cancer is the leading cause
     S2 → It is distinguished by its elevated                of US cancer-related deaths.
     fatality rate and the second-highest inci-              S2 → Lung cancer screening with a low-
     dence rate among malignancies.                          dose chest computed tomography scan
     S3 → The metastatic dissemination of                    is now standard of care for a high-risk
     lung cancer stands as a primary determi-                eligible population.
     nant of its elevated mortality and recur-               S3 → Lung cancer screening is recom-
     rence rates.                                            mended for a high-risk, eligible popula-
                                                             tion.
F.3 Converting Compound-Complex
                                                             S4 → Clinicians and surgeons must eval-
    Sentences to Simple Sentences
                                                             uate the trade-offs of benefits and harms,
Given the sentence:                                          S5 → Evaluated trade-offs of benefits
                                                             and harms include the identification of
    “Although lung cancer is the leading
                                                             many benign lung nodules.
    cause of US cancer-related deaths, lung
                                                             S6 → Evaluated trade-offs of bene-
    cancer screening with a low radiation
                                                             fits and harms include the risk of over-
    dose chest computed tomography scan
                                                             diagnosis.
    is now standard of care for a high-risk
                                                             S7 → Evaluated trade-offs of benefits
    eligible population, and clinicians and
                                                             and harms include complications from
    surgeons must evaluate the trade-offs of
                                                             lung-cancer screening
    benefits and harms, including the iden-
    tification of many benign lung nodules,           F.4     Prompts
    overdiagnosis, and complications.”
                                                            COT+FICL Complex Sentence Conver-
   • Independent Clause 1 (IC1): “Lung cancer               sion
     screening with a low radiation dose chest com-         Below is a step-by-step process. For each
     puted tomography scan is now standard of               example, think step by step, then output
     care for a high-risk eligible population”              only the simplified sentences in the form:
   • Independent Clause 2 (IC2): “Clinicians and                  S1 → . . .     S2 → . . .      ...
     surgeons must evaluate the trade-offs of bene-
     fits and harms, ”                                      one per line, and nothing else.

   • Dependent Clause (DC): “Although lung can-             Example 1:
     cer is the leading cause of US cancer-related          Input:
     deaths”
                                                                “A prospective cohort study was
   • Modifiers: "including the identification of                conducted in Leeds, UK, based
     many benign lung nodules, overdiagnosis, and               on routinely collected data from a
     complications"
                                                  15534
    service that allowed patients with                 3. Rewrite each as standalone simple sen-
    symptoms of lung cancer to re-                        tences:
    quest CXR.”
                                                     Output:
Chain-of-Thought:
                                                        • S1 → We measured the change in flu-
 1. Identify the independent clause: “A                   orescence using a spectrophotometer.
    prospective cohort study was con-
    ducted in Leeds, UK.”                               • S2 → The cells were treated with the
                                                          drug.
 2. Identify dependent clauses/modifiers:
                                                        • S3 → The drug had been synthesized
       • Modifier A: “based on routinely                  in our lab.
         collected data from a service”
       • Dependent clause B: “that al-
                                                     Now apply the same process to this new
         lowed patients with symptoms of
                                                     sentence:
         lung cancer to request CXR”
                                                     Input: "{{sentence}}"
 3. Rewrite each as a standalone simple              ***OUTPUT ONLY the simplified sen-
    sentence:                                        tences, one per line in the form S1 → . . . ,
Output:                                              S2 → . . . , etc., and nothing else.***
                                                     Now process this abstract:
  • S1 → A prospective cohort study was              “‘ABSTRACT GIVEN HERE”’
    conducted in Leeds, UK.
                                                     COT+FICL Compound Sentence Conver-
  • S2 → The study was based on rou-
                                                     sion
    tinely collected data from a service.
                                                     Below is a process to convert a compound
  • S3 → The service allowed patients                sentence into simple sentences. For each
    with symptoms of lung cancer to re-              example, think step by step, then output
    quest CXR.                                       only the simplified sentences in the form:
                                                             S1 → . . .   S2 → . . .   ...
Example 2:
                                                     one per line, and nothing else.
Input:
                                                     Example 1:
   “After the cells were treated with                Input:
   the drug, which had been synthe-
                                                          “Lung cancer stands prominently
   sized in our lab, we measured the
                                                          among the foremost contributors
   change in fluorescence using a
                                                          to human mortality, distinguished
   spectrophotometer.”
                                                          by its elevated fatality rate and
Chain-of-Thought:                                         the second-highest incidence rate
                                                          among malignancies, and the
 1. Independent clause: “We measured the                  metastatic dissemination of lung
    change in fluorescence using a spec-                  cancer stands as a primary deter-
    trophotometer.”                                       minant of its elevated mortality
                                                          and recurrence rates.”
 2. Dependent clauses/modifiers:
                                                     Chain-of-Thought:
       • Dependent clause A: “After the
         cells were treated with the drug”             1. Identify the independent clauses:
       • Modifier B: “which had been syn-                    • IC1: “Lung cancer stands promi-
         thesized in our lab”                                  nently among the foremost con-
                                                               tributors to human mortality.”

                                             15535
       • IC2: “The metastatic dissemina-                  • S1 → Climate change accelerates the
         tion of lung cancer stands as a pri-               melting of polar ice.
         mary determinant of its elevated
         mortality and recurrence rates.”                 • S2 → Rising sea levels threaten coastal
                                                            communities around the world.
 2. Identify modifiers:
       • Modifier: “distinguished by its                Now apply the same process to this new
         elevated fatality rate and the                 sentence:
         second-highest incidence rate                  Input: "{{sentence}}"
         among malignancies” (modifies                  OUTPUT ONLY the simplified sentences,
         IC1)                                           one per line in the form S1 → . . . , S2 →
                                                        . . . , etc., and nothing else.
 3. Rewrite all parts as simple, standalone
    sentences.

Output:                                                 COT+FICL Compound-Complex Sen-
                                                        tence Conversion
  • S1 → Lung cancer stands prominently
    among the foremost contributors to hu-              Below is a process to split a compound-
    man mortality.                                      complex sentence into standalone simple
                                                        sentences. Think step by step, then apply.
  • S2 → It is distinguished by its elevated
    fatality rate and the second-highest in-            Example 1:
    cidence rate among malignancies.                    Input:

  • S3 → The metastatic dissemination of                    “Although lung cancer is the lead-
    lung cancer stands as a primary deter-                  ing cause of US cancer-related
    minant of its elevated mortality and                    deaths, lung cancer screening
    recurrence rates.                                       with a low radiation dose chest
                                                            computed tomography scan is
                                                            now standard of care for a high-
Example 2:
                                                            risk eligible population, and clin-
Input:
                                                            icians and surgeons must evalu-
   “Climate change accelerates the                          ate the trade-offs of benefits and
   melting of polar ice, and rising                         harms, including the identifica-
   sea levels threaten coastal com-                         tion of many benign lung nod-
   munities around the world.”                              ules, overdiagnosis, and compli-
                                                            cations.”
Chain-of-Thought:
                                                        Chain-of-Thought:
 1. Identify the independent clauses:
                                                         1. Dependent Clause (DC): “Although
       • IC1: “Climate change accelerates
                                                            lung cancer is the leading cause of US
         the melting of polar ice.”
                                                            cancer-related deaths”
       • IC2: “Rising sea levels threaten
         coastal communities around the                  2. Independent Clause 1 (IC1): “Lung
         world.”                                            cancer screening with a low-dose chest
                                                            computed tomography scan is now
 2. No dependent clauses or modifiers.                      standard of care for a high-risk eligible
 3. Rewrite each as a standalone simple                     population”
    sentence.                                            3. Independent Clause 2 (IC2): “Clini-
Output:                                                     cians and surgeons must evaluate the
                                                            trade-offs of benefits and harms”

                                                15536
 4. Modifier list: “including the identifi-            Output:
    cation of many benign lung nodules,
    overdiagnosis, and complications”                     • S1 → The sun warmed the fields.

 5. Rewrite into standalone simple sen-                   • S2 → The fields remained dry.
    tences.                                               • S3 → Farmers worried about the
Output:                                                     drought.

  • S1 → Lung cancer is the leading cause
    of US cancer-related deaths.                       Now apply to this new sentence:
                                                       Input: "{{sentence}}"
  • S2 → Lung cancer screening with a
                                                       OUTPUT ONLY the simplified sentences,
    low-dose chest computed tomography
                                                       one per line in the form S1 → . . . , S2 →
    scan is now standard of care for a high-
                                                       . . . , etc., and nothing else.
    risk eligible population.

  • S3 → Lung cancer screening is recom-
    mended for a high-risk, eligible popu-             COT + FICL Relationship Extraction for
    lation.                                            Knowledge Graph

  • S4 → Clinicians and surgeons must                  You are a knowledge graph relationship ex-
    evaluate the trade-offs of benefits and            traction agent. Your task is to extract struc-
    harms.                                             tured relationships from simple sentences to
                                                       create knowledge graph triples. Each triple
  • S5 → Evaluated trade-offs include the              should contain two entities and the relation-
    identification of many benign lung nod-            ship between them.
    ules.
                                                          • Analyze the sentence structure and
  • S6 → Evaluated trade-offs include the
                                                            identify key components.
    risk of overdiagnosis.
                                                          • Extract all meaningful entities (nouns,
  • S7 → Evaluated trade-offs include
                                                            noun phrases, proper nouns, concepts).
    complications from lung-cancer
    screening.                                            • Identify relationships between entities
                                                            based on verbs, prepositions, and se-
Example 2:                                                  mantic meaning.
Input:                                                    • Form triples (Entity 1 → Relationship
   “Although warmed by the sun, the                         → Entity 2) as structured relationships.
   fields remained dry, and farmers                       • Validate that each triple captures mean-
   worried about the drought.”                              ingful semantic information.
Chain-of-Thought:
                                                       Examples:
 1. Dependent Clause (DC): “Although
    warmed by the sun”                                      "Regulating miR-497-5p pro-
                                                            vides a potential targeted therapy
 2. Independent Clause 1 (IC1): “The                        for lung cancer treatment."
    fields remained dry”
                                                       [{
 3. Independent Clause 2 (IC2): “Farmers               "Entity 1": "regulating miR-497-5p",
    worried about the drought”                         "Entity 2": "lung cancer targeted
                                                       treatment",
                                                       "Relationship": "provides"
 4. Rewrite into standalone simple sen-                }]
    tences.

                                               15537
      "The activation of caspase sig-
      nal pathway was the reason for
      stronger apoptosis."

 [{
 "Entity 1": "activation of caspase
 signal pathway",
 "Entity 2": "stronger apoptosis",
 "Relationship": "was the reason for"
 }]


     "With clinical significance fea-
     tures selection, over-sampling
     methods achieved the highest
     AUC results."

 [{
 "Entity 1": "clinical significance
 features selection",
 "Entity 2": "over-sampling methods",
 "Relationship": "With"},
 {"Entity 1": "over-sampling methods",
 "Entity 2": "highest AUC results",
 "Relationship": "achieved"}]

 Now extract knowledge graph relationships
 from this sentence: {sentence}

All other prompt types are in our code base.




                                               15538
Figure 3: Subsection of the Knowledge Graph.




                   15539
