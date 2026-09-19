                     Rethinking the Role of LLMs for Document-level Relation Extraction:
                           a Refiner with Task Distribution and Probability Fusion

                             Fu Zhang† , Xinlong Jin† , Jingwei Cheng∗ , Hongsen Yu, Huangming Xu
                            School of Computer Science and Engineering, Northeastern University, China
                             {zhangfu,chengjingwei}@mail.neu.edu.cn; drasick59596@163.com



                                        Abstract

                      Document-level relation extraction (DocRE)
                      provides a broad context for extracting one
                      or more relations for each entity pair. Large
                      language models (LLMs) have made great
                      progress in relation extraction tasks. How-
                      ever, one of the main challenges we face is that
                      LLMs have difficulty in multi-label relation
                      prediction tasks. Additionally, another note-
                      worthy challenge and discovery we reveal: the
                      small language models (SLMs) for DocRE tend
                      to classify existing relations as “no relation"         Figure 1: A DocRE example: predicting the existence
                      (NA), while LLMs tend to predict existing re-           of one or more relations, or no relations for entity pairs.
                      lations for all entity pairs. To address these
                      challenges, we propose a novel method that
                      utilizes LLMs as a refiner, employing task dis-
                      tribution and probability fusion. The task dis-         on predicting a relation between two entities men-
                      tribution we carefully designed aims to distin-         tioned in a single sentence which is called sentence-
                      guish hard and easy tasks, and feed hard tasks          level relation extraction. By contrast, document-
                      to our LLMs-based framework to reevaluate               level relation extraction (DocRE) (Yao et al., 2019)
                      and refine. Further, in order to effectively solve      offers a broader context for analysis and poses
                      the multi-label relation prediction problem in          greater challenges, as it involves identifying one
                      the refinement process, we propose a proba-
                                                                              or more relations for entities that span multiple
                      bility fusion method, ensuring and enhancing
                      fusion predictions by maintaining a balance             sentences or paragraphs as illustrated in Figure 1.
                      between SLMs and LLMs. Extensive exper-                    Recent advancements in DocRE focus on the use
                      iments on widely-used datasets demonstrate              of neural models for sequence-based, graph-based,
                      that our method outperforms existing LLM-               and transformer-based approaches (Delaunay et al.,
                      based methods without fine-tuning by an av-
                                                                              2023). Meanwhile, large language models (LLMs)
                      erage of 25.2% F1. Refining SLMs using our
                      method consistently boosts the performance of
                                                                              have achieved significant success in a wide range of
                      the SLMs, achieving new state-of-the-art re-            natural language processing tasks, leveraging emer-
                      sults compared to existing SLMs and LLMs1 .             gent capabilities of reasoning and in-context learn-
                                                                              ing across diverse domains such as commonsense
               1      Introduction                                            reasoning and open-domain question answering
                                                                              (Yu et al., 2023; Sun et al., 2023). Recent studies
               Relation extraction (RE) is the task of extracting se-         have utilized LLMs for relation extraction tasks,
               mantic relations among entities within a given text,           including few-shot RE (Wei et al., 2023; Xu et al.,
               which has abundant applications such as knowl-                 2023b; Ma et al., 2023b) and sentence-level RE
               edge graph construction, question answering, and               (Wadhwa et al., 2023; Zhang et al., 2023). Meth-
               text analysis (Vaswani et al., 2017; Distiawan et al.,         ods that involve designing prompts and fine-tuning
               2019; Shi et al., 2019). Prior studies mostly focus            LLMs for specific tasks have been shown to out-
               1
                   Our code: https://github.com/Drasick/Drell.                perform fine-tuned small language models (SLMs)
                   †
                     Equal contribution. ∗ Corresponding author.              in several relation extraction domains (Gutierrez
                                                                           6293
Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies
                                                        (Volume 1: Long Papers), pages 6293–6312
                                         April 29 - May 4, 2025 ©2025 Association for Computational Linguistics
et al., 2022; Xu et al., 2023a).                                 effective, saving both time and cost without
   However, there are few studies on applying                    requiring additional fine-tuning2 .
LLMs to DocRE, including PromptRE (Gao et al.,
2023) and DocGNRE (Li et al., 2023) without             2       Related Work
fine-tuning, and AutoRE (Lilong et al., 2024) with
                                                        DocRE involves extracting relations between en-
fine-tuning. Building on our in-depth research, we
                                                        tities within a document. Let the document
identify several key focal points that warrant at-
                                                        be defined as D, composed of N sentences
tention and resolution: (i) LLMs struggle to han-
                                                        {sn }N n=1 . We need to combine all mentioned
dle datasets with a large number of negative
                                                        entities {eq }E  q=1 pairwise to form entity pairs
samples effectively. Our research reveals that the
                                                        (eh , et )h,t∈{1,2,...,E};h̸=t , where h represents the
limited predictive performance of SLMs owing to
                                                        head entity and t represents the tail entity. The task
its tendency to classify existing relations as “no
                                                        is to predict the relation r for this entity pair, where
relation” (NA), while LLMs tend to predict exist-
                                                        r belongs to the pre-defined set {ri }R    i=1 ∪ {NA}.
ing relations for all entity pairs. (ii) LLMs, which
                                                        For the entity pair (eh , et ), we define its probability
are essentially generative models, struggle with
                                                        relative to relation r as P (r). Therefore, we define
multi-label relation prediction tasks. When mul-
                                                        the probability distribution F (h,t) of the entity pair
tiple relations need to be predicted for an entity
                                                        (eh , et ) as:
pair, the answer generated by LLMs may not ex-
actly match the relation labels. Even when LLMs                   F(h,t) = {P (r)|r ∈ {ri }R                         (1)
                                                                                           i=1 ∪ {NA}}
are employed to choose from multiple labels, they
prefer to choose a single label.                        2.1      DocRE based on SLMs
   To address these points, aimed at exploratory of
                                                        The utilization of SLMs for DocRE can be roughly
the ability differed between SLMs and LLMs for
                                                        categorized into sequence-based (e.g. CNN (Yao
DocRE, we propose a method that utilizes LLMs
                                                        et al., 2019), BiLSTM (Yao et al., 2019)), graph-
as a refiner, employing task distribution and proba-
                                                        based (e.g. GAIN (Zeng et al., 2020), SIRE (Zeng
bility fusion to complete the DocRE task. The task
                                                        et al., 2021)), and transformer-based. We prefer
distribution initially uses SLMs to address the issue
                                                        transformer-based models with the same underly-
of a large number of negative samples and subse-
                                                        ing architecture as LLMs, so we select several re-
quently uses LLMs as a refiner to resolve tasks
                                                        cently representative and competitive transformer-
that are difficult for SLMs. The probability fusion
                                                        based models: ATLOP (Zhou et al., 2021), Eider
provides a stable solution to the multi-label classi-
                                                        (Xie et al., 2022), DREEAM (Ma et al., 2023a), and
fication problem, ensuring that the predictions are
                                                        AA (Lu et al., 2023) as SLMs baselines for sub-
not overly dependent on either the SLMs or the
                                                        sequent experiments. Descriptions of these SLMs
LLMs. Our contributions are as follows:
                                                        can be found in the review work (Delaunay et al.,
   • We explore the performance of SLMs and             2023) in details.
     LLMs in DocRE and reveal several notewor-             Transformer-based models utilize pre-trained
     thy findings.                                      models (such as BERT (Devlin et al., 2019),
                                                        RoBERTa (Liu et al., 2019)) to produce probability
   • We propose a task distribution method that         predictions for relations in R∪{NA}. By using the
     allows LLMs, acting as a refiner, to effec-        predicted value for the NA relation as a threshold,
     tively assist in DocRE tasks that are difficult    and considering all relations with values greater
     for SLMs.                                          than this threshold as the predicted relation labels
                                                        for (eh , et ), we define this process as:
   • We innovatively propose a probability fusion
     method for multi-label classification, balanc-                                      k
                                                                                         X (h,t)         (h,t)
     ing and enhancing the predictions made by              Pslm (r|eh , et ; D) = σ(          zh   Wri zt       + br )
     both SLMs and LLMs.                                                                 i=1
                                                                                                     (2)
                                                                (h,t)      (h,t)
   • Experiments on widely-used DocRE datasets          where zh      and zt     are the embeddings with
     demonstrate that using LLMs as a refiner           the in-context information for the head and tail
     can consistently enhance the performance           2
                                                            All LLMs discussed in this paper are not fine-tuned, and the
     of SLMs. Moreover, our method is cost-                 results for LLMs are based on default weights or APIs.

                                                    6294
entities, Wri and br is the weight matrix and bias                                  DocRED         Re-DocRED
for relation r, σ is the activation. For simplicity,              # Train               3053              3053
Pslm (r|eh , et ; D) is abbreviate as Pslm (r). Hence,            # Dev                 1000               500
                                 (h,t)
the probability distribution Fslm is denoted as :                 # Test                1000               500
                                                                  Triples             50,503           120,664
          (h,t)
        Fslm = {Pslm (r)|r ∈ R ∪ {NA}}               (3)          Relation types          96                96

The final probability set P (h,t) of relation labels for    Table 1: Dataset statistics. # indicates document count.
the entity pair (eh , et ) is defined as:

      P(h,t) = {Pslm |Pslm (r) > Pslm (NA)}          (4)

This approach is widely used in SLMs and com-
pletes the multi-label classification for DocRE.

2.2 DocRE based on LLMs
Currently, there are few LLMs-based models eval-
uated on document-level datasets. Two notable
studies without fine-tuning conducted on DocRE
are discussed: PromptRE (Gao et al., 2023) and              Figure 2: Comparison of relation prediction accuracy.
DocGNRE (Li et al., 2023). PromptRE combines
diverse prompting, integrating label distribution           3.1    Experiment Setup
and entity types to enhance DocRE. DocGNRE
autonomously generates relation triples to apply            Datasets We evaluate on widely-adopted datasets
LLMs for dataset augmentation, instead of using             for DocRE, including DocRED (Yao et al., 2019)
LLMs for DocRE.                                             and Re-DocRED (Tan et al., 2022) in Table 1. Re-
   The smallest unit of LLM generation is defined           DocRED improves annotation labels upon the pop-
as a token wi , its probability among all tokens in         ular DocRED dataset.
the vocabulary V can be positioned according to             Implementation Settings Regarding the LLMs
the previous token sequence [w1 , . . . , wi−1 ] (com-      in E1 , we apply GPT-3.5-turbo-0613 following
monly called as prompt) as follows:                         PromptRE and DocGNRE, and comparing on Re-
                                                            DocRED. In E2 , we employ the SLM ATLOP
             Pllm (wi |w1 , w2 , . . . , wi−1 )      (5)
                                                            (Zhou et al., 2021) and compare it with our LLM-
         Fllm (wi ) = {Pllm (wj )|wj ∈ V }           (6)    based method on DocRED.
where Fllm (wi ) is denoted as the probability distri-      Evaluation Metric We adopt F1 and Ign F1 as
bution of the output token. The LLM-based meth-             the evaluation metrics for DocRE, as established
ods typically output tokens until the end. Subse-           in previous research. Ign F1 disregards triples that
quently, the output [wi , wi+1 . . . , wend ] is regular-   are present in training set.
ized to predict multi-relation labels.                         For each document, we will consider relation
                                                            accuracy (defined as the proportion of correctly
3   Exploratory Analysis of SLMs and                        predicted labels among all predicted labels), and
    LLMs for DocRE                                          the prediction of NA_but_relation (defined as the
To investigate the efficacy of LLMs in DocRE and            number of entity pairs with relation labels predicted
explore their performance distinctions with SLMs,           as NA).
we devise two experiments: (E1 ) Similar to previ-
                                                            3.2 E1 : Observation of LLMs on DocRE
ous approaches based on LLMs, we mainly utilize
prompts to predict relation labels, aiming to ini-          To validate the inherent capability of LLMs in
tially assess the performance of LLMs in DocRE.             DocRE, we design a document-fitting prompt tem-
(E2 ) We conduct an in-depth comparative analysis           plate (as will be introduced in Section 4.1 and Fig-
of relation labels predicted by LLMs and SLMs to            ure 4)3 to test each entity pair. This prompt is then
insight the potential capabilities of LLMs in DocRE         3
                                                            All prompt templates mentioned in this paper can be found in
and uncover any limitations in SLMs.                        Appendix A.

                                                        6295
                                                             Model                              F1    Ign F1
                                                             PromptRE (Gao et al., 2023)∗    10.56      9.04
                                                             DocGNRE (Li et al., 2023)∗      11.73         -
                                                             Only-LLM (Ours)                 22.95     22.36

                                                         Table 2: Performance of our LLM prompting-based
                                                         method. Results of ∗ are from their original papers.


                                                         3.4     Discussion: Why not leverage LLMs to
Figure 3: Comparison of predicting NA_but_relation.              refine predictions by SLMs?
                                                         Based on Sections 3.2 and 3.3, the following is-
                                                         sues arise: The potential of LLMs in DocRE still
fed to GPT-3.5 for querying and reasoning to derive
                                                         has significant room for exploration. And, leverag-
the predicted relation labels.
                                                         ing characteristics of LLMs and SLMs to jointly
   Table 2 shows that our method achieves improve-       enhance DocRE may be valuable.
ments over the baselines, but the performance of            Hence, our main ideas are as follows: First,
directly using LLM prompts in DocRE is still un-         we propose a task distribution method that allows
satisfactory, as well as other LLMs-based methods.       LLMs as a refiner, to effectively assist SLMs. This
This suggests that the direct utilization of prompts     involves identifying tasks where the SLMs discard
currently may not leverage the reasoning capabili-       many relations as NA, which should be refined
ties inherent in LLMs like other domains.                by the LLMs. Second, even with LLMs assist-
                                                         ing SLMs, LLMs themselves still struggle with
                                                         multi-label relation prediction tasks. We need to de-
3.3   E2 : Tendency Analysis of SLMs & LLMs
                                                         sign a method tailored for multi-label classification.
                                                         Moreover, to further enhance fusion predictions by
To investigate why direct use of prompts does not
                                                         maintaining a balance between SLMs and LLMs,
achieve the same or better effectiveness as SLMs,
                                                         we propose a probability balance fusion method.
we conduct experiments on each document, an-
alyze the experimental results in depth, and ran-
                                                         4     Methodology
domly select 100 documents for visualization. The
experiments’ outcomes are illustrated in Figures         Our refining method consists of Task Distribution
2 and 3: (1) The performance of the SLM is con-          and Self-supervised Probability Fusion as shown
strained because, while it ensures high predic-          in Figure 4.
tion relation accuracy, it discards many rela-
tions as NA. Our in-depth analysis reveals that the      4.1     Task Distribution
high performance of SLMs is largely attributed to        As mentioned in Section 2, when given an en-
their higher prediction accuracy. Additionally, an       tity pair (eh , et ) to be predicted in a document D,
interesting finding emerges from Figure 3: among         SLMs can derive the probability Pslm (r) and dis-
the results of SLM prediction errors, there is a         tribution Fslm using Eq. (2) and (3).
tendency to predict entity pairs that have relation
                                                            For the tendency as discussed in Section 3.3,
labels as NA, thereby limiting their performance.
                                                         we posit that when Pslm (NA) is greater than
(2) The diminished efficacy of LLMs stems from
                                                         Pslm (r)r∈R , but there is at least one Pslm (r) in
their tendency to predict existence labels for
                                                         close proximity to Pslm (NA), the task is defined
many NA relations. In Figure 2, compared to
                                                         as Hard. We define the close proximity as γ(h,t) .
the SLM, the proportion of correct predictions by
                                                         In light of this, we introduce a partition method
LLMs, excluding NA, is relatively low. Further,
                                                         to distinguish between easy and hard tasks based
Figure 3 illustrates that LLMs tend to predict exis-
                                                         on Pslm (NA) as the threshold. The formulation of
tence labels for many NA relations. This over-
                                                         this method is defined:
prediction phenomenon has also been observed
in the few-shot relation extraction GPT-RE (Wan                     Pslm (NA) − max(Pslm (r))
et al., 2023). We suspect that LLMs may rely on          γ(h,t) =                             , γ(h,t) ≤ δ
                                                                            Pslm (NA)
their own prior knowledge to make predictions.                                                          (7)
                                                      6296
Figure 4: Illustration for our LLMs-based refiner framework. The task distribution is to distinguish hard and easy
tasks, and feed hard tasks to refine (Step ①). The question templates converted from the top-k relations of SLMs
and the document are together fed into the LLM, to get the distribution Fllmk of LLM’s relation predictions (Step
②). With the probability balance fusion of two distributions Fslmk and Fllmk and self-supervision, we obtain the
final relation predictions (Step ③).


where γ(h,t) is denoted as the threshold, and δ is a       the LLM needs to reason about and answer. The
parameter, controlling the range scaling of the task       question includes the top-k relation probability la-
difficulty. If the task is regarded as hard, it will be    bels (obtained by Fslmk , where k indicates only
distributed to the LLM for refinement on the basis         the top-k Pslm (r) are retained), filled with head
of the distribution Fslm .                                 and tail entities, and converted into a question tem-
   The idea of distinguishing hard tasks is similarly      plate designed by ourselves. After our prompt is
proposed by Ma et al. (2023b). Inspired by their           input into the LLM, unlike the regularization-based
work but differing from theirs, first, they focus on       multi-label classification methods mentioned at the
few-shot RE task (where a relation between two             end of Section 2.2, we focus only on the proba-
entities is predicted from a single sentence), and         bility distribution Fllm of the first output token
use a fixed threshold to give all relation prediction      (i.e., Eq. (6)). We then identify all tokens in this
scores. We propose the idea of dynamic thresh-             distribution Fllm that match our multiple-choice
old above, which allows us to adjust the threshold         options. The probability values of these tokens are
adaptively according to the predicted score of each        taken as the final distribution Fllmk of the LLM’s
entity pair. Second, our work after task distribution      relation label predictions.
is completely different from theirs, we for the first         Now, we have two relation probability distribu-
time propose the idea and method of balancing and          tions for the entity pair (eh , et ): Fslmk and Fllmk .
fusing the probability distributions of LLMs and
SLMs to achieve multi-label classification tasks.          4.2      Self-supervised Probability Fusion
   After distributing the hard tasks to the LLM,          Probability Fusion A challenge arises because
considering that the LLM still struggles with multi-      Fslmk is obtained through the prediction results
label relation prediction tasks, we carefully design      of all relations, while Fllmk is based on the token
a document-fitting prompt template consisting of          probability distribution of the entire vocabulary.
Instruction + Document + Question + Answer: In-           Although we can add the probabilities from the
struction informs the LLM that it needs to under-         two distributions one by one, intuitively, the two
take a multiple-choice task. Document contains            distributions are unbalanced4 .
the document information where (eh , et ) belongs          4
                                                               We also demonstrate in subsequent experiments of Section
to. The most important part, Question, is what                 5.3 that direct addition is unreasonable.

                                                      6297
   Therefore, we propose a probability balance es-        5     Experiments
timation. Through this estimation, we aim to make
                                                          5.1    Experimental Setup and Baselines
the distributions of the two models balance, which
helps the final prediction with the judgment of both      Datasets As detailed in Section 3.1, we evalu-
Fslmk and Fllmk .                                         ate our methods on two widely-adopted DocRE
   Considering that the probability distribution          datasets DocRED and Re-DocRED, and report Pre-
Fllmk is calculated based on the softmax function         cision, Recall, F1 and Ign F1.
including the temperature parameter τ , which trans-
                                                          Baselines As detailed in Section 2, we choose
forms the logits (log-odds) zi for each token into a
                                                          two earlier SLMs CNN and BiLSTM that are often
distribution over the vocabulary V :
                                                          compared as baselines.
                                         zi
                                                             As our main baselines, transformer-based mod-
                                       eτ
              Pllm (ri |zi , τ ) = P          z
                                              j
                                                   (8)    els, which have the same underlying architecture
                                       V
                                       j=1 e τ            as LLMs, we select four representative state-of-the-
                                                          art SLMs as the refining models, including ATLOP,
thus, we adjust the distribution of next token output
                                                          Eider, DREEAM, and AA. We also compare with
in LLMs by τ , which helps control the diversity of
                                                          the existing LLMs-based DocRE methods without
generated text.
                                                          fine-tuning, PromptRE and DocGNRE.
   As the hard task we defined in Eq. (7), we clas-
sify tasks where the probabilities Pslm (NA) and          Implementation Details We set parameters, in-
Pslm (r) are close as hard tasks. This closeness may      cluding AdamW (Loshchilov and Hutter, 2019),
affect the variance of the distribution Fslmk . At the    warmup (Goyal et al., 2017), and learning rates
same time, as shown in Eq. (8), adjusting τ can also      as origin for SLMs to train and obtain Fslmk . In
change the variance of Fllmk (we denote the ad-           all subsequent experiments, we test on an A100
justed Fllmk as Fllm
                   τ
                      k ), where we observe that as τ     with 40GB GPU, use LLaMA3-8B (Touvron et al.,
increases, the variance will decrease. The variances      2023) as the LLM, set τ to 1.8, top-k to 4, ξ to 0.03,
σslm and σllm are defined based on their respective       and δ to 0.6/0.5 for two datasets.
distributions Fslmk and Fllm τ
                                k . Therefore, we pro-       Moreover, we conduct experiments without
pose to further determine whether Fslmk and Fllm  τ
                                                     k    SLMs (i.e., Only-LLM as also mentioned in Sec-
are balanced by calculating the difference between        tion 3.2), and without the task distribution and prob-
their variances with a threshold ξ:                       ability fusion (denote as Refiner¬T D&P F , which
                                  τ
                                                          integrates the top-k relations predicted by SLM AT-
         |σslm (Fslmk ) − σllm (Fllmk )| ≤ ξ       (9)    LOP for each entity pair into the LLM’s prompt to
                                                          obtain the final prediction result).
After estimating probability balance, we combine
Pslm (ri ) and Pllm (ri ) in Fslmk and Fllm
                                         τ
                                           k:             5.2    Main Results
 Fref = {Pslm (ri ) + Pllm (ri )|i = 1, . . . , k} (10)   Table 3 and 4 report the performance of our method
                                                          and existing methods on two DocRE datasets.
   According to the Pref (r) in Fref , we finally ob-        (1) Compared with LLMs and earlier SLMs:
tain the refined relation prediction probability set      In Table 3, our Only-LLM method demonstrates
  (h,t)
Pref of the entity pair (eh , et ):                       more competitive than previous LLM-based meth-
                                                          ods, and obtains improvements of 10.95 and 21.36
      (h,t)
    Pref = {Pref |Pref (r) > Pref (NA)}           (11)    F1 on two datasets. Further, Refiner¬T D&P F
                                                          achieves an average F1 improvement of 17.99 on
Self-supervision To further enhance the accuracy          DocRED and 32.49 on Re-DocRED. However, the
of the probability fusion results, we utilize LLM’s       performance of LLM-based methods is still unsat-
self-supervision to enable it to make a second judg-      isfactory even compared to the earlier SLMs, as
                                                  (h,t)
ment on the document and each relation in Pref .          there remains a gap in F1 scores, as shown in Table
The method involves posing the question again,            3 and Table 4, which suggests that the refinement
asking whether there is a relation. We judge based        may be necessary and valuable.
on the probability distribution of the first token out-      (2) Compared SLMs with our Refiner: The
put by the LLM, focusing on the token predictions         F1 and Ign F1 scores of SLMs with our Refiner
of “T" (true) and “F" (false) as showed in Figure 4.      show consistent improvement across the DocRED
                                                      6298
     Model                                         DocRED                                 Re-DocRED
                                           P         R    F1         Ign F1         P        R    F1         Ign F1
     PromptRE (Gao et al., 2023)           -          -    -              -      6.56    27.00 10.56           9.04
     DocGNRE (Li et al., 2023)         14.61       9.8 11.73              -     24.45     5.77  9.33              -
     Only-LLM (Ours)                   23.63     21.81 22.68          21.37     43.83    25.11 31.92          30.48
     Refiner¬T D&P F (Ours)            37.14     24.77 29.72          28.55     63.33    31.89 42.43          41.67

Table 3: Evaluation results of our method based mainly on LLM, with best scores bold. The results of LLMs-based
PromptRE and DocGNRE are from their original papers.

       Model                                       DocRED                                Re-DocRED
                                           Dev                Test                 Dev                Test
                                      Ign F1       F1    Ign F1        F1     Ign F1       F1    Ign F1       F1
       (a) Earlier SLMs
       CNN (Yao et al., 2019)          41.58     43.45    40.33      42.26         -         -        -         -
       BiLSTM (Yao et al., 2019)       48.87     50.94    48.78      51.06         -         -        -         -
       (b) SLMs with Refiner
       ATLOP (Zhou et al., 2021)       59.11     61.01    59.31    61.30       76.79     77.46    76.82    77.56
          ATLOP +Refiner              ↑59.42    ↑61.31   ↑59.55   ↑61.62      ↑77.48    ↑78.05   ↑77.32   ↑78.09
       Eider (Xie et al., 2022)        60.51     62.48    60.42    62.47      †75.91    †76.99   †76.25   †77.13
          Eider +Refiner              ↑60.81    ↑62.90   ↑60.71   ↑62.88      ↑76.43    ↑77.52   ↑76.84   ↑77.61
       DREEAM (Ma et al., 2023a)       63.47     65.30    63.31    65.30      †79.51    †80.66    79.66    80.73
          DREEAM +Refiner             ↑63.71    ↑65.69   ↑63.47   ↑65.82      ↑80.62    ↑81.58   ↑80.45   ↑81.69
       AA (Lu et al., 2023)            61.31     63.38    60.84    63.10       80.04     81.15    80.12    81.20
          AA +Refiner                 ↑61.82    ↑63.88   ↑61.46   ↑63.67      ↑80.92    ↑82.01   ↑80.93   ↑82.03

Table 4: Evaluation results of SLMs and refined with Refiner on DocRED and Re-DocRED, with best scores bold,
and the effect of refine is indicated by arrows. The results of SLMs are from their original papers while others with
† are obtained by our reproduction.


   Model                          Ign F1        F1          results in a dramatic decline in performance (35.13
   Refiner +DREEAM                 80.45       81.69        and 34.91 drop in terms of Ign F1 and F1). It
   w/o self-supervision            80.35       81.48        indicates that filtering a large number of negative
   w probability fusionadd         79.57       80.01        samples (i.e., NA) further enhances the LLM’s at-
   w/o probability fusion          78.51       79.24        tention to the predicted relation labels.
   w/o task distribution           45.32       46.78           (2) The use of balanced probability fusion in-
                                                            deed improves performance. Removing it (i.e.,
Table 5: Ablation study on Re-DocRED test set. “w           w/o probability fusion), F1 score decrease by 2.45.
probability fusionadd ” means that we simply add the        Also, instead of our probability balance fusion
probabilities.                                              method, we simply add the probability distribu-
                                                            tions Fslmk and Fllm τ
                                                                                   k together, which also leads

and Re-DocRED datasets (average improvement                 to the decline of 1.68 F1. All of these highlight the
of 0.4∼0.9 F1). Notably, the F1 scores of the               effectiveness of probability balance fusion.
DREEAM and AA refiner achieve new state-of-                    (3) Self-supervision technique brings a slight
the-art performance compared with the existing              improvement by guiding LLMs to confirm whether
SLMs, indicating that our refiner is effective for          the refined labels are correct, thereby enhancing
enhancing the performance of DocRE tasks.                   the prediction accuracy.

                                                            5.4      Impact of Probability Balance Estimation
5.3 Ablation Study
                                                            In order to further verify the impact of whether two
We investigate the effectiveness of modules in Re-          distributions are balanced on model performance,
finer by removing them in turn. We show our re-             we visualize the distribution difference of variances
sults in Table 5:                                           σllm and σslm for each entity pair at τ = 1.1 and
   (1) Task distribution is extremely effective for         τ = 1.8 in Figure 5. The results show that the
the use of LLMs in refined method. Removing it              variance difference of the LLM and SLM are more
                                                         6299
                                                           Figure 7: The impact of δ on F1 score, the number
Figure 5: Visualization of balanced and imbalanced         of correctly and incorrectly refined entity pairs (#Ref
distributions on the Re-DocRED dataset.                    correct and #Ref error) for SLM ATLOP on DocRED.

                                                               Experiment                      Time(Hour)   Cost(US$)
                                                               (a) without task distribution
                                                                  Only-LLM                          24.21      100.91
                                                                  Refiner¬T D&P F                   28.69           -
                                                                  Self-supervision prompting        28.77           -
                                                               (b) with task distribution
                                                                  Refiner                            1.21           -
                                                                  Self-supervision prompting         1.07           -

                                                           Table 7: Analyze average time and cost of reasoning
                                                           using different LLMs and templates with/without task
                                                           distribution on DocRED and Re-DocRED. We use GPT-
Figure 6: The impact of τ on F1 and Ign F1 scores for
                                                           3.5-turbo-0613 for Only-LLM Template as mentioned
refined SLM ATLOP on Re-DocRED dataset.
                                                           in Section 3.1 and LLaMA3-8B for other Templates as
                                                           mentioned in Section 5.1.
discrete when τ is 1.1. As τ increases, the vari-
ance difference tend to converge. Further, Figure 6        5.6     Cost Analysis and Case Study
demonstrates that as τ increases, F1 and Ign F1 ex-
hibit an increasing trend followed by a decreasing         Table 7 shows our method’s efficiency. Without
trend. These indicate that better balance between          task distribution, inference time increases signif-
the SLM and LLM prediction distributions indeed            icantly due to the presence of a large number of
helps improve prediction outcomes.                         negative samples (i.e., NA). Applying task distribu-
                                                           tion drastically reduces this time, highlighting the
5.5   Impact of δ and top-k in Task Distribution           efficiency gains. For deep cost-benefit analysis, we
                                                           provide a more detailed explanation of the compu-
δ is denoted as the hard task threshold in Eq. (7).
                                                           tational requirements of our refiner framework in
As δ increases, tasks where Pslm (r) is much lower
                                                           Appendix D.
than Pslm (N A) will also be considered hard tasks.
                                                              Several interesting case studies illustrate the dif-
That is, the LLM will handle more tasks. Conse-
                                                           ference in probability fusion before and after bal-
quently, the number of correctly and incorrectly
                                                           ance adjustment. Due to space limitations, cases
refined entity pairs increases, as shown in Figure 7.
                                                           are provided in Appendix E.
   We also evaluate the top-k results in Table 6,
which show that, the proportion of the first k out-
put labels containing the correct relation labels, in-         Proportion(%)              DocRED       Re-DocRED
creases significantly from top-1 to top-3 but more             #train NA                     96.81           92.80
modest for top-4 and top-5.                                    #dev NA                       96.90           91.06
                                                               #test Predicted as NA         33.02           24.20
                                                               #test Ign F1                  63.48           79.63
            top-1    top-2    top-3    top-4    top-5          #test F1                      65.31           80.71
  Hit(%)    72.51    89.02    93.69    95.47    96.98
                                                           Table 8: Effect of proportion of NA in different dataset
   Table 6: The impact of top-k in task distribution.      on SLM DREEAM.

                                                        6300
    Case                                                               True Label            LLM Prediction
    Robert Kingsbury Huntington (13 March 1921 – 5 June
    1942), was a naval aircrewman and member of Torpedo
    Squadron 8 (or VT-8). ... Huntington was one of 29 from
    Torpedo Squadron 8 who gave their lives in this attack.
    ##QUESTION: Which of the following is right?
                                                                            D                      A
    A. Battle of Midway(MISC) isn’t an administrative entity,
    and Battle of Midway(MISC) was a physical object or event
    in Japanese(LOC).
    ...
    D. None of the above options is correct.
    "More" is a song by The Sisters of Mercy, from their album
    Vision Thing. ... The song has also been re-recorded by Meat
    Loaf for his 2016 album Braver Than We Are.
    ##QUESTION: Which of the following is right?
    ...                                                                     D                      B
    B. MTV(MISC) is a production company, and Wuthering
    Heights(MISC) was produced by MTV(MISC).
    ...
    D. None of the above options is correct.

                     Table 9: Case study of LLM predictions conflicting with ground truth.


5.7 Relationship Validation between Dataset              and LLMs methods and achieves state-of-the-art
    Characteristics and Model Behaviors                  performance. Our methods do not require retrain-
To enhance rigor of our validation between dataset       ing of LLMs and can be flexibly integrated with
characteristics and model behaviors, we provide          various SLM approaches on DocRE, incurring an
additional examining experiments.                        acceptable cost, while significantly enhancing the
   In Table 8, we observe that compared to Do-           performance of all SLMs’ original capabilities.
cRED, the proportion of NA in the Re-DocRED
train set decreases by 4.01. With consistent NA pro-     Limitations
portions in train/dev sets, the same model structure,    We do not test our refiner on the graph-based mod-
DREEAM, shows an 8.82 reduction in incorrectly           els considering of LLMs and transformer-based
predicting relations as NA. Performance improve-         SLMs have the same architecture. Moreover, we
ment is also observed, suggesting that SLM NA            conduct experiments on LLMs (LLaMA3-8B for
prediction bias is related to label distribution.        our refiner) without larger model parameters, and
   Table 9 shows cases about LLM predictions con-        we may not have measured the upper limit of
flicting with ground truth. From the cases, we ob-       refiner. In addition, when we adjust the threshold δ,
serve that when certain information is not explicitly    we increase both the number of correctly predicted
presented or requires additional evidence, the LLM       entity pairs and the number of incorrectly predicted
prefers relation-indicating options over option D.       entity pairs by LLM. The incorrectly results may
The complete cases can be found in Appendix F.           also affect the final prediction performance. In
                                                         our future work, we will conduct more extensive
6     Conclusion
                                                         research on graph-based SLMs, and try to select
In this paper, we present a noteworthy finding that      models with larger parameters for refinement.
LLMs tend to predict the existence of relations for
all entity pairs, while SLMs often classify existing     Acknowledgments. The authors thank reviewers
relations as NA. We propose a novel method utiliz-       for their valuable comments. The work is supported
ing LLMs as refiners, employing task distribution        by National Natural Science Foundation of China
and self-supervised probability fusion. Our refiner      (62276057), and Sponsored by CAAI-MindSpore
demonstrates significant improvement over SLMs           Open Fund, developed on OpenI Community.
                                                     6301
References                                                 Youmi Ma, An Wang, and Naoaki Okazaki. 2023a.
                                                             Dreeam: Guiding attention with evidence for im-
Julien Delaunay, Hanh Thi Hong Tran, Carlos-Emiliano         proving document-level relation extraction. The 17th
   González-Gallardo, Georgeta Bordea, Nicolas Sidere,       Conference of the European Chapter of the Associa-
   and Antoine Doucet. 2023. A comprehensive survey          tion for Computational Linguistics (EACL).
   of document-level relation extraction (2016-2023).
   arXiv e-prints, pages arXiv–2309.                       Yubo Ma, Yixin Cao, Yong Ching Hong, and Aixin Sun.
                                                             2023b. Large language model is not a good few-
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
                                                             shot information extractor, but a good reranker for
   Kristina Toutanova. 2019. Bert: Pre-training of deep
                                                             hard samples! In The 2023 Conference on Empirical
   bidirectional transformers for language understand-
                                                             Methods in Natural Language Processing (EMNLP).
   ing. In Proceedings of the 2019 Conference of the
  North American Chapter of the Association for Com-
                                                           Yu Shi, Jiaming Shen, Yuchen Li, Naijing Zhang, Xin-
   putational Linguistics (NAACL).
                                                             wei He, Zhengzhi Lou, Qi Zhu, Matthew Walker,
Bayu Distiawan, Gerhard Weikum, Jianzhong Qi, and            Myunghwan Kim, and Jiawei Han. 2019. Discover-
  Rui Zhang. 2019. Neural relation extraction for            ing hypernymy in text-rich heterogeneous informa-
  knowledge base enrichment. In Proceedings of the           tion network by exploiting context granularity. In
  57th Annual Meeting of the Association for Compu-          Proceedings of the 28th ACM International Confer-
  tational Linguistics (ACL), pages 229–240.                 ence on Information and Knowledge Management
                                                             (CIKM), pages 599–608.
Chufan Gao, Xulin Fan, Jimeng Sun, and Xuan Wang.
  2023. Promptre: Weakly-supervised document-level         Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and
  relation extraction via prompting-based data program-      Denny Zhou. 2023. Recitation-augmented language
  ming. arXiv preprint arXiv:2310.09265.                     models. In International Conference on Learning
                                                             Representations (ICLR).
Priya Goyal, Piotr Dollár, Ross Girshick, Pieter No-
   ordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew         Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and
   Tulloch, Yangqing Jia, and Kaiming He. 2017. Ac-          Sharifah Mahani Aljunied. 2022. Revisiting docred-
   curate, large minibatch sgd: Training imagenet in 1       addressing the false negative problem in relation ex-
   hour. arXiv preprint arXiv:1706.02677.                    traction. In Proceedings of the 2022 Conference on
                                                             Empirical Methods in Natural Language Processing
Bernal Jimenez Gutierrez, Nikolas McNeal, Clay Wash-         (EMNLP), pages 8472–8487.
  ington, You Chen, Lang Li, Huan Sun, and Yu Su.
  2022. Thinking about gpt-3 in-context learning for       Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
  biomedical ie? think again. Proceedings of the 60th        Martinet, Marie-Anne Lachaux, Timothée Lacroix,
  Annual Meeting of the Association for Computational        Baptiste Rozière, Naman Goyal, Eric Hambro,
  Linguistics (ACL).                                         Faisal Azhar, et al. 2023. Llama: Open and effi-
                                                             cient foundation language models. arXiv preprint
Junpeng Li, Zixia Jia, and Zilong Zheng. 2023. Semi-         arXiv:2302.13971.
  automatic data enhancement for document-level re-
  lation extraction with distant supervision from large    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
  language models. Conference on Empirical Methods           Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
  in Natural Language Processing 2023 (EMNLP).               Kaiser, and Illia Polosukhin. 2017. Attention is all
                                                             you need. Advances in neural information processing
Xue Lilong, Zhang Dan, Dong Yuxiao, and Tang
                                                             systems (NeurIPS), 30.
  Jie. 2024. Autore: Document-level relation extrac-
  tion with large language models. arXiv preprint
                                                           Somin Wadhwa, Silvio Amir, and Byron C Wallace.
  arXiv:2403.14888.
                                                             2023. Revisiting relation extraction in the era of
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-          large language models. Proceedings of the 61th An-
  dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,              nual Meeting of the Association for Computational
  Luke Zettlemoyer, and Veselin Stoyanov. 2019.              Linguistics (ACL).
  Roberta: A robustly optimized bert pretraining ap-
  proach. arXiv preprint arXiv:1907.11692.                 Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu,
                                                             Haiyue Song, Jiwei Li, and Sadao Kurohashi. 2023.
Ilya Loshchilov and Frank Hutter. 2019. Decoupled            Gpt-re: In-context learning for relation extraction us-
   weight decay regularization. In International Confer-     ing large language models. In The 2023 Conference
   ence on Learning Representations (ICLR).                  on Empirical Methods in Natural Language Process-
                                                             ing (EMNLP).
Chonggang Lu, Richong Zhang, Kai Sun, Jaein Kim,
  Cunwang Zhang, and Yongyi Mao. 2023. Anaphor             Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang,
  assisted document-level relation extraction. In Pro-       Xin Zhang, Shen Huang, Pengjun Xie, Jinan Xu,
  ceedings of the 2023 Conference on Empirical Meth-         Yufeng Chen, Meishan Zhang, et al. 2023. Zero-
  ods in Natural Language Processing (EMNLP),                shot information extraction via chatting with chatgpt.
  pages 15453–15464.                                         arXiv preprint arXiv:2302.10205.
                                                       6302
Yiqing Xie, Jiaming Shen, Sha Li, Yuning Mao, and Ji-       Table 17. NA is used as the last option. We aim to
  awei Han. 2022. Eider: Empowering document-level          obtain the probability distribution of the first token
  relation extraction with efficient evidence extraction
                                                            after the Answer and select all the multiple-choice
  and inference-stage fusion. Proceedings of the 60th
  Annual Meeting of the Association for Computational       question letters greater than NA as the final relation
  Linguistics (ACL).                                        prediction results.
Canwen Xu, Yichong Xu, Shuohang Wang, Yang Liu,             A.2    Multi-choice Template
  Chenguang Zhu, and Julian McAuley. 2023a. Small
  models are valuable plug-ins for large language mod-      Our multiple-choice prompt template consists of
  els. arXiv preprint arXiv:2305.08848.                     Instruction + Document + Question + Answer as
Xin Xu, Yuqi Zhu, Xiaohan Wang, and Ningyu Zhang.           well. Unlike the only-LLM template, the prompt
  2023b. How to unleash the power of large language         we use for task distribution selects only the most
  models for few-shot relation extraction? Proceedings      likely top-k relations predicted by the SLM. We
  of the 61th Annual Meeting of the Association for         obtain the probability distribution of the first token
  Computational Linguistics (ACL).
                                                            after the Answer and select all the multiple-choice
Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin,           letters with probabilities greater than NA as the
  Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou,         final multi-label relation prediction results. The
  and Maosong Sun. 2019. Docred: A large-scale              multiple-choice prompt is shown in Table 15.
  document-level relation extraction dataset. Proceed-
  ings of the 57th Annual Meeting of the Association
  for Computational Linguistics (ACL).                      A.3    Self-supervision Judgment Template
                                                            Our judgment prompt template consists of Instruc-
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu,
 Mingxuan Ju, Soumya Sanyal, Chenguang Zhu,                 tion + Document + Question + Answer as shown in
 Michael Zeng, and Meng Jiang. 2023. Generate               Table 16. Unlike the multiple-choice template, the
 rather than retrieve: Large language models are            self-supervision part does not use multiple-choice
 strong context generators. In International Confer-        questions but judgment questions. We provide a
 ence for Learning Representation (ICLR).
                                                            document and a sentence (i.e., each multiple-choice
Shuang Zeng, Yuting Wu, and Baobao Chang. 2021.             with probabilities greater than NA as the final rela-
  Sire: Separate intra-and inter-sentential reasoning for   tion prediction) and let the LLM judge whether the
  document-level relation extraction. In Findings of
  the Association for Computational Linguistics: ACL-
                                                            sentence is T or F. If the probability of T is greater
  IJCNLP 2021 (ACL), pages 524–534.                         than the probability of F, we consider the result of
                                                            this refinement acceptable; otherwise, we discard
Shuang Zeng, Runxin Xu, Baobao Chang, and Lei Li.           the result of this refinement.
  2020. Double graph based reasoning for document-
  level relation extraction. Proceedings of the 2020
  Conference on Empirical Methods in Natural Lan-           B Hyper-Parameters of SLMs and LLMs
  guage Processing (EMNLP).
                                                            For the refinement of SLMs, we reproduce their
Kai Zhang, Bernal Jiménez Gutiérrez, and Yu Su. 2023.       prediction logits using the same parameter set-
  Aligning instruction tasks unlocks large language         tings as outlined in their respective papers. For
  models as zero-shot relation extractors. Findings of
  the Association for Computational Linguistics (ACL        Refiner¬T D&P F , we set the temperature to 0 to
  Findings).                                                make the results more stable, and we choose dif-
                                                            ferent δ thresholds for different DocRE datasets.
Wenxuan Zhou, Kevin Huang, Tengyu Ma, and Jing
                                                            Detailed parameter selections of SLMs and LLMs
 Huang. 2021. Document-level relation extraction
 with adaptive thresholding and localized context pool-     are provided in Table 10.
 ing. In Proceedings of the AAAI conference on artifi-
 cial intelligence (AAAI), pages 14612–14620.               C     Impact of δ on Re-DocRED Dataset

A    Prompt Templates                                       δ is denoted as the hard task threshold in Eq. (7).
                                                            As the equation defines, if δ increases, the hard task
A.1 Only-LLM Template                                       threshold will range up, meaning that tasks with
Our only-LLM prompt template used in Section                Pslm (r) much lower than Pslm (N A) will also be
3.1 consists of Instruction + Document + Question           considered as hard tasks. Consequently, the LLM
+ Answer as shown in Table 14. The Question is              will handle more tasks. Figure 8 shows the changes
constructed by filling in the head and tail entities        of F1, we can see that the best F1 score of the
of the question through the question template of            Refiner on Re-DocRED is 78.05 with δ set to 0.5.
                                                        6303
     Model           Parameters
     ATLOP           adm ϵ = 1e-6
                     learning_rate = 5e-5
                     warmup_ratio = 0.06
                     num_train_epochs = 30
     Eider           learning_rate = 5e-5
                     learning_rate(other) = 1e-4
                     warmup_ratio = 0.06
                     num_train_epochs = 30
     DREEAM          learning_rate(encoder) = 5e-5
                     learning_rate(classifier) = 1e-4
                     warmup_ratio = 0.06                   Figure 8: Impact of δ on F1 score, the number of cor-
                     num_train_epochs(teacher) = 30        rectly and incorrectly refined entity pairs for SLM AT-
                     num_train_epochs(student) = 10        LOP on Re-DocRED dataset.
     AA              learning_rate(encoder) = 5e-5
                     learning_rate(classifier) = 1e-4
                     warmup_ratio = 0.06
                     num_train_epochs = 30
     Refiner¬T D&P F max_new_tokens = 1
                     top-k = 4
                     top_p = 0.9
                     temperature = 0
     Refiner         top-k = 4
                     temperature = 1.8
                     ξ = 0.03
                     δ(DocRED) = 0.6
                     δ(Re-DocRED) = 0.5
                     top_p = 0.9
                     max_new_tokens = 1

Table 10: The hyper-parameters of SLMs and LLMs.


D   Computational Requirements
Combined with our cost analysis in Section 5.6, we
                                                                Figure 9: Case Study of Probability Fusion.
further provide a more detailed analysis of the com-
putational requirements of our refiner framework.
It is important to emphasize that the LLaMA3-8B
                                                           vidually would create a significant computational
model we used does not involve the fine-tuning pro-
                                                           burden.
cess. We analyze the hour time taken for training
the SLM and the inference for LLaMA3-8B on the                In Table 12, we can observe that due to the large
refiner task. Our hardware environment consists of         number of NA entity pairs, direct inference incurs
an A100 with 40GB GPU, and the LLaMA3-8B                   significantly higher inference costs compared to
model runs and performs inference based on the             using Task Distribution, and the performance is not
Transformer library.                                       optimal. This further highlights that, for the DocRE
   From Table 11, we can analyze that the                  task, considering Task Distribution for inference
LLaMA3-8B inference time does not significantly            costs is a more reasonable approach to assess the
contribute to the overall inference cost. Therefore,       real-world applicability.
when using LLMs for single entity pair prediction,
the computational requirements increase in a rea-          E   Case Study of Probability Fusion
sonable way compared to SLM-only approaches.
   Our proposed Task Distribution effectively en-          In Figure 9, we present cases illustrating how the
hances the cost-benefit of our model framework.            probability fusion affects the final decision. It is
This is because NA entities make up a very large           evident that LLM tends to be overly confident in
proportion of the dataset. If all entity pairs are         its final prediction, leading to inaccuracy. However,
handed over to the LLM for relation judgment,              when a better balance between SLM and LLM is
requiring the LLM to analyze each entity pair indi-        achieved, which results in an accurate prediction.
                                                        6304
                           SLM Train        LLM Inference         Total     Inference Proportion(%)
    ATLOP                       2.23                     -         2.23                            -
    ATLOP+Ref iner              2.23                  0.44         2.67                        16.48
    Eider                       1.61                     -         1.61                            -
    Eider+Ref iner              1.61                  0.49          2.1                        23.33
    DREEAM                      2.15                     -         2.15                            -
    DREEAM+Ref iner             2.15                  0.52         2.67                        19.48
                  Table 11: Analysis the SLM training time and LLM inference time (hours).

                                    GPT-3.5-Turbo                          LLaMA3-8B
                                Cost ($) Ign F1    F1              Time (hours) Ign F1                 F1
    Direct Inference             100.91   24.17 27.65                     28.69  30.48              31.92
    With Task Distribution         4.96   33.34 34.01                      1.21  41.67              42.43
Table 12: Analysis comparing the inference cost of GPT-3.5-Turbo and LLaMA3-8B under two scenarios: using
Task Distribution and direct inference.


F    Case study of LLM Predictions                      entity should be an individual, or a region.), second,
     Conflicting with Ground Truth                      construct a grammatical sentence for the head and
                                                        tail entities and relations. The question relation
As the LLM used in our study was not fine-tuned,        template we designed is shown in Table 17.
we conducted a case study to illustrate that such
conflicts are indeed worth investigating and may
contribute to the suboptimal performance of LLMs
in their current, non-fine-tuned state. Below, we
provide examples highlighting the discrepancies
between LLM predictions and the ground truth,
underscoring the need for further exploration of
this phenomenon. The case results are presented in
Table 13.
   From these cases, we observe that when certain
information is not explicitly presented in the doc-
ument or requires additional evidence for verifica-
tion, the LLM tends to rely on its own knowledge
to infer the correctness of a given option. This
often leads the LLM to favor relation-indicating
options over selecting option D. We hope these
examples provide a clearer understanding of how
LLM predictions can conflict with the ground truth.

G     Question Relation Templates
By observing the description of each relation in
DocRED, we can find that there are many relations,
which only using the form of [head + relationship
+ tail] cannot express the true meaning of these
relations. To help LLMs better understand each
relation label, we design our own question relation
templates for DocRED and Re-DocRED datasets.
   Our design idea mainly includes two parts: first,
the template needs to describe the entity (e.g, the
                                                   6305
                  Table 13: Case study of LLM predictions conflicting with ground truth.

Case                                                                True Label         LLM Prediction
Robert Kingsbury Huntington (13 March 1921 – 5 June
1942), was a naval aircrewman and member of Torpedo
Squadron 8 (or VT-8). He was radioman / gunner to En-
sign George Gay’s TBD Devastator aircraft. Along with his
entire squadron, Huntington was shot down during the Battle
of Midway, on 4–5 June 1942. Born in Los Angeles, Cal-
ifornia, enlisted in the United States Navy 21 April 1941.
He received the Distinguished Flying Cross for heroism and
extraordinary achievement as rear gunner in a torpedo plane
during an attack against enemy Japanese forces in the Battle
of Midway 4 June 1942. Flying without fighter support and
with insufficient fuel to return to their carrier, Huntington
and his fellow crewmember pressed home their attack with
                                                                          D                  A
utter disregard for their own personal safety, in the face of a
tremendous antiaircraft barrage and overwhelming fighter op-
position. Huntington was one of 29 from Torpedo Squadron
8 who gave their lives in this attack.
##QUESTION: Which of the following is right?
A. Battle of Midway(MISC) isn’t an administrative entity,
and Battle of Midway(MISC) was a physical object or event
in Japanese(LOC).
B. Japanese(LOC) is a country, and Battle of Midway(MISC)
isn’t a person. Japanese(LOC) is the sovereign state of Battle
of Midway(MISC).
C. Battle of Midway(MISC) is owned by Japanese(LOC).
D. None of the above options is correct.
"More" is a song by The Sisters of Mercy, from their album
Vision Thing. It was the first single from the album, reaching
number one on the Billboard Modern Rock Tracks chart
for five weeks, starting 15 December 1990. The song was
co-written and co-produced by Andrew Eldritch and Jim
Steinman. It was covered by Shaaman on their album Reason,
and Gregorian for their album The Dark Side. Steinman
produced a cover of the song, by Mike Vogel and Erika
Christensen, for the soundtrack of the MTV film Wuthering
Heights. He also used the song’s main guitar riff and the" I
need all the love I can get" vocal in a song for his musical
                                                                          D                  B
Batman. The song has also been re-recorded by Meat Loaf
for his 2016 album Braver Than We Are.
##QUESTION: Which of the following is right?
A. Wuthering Heights(MISC) was a radio or television show,
and Wuthering Heights(MISC) was aired on or included by
MTV(MISC).
B. MTV(MISC) is a production company, and Wuthering
Heights(MISC) was produced by MTV(MISC).
C. MTV(MISC) is a class, and Wuthering Heights(MISC) is
an individual member of MTV(MISC).
D. None of the above options is correct.

                                                  6306
Dollar General Corporation is an American chain of variety
stores headquartered in Goodlettsville, Tennessee. As of July
2018, Dollar General operates 15,000 stores in 45 of the 48
contiguous United States (the exceptions being three states
in the northwest : Idaho, Montana, and Washington). The
company first began in 1939 as a family-owned business
called J.L. Turner and Son in Scottsville, Kentucky by James
Luther Turner and Cal Turner. In 1968, the name changed
to Dollar General Corporation and the company went public
on the New York Stock Exchange. Fortune 500 recognized
Dollar General in 1999 and in 2018 reached # 123. Dollar
General has grown to become one of the most profitable
                                                                 D   B
stores in the rural United States with revenue reaching around
$ 21 billion in 2017.
##QUESTION: Which of the following is right?
A. American(LOC) is a country, and Dollar General(ORG)
isn’t a person. American(LOC) is the sovereign state of
Dollar General(ORG).
B. American(LOC) is a country, and Dollar General(ORG)
was originally made in American(LOC).
C. American(LOC) is a country, and Dollar General Corpo-
ration(ORG) is a person. Dollar General Corporation(ORG)
is a citizen of American(LOC).
D. None of the above options is correct.
Live in New York was a 2-CD live album released by perfor-
mance artist Laurie Anderson on Nonesuch Records in 2002.
It was her ninth album of new recordings released since 1982.
The front cover of the CD has the title Live at Town Hall,
New York City September 19–20, 2001, however the official
title of the album is just Live in New York. Recorded less
than 10 days after the September 11, 2001, attacks on New
York City, the album was produced during a tour Anderson
gave of the United States in which she performed a mixture
of older pieces from earlier in her career and newer works,
including songs from her then-recent album Life on a String,
                                                                 D   C
as well as earlier albums such as United States Live, Big
Science, Bright Red, Home of the Brave and Strange Angels.
##QUESTION: Which of the following is right?
A. Big Science(MISC) is the prior version in series, followed
by Bright Red(MISC).
B. Big Science(MISC) and Bright Red(MISC) aren’t in series,
and Big Science(MISC) is replaced by Bright Red(MISC),
so Big Science(MISC) will never take place again.
C. Big Science(MISC) is the next version in series, following
Bright Red(MISC).
D. None of the above options is correct.




                                                 6307
Only-LLM Prompt                             Example
## INSTRUCTION:                             ## INSTRUCTION:
[The instruction for the LLM]               Read the ##DOCUMENT and answer the ##QUESTION.
## DOCUMENT:                                Write the answers in ##ANSWER.
[The related document of the entity pair]   ## DOCUMENT:
## QUESTION:                                Skai TV is a Greek free-to-air television network based in
[The statements for LLM to choose]          Piraeus. It is part of the Skai Group, one of the largest media
## ANSWER:                                  groups in the country. It was relaunched in its present form
                                            on 1st of April 2006 in the Athens metropolitan area, and
                                            gradually spread its coverage nationwide. Besides digital
                                            terrestrial transmission, ..., all foreign shows.
                                            ## QUESTION:
                                            Which of the following is right?
                                            1. Greece(LOC) is a country, and Skai TV(ORG) isn’t a per-
                                            son. Greece(LOC) is the sovereign state of Skai TV(ORG).
                                            2. Greece(LOC) is an administrative entity, and Skai
                                            TV(ORG) is located on the territory of Greece(LOC).
                                            3. Greece(LOC) is a country, and Skai TV(ORG) was
                                            originally made in Greece(LOC).
                                            4. ....
                                            .... [all relations]
                                            97. None of the above options is correct.
                                            ## ANSWER: [TOKEN]

               Table 14: The only-LLM prompt template and an example for an entity pair.



Multiple-choice Prompt                      Example
## INSTRUCTION:                             ## INSTRUCTION:
[The instruction for the LLM]               Read the ##DOCUMENT and answer the ##QUESTION.
## DOCUMENT:                                Write the answers in ##ANSWER.
[The related document of the entity pair]   ## DOCUMENT:
## QUESTION:                                Skai TV is a Greek free-to-air television network based in
[The statements for LLM to choose]          Piraeus. It is part of the Skai Group, one of the largest media
## ANSWER:                                  groups in the country. It was relaunched in its present form
                                            on 1st of April 2006 in the Athens metropolitan area, and
                                            gradually spread its coverage nationwide. Besides digital
                                            terrestrial transmission, ..., all foreign shows.
                                            ## QUESTION:
                                            Which of the following is right?
                                            A. Greece(LOC) is a country, and Skai TV(ORG) isn’t a per-
                                            son. Greece(LOC) is the sovereign state of Skai TV(ORG).
                                            B. Greece(LOC) is an administrative entity, and Skai
                                            TV(ORG) is located on the territory of Greece(LOC).
                                            C. Greece(LOC) is a country, and Skai TV(ORG) was orig-
                                            inally made in Greece(LOC).
                                            D. None of the above options is correct.
                                            ## ANSWER: [TOKEN]

             Table 15: The multiple choice prompt template and an example for an entity pair.


                                                  6308
Judgment Prompt                             Example
## INSTRUCTION:                             ## INSTRUCTION:
[The instruction for the LLM]               Read the ##DOCUMENT and answer the ##QUESTION.
## DOCUMENT:                                Write the answers in ##ANSWER.
[The related document of the entity pair]   ## DOCUMENT:
## QUESTION:                                Skai TV is a Greek free-to-air television network based in
[The statements for LLM to choose]          Piraeus. It is part of the Skai Group, one of the largest media
## ANSWER:                                  groups in the country. It was relaunched in its present form
                                            on 1st of April 2006 in the Athens metropolitan area, and
                                            gradually spread its coverage nationwide. Besides digital
                                            terrestrial transmission, ..., all foreign shows.
                                            ## QUESTION:
                                            True or False? Only return T or F.
                                            Greece(LOC) is a country, and Skai TV(ORG) isn’t a per-
                                            son. Greece(LOC) is the sovereign state of Skai TV(ORG).
                                            ## ANSWER: [TOKEN]

                Table 16: The judgment prompt template and an example for an entity pair.




                                                  6309
Table 17: Question Relation Templates, where {head} and {tail} are the placeholders for subject and object.

ID     Relation                           Template
P6     head of government                 {tail} is a person, and {head} is a governmental body.
                                          {tail} is the head of {head}.
P17    country                            {tail} is a country, and {head} isn’t a person. {tail} is the
                                          sovereign state of {head}.
P19    place of                           birth {tail} is a specific location, and {head} was born in
                                          {tail}.
P20    place of                           death {tail} is a specific location, and {head} died in
                                          {tail}.
P22    father                             {head} and {tail} are both people, {tail} is {head}’s bio-
                                          logical father.
P25    mother                             {head} and {tail} are both people, {tail} is {head}’s bio-
                                          logical mother.
P26    spouse                             {head} and {tail} are spousal.
P27    country of citizenship             {tail} is a country, and {head} is a person. {head} is a
                                          citizen of {tail}.
P30    continent                          {tail} is a continent, and {head} is part of {tail}.
P31    instance of                        {tail} is a class, and {head} is an individual member of
                                          {tail}.
P35    head of state                      {tail} is a person, and {head} is a country or state. {tail}
                                          is the head of {head}.
P36    capital                            {head} is an administrative territorial entity, and {tail} is
                                          a captial of {head}.
P37    official language                  {head} ’s official language is {tail}.
P39    position held                      {head} is a person, and {head} holds {tail} position.
P40    child                              {head} has {tail} in their family as their child.
P50    author                             {tail} isn’t a person, and the author of {tail} is {head}.
P54    member of sports team              {tail} is a sports team or club, and {head} plays for {tail}.
P57    director                           {head} is a work directed by {tail}.
P58    screenwriter                       {tail} is the author of the script for {head}.
P69    educated                           {tail} is an educational institution, and {head} was edu-
                                          cated in {tail}.
P86    composer                           {head} is the music wrote by {tail}.
P102   member of political party          {tail} is a political party, and {head} has been a member
                                          of {tail}.
P108   employer                           {tail} is a person or organization, and {head} worked for
                                          {tail}.
P112   founded by                         {head} is a organization, religion or place, and {tail} is a
                                          founder or co-founder of {head}.
P118   league                             {head} is a team, and {head} is in {tail} league.
P123   publisher                          {tail} is organization or person, and {head} is published
                                          by {tail}.
P127   owned by                           {head} is owned by {tail}.
P131   located in the administrative ter- {tail} is an administrative entity, and {head} is located on
       ritorial entity                    the territory of {tail}.
P136   genre                              {tail} is a work’s genre in which {head} worked.
P137   operator                           {tail} is a person or organization, and {tail} is the operator
                                          of {head}.
P140   religion                           {tail} is a religion, and {tail} is the religion of {head}.

                                                   6310
P150   contains administrative territorial
                                         {head} is administrative territorial entity, and {tail} is the
       entity                            direct subdivision of {head}.
P155   follows                           {head} is the next version in series, following {tail}.
P156   followed by                       {head} is the prior version in series, followed by {tail}.
P159   headquarters location             {head} is a organization, and {head} is based in {tail}.
P161   cast member                       {tail} is a cast member of {tail}.
P162   producer                          {tail} is a person, and {tail} is the producer of {head}.
P166   award received                    {head} received the award called {tail}.
P170   creator                           {head} is a work or fictional object, created by {tail}.
P171   parent taxon                      {tail} is the closest parent taxon of {head}.
P172   ethnic group                      The ethnic group of {head} is {tail}.
P175   performer                         {tail} is the performer of {head}.
P176   manufacturer                      {head} are made by manufacturer {tail}.
P178   developer                         {tail} is a person or organisation, and is the developer of
                                         {head}.
P179   series                            {tail} is a series, and {head} is part of this series.
P190   sister city                       {head} is twinned with {tail}, so they ’re sister cities to
                                         each other.
P194   legislative body                  {tail} is a political institution, as the legislative body
                                         governing {head}.
P205   basin country                     {tail} is a country, and {head} is the drainage or body of
                                         water created by {tail}.
P206   located in or next to body of wa- {tail} is sea, lake or river, and {head} is located in or next
       ter                               to body of {tail}.
P241   military branch                   {tail} is a military branch, and {head} belongs to {tail}.
P264   record label                      {tail} is a record company, and {head} was released on
                                         {tail}.
P272   production company                {tail} is a production company, and {head} was produced
                                         by {tail}.
P276   location                          {head} isn’t an administrative entity, and {head} was a
                                         physical object or event in {tail}.
P279   subclass of                       {head} and {tail} are two classes, and {head} is a subclass
                                         of {tail}.
P355   subsidiary                        {head} is a company or organization, and {tail} is the
                                         subsidiary of {head}.
P361   part of                           {head} is a part of {tail}.
P364   original language of work         {head} is a film or performance work, and {tail} is the
                                         original language of {head}.
P400   platform                          {tail} is a released platform, and {head} was released for
                                         {tail}.
P403   mouth of the watercourse          {head} is a watercourse, and {head} drains into the body
                                         of {tail}.
P449   original network                  {head} was a radio or television show, and {head} was
                                         aired on or included by {tail}.
P463   member of                         {tail} isn’t an ethinc or social groups, and {head} belongs
                                         to {tail}.
P488   chairperson                       {head} is an organization, group or body, and {tail} is the
                                         chairperson of {head}.
P495   country of origin                 {tail} is a country, and {head} was originally made in
                                         {tail}.
P527   has part                          {tail} is a part of {head}.

                                                6311
P551    residence                    {tail} is a place, and {head} is a person. The {tail} is the
                                     place where {head} is, or has been, resident.
P569    date of birth                {tail} is a time, and {head} was born on {tail}.
P570    date of death                {tail} is a time, and {head} was died on {tail}.
P571    inception                    {tail} is a time point, and {head} was firstly founded on
                                     {tail}.
P577    publication date             {tail} is a time point, and {head} was a work firstly pub-
                                     lished on {tail}.
P580    start time                   {tail} is a time, and {head} started being valid in {tail}.
P582    end time                     {tail} is a time, and {head} stopped being valid in {tail}.
P585    point in time                {tail} is a time point, and {head} took place at this point
                                     in {tail}.
P607    conflict                     {tail} is a battle, was or other military engagement, and
                                     {head} participated in {tail}.
P674    characters                   {head} is a work, and {tail} is one of characters in {head}.
P676    lyrics by                    {head} is a song, and the lyrics of {head} were written by
                                     {tail}.
P706    located on terrain feature   {tail} is a specified landform, and {head} is located on
                                     {tail} according to the terrain feature.
P710    participant                  {head} is an event, and {tail} participated in {head}.
P737    influenced by                {head} was a person or idea, etc, and {head} was influ-
                                     enced by {tail}.
P740    location of formation        {tail} is a location, and {head} is a group or organization
                                     formed in {tail}.
P749    parent organization          {head} is a company or organization, and {tail} is the
                                     parent organization of {head}.
P800    notable work                 {tail} is a notable work, and {tail} is one of {head}’s
                                     works.
P807    separated from               {head} emerged after the collapse or separation of {tail}.
P840    narrative location           {head} is a work or story, and {head} is about what hap-
                                     pened in {tail}.
P937    work location                {tail} is a location, and {head} worked in the past or is
                                     working now.
P1001   applies to jurisdiction      {head} has the territorial jurisdiction of {tail}.
P1056   product or material produced {tail} was the material or product produced by {head}.
P1198   unemployment rate            {tail} as the best competition record of {head} in some
                                     event.
P1336   territory claimed by         {head} is an area, and {head} is administered by {tail}.
P1344   participant of               {head} is a person or an organization, and {tail} is an
                                     event. {head} participated in {tail}.
P1365   replaces                     {head} and {tail} aren’t in series, and {head} replaces
                                     {tail}, so {tail} will never take place again.
P1366   replaced by                  {head} and {tail} aren’t in series, and {head} is replaced
                                     by {tail}, so {head} will never take place again.
P1376   capital of                   {tail} is an administrative division, and {head} is the
                                     capital of {tail}.
P1412   languages spoken, written or {tail} is a person, and {head} is the language that {tail}
        signed                       speaks or writes.
P1441   present in work              {head} is a fictional entity or historical person, and {head}
                                     is present in the work named {tail}.
P3373   sibling                      {head} and {tail} are siblings.

                                              6312
