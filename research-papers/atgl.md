     ATGL: An Adaptive-Threshold Global Loss for Document-level Relation
                                Extraction

              Huangming Xu, Fu Zhang* , Zhixuan Yang, Lu Zhang, Jingwei Cheng
   School of Computer Science and Engineering, Northeastern University, Shenyang 110819, China
                       xuhuangming@foxmail.com, zhangfu@neu.edu.cn



                              Abstract                             contexts where entities may appear across multi-
                                                                   ple sentences, thereby increasing the complexity of
            Document-level relation extraction (DocRE)
            aims to determine which relations hold between
                                                                   relation modeling. DocRE plays a crucial role in
            a given entity pair within a document. As a            downstream applications, such as question answer-
            multi-label classification task, the most com-         ing (Baek et al., 2023; Pan et al., 2024) and knowl-
            monly adopted paradigm introduces a learnable          edge graph construction (Zhang and Soh, 2024).
            threshold to distinguish positive and negative            Given that DocRE is a multi-label classification
            classes for an entity pair. Under this paradigm,       task where an entity pair may have multiple re-
            existing losses decouple the optimization into         lations, most existing works distinguish between
            independent positive and negative losses, which
            interact solely with a shared threshold. This
                                                                   positive and negative classes1 for a given entity pair
            leads to two inherent limitations: (i) thresh-         by applying a threshold. To achieve this, the Bi-
            old instability caused by conflicting gradient         nary Cross-Entropy (BCE) loss (Goodfellow et al.,
            updates from the decoupled losses; and (ii) op-        2016) is adopted for the DocRE task (Yao et al.,
            timization bias exacerbated by the severe im-          2019; Zeng et al., 2020). BCE decomposes the
            balance between limited positive samples and           multi-label classification task into multiple inde-
            abundant negative samples inherent in DocRE,           pendent binary subtasks and determines a fixed
            which makes the model more likely to predict
                                                                   threshold for all entity pairs. However, such a fixed
            that no relation exists. To address these issues,
            we propose the Adaptive-Threshold Global               threshold cannot effectively adapt to different en-
            Loss (ATGL). Unlike prior work, ATGL in-               tity pairs. To address this inflexibility, Zhou et al.
            tegrates positive, negative, and threshold opti-       (2021) propose the Adaptive Threshold Loss (ATL).
            mization into a unified logit space and explic-        ATL introduces an adaptive threshold class T H,
            itly enforces ranking constraints on their con-        which allows each entity pair to apply an adaptive
            tributions to the objective. Furthermore, ATGL         threshold and decomposes the optimization into
            incorporates an imbalance-aware optimization           two independent parts2 : positive loss LPT and neg-
            mechanism, thereby effectively addressing the
            severe class imbalance in DocRE. Our ATGL
                                                                   ative loss LNT .
            serves as a general optimization objective that           Building upon the success of ATL, subsequent
            can be readily applied to different DocRE mod-         loss optimization strategies primarily focus on
            els. Experiments on four datasets show that            learning a clearer separation between the thresh-
            ATGL outperforms other DocRE losses and                old class and positive/negative classes by enhanc-
            achieves state-of-the-art results, while consis-       ing discriminative margins (e.g., NCRL (Zhou and
            tently improving the performance of existing
                                                                   Lee, 2022), AML (Wei and Li, 2022), HingeABL
            DocRE models. Code is available at https:
            //github.com/xhm-code/ATGL.
                                                                   (Wang et al., 2023), AFL (Tan et al., 2022a), AMTL
                                                                   (Xu et al., 2025a), ARPDL (Xu et al., 2025b), and
    1       Introduction                                           CMM (Duan et al., 2025)). Despite these advances,
    Relation Extraction (RE) aims to identify relations                 1
                                                                         Given a predefined set of relations R, the positive classes
    between an entity pair in a given text. Compared               PT ⊆ R for an entity pair represent the relations that exist,
                                                                   while the negative classes NT ⊆ R represent the relations
    to sentence-level RE, document-level relation ex-              that do not exist.
    traction (DocRE) (Yao et al., 2019) is more chal-                  2
                                                                         For ATL-based losses, a threshold class (denoted as T H)
    lenging because it requires reasoning over longer              is used to divide R into PT and NT . The positive loss LPT
                                                                   measures the distance between T H and PT , while the nega-
        *    Corresponding author.                                 tive loss LNT measures the distance between T H and NT .

                                                                34702
Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 34702–34716
                                 July 2-7, 2026 ©2026 Association for Computational Linguistics
                            (a) Threshold Instability via                 (b) Instability Exacerbated                   (c) Ours: ATGL loss
                              Decoupled Optimization                           by Class lmbalance                      (Unified Logit Space)


                                                                                                                                                 Constraints and Strategies:


              Logit Value                                   Logit Value                                  Logit Value


                                                                                                                                               Positive > Threshold > Negative


                             Positive      Negative             Ideal                      Learned      Positive loss 
                             Classes        Classes           Threshold                   Threshold                          T
                                                                                                                                 Negative loss 
                                                                                                                                                             T




Figure 1: The degree of deviation between the learned threshold and the ideal threshold reflects the level of stability,
and the thickness of the arrows represents the magnitude of the loss. (a) Existing ATL-based losses optimize
positive and negative samples independently, inducing conflicting updates on the shared threshold and causing
threshold instability. (b) The severe class imbalance in DocRE further amplifies this instability: abundant no-relation
samples dominate the negative loss and bias the model toward predicting no relation for an entity pair. (c) Our
proposed ATGL optimizes all classes in a unified logit space, explicitly enforcing a global ranking among them
while strategically suppressing negative classes and amplifying positive ones.


several key challenges remain to be fully addressed:                                        that decouple the optimization of positive and nega-
   (i) Threshold Instability via Decoupled Op-                                              tive classes, ATGL integrates positive, negative,
timization. Existing ATL-based losses indepen-                                              and threshold classes into a unified logit space.
dently optimize the positive loss LPT and the neg-                                          Through a unified optimization paradigm that ex-
ative loss LNT , interacting solely with the shared                                         plicitly enforces ranking constraints on the contri-
threshold class T H. As shown in Fig. 1(a), this                                            butions of positive, negative, and threshold classes
decoupled optimization makes the threshold unsta-                                           to the objective, ATGL ensures the stability of the
ble: on one hand, to minimize the positive loss                                             threshold. Furthermore, by strategically suppress-
LPT , the threshold may be pushed too low; on the                                           ing the contribution of negative classes while ampli-
other hand, to minimize the negative loss LNT , the                                         fying positive ones, ATGL effectively counteracts
threshold may be excessively raised. Consequently,                                          the gradient dominance of the negative loss LNT
this leads to an ultimately learned threshold that is                                       caused by class imbalance.
either high or low compared to the ideal threshold,                                            Our main contributions are as follows:
which causes a shift in the decision boundary.
   (ii) Instability Exacerbated by Class Imbal-                                                 • We propose ATGL, a unified loss that resolves
ance. As shown in Fig. 1(b), this instability is                                                  the threshold instability issue inherent in ATL-
further exacerbated by the severe class imbalance3                                                based losses by establishing a direct, global
inherent to DocRE (e.g., 94% of entity pairs in Re-                                               interaction between positive, negative, and
DocRED (Tan et al., 2022b) have no relation). The                                                 threshold classes.
large number of samples with no relation dominates
the optimization of the negative loss LNT , making                                              • We further incorporate an imbalance-aware
the model more likely to predict that no relation ex-                                             optimization mechanism into ATGL that miti-
ists. Under the decoupling optimization paradigm,                                                 gates the gradient dominance of the negative
the positive loss LPT struggles to counterbalance                                                 loss, thereby addressing severe class imbal-
this dominance, resulting in a sub-optimal separa-                                                ance in DocRE.
tion threshold that fails to effectively distinguish
between positive and negative classes.                                                          • We provide detailed theoretical analysis and
   To address these issues, we propose a novel                                                    formal proofs for ATGL, offering insights into
loss, Adaptive-Threshold Global Loss (ATGL), as                                                   its optimization behavior under threshold in-
shown in Fig. 1(c). Unlike previous approaches                                                    stability and class imbalance.
   3
     Class imbalance: The number of entity pairs containing                                     • Extensive evaluations on four DocRE datasets
at least one valid relation is significantly smaller than those
without any relation, resulting in severe class imbalance (often                                  show that ATGL loss consistently surpasses
termed the N A problem).                                                                          other DocRE losses, achieving state-of-the-art
                                                                                      34703
     results and exhibiting strong generalization        Specifically, AFL is applied in the KD-DocRE (Tan
     across various DocRE backbone models.               et al., 2022a) model, and PEMSCL is used in the
                                                         VaeDiff-DocRE (Tran et al., 2025) model.
2   Related Work                                            Although these methods improve different as-
                                                         pects of ATL, their losses still optimize the positive
Document-level relation extraction (DocRE) aims          and negative parts independently, interacting solely
to predict one or more relations for an entity pair      through the shared threshold class, as detailed in
and is typically formulated as a multi-label clas-       Section 1. This can lead to a learned threshold that
sification problem. Traditionally, DocRE tasks           deviates from the ideal threshold (either too high
commonly adopt the Binary Cross-Entropy (BCE)            or too low), causing instability and a shift in the
(Goodfellow et al., 2016) loss, which models each        decision boundary. Additionally, this instability
relation independently using a sigmoid function          is further exacerbated by severe class imbalance.
and applies a fixed threshold during inference.          To address these issues, we propose the Adaptive
However, since a fixed threshold is not suitable         Threshold Global Loss (ATGL).
for all entity pairs, Zhou et al. (2021) propose the
Adaptive Threshold Loss (ATL) to overcome this           3     Preliminary
limitation. ATL introduces a threshold class T H,
requiring positive classes to yield scores higher        3.1    Problem Formulation
than this threshold, while negative classes remain       Given a document D and a set of entities {ei }ni=1
below it. During inference, each entity pair is as-      it contains, where n denotes the total number of
signed an adaptive threshold, and relations whose        entities, DocRE is formulated as a multi-label task
scores exceed this threshold are regarded as valid.      that aims to identify relations in R ∪ {N A} that
Due to the effectiveness of ATL in DocRE tasks,          hold between each entity pair (es , eo ). Here, R
subsequent DocRE models commonly adopt ATL               denotes a predefined set of relations, and es and
as their optimization objective, including ALTOP         eo correspond to the subject and object entities,
(Zhou et al., 2021), DREEAM (Ma et al., 2023),           respectively. For each pair T = (es , eo ), we denote
SA-KD (Zhang et al., 2023), and SRF (Zhang et al.,       by PT ⊆ R the set of relations that hold, and by
2024).                                                   NT ⊆ R the set of relations that do not. If PT = ∅,
   To further enhance the discrimination between         the pair is assigned the N A label.
positive and negative classes, subsequent studies
extend ATL by incorporating margins or strength-         3.2    Adaptive Threshold Loss (ATL) for
ening the separation between positive and nega-                 Multi-label Classification
tive logits, leading to variants such as Balanced-       ATL (Zhou et al., 2021) is designed for multi-label
Softmax (Zhang et al., 2021), AML (Wei and Li,           classification in DocRE by introducing a learnable
2022), NCRL (Zhou and Lee, 2022), and Hinge              threshold class T H. To realize this mechanism, the
loss (Wang et al., 2023). Specifically, Balanced-        ATL loss optimizes positive loss LPT and negative
Softmax is adopted in DocuNet (Zhang et al.,             loss LNT independently, interacting solely with the
2021), the AML loss is applied in the SagDRE             shared threshold class T H, as shown in Eq. (1).
(Wei and Li, 2022) model, while NCRL is adopted
                                                                                                                       !
in the REwNCRL (Xu et al., 2024) model.                                X                     exp(logitr )
                                                             LPT = −          log     P                                    ,
   In parallel, ATL remains limited in handling                        r∈PT            r′ ∈PT ∪{T H} exp(logitr′ )
the severe class imbalance in DocRE, where the                                                                     !
                                                                                          exp(logitT H )                       (1)
large number of negative samples tend to domi-                 LNT = − log          P                                  ,
nate its optimization. To mitigate this issue, several                               r′ ∈NT ∪{T H} exp(logitr′ )

studies propose losses that rebalance the contribu-                        LAT L = LPT + LNT .
tions of positive and negative classes or incorporate
prior knowledge, such as AFL (Tan et al., 2022a),          During training, the positive loss LPT encour-
ARPDL (Xu et al., 2025b), and CMM (Duan et al.,          ages logits of positive classes PT to exceed thresh-
2025). Meanwhile, other works aim to enhance             old class T H, while the negative loss LNT ensures
the discriminability of positive classes or introduce    logits of negative classes NT remain below thresh-
multi-threshold mechanisms, as seen in PEMSCL            old class T H. At inference, relations whose logits
(Guo et al., 2023) and AMTL (Xu et al., 2025a).          exceed the threshold are predicted to hold.
                                                    34704
4    Methodology                                         clarity, the negative loss LNT can be rewritten as:
                                                                                                              !
4.1 Our Analysis of ATL-based Losses                                                 exp(logitT H )
                                                            LNT = − log     P
Threshold Instability via Decoupled Optimiza-                                   r′ ∈NT ∪{T H} exp(logitr′ )
                                                                                                                       !           (5)
tion. As shown in Eq. (1), the ATL loss con-                                     P
                                                                                               1
                                                                = − log                                                    .
sists of two decoupled components: the positive                             1+       r′ ∈NT exp(logitr′ − logitT H )

loss LPT and the negative loss LNT . These two
losses are optimized independently, each within its      4.2   Adaptive-Threshold Global Loss
own logit space, interacting solely with the shared      To overcome the above limitations of ATL-based
threshold class T H. This design introduces an in-       losses, we propose the Adaptive-Threshold Global
herent limitation: independent optimization of the       Loss (ATGL), which integrates positive, threshold,
positive and negative losses can lead to inconsis-       and negative classes into a unified logit space and
tent updates of the shared threshold.                    explicitly enforces a global ranking among them.
   As shown in Eq. (2) and Eq. (3), the gradients of     ATGL is formally defined as:
the threshold logitT H with respect to the positive                              
and negative losses are given by:                                                
                                                                                 α > 1, r ∈ PT
                                                                             wr = 1,     r=TH,
                                                                                 
                                                                                 
                    X        ∂                                                    0,     r ∈ NT
    ∇T H LPT = −                                  (2)                                                                          !
                          ∂logitT H                                        X                          exp(logitr )
                   r∈PT                                                                         P
                                                            LAT GL = −               wr · log                                      .
                       exp(logitr )                                      r∈R∪{T H}               r′ ∈R∪{T H} exp(logitr′ )
      log                 P                     ,                                                        (6)
          exp(logitT H ) + r′ ∈PT exp(logitr′ )
                                                         For each relation r ∈ R ∪ {T H}, logitr denotes
                                                         the unnormalized logit for the relation r, and wr
                         ∂                               denotes the weight associated with the relation r.
    ∇T H LNT = −                                 (3)
                     ∂logitT H                              To further explain the ATGL loss, we answer the
                     exp(logitT H )                      following three questions. Additional theoretical
     log                  P                    .         analysis of ATGL is provided in Appendix A.
         exp(logitT H ) + r′ ∈NT exp(logitr′ )
                                                         (i) How can positive, threshold, and nega-
  Thus, the total gradient with respect to the thresh-
                                                         tive classes be represented in a unified logit
old can be expressed as:
                                                         space while establishing a global ranking among
      ∇T H LAT L = ∇T H LPT + ∇T H LNT .          (4)    them? To place all classes in a unified logit space,
                                                         ATGL integrates positive, threshold, and negative
   Intuitively, ∇T H LPT > 0 encourages the thresh-      classes into a single softmax function, where each
old to decrease relative to logits of positive classes   class’s logit contributes to a shared normalization.
PT , whereas ∇T H LNT < 0 encourages it to in-           This ensures that their probabilities are mutually de-
crease relative to logits of negative classes NT .       pendent and directly comparable. To further estab-
These opposing gradients can interfere with each         lish a global ranking, ATGL assigns larger weights
other, leading to instability in the threshold. As       to positive classes PT and smaller weights to neg-
a result, the ultimately learned threshold is either     ative classes NT , with the threshold class lying
higher or lower than the ideal threshold, which          in between, thereby encoding the desired ordering
causes a shift in the decision boundary.                 within the unified logit space.
Instability Exacerbated by Class Imbalance.              (ii) What is the rationale for assigning a weight
This instability is further exacerbated by the se-       of 1 to the threshold class and 0 to the negative
vere class imbalance (Tan et al., 2022a,b). As ob-       classes? Due to severe class imbalance inherent
served by Wang et al. (2023), LNT in Eq. (5) → 0         to DocRE, the large number of examples with no
when logitr′ − logitT H → −∞, indicating that            relation dominates loss optimization, making the
logitT H ≫ logitr′ . This implies that ATL tends         model more likely to predict that no relation ex-
to learn a threshold well above the scores of most       ists. By assigning a weight of 1 to the threshold
candidate relations, which results in a sub-optimal      class T H and 0 to the negative classes NT , ATGL
separation threshold that fails to effectively distin-   removes the explicit loss contributions from nega-
guish between positive and negative classes. For         tive classes and prevents them from dominating the
                                                    34705
training objective. In this setting, if the prediction             Dataset      Split   #Docs.   #Rels.   #Triples.
for an entity pair is that there is no relation, the loss                       train     500      2        5240
                                                                    CDR          dev      500      2        5087
term associated with the threshold T H relative to                               test     500      2        5204
negative classes NT can be written as a standard                                train     602     65       14,403
cross-entropy objective:                                            DWIE         dev       98     65        2,624
                                                                                 test      99     65        2,495
                           exp(logitT H )                                       train    3,053    96       85,932
    LT H = − log P                                . (7)           Re-DocRED      dev      500     96       17,284
                      r′ ∈NT ∪{T H} exp(logitr′ )
                                                                                 test     500     96       17,448
                                                                                train    3,053    96       96,505
Minimizing this term encourages the logit of thresh-              DocGNRE        dev      500     96       17,284
                                                                                 test     500     96       19,526
old to rise above the logits of negative classes NT ,
thereby stabilizing the learned threshold without re-                     Table 1: Statistics of datasets.
quiring explicit supervision on each negative class.
   Moreover, fixing the weights of the threshold
T H and negative classes NT to wT H = 1 and                 from the dev/test evaluation set the relational facts
wr = 0 for r ∈ NT yields a simple and inter-                that overlap with the training set.
pretable weighting scheme: positive classes PT are
emphasized more strongly with wr = α > 1, the
                                                            6     Main Results and Analysis
threshold T H serves as an anchor with intermedi-           6.1    Different DocRE Models with ATGL
ate importance, and negative classes NT contribute
                                                            To evaluate the generality and effectiveness of
only through the softmax normalization term in the
                                                            ATGL, we apply it to various DocRE models by
unified logit space.
                                                            replacing their original losses. As shown in Ta-
(iii) How to alleviate class imbalance? As dis-             ble 2, ATGL consistently improves both F1 and
cussed above in (ii), assigning wr = 0 for r ∈ NT           Ign-F1 scores across different datasets and encoder
removes the explicit loss contributions from nega-          backbones. On the Re-DocRED dataset, compar-
tive classes and prevents them from dominating the          atively earlier models such as ATLOP, DocuNet,
training objective. Furthermore, assigning a larger         and DREEAM obtain substantial gains of about
weight α > 1 to positive classes than to the thresh-        2 to 5 F1 points, while more recent and stronger
old (wT H = 1) and negative classes (wr = 0)                models like TTM-RE and VaeDiff-DocRE are fur-
amplifies the gradients from scarce positive exam-          ther boosted to F1 scores of 82.13 and 79.37, re-
ples and encourages the model to focus on correctly         spectively. Similar trends are observed on the
identifying positive classes.                               DocGNRE dataset, where ATGL yields consistent
                                                            improvements in Test F1 ranging from 1.9 to 4.7
5    Experiments                                            points. On the DWIE dataset, the performance is
                                                            also improved, with average gains of 2.62 in Test
Datasets. To comprehensively evaluate our pro-
                                                            F1 and 3.86 in Test Ign-F1. Quantitatively, as indi-
posed ATGL loss, we conduct experiments on four
                                                            cated by the "Avg.", ATGL achieves a remarkable
DocRE datasets: CDR (Li et al., 2016), DWIE
                                                            average improvement of 2.80 points in both Test F1
(Zaporojets et al., 2021), Re-DocRED (Tan et al.,
                                                            and Ign-F1 across all experiments. These results
2022b) and DocGNRE (Li et al., 2023). These
                                                            suggest that ATGL serves as a general and effective
datasets cover both general-domain and biomedical
                                                            loss, yielding consistent performance gains across
DocRE. We report basic statistics in Table 1, and
                                                            different models, encoders, and datasets.
provide more detailed descriptions in Appendix B.
Implementation Details and Evaluation Met-                  6.2    Comparison of Different Losses
rics. We adopt BERTbase (Devlin et al., 2019)               To further evaluate the effectiveness of ATGL, we
and RoBERTalarge (Liu et al., 2019) as encoders,            compare it with the existing state-of-the-art losses
and conduct experiments on a GeForce RTX 3090               for DocRE. ATGL loss consistently achieves the
GPU. Experimental results are averaged over mul-            best performance across four datasets, as shown in
tiple random seeds to ensure robustness.                    Table 3. On Re-DocRED dataset, it reaches 77.26
   Following Yao et al. (2019), we adopt the F1             F1 and 76.05 Ign-F1, outperforming the previous
score and the Ign F1 score for performance evalua-          best loss (CMM) by +1.14 and +1.29 points, re-
tion. The Ign F1 score is calculated by removing            spectively. On DocGNRE dataset, ATGL obtains
                                                       34706
 Model                                                            Dev                                                        Test
                                       F1       F1 with ATGL        Ign-F1    Ign-F1 with ATGL       F1      F1 with ATGL     Ign-F1      Ign-F1 with ATGL
                                                           Re-DocRED Dataset with BERTbase
 ATLOP (Zhou et al., 2021)           74.22 *     77.01 (+2.79) 73.35 *    75.75 (+2.40)            74.02 *   77.26 (+3.24)    73.22 *       76.05 (+2.83)
 DocuNet (Zhang et al., 2021)        74.62 *     76.93 (+2.31) 73.60 *    75.71 (+2.11)            74.48 *   77.09 (+2.61)    73.53 *       75.92 (+2.39)
 KD-DocRE (Tan et al., 2022a)        74.66 *     76.80 (+2.14) 73.68 *    75.58 (+1.90)            74.55 *   76.68 (+2.13)    73.64 *       75.51 (+1.87)
 DREEAM (Ma et al., 2023)            74.13 *     76.85 (+2.72) 73.68 *    75.67 (+1.99)            73.75 *   76.87 (+3.12)    73.33 *       75.76 (+2.43)
 TTM-RE (Gao et al., 2024)           75.51 *     80.40 (+4.89) 74.31 *    79.26 (+4.95)            75.71 *   80.58 (+4.87)    74.55 *       79.50 (+4.95)
 VaeDiff-DocRE (Tran et al., 2025)   75.89 †     77.51 (+1.62) 74.96 †    76.29 (+1.33)            75.07 †   77.44 (+2.37)    74.13 †       76.26 (+2.13)
                                                         Re-DocRED Dataset with RoBERTalarge
 ATLOP (Zhou et al., 2021)           77.63 *     80.35 (+2.72) 76.88 *    79.28 (+2.40)      77.73 *         80.68 (+2.95)    76.94 *       79.64 (+2.70)
 DocuNet (Zhang et al., 2021)        78.16 *     79.97 (+1.81) 77.53 *    78.96 (+1.43)      77.92 *         79.85 (+1.93)    77.27 *       78.88 (+1.61)
 KD-DocRE (Tan et al., 2022a)        78.65 *     80.09 (+1.44) 77.92 *    79.07 (+1.15)      78.35 *         80.13 (+1.78)    77.63 *       79.18 (+1.55)
 DREEAM (Ma et al., 2023)            77.60 *     80.55 (+2.95) 77.20 *    79.50 (+2.30)      77.94 *         80.67 (+2.73)    77.34 *       79.65 (+2.31)
 TTM-RE (Gao et al., 2024)           78.13 *     81.68 (+3.55) 78.05 *    80.75 (+2.70)      79.95 *         82.13 (+2.18)    78.20 *       81.25 (+3.05)
 VaeDiff-DocRE (Tran et al., 2025)   79.19 †     79.48 (+0.29) 78.35 †    78.49 (+0.14)      79.03 †         79.37 (+0.34)    78.22 †       78.41 (+0.19)
                                                            DocGNRE Dataset with BERTbase
 ATLOP (Zhou et al., 2021)           73.89 *     76.90 (+3.01) 73.07 *    75.64 (+2.57)            68.74 *   72.80 (+4.06)    68.06 *       71.74 (+3.68)
 DocuNet (Zhang et al., 2021)        74.95 †     77.00 (+2.05) 73.99 †    75.84 (+1.85)            69.98 †   72.68 (+2.70)    69.16 †       71.70 (+2.54)
 KD-DocRE (Tan et al., 2022a)        75.19 †     76.86 (+1.67) 74.20 †    75.65 (+1.45)            70.21 †   72.13 (+1.92)    69.38 †       71.07 (+1.69)
 DREEAM (Ma et al., 2023)            74.23 *     76.99 (+2.76) 73.76 *    75.73 (+1.97)            68.24 *   72.74 (+4.50)    68.89 *       71.68 (+2.79)
 TTM-RE (Gao et al., 2024)           75.44 *     80.12 (+4.68) 74.33 *    78.88 (+4.55)            71.14 *   75.85 (+4.71)    70.19 *       74.80 (+4.61)
 VaeDiff-DocRE (Tran et al., 2025)   75.60 †     77.28 (+1.68) 74.61 †    75.98 (+1.37)            71.05 †   73.52 (+2.47)    70.21 †       72.41 (+2.20)
                                                          DocGNRE Dataset with RoBERTalarge
 ATLOP (Zhou et al., 2021)           77.61 *     80.53 (+2.92) 76.96 *    79.42 (+2.46)     72.90 *          76.31 (+3.41)    72.36 *       75.38 (+3.02)
 DocuNet (Zhang et al., 2021)        77.70 †     79.85 (+2.15) 76.97 †    78.79 (+1.82)     73.29 †          75.60 (+2.31)    72.71 †       74.73 (+2.02)
 KD-DocRE (Tan et al., 2022a)        77.62 †     79.83 (+2.21) 76.87 †    78.82 (+1.95)     72.95 †          75.65 (+2.70)    72.32 †       74.83 (+2.51)
 DREEAM (Ma et al., 2023)            77.75 *     80.56 (+2.81) 77.28 *    79.49 (+2.21)     72.90 *          76.29 (+3.39)    72.97 *       75.40 (+2.43)
 TTM-RE (Gao et al., 2024)           78.16 *     81.83 (+3.67) 77.30 *    80.91 (+3.61)     73.72 *          77.56 (+3.84)    73.01 *       76.78 (+3.77)
 VaeDiff-DocRE (Tran et al., 2025)   78.06 †     79.62 (+1.56) 77.24 †    78.37 (+1.13)     73.57 †          75.64 (+2.07)    72.90 †       74.59 (+1.69)
                                                                 DWIE Dataset with BERTbase
 ATLOP (Zhou et al., 2021)           69.96 ‡     72.45 (+2.49)     63.57 ‡    67.37 (+3.80)        74.36 ‡   77.34 (+2.98)    67.56 ‡       70.55 (+2.99)
 KD-DocRE (Tan et al., 2022a)        71.78 ‡     75.53 (+3.75)     65.84 ‡    69.63 (+3.79)        77.01 ‡   78.45 (+1.44)    70.27 ‡       71.32 (+1.05)
 TTM-RE (Gao et al., 2024)           73.01 †     76.35 (+3.34)     65.33 †    70.90 (+5.57)        75.59 †   78.89 (+3.30)    65.98 †       72.01 (+6.03)
                                                            DWIE Dataset with RoBERTalarge
 ATLOP (Zhou et al., 2021)           76.65 *     76.97 (+0.32) 72.47 *      72.77 (+0.30)          81.39 *   81.60 (+0.21)    76.83 *      77.04 (+0.21)
 KD-DocRE (Tan et al., 2022a)        76.55 *     76.94 (+0.39) 72.01 *      72.24 (+0.23)          80.92 *   81.40 (+0.48)    75.67 *      76.41 (+0.74)
 TTM-RE (Gao et al., 2024)           72.63 †     80.18 (+7.55) 64.35 †     76.00 (+11.65)          75.21 †   82.49 (+7.28)    65.40 †      77.54 (+12.14)
 Avg.                                75.84       78.44 (+2.60)      73.99       76.56 (+2.57)       74.72    77.52 (+2.80)     72.73        75.53 (+2.80)


Table 2: Performance of various DocRE models when their original losses are replaced with the ATGL loss. ATL
loss (Zhou et al., 2021) for ATLOP and DREEAM, Balanced-Softmax loss (Zhang et al., 2021) for DocuNet, AFL
loss (Tan et al., 2022a) for KD-DocRE, SSR-PU loss (Wang et al., 2022) for TTM-RE, and PEMSCL loss (Guo
et al., 2023) for VaeDiff-DocRE. † indicates our reproduction; * indicates results from Xu et al. (2025a); ‡ from
Zhang et al. (2023). Avg. reports the average performance scores and improvements across all models and datasets.

 Loss Function                                        Re-DocRED                        DocGNRE                           DWIE                     CDR
                                                 F1              Ign-F1           F1             Ign-F1           F1            Ign-F1             F1
 ATL (Zhou et al., 2021)                       73.29 *           72.46 *        68.74 *          68.06 *       74.36 §          67.56 §          68.84 †
 Balanced-Softmax (Zhang et al., 2021)         73.68 *           72.85 *        68.84 *          68.13 *       67.50 †          57.68 †          68.26 †
 AML (Wei and Li, 2022)                        72.60 *           71.78 *        67.86 *          67.11 *       72.36 †          63.69 †          68.71 †
 AFL (Tan et al., 2022a)                       74.15 *           73.20 *        69.45 *          68.69 *       75.83 †          67.75 †          69.29 †
 NCRL (Zhou and Lee, 2022)                     73.87 *           72.79 *        69.20 *          68.27 *       74.55 †          65.96 †          68.82 †
 PEMSCL (Guo et al., 2023)                     73.98 *           73.06 *        69.46 *          68.70 *       75.39 †          67.12 †          68.96 †
 HingeABLSAT (Wang et al., 2023)               73.46 *           72.61 *        69.15 *          68.41 *       75.18 †          67.14 †          69.08 †
 HingeABLMeanSAT (Wang et al., 2023)           74.68 *           72.90 *        70.83 *          69.25 *       59.40 †          45.82 †          67.18 †
 HingeABL (Wang et al., 2023)                  75.15 *           73.84 *        70.98 *          69.90 *       73.38 †          63.81 †          68.84 †
 AMTL (Xu et al., 2025a)                       75.63 *           74.44 *        71.34 *          70.34 *       75.71 †          67.85 †             -
 ARPDL (Xu et al., 2025b)                      75.90 ‡           74.81 ‡           -                -             -                -                -
 CMM (Duan et al., 2025)                       76.12 ‡           74.76 ‡           -                -             -                -                -
 ATGL (Ours)                                77.26(+1.14)   76.05(+1.24)       72.80(+1.46)   71.74(+1.40)    77.34(+1.51)    70.05(+2.20)     70.38(+1.09)


Table 3: Performance of different losses on four datasets: Re-DocRED, DocGNRE, DWIE, and CDR. * marks
results from Xu et al. (2025a), † indicates our reproduction, § from Zhang et al. (2023), and ‡ refers to the original
paper. Consistent with prior work, all experiments employ the model ATLOP (Zhou et al., 2021) for representation
and BERTbase for encoding. Since the implementation of ARPDL and CMM are not publicly available and their
results are only reported on Re-DocRED, we compare with ARPDL and CMM only on Re-DocRED.

                                                                             34707
              8 ATL Std(RO):0.167
                  AFL Std(RO):0.182
                  HingeABL Std(RO):0.304
              6 ATGL Std(RO):0.098

    Density   4
              2
              0         0.0      0.2       0.4      0.6     0.8   1.0
                                       Relative Offset (RO)

Figure 2: Distribution of the Relative Offset (RO) across                     Figure 3: Comparison of global ranking consistency be-
different losses for evaluating threshold instability. RO                     tween ATL-based and ATGL losses. A lower Ordering
quantifies the position of a threshold relative to the posi-                  Violation Rate (OVR) indicates better consistency.
tive and negative classes. Density reflects the occurrence
frequency of the RO values associated with entity pairs.
                                                                                 Furthermore, as shown in the legend of Fig. 2,
                                                                              ATGL achieves the lowest Std(RO) value (0.098),
72.80 F1 and 71.74 Ign-F1, surpassing the strongest                           followed by ATL (0.167), AFL (0.182), and Hinge-
baseline (AMTL) by +1.46 and +1.40 points. For                                ABL (0.304). Lower Std(RO) values indicate that
DWIE and CDR datasets, ATGL also improves                                     the threshold is more stably positioned between
Test F1 by +1.51 and +1.09 points compared to                                 the positive and negative classes across samples,
the best reported results. ATGL consistently sur-                             which empirically confirms that the global ordering
passes other losses across four datasets, improving                           enforced by ATGL leads to more stable thresholds.
performance by about 1.30 F1 and 1.61 Ign-F1 on
average, further demonstrating its effectiveness.                             7.2   Global Ranking Consistency Analysis
                                                                              To further examine whether different losses pre-
7        Further Analysis                                                     serve the desired global ranking among positive,
                                                                              threshold, and negative logits, we use the Order-
7.1 Empirical Verification of Threshold                                       ing Violation Rate (OVR), which quantifies the
    Instability                                                               consistency of this ranking. For each entity pair,
Since the thresholds in the evaluated losses are                              let logitPT and logitNT denote the average logits
adaptive and each entity pair has its own thresh-                             of positive and negative classes, respectively, and
old value, we first define the relative position of a                         logitT H denote logit of the threshold class. A vi-
threshold using the Relative Offset (RO), which is                            olation occurs if logitPT < logitT H or logitT H <
defined as:                                                                   logitNT , and the overall OVR is defined as:
                                                                                         PN                                                   
                              logitT H − logitNT                              OVR = N1    i=1 I
                                                                                                           (i)       (i)        (i)        (i)
                                                                                                      logitPT < logitT H ∨ logitT H < logitNT ,
                         RO =                    .                      (8)
                              logitPT − logitNT                                                                                 (9)
                                                                              where I(·) is the indicator function, which returns 1
RO thus provides a measure of where the threshold                             if the condition inside is satisfied and 0 otherwise,
lies between the negative and positive classes. To                            and N is the total number of entity pairs.
quantify the stability of these adaptive thresholds                              Fig. 3 reports the OVR performance of various
across samples, we further compute the standard                               losses. Among the baselines, HingeABL demon-
deviation of RO, i.e., Std(RO). A smaller Std(RO)                             strates relatively better consistency with an OVR
indicates that the threshold is more stably posi-                             of 0.302, outperforming AFL (0.325) and ATL
tioned between the positive and negative classes.                             (0.366). Notably, our proposed ATGL loss achieves
   Fig. 2 shows the distribution of RO across en-                             the lowest OVR of 0.259, significantly reducing the
tity pairs with existing relations for different losses.                      violation rate by 14.2% compared to the strongest
ATL, AFL, and HingeABL exhibit relatively wide                                baseline (HingeABL). This substantial reduction
and dispersed RO distributions, reflecting higher                             demonstrates that ATGL effectively enforces a
variability in threshold positioning across entity                            more rigorous global ranking. Furthermore, we
pairs. In contrast, the proposed ATGL loss pro-                               observe that lower OVR values generally align
duces a distribution that is sharply concentrated                             with higher F1 scores, suggesting that maintain-
near 1.0, suggesting that the thresholds are much                             ing global ranking consistency is a critical factor
more consistently positioned across samples.                                  for improving DocRE performance.
                                                                         34708
                              FN      FN      FN_NA         FN                              FP           ATGL FP@NA        ATGL type-consistent FP
    Loss       FP ↓   FN ↓
                             _NA
                                 ↓   _Rel
                                         ↓   /(FP+FN)
                                                     ↓   /(FP+FN)
                                                                 ↓           Relation   (ATL→ATGL)        / ATGL FP             / ATGL FP
 ATL *         1887   6253   5498    755      67.54       76.82
 AML *         2032   6363   5515    848      65.69       75.80              P131       350→638             551/638                637/638
 AFL *         2300   5744   4898    846      60.89       71.41              P17        229→383             323/383                383/383
 NCRL *        2770   5603   4872    731      58.19       66.92              P27        113→210             200/210                208/210
 PEMSCL *      2264   5746   4926    820      61.50       71.74
 SAT *         1749   6241   5363    878      67.12       78.11              P150        37→101              96/101                101/101
 HingeABL *    2935   5083   4306    777      53.70       63.39              P361        37→70                46/70                 70/70
 ARPDL *       2781   5076   4426    650      56.33       64.60              P1001       20→56                24/56                 56/56
 AMTL †        2890   5066   4295    771      53.98       63.68
 ATGL (Ours)   3159   4373   3730    643      49.52       58.06
                                                                        Table 5: Relation-level attribution of FP increases on
Table 4: Distribution of prediction errors for ATGL and                 the Re-DocRED test set. FP@NA denotes cases where
baseline losses, illustrating effectiveness in mitigating               the gold label is N A but the model predicts a relation.
class imbalance. * marks results from ARPDL (Xu
et al., 2025b). † indicates our reproduction. FP (False                  Model            F1 (Dev)        F1 (Test)     Ign-F1 (Dev)    Ign-F1 (Test)
Positive): a negative sample incorrectly predicted as                    ATLOP
positive. FN (False Negative): a positive sample incor-                   + CMM *           76.30           76.10           75.00           74.80
                                                                          + ATGL        77.01 (+0.71)   77.26 (+1.16)   75.75 (+0.75)   76.05 (+1.25)
rectly predicted as negative. FN_NA: a positive sample                   DocuNet
misclassified as negative, with the predicted label being                 + CMM *           76.40           76.30           75.00           75.00
                                                                          + ATGL        76.93 (+0.53)   77.09 (+0.79)   75.71 (+0.71)   75.92 (+0.92)
N A. FN_Rel: a positive sample misclassified as nega-                    KD-DocRE
tive, with the label belonging to the negative classes.                   + CMM *           75.90           76.10           74.30           74.60
                                                                          + ATGL        76.80 (+0.90)   76.68 (+0.58)   75.58 (+1.28)   75.51 (+0.91)



7.3 Analysis of Class Imbalance                                         Table 6: Results on Re-DocRED comparing our ATGL
                                                                        loss with the prior SOTA CMM loss, evaluated with
To investigate the effectiveness of ATGL in allevi-                     BERTbase . * denotes results reported in the original
ating class imbalance in DocRE, we analyze the                          CMM paper.
distribution of prediction errors. Table 4 reports
four types of false prediction patterns. As shown,                      plausible near-misses rather than arbitrary noise,
prior losses such as ARPDL and PEMSCL produce                           indicating that ATGL mainly gains by recovering
a large number of false negatives. ARPDL predicts                       missed positive relations.
5076 FN and 4426 FN_NA, while PEMSCL pre-
dicts 5746 FN and 4926 FN_NA. In contrast, our                          7.5       Analysis of Hyperparameter α
ATGL loss predicts 4373 FN and 3730 FN_NA,                              To evaluate the impact of the hyperparameter α in
showing a clear reduction in false negatives. The                       Eq. (6) on ATGL’s performance, we report detailed
number of FN_Rel is also lower for ATGL (643)                           results across four datasets in Appendix C and
compared to HingeABL (777), while FP increases                          provide a practical guideline for selecting α.
slightly to 3159. Moreover, ATGL achieves the
lowest FN_NA/(FP+FN) and FN/(FP+FN) ratios                              7.6       ATGL vs. the Previous SOTA Loss CMM
among all methods (49.52 and 58.06, respectively),                                Across DocRE Models
indicating that a larger proportion of positive sam-                    To directly compare ATGL against the previous
ples are correctly recovered. These results suggest                     SOTA loss CMM (Duan et al., 2025), we evaluate
that ATGL effectively mitigates class imbalance.                        both losses on several representative DocRE back-
                                                                        bones. As shown in Table 6, replacing CMM with
7.4 Analysis of False Positive
                                                                        ATGL consistently improves both F1 and Ign-F1.
While ATGL substantially reduces false negatives,                          On ATLOP, ATGL improves test F1 from 76.10
it also increases false positives, as shown in Ta-                      to 77.26 and test Ign-F1 from 74.80 to 76.05 (+1.16
ble 4. To better understand this trade-off, we fur-                     and +1.25, respectively). DocuNet and KD-DocRE
ther analyze the relation-level distribution of the ad-                 also benefit from ATGL, with gains of up to +0.92
ditional false positives. The increase is not evenly                    F1 and +1.28 Ign-F1. These consistent improve-
distributed across relations, but concentrated in a                     ments indicate that ATGL is a strong and broadly
few frequent ones: P131, P17, and P27 account for                       applicable loss, serving as a reliable drop-in re-
38% of the total FP increase from ATL to ATGL                           placement for current SOTA DocRE losses.
(Table 5). Moreover, nearly all of these additional
false positives are type-consistent, with about 99%–                    7.7       Computational Cost of ATGL
100% satisfying type constraints across major rela-                     To assess the computational cost of the proposed
tions. This suggests that many of these errors are                      ATGL, we compare its training time against sev-
                                                                     34709
             Category   Entity pair (h → t)              Rel. r       Change (ATL→ATGL)     Margin (ATL→ATGL)
             FN→TP      Jerry Garcia → Grateful Dead     P463            ¬P463 → P463             −0.31 → 3.95
             FN→TP      Salesis house → Appenzell        P131             NA → P131               −1.98 → 3.60
             FN→TP      The Late Edition → BBC Radio     P449             NA → P449               −1.93 → 3.91
             TP→FN      Scandinavian → Danish            P527             P527 → NA               7.16 → −0.30
             TP→FN      Russia → Magnitogorsk            P150             P150 → NA               5.79 → −0.83
             FP@NA      Australia → Australian           P172             NA → P172               −3.23 → 3.41
             FP@NA      Marlborough → Worcester County   P131             NA → P131               −3.41 → 3.05

Table 7: Representative qualitative examples of ATL and ATGL on the Re-DocRED test set. Margin is defined as
logitr - logitT H . FN→TP denotes cases where a relation exists but ATL misses it and ATGL predicts it correctly.
TP→FN denotes cases where ATL predicts the relation correctly but ATGL misses it. FP@NA denotes cases where
no labeled relation exists (N A) but ATGL predicts relation r.


            Loss Function        Training Time                                                                        Test
                                                                      Model                        PLM
            ATL                  40.37 minutes                                                                    Ign-F1      F1
            Balanced-Softmax     40.03 minutes                        DocGNRE *                    Llama3-8B      11.04      11.12
            AML                  40.46 minutes
                                                                      LMRC *                       Llama3-8B      52.15      52.45
            AFL                  40.43 minutes
                                                                      D-F *                        Llama3-8B      52.50      53.33
            NCRL                 39.44 minutes
                                                                      D-R-F *                      Llama3-8B      54.35      54.84
            PEMSCL               40.66 minutes
            HingeABL             41.50 minutes                        AutoRE *                     Llama3-8B      58.33      59.29
            AMTL                 40.46 minutes                        EP-RSR *                     Llama3-8B      63.03      64.24
            ATGL (Ours)          40.71 minutes                        DocKG-RAG †                  Llama3-8B      74.32      75.49
                                                                      ATLOP + ATGL loss (ours)     BERTbase       76.05      77.26
                                                                      ATLOP + ATGL loss (ours)     RoBERTalarge   79.64      80.68
Table 8: Comparison of losses in terms of training time               TTM-RE + ATGL loss (ours)    BERTbase       79.50      80.58
on the ATLOP model with a BERTbase encoder, trained                   TTM-RE + ATGL loss (ours)    RoBERTalarge   81.25      82.13
for 30 epochs on Re-DocRED with a batch size of 4.
                                                                  Table 9: Performance of LLMs on Re-DocRED. * from
                                                                  Zhang et al. (2025); † from Xu et al. (2025c).
eral baseline losses. As shown in Table 8, training
ATLOP with ATGL takes 40.71 minutes, which
is comparable to competitive losses such as ATL                   correcting high-logit misses, i.e., cases where the
(40.37 minutes), AFL (40.43 minutes), and Hinge-                  target relation has a negative margin under ATL
ABL (41.50 minutes). These results demonstrate                    but a large positive margin under ATGL. We also
that ATGL achieves performance gains without in-                  include cases where ATL is correct but ATGL fails,
troducing an additional computational burden.                     as well as representative false positives made by
                                                                  ATGL on entity pairs labeled as N A. These exam-
7.8 Comparison with LLM-based Models                              ples provide a balanced view of both the gains and
To contextualize ATGL beyond the ATL-based                        trade-offs of ATGL.
losses, we further compare it with representative
                                                                  8      Conclusion
LLM-based DocRE models on the Re-DocRED
dataset, as shown in Table 9. ATLOP trained                       We propose a novel Adaptive-Threshold Global
with ATGL achieves 77.26/80.68 F1 on the test                     Loss, ATGL, which effectively mitigates the thresh-
set with BERTbase /RoBERTalarge , outperforming                   old instability caused by decoupled optimization
DocKG-RAG (75.49) and substantially exceeding                     and the optimization bias exacerbated by severe
earlier prompting-based methods such as EP-RSR                    class imbalance in DocRE tasks. ATGL proposes
(64.24). With a stronger backbone, TTM-RE com-                    to integrate positive, negative, and threshold classes
bined with ATGL further improves performance to                   into a unified logit space and explicitly enforces
80.58/82.13 F1, indicating that the effectiveness of              global ranking constraints on their contributions.
ATGL persists across different encoder strengths.                 Experiments on four datasets show that ATGL con-
                                                                  sistently surpasses other DocRE losses, and im-
7.9 Case Study                                                    proves different DocRE models. As a loss that does
To further understand the behavior of ATGL, we                    not depend on any specific model and only changes
present representative qualitative examples from                  the optimization objective, we expect ATGL to be
the Re-DocRED test set in Table 7. Compared                       readily applicable to broader multi-label classifica-
with ATL, ATGL mainly improves performance by                     tion tasks beyond DocRE.
                                                         34710
Limitations                                               Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
                                                             2016. Deep learning, volume 1.
Despite the substantial improvements that ATGL
brings to DocRE, it still exhibits noteworthy lim-        Jia Guo, Stanley Kok, and Lidong Bing. 2023. To-
                                                             wards integration of discriminability and robustness
itations. In particular, while ATGL significantly            for document-level relation extraction. In Proceed-
enhances overall performance and reduces the false           ings of the 17th Conference of the European Chap-
negative (FN) rate, these gains come at the cost of          ter of the Association for Computational Linguistics
an increased false positive (FP) rate. As shown in           (EACL), pages 2598–2609.
Section 7.3 Analysis of Class Imbalance (Table            Jiao Li, Yueping Sun, Robin J Johnson, Daniela Sci-
4), although the overall comparative results show            aky, Chih-Hsuan Wei, Robert Leaman, Allan Peter
that ATGL effectively mitigates class imbalance              Davis, Carolyn J Mattingly, Thomas C Wiegers, and
compared to prior work, ATGL yields more false               Zhiyong Lu. 2016. Biocreative v cdr task corpus:
                                                             a resource for chemical disease relation extraction.
positives than other losses such as ATL, PEMSCL,             Database, 2016.
and HingeABL. This indicates that, by strategically
suppressing the contribution of negative classes          Junpeng Li, Zixia Jia, and Zilong Zheng. 2023. Semi-
                                                            automatic data enhancement for document-level re-
while amplifying positive ones, ATGL behaves                lation extraction with distant supervision from large
more aggressively when predicting the existence of          language models. In Proceedings of the 2023 Con-
relations for an entity pair, thereby inducing a trade-     ference on Empirical Methods in Natural Language
off between precision and recall. Such a trade-off          Processing (EMNLP), pages 5495–5505.
may require careful calibration depending on the          Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
needs of specific application scenarios.                    dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
                                                            Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Acknowledgments                                             Roberta: A robustly optimized BERT pretraining
                                                            approach. arXiv, abs/1907.11692.
The authors sincerely thank the anonymous review-
ers for their valuable comments and suggestions,          Youmi Ma, An Wang, and Naoaki Okazaki. 2023.
                                                            Dreeam: Guiding attention with evidence for improv-
which have greatly improved this paper. This work           ing document-level relation extraction. In Proceed-
is supported by the National Natural Science Foun-          ings of the 17th Conference of the European Chap-
dation of China (62276057).                                 ter of the Association for Computational Linguistics
                                                            (EACL), pages 1971–1983.
                                                          Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-
References                                                  apu Wang, and Xindong Wu. 2024. Unifying large
Jinheon Baek, Soyeong Jeong, Minki Kang, Jong C             language models and knowledge graphs: A roadmap.
   Park, and Sung Hwang. 2023.           Knowledge-         IEEE Transactions on Knowledge and Data Engi-
   augmented language model verification. In Proceed-       neering (TKDE), 36(7):3580–3599.
   ings of the 2023 Conference on Empirical Methods
   in Natural Language Processing (EMNLP), pages          Qingyu Tan, Ruidan He, Lidong Bing, and Hwee Tou
   1720–1736.                                               Ng. 2022a. Document-level relation extraction with
                                                            adaptive focal loss and knowledge distillation. In
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and               Findings of the Annual Meeting of the Association
   Kristina Toutanova. 2019. Bert: Pre-training of deep     for Computational Linguistics (ACL), pages 1672–
   bidirectional transformers for language understand-      1681.
   ing. In Proceedings of the 2019 Conference of the
  North American Chapter of the Association for Com-      Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and
   putational Linguistics: Human Language Technolo-         Sharifah Mahani Aljunied. 2022b. Revisiting docred-
   gies, Volume 1 (NAACL-HLT), pages 4171–4186.             addressing the false negative problem in relation ex-
                                                            traction. In Proceedings of the 2022 Conference on
Zhichao Duan, Tengyu Pan, Zhenyu Li, Xiuxing Li, and        Empirical Methods in Natural Language Processing
  Jianyong Wang. 2025. Comm: Concentrated mar-              (EMNLP), pages 8472–8487.
  gin maximization for robust document-level relation
  extraction. In Proceedings of the AAAI Conference       Khai Phan Tran, Wen Hua, and Xue Li. 2025. Vaediff-
  on Artificial Intelligence (AAAI), volume 39, pages       docre: End-to-end data augmentation framework for
  23841–23849.                                              document-level relation extraction. In Proceedings of
                                                            the 31st International Conference on Computational
Chufan Gao, Xuan Wang, and Jimeng Sun. 2024. TTM-           Linguistics (COLING), pages 7307–7320.
  RE: memory-augmented document-level relation ex-
  traction. In Proceedings of the 62nd Annual Meet-       Jize Wang, Xinyi Le, Xiaodi Peng, and Cailian Chen.
  ing of the Association for Computational Linguistics       2023. Adaptive hinge balance loss for document-
  (ACL).                                                     level relation extraction. In Findings of Conference
                                                     34711
  on Empirical Methods in Natural Language Process-        Bowen Zhang and Harold Soh. 2024. Extract, define,
  ing (EMNLP), pages 3872–3878.                              canonicalize: An llm-based framework for knowl-
                                                             edge graph construction. In Proceedings of the 2024
Ye Wang, Xinxin Liu, Wenxin Hu, and Tao Zhang. 2022.         Conference on Empirical Methods in Natural Lan-
  A unified positive-unlabeled learning framework for        guage Processing (EMNLP), pages 9820–9836.
  document-level relation extraction with different lev-
  els of labeling. In Proceedings of the 2022 Con-         Fu Zhang, Qi Miao, Jingwei Cheng, Hongsen Yu,
  ference on Empirical Methods in Natural Language           Yi Yan, Xin Li, and Yongxue Wu. 2024. Srf: en-
  Processing (EMNLP), pages 4123–4135.                       hancing document-level relation extraction with a
                                                             novel secondary reasoning framework. In Proceed-
Ying Wei and Qi Li. 2022. Sagdre: Sequence-aware             ings of the 2024 Conference on Empirical Methods
  graph-based document-level relation extraction with        in Natural Language Processing (EMNLP), pages
  adaptive margin loss. In Proceedings of the 28th           15426–15439.
  ACM SIGKDD Conference on Knowledge Discovery
  and Data Mining (KDD), pages 2000–2008.
                                                           Fu Zhang, Hongsen Yu, Jingwei Cheng, and Huang-
                                                             ming Xu. 2025. Entity pair-guided relation sum-
Huangming Xu, Fu Zhang, and Jingwei Cheng. 2025a.            marization and retrieval in llms for document-level
  An adaptive multi-threshold loss and a general frame-      relation extraction. In Findings of the Association
  work for collaborating losses in document-level re-        for Computational Linguistics: NAACL 2025, pages
  lation extraction. In Findings of the Association          4022–4037.
  for Computational Linguistics (ACL), pages 20996–
  21007.
                                                           Liang Zhang, Zijun Min, Jinsong Su, Pei Yu, Ante
Huangming Xu, Fu Zhang, Jingwei Cheng, and Xin Li.           Wang, and Yidong Chen. 2023. Exploring effec-
  2025b. ARPDL: adaptive relational prior distribu-          tive inter-encoder semantic interaction for document-
  tion loss as an adapter for document-level relation        level relation extraction. In Proceedings of the Thirty-
  extraction. In Proceedings of the Thirty-Fourth Inter-     Second International Joint Conference on Artificial
  national Joint Conference on Artificial Intelligence       Intelligence (IJCAI), pages 5278–5286.
  (IJCAI), pages 8313–8321.
                                                           Ningyu Zhang, Xiang Chen, Xin Xie, Shumin Deng,
Xiaolong Xu, Chenbin Li, Haolong Xiang, Lianyong Qi,         Chuanqi Tan, Mosha Chen, Fei Huang, Luo Si, and
  Xuyun Zhang, and Wanchun Dou. 2024. Attention              Huajun Chen. 2021. Document-level relation extrac-
  based document-level relation extraction with none         tion as semantic segmentation. In Proceedings of the
  class ranking loss. In Proceedings of the Thirty-Third     Thirtieth International Joint Conference on Artificial
  International Joint Conference on Artificial Intelli-      Intelligence (IJCAI), pages 3999–4006.
  gence (IJCAI), pages 6569–6577.
                                                           Wenxuan Zhou, Kevin Huang, Tengyu Ma, and Jing
Xiaolong Xu, Yibo Zhou, Haolong Xiang, Xiaoyong             Huang. 2021. Document-level relation extraction
  Li, Xuyun Zhang, Lianyong Qi, and Wanchun Dou.            with adaptive thresholding and localized context pool-
  2025c. Docks-rag: Optimizing document-level rela-         ing. In Proceedings of the AAAI Conference on Arti-
  tion extraction through llm-enhanced hybrid prompt        ficial Intelligence (AAAI), volume 35, pages 14612–
  tuning. In Forty-second International Conference          14620.
  on Machine Learning, ICML 2025, Vancouver, BC,
  Canada, July 13-19, 2025.                                Yang Zhou and Wee Sun Lee. 2022. None class rank-
                                                             ing loss for document-level relation extraction. In
Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin,            Proceedings of the Thirty-First International Joint
  Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou,          Conference on Artificial Intelligence (IJCAI), pages
  and Maosong Sun. 2019. Docred: A large-scale               4538–4544.
  document-level relation extraction dataset. In Pro-
  ceedings of the 57th Annual Meeting of the Associ-
  ation for Computational Linguistics (ACL), pages         A    Theoretical Analysis of ATGL
  764–777.
                                                           Overview. Our goal is to formalize three key prop-
Klim Zaporojets, Johannes Deleu, Chris Develder, and
                                                           erties: (i) ATGL corresponds to matching a struc-
  Thomas Demeester. 2021. DWIE: an entity-centric
  dataset for multi-task document-level information        tured target distribution that encodes the desired
  extraction. Inf. Process. Manag., 58(4):102563.          group-wise ordering; (ii) the optimization land-
                                                           scape in logits is convex (modulo translation) and
Shuang Zeng, Runxin Xu, Baobao Chang, and Lei Li.          admits a unique minimizer in probability space;
  2020. Double graph based reasoning for document-
  level relation extraction. In Proceedings of the 2020
                                                           (iii) the optimum induces a strict group-wise logit
  Conference on Empirical Methods in Natural Lan-          ranking (e.g., srp > sT H > srn ) and yields stable,
  guage Processing (EMNLP), pages 1630–1640.               interpretable gradient dynamics for the threshold.
                                                      34712
A.1 Group-Structured Objective of ATGL                             A.2   Convexity, Hessian, and Uniqueness
Purpose (A.1). We first rewrite ATGL in a unified                  Purpose (A.2). We analyze the optimization prop-
softmax form and show that its class weights in-                   erties of ATGL: we derive the gradient and Hessian
duce a structured target distribution over C, where                with respect to logits, establish convexity (modulo
positive classes, the threshold, and negative classes              the translation invariance of softmax), and charac-
play distinct roles. This view will be the basis for               terize the uniqueness of the optimal distribution in
the convexity and ranking analyses below.                          the probability simplex together with its attainabil-
   Recall that for an entity pair T = (es , eo ), ATGL             ity under finite logits.
integrates positive, threshold, and negative classes
                                                                   Gradient.    Using log pr = sr − log Z with Z =
into a single softmax (Eq. (6)). For notational                    P
                                                                         exp(s j ), we can rewrite
convenience, we define                                               j∈C
                                                                                        X
                   R = PT ∪ NT ,                            (10)           LATGL = −          wr (sr − log Z)      (18)
                                                                                        r∈C
                    C = R ∪ {T H},                          (11)                        X
                                                                                   =−         wr sr + W log Z.     (19)
and for each class r ∈ C,                                                               r∈C

                                          exp(sr )                 Here j, k ∈ C index classes (relations and the
   sr = logitr ,       pr = P                           .   (12)
                                        r′ ∈C exp(sr′ )            threshold), and sk denotes the logit of class k. Tak-
The per-instance ATGL loss can be written as                       ing the derivative with respect to sk gives
                      X                                             ∂LATGL             exp(sk )
            LATGL = −     wr log pr ,        (13)                            = −wk + W          = W pk − wk .
                             r∈C
                                                                      ∂sk                Z
                                                                                                          (20)
where the weights are                                              Equivalently,
                
                
                α > 1, r ∈ PT ,                                                1 ∂LATGL
                                                                                         = pk − w̃k ,              (21)
          wr = 1,          r = T H,                                             W ∂sk
                
                
                   0,      r ∈ NT ,                         (14)   i.e., the gradient is proportional to the difference
                X                                                  between model probability and target probability.
           W =       wr = α|PT | + 1.
                     r∈C                                           Hessian and convexity. The Jacobian of p with
  We normalize these weights as:                                   respect to s is the standard softmax Jacobian:
                           wr                                                  ∂pk
                   w̃r =      ,     r ∈ C.                  (15)                   = pk (I[k = j] − pj ),          (22)
                           W                                                   ∂sj
       P
Then       r∈C w̃r = 1, and Eq. (13) can be rewritten:             so the Hessian of LATGL reads
                                  X
             LATGL = −W                 w̃r log pr .        (16)     ∂ 2 LATGL       ∂pk                        
                                  r∈C                                           =W       = W pk I[k = j] − pk pj .
                                                                      ∂sj ∂sk        ∂sj
   The normalized weights w̃r can be interpreted                                                               (23)
as the relative contribution of positive, threshold,               In matrix form,
and negative classes for a given entity pair:                                                            
                                                                           ∇2s LATGL = W Diag(p) − pp⊤ .      (24)
                   
                   α/W, r ∈ PT ,
                   
                                                                   For any vector v ∈ R|C| ,
             w̃r = 1/W, r = T H,                (17)
                   
                                                                                         X             X           
                   0,        r∈N .                                  v ⊤ ∇2s LATGL v = W     pr vr2 − (   p r vr )2
                                               T
                                                                                              r           r
   In particular, the positive classes r ∈ PT share                                                                (25)
the same normalized weight w̃r = α/W , the                                          = W · Varr∼p (vr ) ≥ 0.        (26)
threshold class satisfies w̃T H = 1/W , and the
negative classes r ∈ NT have w̃r = 0 while still                     Therefore ∇2s LATGL is positive semi-definite,
entering the softmax normalization through pr .                    and LATGL is convex in s. Moreover, the Hessian
                                                              34713
is not positive definite: its nullspace is spanned by          Therefore the unique minimizer in the probability
the all-ones vector, reflecting the standard softmax           simplex is
translation invariance s 7→ s + c · 1, i.e.,                                    
                                                                                
   LATGL (s + c1) = LATGL (s),             ∀c ∈ R.      (27)                    α/W, r ∈ PT ,
                                                                                
                                                                            ∗
                                                                           pr = 1/W, r = T H,              (34)
Hence LATGL is convex but not strictly convex in                                
                                                                                
                                                                                0,      r∈N ,       T
s (modulo the translation subspace); in particular,
any local minimizer (if it exists) is globally optimal         which coincides with the normalized weights in
up to a global shift.                                          Eq. (17).
Unique minimizer in probability space. Con-                       In the probability simplex, p∗ in Eq. (34) is the
sider the constrained optimization problem                     unique minimizer. Under the softmax parameter-
               X                                               ization, the solution p∗ is not attained by any fi-
     min −         wr log pr ,                                 nite logits s; rather, inf s LATGL (s) is approached
     {pr }
              X
                  r∈C
                                                        (28)   in the limit srn → −∞ for all rn ∈ NT . The
      s.t.          pr = 1, pr ≥ 0        ∀r ∈ C.              relative logits of the positive and threshold classes
              r∈C                                              are nevertheless unique up to a global shift, since
   Since wr = 0 for all r ∈ NT , the objective is              LATGL (s) = LATGL (s + c1) for any c ∈ R.
independent of {pr }r∈NT ; if some pr > 0 with                 A.3     Global Ranking
r ∈ NT , we can shift an arbitrarily small amount
of probability from pr to a class in PT ∪ {T H} to             Purpose (A.3). Based on the optimal probability
strictly decrease the objective while preserving the           distribution obtained in Appendix A.2, we show
constraints, so at any minimizer we must have                  that ATGL enforces a strict group-wise ordering
                                                               among logits. In particular, positive classes are sep-
                  p∗r = 0,      ∀r ∈ NT .               (29)   arated from the threshold by a logit gap controlled
  Restricting to the remaining classes PT ∪ {T H},             by α, while negative classes are pushed below the
the problem in Eq. (28) reduces to                             threshold in the softmax limit.
                    X                                             At the optimal probability distribution p∗ in
      min −                 wr log pr                          Eq. (34), we obtain a strict group-wise ranking
      {pr }
                  r∈PT ∪{T H}                                  in the logit space. For any positive class rp ∈ PT ,
                    X                                          we have
       s.t.                    pr = 1,                  (30)
              r∈PT ∪{T H}                                                        p∗rp    α/W
                                                                                      =        = α > 1.          (35)
              pr ≥ 0        ∀r ∈ PT ∪ {T H}.                                    p∗T H   1/W
The associated Lagrangian is                                   Under the softmax parameterization,
                    X
  J (p, λ) = −               wr log pr                               p∗rp        exp(srp )
                     r∈PT ∪{T H}                                            =              = exp(srp − sT H ),   (36)
                                                                p∗T H         exp(sT H )
                                                        (31)
                               X
                  + λ                    pr − 1  .           which yields
                            r∈PT ∪{T H}
                                                                                srp − sT H = log α > 0.          (37)
Taking derivatives with respect to pr and setting
them to zero yields, for all r ∈ PT ∪ {T H},                   Hence srp > sT H for all rp ∈ PT .
                                                                  For any negative class rn ∈ NT , Eq. (34) im-
   ∂J     wr                                     wr
       =−    +λ=0                 ⇒       pr =      .   (32)   plies p∗rn = 0. In the softmax parameterization, this
   ∂pr    pr                                     λ
                                 P                             corresponds to the limit srn → −∞, and therefore
Enforcing the constraint             r∈PT ∪{T H} pr     = 1    for any finite sT H we have sT H > srn . Combining
gives                                                          the two yields
      X       wr
                 =1                                               srp > sT H > srn ,    ∀rp ∈ PT , ∀rn ∈ NT ,
              λ
   r∈PT ∪{T H}
                                X                       (33)                                                  (38)
              ⇒     λ=                    wr = W.              which formalizes the global ranking between posi-
                            r∈PT ∪{T H}                        tive, threshold, and negative classes at optimum.
                                                          34714
A.4 Gradient Dynamics and Threshold                     Monotone suppression of negatives.          For any
    Stability                                           negative r ∈ NT , Eq. (41) implies
Purpose (A.4). Finally, we study the gradient dy-                     ∂LATGL
namics implied by ATGL to explain why the thresh-                            = W pr ≥ 0.                (45)
                                                                       ∂sr
old is stable during training. We show that the
logit of threshold is driven toward a fixed target      Whenever pr > 0, gradient descent satisfies
probability, positive classes are amplified toward a
                                                                 s(t+1)
                                                                  r     = s(t)     (t)  (t)
                                                                           r − ηW pr < sr ,             (46)
larger target mass, and negative classes are mono-
tonically suppressed, interacting with the threshold    so the logits of negative classes are always pushed
only through the shared normalization.                  downward. Importantly, negative classes never
   The gradients in Eq. (20) describe how positive      contribute negative gradients to sT H or the pos-
classes, the threshold, and negative classes interact   itive classes; they influence them only through the
during optimization.                                    shared softmax normalization.
Group-specific gradients.      From Eq. (20) and
                                                        B   Details of Datasets
Eq. (14), we have
                                                        • CDR (Li et al., 2016) is a high-quality,
     ∂LATGL
            = W pr − α,           r ∈ PT ,      (39)      community-annotated corpus that is essential for
       ∂sr
                                                          biomedical text-mining and relation extraction.
     ∂LATGL
            = W pT H − 1,                       (40)    • DWIE (Zaporojets et al., 2021) is a document-
      ∂sT H                                               level, entity-centered information extraction
     ∂LATGL                                               dataset covering four subtasks. It is sourced from
            = W pr ,              r ∈ NT .      (41)
       ∂sr                                                Deutsche Welle’s English online content, pro-
Assuming gradient descent with learning rate η >          viding realistic annotations and rule labels for
0,                                                        evaluating DocRE methods, with 602 training,
                           ∂LATGL                         98 development, and 99 test documents.
          s(t+1)
           r     = s(t)
                    r −η            .        (42)
                             ∂sr
                                (t)                     • Re-DocRED (Tan et al., 2022b) is constructed
Self-correcting threshold. Eq. (40) shows that            based on the DocRED (Yao et al., 2019) dataset
the sign of the threshold gradient is determined          to address the annotation incompleteness in the
solely by pT H :                                          original dataset and undergoes further manual
                                                         verification. Its training set is derived from the
                  > 0, pT H > 1/W,                       training set of DocRED, comprising a total of
         ∂LATGL                                          3,053 documents. The development and test sets
                    = 0, pT H = 1/W,         (43)
          ∂sT H                                         are split from the original development set, each
                    < 0, pT H < 1/W.
                                                          containing 500 documents, and these subsets are
Consequently, under gradient descent, the thresh-         also manually re-verified.
old logit is self-correcting around its target prob-    • DocGNRE (Li et al., 2023) is an enhanced ver-
ability 1/W : if pT H > 1/W , sT H is decreased,          sion of Re-DocRED (Tan et al., 2022b) dataset.
and if pT H < 1/W , sT H is increased.                    It is built upon the original Re-DocRED by semi-
                                                          automatically supplementing missing relation
Amplified positive classes. Similarly, for any
                                                          triples. DocGNRE follows the same document
positive class r ∈ PT ,
                                                          split as Re-DocRED, with 3,053 documents in
                   
                   < 0, pr < α/W,                        the training set and 500 documents in the test set.
           ∂LATGL 
                     = 0, pr = α/W,        (44)         C   Analyzing Hyperparameters
             ∂sr  
                     > 0, pr > α/W,
                                                        To analyze the effect of the hyperparameter α on
thus the logits of positive classes are increased       ATGL, we conduct experiments on four DocRE
when their probabilities are below the target α/W       datasets. As shown in Fig. 4, ATGL exhibits stable
and decreased otherwise, giving positive classes a      performance over a wide range of α values. Specif-
larger target probability than the threshold (α/W       ically, the F1 first increases as α grows, reaches a
vs. 1/W ), which is beneficial when positive exam-      peak, and then shows slight degradation when α
ples are scarce.                                        becomes excessively large. For instance, the best
                                                   34715
                                       Dataset                            Task Type                   #Rels       #Triples   #Triples/#Rels   Selected α
                                       CDR                        Binary (no NA)                          2       5,240         2620.0          7.5
                                       DWIE                         Multi-label                           65      14,403         221.6           6.0
                                       Re-DocRED                    Multi-label                           96      85,932         895.1          16.5
                                       DocGNRE                      Multi-label                           96      96,505        1005.3          20.0

                                                                       Table 10: Dataset statistics for selecting α.


                           CDR
                         70.23 70.19                                      72.45 DWIE
                                                                                            72.24
  70.2                                                              72.07    72.35 72.16
                 70.07                                    72                    71.73
                                                                                   71.63          71.46
  70.0                                           69.99                 71.47                   71.46
                                                                 71.14                               71.11
                                                          71                             71.08
                                         69.81
F169.8                                                   F1
         69.68                                            70

  69.6
                                                          69

  69.469.36                                                 68.06
                                                          68
         4        6          8       10           12          2       4      6     8    10      12   14
                      hyperparameter                                         hyperparameter


                  (a) CDR                                                   (b) DWIE
                         Re-DocRED                                              DocGNRE 76.90
  77.0                       77.01                        76.9
                                 76.94          76.98                                           76.87
                                                                                                   76.82
                                                          76.8                       76.79             76.78
  76.9                                 76.9176.8776.89                                  76.79
             76.8476.78 76.80                             76.7
  76.8                                                                           76.69
         76.75        76.78
F176.7 76.72                                              76.6
                                                         F1
                                                              76.55

  76.6                                                    76.5    76.51 76.52
                                                                     76.44
  76.5                                                    76.4
      76.43                                                                  76.32
  76.47.5 10.0 12.5 15.0 17.5 20.0 22.5                   76.3
                                                                      10        15          20            25
                hyperparameter                                               hyperparameter


             (c) ReDocRED                                             (d) DocGNRE

Figure 4: Performance of ATGL loss across different α
values on four datasets, using BERTbase with ATLOP.


performance is achieved at α=6 on DWIE, α=16.5
on Re-DocRED, α=20 on DocGNRE and α=7.5
on CDR. These results indicate that while ATGL
is generally robust to the choice of α, moderate
tuning within the high-performing region yields ad-
ditional gains without requiring exhaustive search.
   From these results and the dataset statistics in Ta-
ble 10, we derive a practical guideline for selecting
α. For typical multi-label DocRE benchmarks with
a dominant NA class, α is mainly determined by
the label space size (#Rels), while #Facts/#Rels can
be used as a secondary cue for minor adjustment.
In practice, we first choose a default α0 according
to #Rels (e.g., around 7 for smaller label spaces
and 16–20 for larger ones), and then optionally re-
fine it using #Facts/#Rels. If a development set is
available, a simple 3-point search around α0 , i.e.,
{0.5α0 , α0 , 2α0 }, is usually sufficient. Since CDR
is a binary task without an N A label, its optimal
α is not directly comparable; for binary or non-NA
settings, a larger α (e.g., α ≈ 7.5) is recommended.

                                                                                                               34716
