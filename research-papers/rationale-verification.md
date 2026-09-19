                                     The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)




                                      Enhancing Relation Extraction
                            via Supervised Rationale Verification and Feedback
               Yongqi Li1 , Xin Miao1 , Shen Zhou1 , Mayi Xu1 , Yuyang Ren1,3 , Tieyun Qian1,2 *
                                         1
                                      School of Computer Science, Wuhan University, China
                        2
                        Intellectual Computing Laboratory for Cultural Heritage, Wuhan University, China
                                      3
                                        Research Institute of Nuclear Power Operation, China
                  {liyongqi, miaoxin, shenzhou, xumayi}@whu.edu.cn, renyy@cnnp.com.cn, qty@whu.edu.cn

                             Abstract                                                                                   Feedback
                                                                                        In-context
                                                                                                                       Rationale        Find code,
  Despite the rapid progress that existing automated feedback                     (a)
                                                                                          Demos
                                                                                                                                        calculation
                                                                                                                                                        Final
  methods have made in correcting the output of large language                                                                                         Output
                                                                                        Instance          LLMs         Prediction       error, etc.
  models (LLMs), these methods cannot be well applied to the
  relation extraction (RE) task due to their designated feedback                                     Re-selected Demos for RE as Feedback
  objectives and correction manner. To address this problem,
                                                                                        In-context
  we propose a novel automated feedback framework for RE,                                 Demos
                                                                                                                       Rationale
                                                                                                                                       Relation Bias    Final
  which presents a rationale supervisor to verify the rationale                   (b)                                                  Verification    Output
                                                                                        Instance           LLMs        Prediction
  and provides re-selected demonstrations as feedback to cor-
  rect the initial prediction. Specifically, we first design a causal
  intervention and observation method to collect biased/unbi-                   Figure 1: Comparison between current automated feedback
  ased rationales for contrastive training the rationale supervi-               methods (a) and ours (b). The main difference is that our
  sor. Then, we present a verification-feedback-correction pro-                 rationale supervisor can verify whether the relation bias oc-
  cedure to iteratively enhance LLMs’ capability of handling                    curs and provide re-selected demonstrations as feedback.
  the RE task. Extensive experiments prove that our proposed
  framework significantly outperforms existing methods.

Code — https://github.com/NLPGM/SRVF                                            and Su 2023) to improve the performance. The verification
                                                                                and feedback mechanism for correcting the biased predic-
                                                                                tion is still missing from current LLM based RE research.
                         Introduction                                              To fill this gap, in this study, we focus on exploring the
The relation extraction (RE) task aims to extract the se-                       verification and feedback mechanism (Pan et al. 2023) of
mantic relation between entities in the text, which is an                       LLMs for RE. Specifically, we aim to examine whether the
important task in information extraction. Unlike previous                       relation prediction of LLMs is biased by verifying the ratio-
fine-tuning strategies based on small language models (Wu                       nale (the generated explanation when LLMs perform RE)
and He 2019), recent studies (Wan et al. 2023; Ma et al.                        and providing feedback for correction. However, the cur-
2023) leverage the strong instruction understanding abili-                      rent verification and feedback mechanism faces the follow-
ties and rich intrinsic knowledge of large language models                      ing two problems when being applied to RE.
(LLMs) (Ouyang et al. 2022; Touvron et al. 2023; Bai et al.                        Firstly, existing methods are mainly designed for other
2022) to enhance the performance of RE.                                         tasks, e.g., the reasoning task. The objectives of their feed-
   Despite their significant progress, LLM based methods                        back are also tailored for those tasks, e.g., correcting code,
may suffer from relation bias when performing relation ex-                      factual, or calculation errors in initial responses (Zhang et al.
traction. For example, given a sentence “data is derived                        2023; Gou et al. 2023), or choosing an optimal prefix for
from a study”, where “data” and “study” form the “Entity-                       the next step in multi-step reasoning (Khalifa et al. 2023),
Origin” relation, LLMs may be influenced by the pre-                            as shown in Fig. 1 (a). For example, for the mathematical
trained knowledge and have the stereotype that “data is the                     reasoning task, Self-Refine (Madaan et al. 2023) utilizes the
product that someone produces”, thus making a biased rela-                      LLM agent to find calculation errors in the initial answer and
tion prediction “Product-Producer”, which ignores that the                      provide error information as feedback to correct the answer.
real producer is investigators (producer of the study). Fur-                    However, such feedback objectives are based on the logical
thermore, existing LLM based RE methods focus on the pre-                       properties of reasoning tasks, which are not available for RE.
selection of in-context demonstrations (Wan et al. 2023; Ma,                       Secondly, existing methods (Madaan et al. 2023; Nathani
Li, and Zhang 2023) or instruction design (Zhang, Gutiérrez,                   et al. 2023) do not include demonstrations in their feedback.
    * Corresponding author.                                                     However, the demonstrations are essential for RE even at
Copyright © 2025, Association for the Advancement of Artificial                 the correction stage. This is because without demonstrations
Intelligence (www.aaai.org). All rights reserved.                               in the feedback, the RE task would degrade to zero-shot


                                                                        24521
RE and is harder than the initial few-shot one. Moreover,                                          Related Work
the demonstrations in initial few-shot RE cannot be directly
                                                                           LLMs for Relation Extraction Recently, many stud-
used in feedback since they will mislead the model back to
                                                                           ies (Xu et al. 2023; Li et al. 2023a; Wei et al. 2023; Wad-
the initial one, and thus the impact of feedback is discarded.
                                                                           hwa, Amir, and Wallace 2023; Li, Wang, and Ke 2023) have
   To address the above problems, we propose a novel auto-                 explored how to unlock the potential of LLMs for the RE
mated feedback framework for RE, which trains a rationale                  task, including designing the in-context demonstration se-
supervisor based on a BERT-like small model and utilizes it                lection strategy (Wan et al. 2023; Ma, Li, and Zhang 2023;
to not only verify the prediction but also provide new demon-              Pang et al. 2023) and optimizing instruction patterns (Zhang,
stration improved feedback for correction during the infer-                Gutiérrez, and Su 2023; Wang et al. 2023a; Ma et al. 2023).
ence. As shown in Fig. 1 (b), our rationale supervisor pro-                Despite great success, these methods rely solely on optimiz-
vides re-selected demonstrations as feedback for correcting                ing the initial prompt to improve performance. However, we
the initial prediction of LLMs.                                            find that due to the relation bias, LLMs may still confuse
                                                                           certain relations with similar entities and thus make biased
   In order to train a rationale supervisor, we need to col-               predictions. To alleviate this issue, we introduce the idea of
lect both unbiased and biased rationales, i.e., positive and               automated feedback to RE for the first time, expecting to
negative samples. Though several verification methods have                 correct biased predictions via the provided feedback.
been proposed to collect positive and negative rationales in
other tasks, both their purpose and the collection method are              LLMs with Automated Feedback Some researchers
not suitable for our RE task. (1) Firstly, their collected posi-           have exploited the automated feedback for correcting the
tive and negative rationales are used for training the verifier,           undesirable output of LLMs (Pan et al. 2023; Kamoi et al.
which only needs to discriminate the positive predictions                  2024). However, the feedbacks in existing methods are de-
from negative ones. In contrast, the rationale supervisor in               signed for correcting various reasoning mistakes, e.g., code
our framework is designed to correct biased predictions, thus              errors (Zhang et al. 2023), factual errors (Gou et al. 2023),
needing to further discriminate different negative rationales.             calculation errors (Nathani et al. 2023; Madaan et al. 2023;
(2) Secondly, the way of collecting rationales in current ver-             Paul et al. 2023), or as an optimal prefix for the next step in
ification methods relies on the manually annotated golden                  multi-step reasoning (Khalifa et al. 2023; Li et al. 2023b).
reasoning steps as positive samples and perform rule-based                 These feedbacks are dependent on the reasoning task and
perturbation (Paul et al. 2023; Golovneva et al. 2023) or er-              unavailable for RE. Moreover, they do not include the
ror step alignment (Khalifa et al. 2023; Li et al. 2023b) to               demonstrations which are essential for RE. To address this
obtain negative samples. Unfortunately, such annotated sam-                issue, we propose a novel automated feedback framework
ples and rules for perturbation are not available in RE.                   which provides re-selected demonstrations as feedbacks to
                                                                           help LLMs correct the biased prediction.
   In view of this, we propose a causal intervention and
observation method to address the lack of annotated ratio-                                              Method
nales and collect biased rationales for training the supervi-
sor. Specifically, we first present a label-guided intervention            This section presents our proposed supervised rationale ver-
strategy to collect unbiased rationales, and we also present a             ification and feedback (SRVF) framework for the RE task.
diversified intervention strategy to collect biased rationales.            Task Formulation Given a set of pre-defined relation
In addition, during the inference, we utilize the rationale                types YD , the relation extraction (RE) task aims to predict
supervisor to retrieve new demonstrations from the labeled                 the relation type y ∈ YD between the head entity eh and
samples and include them in the feedback, which are then                   the tail entity et of each test example x = {s, eh , et }, where
used by the LLM for re-generating predictions. Since the                   s denotes the sentence. In this study, we adopt in-context
supervisor has learned the difference among various biased                 learning (ICL) with the rationale to prompt LLMs for the
rationales, the LLM gets the signal to adjust its direction for            RE task. Specifically, for each test example x, we need to
correction. This verification-feedback-correction procedure                randomly select or retrieve m initial in-context demonstra-
iterates until the output rationale is verified as unbiased.               tions Dicl = {{x1 , r1u , y1 }, ..., {xm , rm
                                                                                                                       u
                                                                                                                         , ym }} related to x
                                                                                                            1
                                                                           from the labeled dataset Dl . Then, the LLM fθ with pa-
   Overall, we make three major contributions. 1) We ex-
                                                                           rameters θ is expected to output the relation type y ∈ YD
tend the LLM based RE research to the automated feedback
                                                                           between eh and et , along with the rationale r, denoted as
paradigm, which equips LLM with the ability of correct-
                                                                           {r, y} = fθ (Dicl , x).
ing the biased prediction. 2) We propose a novel supervised
rationale verification and feedback framework, which first                 Overview In this paper, we propose a rationale verifica-
collects rationales with a causal intervention and observa-                tion and feedback framework to guide LLMs towards bet-
tion method for training the supervisor, and then employs                  ter predictions for RE iteratively. Generally, this framework
the supervisor to retrieve sample-related demonstrations as
feedback for guiding the LLM in correction. 3) Extensive                       1
                                                                                 Since there is no annotated golden rationale in the original
experiments prove that our proposed method can improve                     dataset, we add the induced unbiased rationale in the following sec-
the performance of LLM based RE methods and is superior                    tion to Dl to enable it for the setup of ICL with the rationale, i.e.,
to existing automated feedback methods.                                    Dl = {{x1 , r1u , y1 }, ..., {xn , rnu , yn }}.


                                                                   24522
consists of three phases: 1) causal intervention and observa-                             R                   R              do(R)                         R
tion for rationale collection, 2) contrastive training rationale
supervisor, and 3) rationale verification and feedback.                        X      B           X      B           X   B              do(I)   X      B

   Specifically, we first adopt the causal intervention and ob-                           Y                  do(Y)            Y                            Y
servation method to collect unbiased and biased rationales,                    (a) Original SCM   (b) Induce Unbiased Rationale      (c) Observe Biased Rationale
i.e., Ru and Rb . Then, we use Ru and Rb to train the ratio-
nale supervisor Rγ with parameters γ. Finally, as shown in                  Figure 2: The structure causal model for illustrating the pro-
Fig. 3, in the inference time, once the output rationale r is               posed causal intervention and observation strategy.
verified as a biased one by Rγ , we use Rγ to retrieve feed-
back demonstrations Df b based on r, where Df b ⊂ Dl . The
feedback demonstrations are used for re-generating r and y                  these rules are designed based on the logical properties of
using ICL, i.e., {r, y} = fθ (Df b , x). The procedure iterates             reasoning tasks, which are not available in RE. To tackle
until the rationale r is verified as unbiased, and the corre-               this problem, we propose a diversified intervention strategy
sponding relation prediction y will finally be output.                      for collecting the biased rationales.
                                                                               Specifically, for the labeled sample {xi , yi }, we first ran-
Causal Intervention and Observation for Rationale
                                                                            domly select a demonstration set Ddi with diverse labels,
Collection                                                                  where Ddi ⊂ Dl and the label of each demonstration in Ddi
Generally, during this phase, for each labeled sample                       is not equal to yi . The diversity of labels in Ddi is designed
{xi , yi }, we aim to collect the unbiased rationale corre-                 to induce LLMs to make diverse errors on the same sample,
sponding with the golden label {riu , yiu }, as well as the bi-             to increase the diversity of collected biased rationales. Then,
ased rationale with corresponding biased relation prediction                as shown in Fig. 2 (c), we set the in-context demonstration I
{rib , yib }. This process consists of two steps: 1) induce unbi-           as {xj , rju , yj } from Ddi , i.e., do(I = {xj , rju , yj }). Finally,
ased rationale, and 2) observe biased rationale. As shown in                the observed value of rationale R is robs while the observed
Fig. 2, we use the structural causal model (SCM) in causal                  value of rationale Y is yobs . If yobs ̸= yi , we treat the ob-
inference (Pearl et al. 2000) to illustrate the strategy.                   served robs with its corresponding relation prediction yobs as
                                                                            a potentially biased one, i.e., {rib , yib }, and add it to Rb .
Preliminary of SCM As shown in Fig. 2, the SCMs show
the relationships among the input (X), the relation predic-
tion (Y ), the rationale for prediction (R), the certain bias of            Contrastive Training Rationale Supervisor
LLMs (B) and in-context demonstration I. The arrows be-                     We expect the rationale supervisor to 1) verify whether the
tween nodes indicate causal directions. For example, “X →                   output rationale is biased, and 2) provide different feedbacks
R” means that the LLM generates the rationale R related to                  for different bias situations to correct the initial prediction.
the prediction for the sample X. “X → B → R” indicates                      To reach this, we adopt contrastive learning to train the ra-
that the LLM activates some biased knowledge B related                      tionale supervisor to acquire two abilities: 1) discriminating
to the sample X and generates a rationale R influenced by                   biased and unbiased rationales, and 2) learning the differ-
the biased knowledge B. Besides, in Fig. 2 (b), the “do(Y )”                ence of various biased rationales.
indicates that cutting off all factors that could influence the                 We design two kinds of positive and negative pairs for
value of Y and assigning Y a certain value as needed.                       contrastive training.
Induce Unbiased Rationale Previous methods rely on the                          For positive pairs, we treat “unbiased rationales with the
human-annotated rationales, e.g., golden reasoning steps in                 same golden label”, and “biased rationales under the same
mathematical tasks (Khalifa et al. 2023), which are not avail-              bias situation” as the two kinds of positive pairs. For exam-
able in the RE dataset. To address this issue, we propose                   ple, if samples s1 and s2 , which have the same label, are
a label-guided intervention strategy to obtain the unbiased                 also predicted as the same wrong relation, we call “samples
rationale for each labeled sample, which explains why the                   s1 and s2 are in the same bias situation”. Thus, the biased ra-
sample xi should be predicted as the golden label yi .                      tionales (r1b and r2b ) of s1 and s2 , are treated as a positive pair
                                                                            and should be pulled together in the rationale representation
   As shown in Fig. 2 (b), this strategy consists of two steps:
                                                                            space, i.e., r1b →← r2b .
1) cut causal directions that could make bias (B) influence
the prediction (Y ), and let the golden label guide the ratio-                  For negative pairs, we first consider the “biased and un-
nale (R) generation, formally denoted as do(Y = yi ) and                    biased rationales from the same sample” as a negative pair.
do(Y ) → R. The observed generated rationale is R = riu ;                   This is designed to train the rationale supervisor to distin-
2) conduct similar do-operation to the rationale R and let                  guish between biased and unbiased rationales. For example,
do(R) point to Y , i.e., do(R = riu ), do(R) → Y . If the ob-               a sample s1 = {r1u , y1 } where y1 is the golden label and
served value of Y is equal to the golden label yi , we treat                r1u is the corresponding unbiased rationale, is wrongly pre-
{riu , yi } as the unbiased one and add it to Ru .                          dicted as relation y2 and corresponding biased rationale is
                                                                            r1b . Thus, r1b and r1u are treated as a negative pair and should
Observe Biased Rationale In previous methods, incor-                        be pushed away in the rationale representation space, i.e.,
rect rationales are synthesized from golden ones using                      r1b ↔ r1u . Second, we also treat “biased rationales under
perturbation or error step alignment based on certain                       different bias situations” as a negative pair to train the ratio-
rules (Golovneva et al. 2023; Khalifa et al. 2023). However,                nale supervisor, which can distinguish different bias situa-


                                                                    24523
                 Initial In-context Demonstrations                                              Feedback Demonstrations
                                                                                                                                                Prediction
              Instance: The profiles (head entity) are           Feedback                Instance: The profiles (head entity) are              Instrument-
              used by teachers (tail entity) to write better   Demonstration             used by teachers (tail entity) to write better          Agency
              recommendation letters.                            Retrieval               recommendation letters.



                                   Large                                                                      Large                             Unbiased
                                   Language                       Biased                                      Language
                                   Model                                                                      Model



              Rationale: The profiles are the component                                  Rationale: The profiles are tools employed
              that is used by the teachers. Therefore, the                               by teachers to write better letters. Thus,
              "profiles" serves as the "Component" while                                 the "profiles" serves as the "Instrument"
              the "teachers" serves as the "Whole".              Rationale               while "teachers" serves as the "Agency".               Rationale
              Relation Prediction: Component-Whole              Verification             Relation Prediction: Instrument-Agency                Verification

                    (a) Initial Prediction                                                    (b) Refined Prediction

Figure 3: An example of correcting the initial biased prediction of LLMs via the proposed SRVF framework in the inference
time. The rationale supervisor first verifies the initial prediction in (a) as biased. Then, with the feedback demonstrations
retrieved by the rationale supervisor, the LLM makes a correct relation prediction in (b). Note: The rationale supervisor here is
obtained by contrastive training using collected biased and unbiased rationales as described before.


tions and provide feedback based on the biased rationale in                                           Su = {{ru , y u } | {ru , y u } ∈ Ru , y u = y},                   (4)
the inference time.
   In general, the contrastive loss is calculated as:                                       Then, the indicator score to judge whether r is a biased
                                                                                         rationale is calculated as follows:
                   1
                                  P
                ∥S pos ∥                       exp(sim(r1 , r2 )/τ )
                           {r1 ,r2 }∈S pos
 Lcl = − log                                                           , (1)                pb =         max          sim(r, r b ) −          max          sim(r, r u ), (5)
                                                                                                     {r b ,y b }∈Sb                       {r u ,y u }∈Su
                            P
                                               exp(sim(r1 , r2 )/τ )
               {r1 ,r2 }∈(S pos ∪S neg )
                                                                                            where the similarity function sim() is defined in Eq. 2.
              sim(r1 , r2 ) = Rγ (r1 ) · Rγ (r2 )T ,       (2)                           When pb is greater than 0, it implies that the feature of r is
   where S  pos      pos
                = S1 ∪ S2 , Spos   neg        neg    neg
                                         = S1 ∪ S2 . S1    pos                           closer to the feature field of Sb than that of Su , which means
and S2pos denote the two kinds of positive pair set, and S1neg                           r and corresponding relation prediction y should be regarded
and S2neg denote two kinds of negative pair set. Here we                                 as biased, and feedback is needed to correct them.
adopt the dot product as the similarity function sim() and
                                                                                         Feedback Demonstration Retrieval Once the output ra-
add a temperature hyper-parameter τ to focus more on diffi-
                                                                                         tionale r is verified as biased, we need to retrieve a new set
cult pairs (Chen et al. 2020). During the procedure of ratio-
                                                                                         of in-context demonstrations based on the feature of r for
nale contrastive training, the parameters γ of Rγ are updated
                                                                                         guiding LLMs toward correct predictions. Specifically, we
to minimize Lcl .
                                                                                         first select the k most similar biased rationales to r in Sb ,
Rationale Verification and Feedback                                                      denoted as Sbtopk , which is defined as:
As shown in Fig. 3, in the inference time, the trained ra-
                                                                                            Sbtopk = {{rb , y b } | rank{rb ,yb }∈Sb (sim(r, r b )) ≤ k},
tionale supervisor Rγ first verifies whether the prediction
                                                                                                                                                        (6)
is biased. If the prediction is biased, the rationale supervi-
                                                                                         Then, we select the labeled samples corresponding to the
sor will retrieve a feedback demonstration set, which then
guides LLMs toward refined predictions. In this subsection,                              biased rationales in Sbtopk from Dl as the feedback demon-
we will elaborate on the “Rationale Verification” and “Feed-                             strations Df b , which is defined as:
back Demonstration Retrieval” in Fig. 3 in detail. Here we
denote the test example, output rationale, and relation pre-                             Df b = {{xi , riu , yi } | {xi , riu , yi } ∈ Dl , {rib , yib } ∈ Sbtopk },
diction of LLMs as x, r, and y, respectively.                                                                                                                   (7)
                                                                                           where the biased {rib , yib } and unbiased {riu , yi } corre-
Rationale Verification For verification, we need to select                               spond to the same labeled sample {xi , yi }.
the subsets Sb and Su related to the prediction y from Rb
and Ru , respectively, which are then used as anchors to de-                             Correction via In-context Learning After the feedback
termine whether the current output rationale is close to the                             demonstrations Df b are selected, we re-generate r and y us-
biased or unbiased groups. Sb and Su are defined as follows:                             ing the LLM fθ , i.e., {r, y} = fθ (Df b , x). This process will
                                                                                         be iteratively performed until r is verified as unbiased, and
         Sb = {{rb , y b } | {rb , y b } ∈ Rb , y b = y},                  (3)           the corresponding prediction y will be finally output.


                                                                                 24524
                                               SemEval                         TACRED                           Re-TACRED
                Method                                                                                                                    Avg.
                                    5-shot 10-shot 20-shot 50-shot 5-shot 10-shot 20-shot 50-shot 5-shot 10-shot 20-shot 50-shot
                In-context Learning 48.40    49.11   49.65   49.31   24.17   23.69   24.66    24.21    21.37   21.99    21.52    21.10 31.60

Random
                 w/ Self-Refine      47.93   48.50   49.19   48.88   23.15   23.05   24.18    23.12    21.04   21.51    20.71    21.32 31.05
                 w/ Self-Consistency 49.30   49.09   50.11   50.35   25.69   24.76   25.64    25.42    22.08   22.56    21.84    22.01 32.40
                 w/ GRACE            50.80   49.22   54.28   54.83   25.89   25.78   26.49    26.46    22.50   22.67    23.65    24.34 33.91
                w/ our SRVF         54.89 59.67      62.98   71.27 30.07 31.42       32.84    34.58 28.36 31.49         32.87    36.52 42.25
                In-context Learning 57.33    59.13   62.49   64.26   27.48   28.64   30.08    27.81    34.78   41.85    42.82    43.69 43.36

SimCSE
                 w/ Self-Refine      57.01   58.91   62.27   63.89   27.13   26.89   29.11    27.30    34.33   41.87    42.30    43.16 42.85
                 w/ Self-Consistency 57.54   58.81   62.98   65.00   28.82   29.85   30.98    25.42    35.83   42.84    43.71    44.59 43.86
                 w/ GRACE            57.93   58.48   66.32   67.48   28.76   28.60   30.03    26.46    33.95   41.53    42.35    44.37 43.86
                w/ our SRVF         60.76 64.12      69.54   76.32 32.99 33.50       34.81    36.13 39.48 46.54         49.73    54.31 49.85



Task-specific
                In-context Learning 58.68    64.90   65.67   77.32   26.11   26.35   31.15    33.35    42.75   45.53    52.89    56.22 48.41
                 w/ Self-Refine      58.38   64.96   65.68   77.35   25.01   25.48   30.62    32.67    42.10   44.98    52.11    55.62 47.91
                 w/ Self-Consistency 59.62   65.45   65.74   77.48   26.93   26.83   31.67    33.61    43.54   46.04    53.34    56.69 48.91
                 w/ GRACE            60.83   65.14   66.21   76.98   27.12   26.34   30.95    33.40    43.12   45.23    52.61    55.83 48.65
                w/ our SRVF         62.12 67.03      68.94   80.08 30.50 30.92       34.83    36.32 46.13 48.09         55.07    59.82 51.65

Table 1: Results (micro-F1 scores) on the SemEval, TACRED, and Re-TACRED datasets under various few-shot settings. Here
we adopt the Llama-2-7b-chat as the LLM. The best results are in bold.


                                 Experiments                                    • GRACE (Khalifa et al. 2023) trains a verifier to select the
Evaluation Protocal                                                               best intermediate reasoning step, which is then used as
                                                                                  feedback for generating the next step.
Datasets and Metric We adopt three commonly used
datasets for RE, including SemEval (Hendrickx et al. 2010),                    For Self-Consistency, GRACE, and ours, the number of it-
TACRED (Zhang et al. 2017), and Re-TACRED (Stoica,                             erations or candidate responses is set to 5 for fairness. For
Platanios, and Póczos 2021). Besides, compared to the sce-                    Self-Refine, the iteration number is set to 1 since we find that
nario with full data, the potential of LLMs under few-shot                     more iteration rounds result in performance degradation 2 .
settings is of more concern (Ma et al. 2023; Xu et al. 2023).
Hence we adopt the k-shot ( k ∈ {5, 10, 20, 50}) settings                      Main Results
to validate the effectiveness of the proposed method. For all                  Table 1 reports the experimental results with various initial
experiments, we report micro-F1 scores where Other and                         demonstration selection strategies on Llama-2-7b-chat on
no relation are considered negative labels.                                    the SemEval, TACRED, and Re-TACRED datasets. From
                                                                               Table 1, we can draw the following conclusions: 1) Our
Backbones We experiment with three different methods                           proposed SRVF framework yields significant enhancements
as backbones for selecting initial in-context demonstrations                   upon various backbones with different demonstration selec-
for LLM based RE, including: 1) Random, which randomly                         tion strategies. Specifically, the improvement is most sig-
selects initial demonstrations without any retriever. 2) Sim-                  nificant when randomly selecting the initial demonstrations,
CSE, which uses SimCSE (Gao, Yao, and Chen 2021) to re-                        getting a 10.65% absolute micro-F1 score increase on av-
trieve samples that have similar sentence semantics with the                   erage. Besides, when using SimCSE and task-specific re-
test example as initial in-context demonstrations. 3) Task-                    triever as backbones to carefully select initial in-context
specific, which uses a task-specific retriever that has been                   demonstrations, there are also 6.49% and 3.24% absolute
trained on the labeled samples (Wan et al. 2023).                              micro-F1 score boosts on average, respectively. 2) Our pro-
Baselines To the best of our knowledge, we are the first to                    posed method exhibits significant superiority over existing
explore the verification and feedback mechanism for LLM                        verification and feedback methods under all settings. The
based RE. Thus, we can only make modifications on cur-                         multi-agent based Self-Refine method is the worst, which
rent feedback methods in other tasks to adapt them for RE.                     is mainly due to its unsuitable feedback objectives and cor-
Specifically, we choose the following baselines:                               rection manner. Existing methods for verifying the output of
 • Self-Refine (Madaan et al. 2023) consists of three LLM                      LLMs, i.e., Self-Consistency and GRACE, can enhance the
   based agents, i.e., RE agent, verifier agent, and refiner                   performance of in-context learning to some extent. However,
   agent, for iterative feedback and refinement.                               since they do not provide explicit feedback signals for LLMs
                                                                               to correct the prediction, their improvements are limited.
 • Self-Consistency (Wang et al. 2023b) is proposed to con-
   duct verification for the multiple candidate responses and                      2
                                                                                     Please refer to Appendix for detailed implementation details
   choose the best response by majority voting.                                of baselines and ours.


                                                                       24525
              SemEval        TACRED         Re-TACRED                                      SemEval        TACRED       Re-TACRED
Method                                                      Avg.            Method                                                      Avg.
            5-shot 10-shot 5-shot 10-shot 5-shot 10-shot                                 5-shot 10-shot 5-shot 10-shot 5-shot 10-shot
Our SRVF 59.26 63.61 31.19 31.95 37.99 42.04 44.34                          R-BERT     42.75 57.25 9.87 16.24 26.64 35.01 31.29
                                                                            KnowPrompt 53.92 56.42 27.86 30.34 50.08 55.41 45.67
 w/o LGI    55.31   55.46   25.13   29.83   24.44   30.46   36.77
 w/o DI     57.90   62.50   28.52   29.78   35.64   39.99   42.39                                  Llama-2-7b-chat
 w/o RCT    58.37   62.78   30.31   30.87   37.23   43.73   43.88
                                                                            ICL          58.68 64.90 26.11 26.35 42.75 45.53 44.05
 w/o FDR    57.03   60.23   29.27   29.85   35.59   39.39   41.89
                                                                             w/ SRVF     62.12 67.03 30.50 30.92 46.13 48.09 47.47
 w/o RG     52.27   62.09   27.52   29.33   35.14   38.38   40.79
                                                                                                  Llama-2-70b-chat
Table 2: The ablation results (micro-F1) averaged over three                ICL          68.92 69.86 27.32 27.12 43.63 44.94 46.97
backbones. The best results are in bold.                                     w/ SRVF     69.97 70.00 27.69 29.47 45.13 46.93 48.20
                                                                                              Meta-Llama-3-8B-Instruct
Ablation Study                                                              ICL          69.90 69.79 32.63 32.26 48.23 50.69 50.58
                                                                             w/ SRVF     71.14 71.41 35.26 34.29 52.23 55.25 53.26
To validate the effectiveness of components in our method,
we introduce the following variants for ablation studies:                                     Meta-Llama-3-70B-Instruct

 • w/o label-guided intervention (LGI), where the labels do                 ICL          71.21 72.40 34.71 34.97 56.10 57.41 54.47
   not guide the collecting of unbiased rationales.                          w/ SRVF     74.68 74.33 37.05 36.35 59.27 59.96 56.94
 • w/o diversified intervention (DI), which replaces the DI                                         GPT-3.5-turbo
   with random sampling for collecting biased rationales.                   ICL          67.26 70.58 32.46 31.38 43.56 46.88 48.69
 • w/o rational contrastive training (RCT), which trains the                 w/ SRVF     69.62 71.67 37.78 34.63 46.22 49.66 51.60
   rationale supervisor with cross-entropy loss.
 • w/o feedback demonstration retrieval (FDR), which re-                    Table 3: Results (micro-F1 scores) using various LLMs with
   moves the FDR strategy and uses the initially selected                   the task-specific retriever.
   demonstrations as the feedback.
 • w/o RG, which skips the re-generation process and di-
                                                                            experimental results indicate that the “relation bias” issue
   rectly adopts the label of the top-1 retrieved demonstra-
                                                                            exists in LLMs of various scales, and our proposed method
   tion as the final prediction.
                                                                            can function as a plug-in module for various LLMs to
   The results of the ablation study are shown in Table 2.                  effectively mitigate this problem.
From the table, we make the following observations. 1) Re-
moving LGI and DI strategies significantly degrades perfor-                 Comparision with Well-designed Few-shot Methods for
mance, indicating that LLMs struggle to collect unbiased ra-                RE As shown in Table 3, we include two established su-
tionales based solely on generation without causal interven-                pervised fine-tuning methods for RE as baselines: 1) R-
tion. 2) Eliminating RCT also reduces performance, demon-                   BERT (Wu and He 2019), which fine-tunes a BERT for the
strating its effectiveness in helping the rationale supervisor              RE task, and 2) KnowPrompt (Chen et al. 2022), which is
distinguish between unbiased and various biased situations.                 tailored for few-shot scenarios and has shown good few-shot
3) Omitting FDR significantly decreases performance, high-                  performance. As we can see from the results, with the help
lighting its crucial role in guiding LLMs toward corrected                  of our proposed SRVF, even the relatively weak Llama-2-
predictions despite iterative verification. 4) Removing the                 7b-chat can outperform KnowPrompt by 1.80% averagely.
re-generation process results in a substantial performance                  Moreover, when deploying our SRVF on the most power-
drop, showcasing that simple assignment of retrieved top-1                  ful Meta-Llama-3-70B-Instruct, there is an average perfor-
demonstrations isn’t sufficient and that in-context feedback                mance improvement of 11.27% compared to KnowPrompt.
for re-generation adds robustness to the correction process.                Analysis on Successfully Corrected Samples To visual-
                                                                            ize which samples are successfully corrected by the pro-
Analysis                                                                    posed method, we compare the error matrix on the SemEval
Effectiveness on Various-scale LLMs To examine                              dataset before and after correction. The results are obtained
whether the proposed method remains effective for various-                  by summing the number of error predictions of all settings
scale LLMs, we conduct experiments on various sizes of                      in Table 1. The results are shown in Fig. 4.
LLMs from the Llama-2-chat (Touvron et al. 2023), Meta-                        From Fig. 4 (a), we observe that LLMs struggle to distin-
Llama-3-Instruct (AI@Meta 2024), and GPT-3.5 (Ouyang                        guish between relations that share similar entities, e.g., 687
et al. 2022), and present their results in Table 3.                         samples labeled as “Entity-Destination” are incorrectly pre-
   From Table 3, it can be seen that our rationale supervisor               dicted as “Content-Container”. Such error can arise when,
can boost the performance of LLMs with various sizes.                       for example, given sentences “please move the eggs into the
Specifically, even with the most powerful Meta-Llama-                       box” and “there are 5 eggs in the box”, where the same en-
3-70B-Instruct, there is still a 2.47% micro-F1 score                       tity pair “eggs” and “box” form “Entity-Destination” and
improvement over the original in-context learning. The                      “Content-Container” relations, respectively. Such ambigu-


                                                                    24526
                                 (a) Before Verification and Feedback                                     (b) After Verification and Feedback                                                                               ICL                       ICL w/ Self-Consistency         ICL w/ our SRVF
                                                                                                    700                                                                     700

                  Entity-Destination     687       384        27      27          9        131                437          123        25      16          17        99                                                      ICL w/ Self-Refine        ICL w/ GRACE
                                                                                                    600                                                                     600

                  Content-Container       0        192        13          4       14        2                     0        54         12          4       15        2




Golden Relation
                                                                                                    500                                                                     500




                                                                                                                                                                                           Cumulative Time (second)
                  Component-Whole        134        0         51      192         37        7       400       147           0         61      194         87        6       400
                                                                                                                                                                                                                 2000
                       Entity-Origin     154       465        0       13          20       103      300           74       218        0       12          31        74      300



                  Instrument-Agency       15       131        33          0       1         22
                                                                                                    200
                                                                                                                  6        67         27          0       2         16
                                                                                                                                                                            200
                                                                                                                                                                                                                 1500
                                                                                                    100                                                                     100
                  Member-Collection       86       652        51      28          0         74                    47       124        42      22          0         48


                                               r         e
                                                                                                    0                                                                       0
                                                                                                                                                                                                                 1000
                                                              igin        enc               tio                                       igin
                                                                                                                       r         e
                                           ine     Wh                         y                 n                  ine     Wh                     enc y             tio n
                                         ont          ol   Or                     llecuse-Eff                 ont             ol   Or                     llecuse-Eff
                                             a                        -Ag                                         a                           -Ag
                                                 ent     tity                   -Co           ect                        ent     tity                   -Co           ect
                                       nt-           -        -     ent                                     nt-              -        -     ent
                                           C   mp        En                   mb
                                                                                       Ca                       C      mp        En                   mb
                                                                                                                                                               Ca                                                     500
                                 Co               on               trum          er                       Co              on               trum          er
                                   nte     Co                  Ins        Me                                nte    Co                  Ins        Me

                                           Incorrect Relation Prediction                                           Incorrect Relation Prediction                                                                        0
                                                                                                                                                                                                                               Pre-inference      After Initial Generation After First Correction

Figure 4: Error matrix before and after the verification-
feedback-correction procedure. The numbers show how                                                                                                                                       Figure 5: Efficiency comparison of different methods on the
many samples labeled y (on the vertical axis) are incorrectly                                                                                                                             5-shot SemEval setting. The results are accumulated along
predicted as x (on the horizontal axis).                                                                                                                                                  the X axis. For example, “After Initial Generation” refers to
                                                                                                                                                                                          the sum time of “pre-inference” and “initial generation”.

ity often leads LLMs to misclassify relations when they fail                                                                                                                                                                                       DocRED             Re-DocRED
                                                                                                                                                                                                                      Method                                                               Avg.
to focus on context, resulting in numerous errors. However,                                                                                                                                                                                      5-shot 10-shot 5-shot 10-shot
as shown in Fig. 4 (b), the number of samples labeled as
“Entity-Destination” but incorrectly predicted as “Content-                                                                                                                                                           ICL (Random)               7.76 7.82 8.27 7.73 7.90
Container” is reduced by 250. This indicates that our method                                                                                                                                                           w/ our SRVF               15.40 18.00 15.29 15.65 16.09
effectively alleviates the above issue.                                                                                                                                                                               ICL (SimCSE)               15.67 16.40 11.97 12.53 14.14
                                                                                                                                                                                                                       w/ our SRVF               18.39 21.55 17.38 17.87 18.80
Analysis on Method Efficiency Considering possible
concerns on the inference efficiency due to the iterative                                                                                                                                                             ICL (Task-specific) 18.29 18.40 17.44 18.67 18.20
                                                                                                                                                                                                                       w/ our SRVF        20.04 21.55 19.98 21.69 20.82
feedbacks, we compare the inference time on the SemEval
dataset of different methods. Besides, we also evaluate the
pre-inference time of each method, e.g., the time to obtain                                                                                                                               Table 4: Results (micro-F1) on the DocRED (document-
biased/unbiased data and train the rationale supervisor in our                                                                                                                            level RE task). The best results are in bold.
SRVF. The comparison results are shown in Fig. 5.
   From Fig. 5, we can observe that:
   1) Basic in-context learning (ICL) is the most efficient.                                                                                                                                 From Table 4, we can observe that: 1) LLM performs
   2) Self-Refine does not require pre-inference time, but its                                                                                                                            poorly on document-level RE, which is consistent with em-
inference time is more than the sum of our pre-inference                                                                                                                                  pirical observations in Li, Jia, and Zheng (2023); Sun et al.
time and inference time. Moreover, Self-Refine has the worst                                                                                                                              (2024). This is due to the difficulty LLMs face in selecting
performance among all methods (Table 1).                                                                                                                                                  entity pairs that have certain relations from a vast space of
   3) Self-Consistency and GRACE have much higher com-                                                                                                                                    candidate entity pairs. Besides, the large number of candi-
putational costs than our SRVF, especially in terms of in-                                                                                                                                date relation labels (96 in DocRED and Re-DocRED) fur-
ference time. This is mainly because the proposed rationale                                                                                                                               ther increases the difficulty in assigning each entity pair a re-
supervisor can verify whether the LLM prediction is biased.                                                                                                                               lation. 2) Our proposed SRVF effectively enhances the per-
Only the test samples verified as biased by the rationale su-                                                                                                                             formance of LLM under various settings on DocRED and
pervisor will proceed to the correction round for regener-                                                                                                                                Re-DocRED, indicating that our method remains to be ef-
ation. This greatly reduces the time cost of our method in                                                                                                                                fective in such challenging scenarios.
inference time after correction.
   Overall, our SRVF is the second-best in computational ef-                                                                                                                                                                                     Conclusion
ficiency while achieving the best performance (Table 1).
                                                                                                                                                                                          In this paper, we propose a novel automated feedback frame-
Experiments on Document-level RE To explore the ef-                                                                                                                                       work for LLM based relation extraction (RE), which in-
fectiveness of our method for document-level RE, we apply                                                                                                                                 cludes a rationale supervisor to iteratively correct the biased
SRVF on three backbones and conduct experiments on two                                                                                                                                    relation prediction of LLMs. Specifically, we first present
commonly used document-level RE datasets, DocRED (Yao                                                                                                                                     a causal intervention and observation method to collect un-
et al. 2019) and Re-DocRED (Tan et al. 2022). The random                                                                                                                                  biased and biased rationales, which are then used to train
and SimCSE backbones are kept the same as before. For                                                                                                                                     the rationale supervisor. Then, we develop a verification-
the task-specific backbone, we borrow the idea from RE-                                                                                                                                   feedback-correction procedure to iteratively enhance LLMs’
PLM (Ozyurt, Feuerriegel, and Zhang 2024), which obtains                                                                                                                                  ability to correct the biased prediction. Extensive experi-
the final prediction by aggregating the predictions based on                                                                                                                              ments demonstrate the superiority of our framework over
multiple retrieved demonstrations. The experimental results                                                                                                                               existing methods. In the future, we will try to extend the
are reported in Table 4.                                                                                                                                                                  proposed framework to other NLP tasks.


                                                                                                                                                                                  24527
                  Acknowledgments                                        Reasoning. In Bouamor, H.; Pino, J.; and Bali, K., eds.,
This work was supported by the grant from the National                   Findings of the Association for Computational Linguistics:
Natural Science Foundation of China (NSFC) project (No.                  EMNLP 2023, 15299–15328. Singapore: Association for
62276193).                                                               Computational Linguistics.
                                                                         Li, B.; Fang, G.; Yang, Y.; Wang, Q.; Ye, W.; Zhao, W.; and
                       References                                        Zhang, S. 2023a. Evaluating ChatGPT’s Information Ex-
AI@Meta. 2024. Llama 3 Model Card. https://github.com/                   traction Capabilities: An Assessment of Performance, Ex-
meta-llama/llama3/blob/main/MODEL CARD.md.                               plainability, Calibration, and Faithfulness. arXiv preprint
                                                                         arXiv:2304.11633.
Bai, Y.; Jones, A.; Ndousse, K.; Askell, A.; Chen, A.; Das-
Sarma, N.; Drain, D.; Fort, S.; Ganguli, D.; Henighan,                   Li, G.; Wang, P.; and Ke, W. 2023. Revisiting Large
T.; Joseph, N.; Kadavath, S.; Kernion, J.; Conerly, T.; El-              Language Models as Zero-shot Relation Extractors. In
Showk, S.; Elhage, N.; Hatfield-Dodds, Z.; Hernandez, D.;                Bouamor, H.; Pino, J.; and Bali, K., eds., Findings of the
Hume, T.; Johnston, S.; Kravec, S.; Lovitt, L.; Nanda, N.;               Association for Computational Linguistics: EMNLP 2023,
Olsson, C.; Amodei, D.; Brown, T.; Clark, J.; McCandlish,                6877–6892. Singapore: Association for Computational Lin-
S.; Olah, C.; Mann, B.; and Kaplan, J. 2022. Training a                  guistics.
Helpful and Harmless Assistant with Reinforcement Learn-                 Li, J.; Jia, Z.; and Zheng, Z. 2023. Semi-automatic Data
ing from Human Feedback. arXiv:2204.05862.                               Enhancement for Document-Level Relation Extraction with
Chen, T.; Kornblith, S.; Norouzi, M.; and Hinton, G. 2020. A             Distant Supervision from Large Language Models. In Pro-
Simple Framework for Contrastive Learning of Visual Rep-                 ceedings of the 2023 Conference on Empirical Methods in
resentations. In III, H. D.; and Singh, A., eds., Proceedings            Natural Language Processing, 5495–5505.
of the 37th International Conference on Machine Learning,                Li, Y.; Lin, Z.; Zhang, S.; Fu, Q.; Chen, B.; Lou, J.-G.; and
volume 119 of Proceedings of Machine Learning Research,                  Chen, W. 2023b. Making Language Models Better Reason-
1597–1607. PMLR.                                                         ers with Step-Aware Verifier. In Rogers, A.; Boyd-Graber,
Chen, X.; Zhang, N.; Xie, X.; Deng, S.; Yao, Y.; Tan,                    J.; and Okazaki, N., eds., Proceedings of the 61st Annual
C.; Huang, F.; Si, L.; and Chen, H. 2022. Knowprompt:                    Meeting of the Association for Computational Linguistics
Knowledge-aware prompt-tuning with synergistic optimiza-                 (Volume 1: Long Papers), 5315–5333. Toronto, Canada: As-
tion for relation extraction. In Proceedings of the ACM Web              sociation for Computational Linguistics.
conference 2022, 2778–2788.
                                                                         Ma, X.; Li, J.; and Zhang, M. 2023. Chain of Thought with
Gao, T.; Yao, X.; and Chen, D. 2021. SimCSE: Simple Con-                 Explicit Evidence Reasoning for Few-shot Relation Extrac-
trastive Learning of Sentence Embeddings. In Moens, M.-F.;               tion. In The 2023 Conference on Empirical Methods in Nat-
Huang, X.; Specia, L.; and Yih, S. W.-t., eds., Proceedings of           ural Language Processing.
the 2021 Conference on Empirical Methods in Natural Lan-
guage Processing, 6894–6910. Online and Punta Cana, Do-                  Ma, Y.; Cao, Y.; Hong, Y. C.; and Sun, A. 2023. Large Lan-
minican Republic: Association for Computational Linguis-                 guage Model Is Not a Good Few-shot Information Extractor,
tics.                                                                    but a Good Reranker for Hard Samples! In The 2023 Confer-
                                                                         ence on Empirical Methods in Natural Language Process-
Golovneva, O.; Chen, M.; Poff, S.; Corredor, M.; Zettle-                 ing.
moyer, L.; Fazel-Zarandi, M.; and Celikyilmaz, A. 2023.
ROSCOE: A Suite of Metrics for Scoring Step-by-Step Rea-                 Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.;
soning. arXiv:2212.07919.                                                Wiegreffe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang,
                                                                         Y.; et al. 2023. Self-refine: Iterative refinement with self-
Gou, Z.; Shao, Z.; Gong, Y.; Shen, Y.; Yang, Y.; Duan,
                                                                         feedback. arXiv preprint arXiv:2303.17651.
N.; and Chen, W. 2023.            CRITIC: Large Language
Models Can Self-Correct with Tool-Interactive Critiquing.                Nathani, D.; Wang, D.; Pan, L.; and Wang, W. 2023. MAF:
arXiv:2305.11738.                                                        Multi-Aspect Feedback for Improving Reasoning in Large
Hendrickx, I.; Kim, S. N.; Kozareva, Z.; Nakov, P.;                      Language Models. In Bouamor, H.; Pino, J.; and Bali,
Ó Séaghdha, D.; Padó, S.; Pennacchiotti, M.; Romano, L.;              K., eds., Proceedings of the 2023 Conference on Empirical
and Szpakowicz, S. 2010. SemEval-2010 Task 8: Multi-Way                  Methods in Natural Language Processing, 6591–6616. Sin-
Classification of Semantic Relations between Pairs of Nom-               gapore: Association for Computational Linguistics.
inals. In Proceedings of the 5th International Workshop on               Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright,
Semantic Evaluation, 33–38. Uppsala, Sweden: Association                 C. L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray,
for Computational Linguistics.                                           A.; Schulman, J.; Hilton, J.; Kelton, F.; Miller, L.; Simens,
Kamoi, R.; Zhang, Y.; Zhang, N.; Han, J.; and Zhang, R.                  M.; Askell, A.; Welinder, P.; Christiano, P.; Leike, J.; and
2024. When Can LLMs Actually Correct Their Own Mis-                      Lowe, R. 2022. Training language models to follow instruc-
takes? A Critical Survey of Self-Correction of LLMs. arXiv               tions with human feedback. arXiv:2203.02155.
preprint arXiv:2406.01297.                                               Ozyurt, Y.; Feuerriegel, S.; and Zhang, C. 2024. Document-
Khalifa, M.; Logeswaran, L.; Lee, M.; Lee, H.; and Wang,                 Level In-Context Few-Shot Relation Extraction via Pre-
L. 2023. GRACE: Discriminator-Guided Chain-of-Thought                    Trained Language Models.


                                                                 24528
Pan, L.; Saxon, M.; Xu, W.; Nathani, D.; Wang, X.; and                   Wei, X.; Cui, X.; Cheng, N.; Wang, X.; Zhang, X.; Huang,
Wang, W. Y. 2023. Automatically Correcting Large Lan-                    S.; Xie, P.; Xu, J.; Chen, Y.; Zhang, M.; et al. 2023. Zero-
guage Models: Surveying the landscape of diverse self-                   shot information extraction via chatting with chatgpt. arXiv
correction strategies. arXiv:2308.03188.                                 preprint arXiv:2302.10205.
Pang, C.; Cao, Y.; Ding, Q.; and Luo, P. 2023. Guideline                 Wu, S.; and He, Y. 2019. Enriching pre-trained language
Learning for In-Context Information Extraction. In The                   model with entity information for relation classification. In
2023 Conference on Empirical Methods in Natural Lan-                     Proceedings of the 28th ACM international conference on
guage Processing.                                                        information and knowledge management, 2361–2364.
Paul, D.; Ismayilzada, M.; Peyrard, M.; Borges, B.;                      Xu, D.; Chen, W.; Peng, W.; Zhang, C.; Xu, T.; Zhao, X.;
Bosselut, A.; West, R.; and Faltings, B. 2023.   RE-                     Wu, X.; Zheng, Y.; and Chen, E. 2023. Large Language
FINER: Reasoning Feedback on Intermediate Representa-                    Models for Generative Information Extraction: A Survey.
tions. arXiv:2304.01904.                                                 arXiv:2312.17617.
                                                                         Yao, Y.; Ye, D.; Li, P.; Han, X.; Lin, Y.; Liu, Z.; Liu, Z.;
Pearl, J.; et al. 2000. Models, reasoning and inference. Cam-
                                                                         Huang, L.; Zhou, J.; and Sun, M. 2019. DocRED: A Large-
bridge, UK: CambridgeUniversityPress, 19(2): 3.
                                                                         Scale Document-Level Relation Extraction Dataset. In Pro-
Stoica, G.; Platanios, E. A.; and Póczos, B. 2021. Re-tacred:           ceedings of the 57th Annual Meeting of the Association for
Addressing shortcomings of the tacred dataset. In Proceed-               Computational Linguistics, 764–777.
ings of the AAAI conference on artificial intelligence, vol-             Zhang, K.; Gutiérrez, B. J.; and Su, Y. 2023. Aligning In-
ume 35, 13843–13850.                                                     struction Tasks Unlocks Large Language Models as Zero-
Sun, Q.; Huang, K.; Yang, X.; Tong, R.; Zhang, K.; and Po-               Shot Relation Extractors. In Findings of ACL.
ria, S. 2024. Consistency guided knowledge retrieval and de-             Zhang, K.; Wang, D.; Xia, J.; Wang, W. Y.; and Li, L. 2023.
noising in llms for zero-shot document-level relation triplet            ALGO: Synthesizing Algorithmic Programs with Generated
extraction. In Proceedings of the ACM on Web Conference                  Oracle Verifiers. arXiv preprint arXiv:2305.14591.
2024, 4407–4416.                                                         Zhang, Y.; Zhong, V.; Chen, D.; Angeli, G.; and Manning,
Tan, Q.; Xu, L.; Bing, L.; Ng, H. T.; and Aljunied, S. M.                C. D. 2017. Position-aware Attention and Supervised Data
2022. Revisiting DocRED-Addressing the False Negative                    Improve Slot Filling. In Proceedings of the 2017 Conference
Problem in Relation Extraction. In Proceedings of the 2022               on Empirical Methods in Natural Language Processing, 35–
Conference on Empirical Methods in Natural Language                      45. Copenhagen, Denmark: Association for Computational
Processing, 8472–8487.                                                   Linguistics.
Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.;
Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale,
S.; et al. 2023. Llama 2: Open foundation and fine-tuned
chat models. arXiv preprint arXiv:2307.09288.
Wadhwa, S.; Amir, S.; and Wallace, B. 2023. Revisiting
Relation Extraction in the era of Large Language Mod-
els. In Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds.,
Proceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long Papers),
15566–15589. Toronto, Canada: Association for Computa-
tional Linguistics.
Wan, Z.; Cheng, F.; Mao, Z.; Liu, Q.; Song, H.; Li, J.; and
Kurohashi, S. 2023. GPT-RE: In-context Learning for Rela-
tion Extraction using Large Language Models. In Bouamor,
H.; Pino, J.; and Bali, K., eds., Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language Pro-
cessing, 3534–3547. Singapore: Association for Computa-
tional Linguistics.
Wang, F.; Mo, W.; Wang, Y.; Zhou, W.; and Chen, M. 2023a.
A Causal View of Entity Bias in (Large) Language Models.
In The 2023 Conference on Empirical Methods in Natural
Language Processing.
Wang, X.; Wei, J.; Schuurmans, D.; Le, Q. V.; Chi, E. H.;
Narang, S.; Chowdhery, A.; and Zhou, D. 2023b. Self-
Consistency Improves Chain of Thought Reasoning in Lan-
guage Models. In The Eleventh International Conference on
Learning Representations.


                                                                 24529
