    Can LLMs be Good Graph Judge for Knowledge Graph Construction?

            Haoyu Huang1 , Chong Chen2 , Zeang Sheng3 , Yang Li3 , Wentao Zhang3
             1
               Hong Kong University of Science and Technology,2 Huawei Cloud BU
                                      3
                                        Peking University
                    hhuangcp@connect.ust.hk,chenchong55@huawei.com
                  {shengzeang18,liyang.cs,wentao.zhang}@pku.edu.cn



                       Abstract                              tured data is crucial for different downstream appli-
                                                             cations based on KG (Ge et al., 2021; Huang et al.,
     In real-world scenarios, most of the data ob-           2024; Wei et al., 2024; Rabbani et al., 2023).
     tained from the information retrieval (IR) sys-
                                                                Recently, Large Language Models (LLMs) have
     tem is unstructured. Converting natural lan-
     guage sentences into structured Knowledge               demonstrated significant generalization capabili-
     Graphs (KGs) remains a critical challenge. We           ties in various Natural Language Processing (NLP)
     identified three limitations with respect to ex-        tasks (Pan et al., 2024) and KG related tasks, such
     isting KG construction methods: (1) There               as text generation (Li et al., 2024), KG Completion
     could be a large amount of noise in real-world          (KGC) (Yao et al., 2023) and Open Information
     documents, which could result in extracting             Extraction (OpenIE) (Angeli et al., 2015; Dagde-
     messy information. (2) Naive LLMs usually ex-
                                                             len et al., 2024). Consequently, there are many
     tract inaccurate knowledge from some domain-
     specific documents. (3) Hallucination phe-
                                                             works that utilize LLMs to construct KGs from
     nomenon cannot be overlooked when directly              unstructured natural language documents. The in-
     using LLMs to construct KGs. In this paper,             corporation of LLMs can address the issue of gener-
     we propose GraphJudge, a KG construction                alization in open-domain applications (Carta et al.,
     framework to address the aforementioned chal-           2023a). With its robust zero-shot generation ca-
     lenges. In this framework, we designed an               pability, there is no need for us to gather a large
     entity-centric strategy to eliminate the noise in-      volume of annotated data for tasks such as named
     formation in the documents. And we fine-tuned
                                                             entity recognition (NER), entity extraction, or rela-
     a LLM as a graph judge to finally enhance the
     quality of generated KGs. Experiments con-              tion extraction.
     ducted on two general and one domain-specific              Although recent LLM-based methods (Mo et al.,
     text-graph pair datasets demonstrate state-of-          2025; Han et al., 2023; Lairgi et al., 2024) have
     the-art performance against various baseline            gained some success in the KG construction task,
     methods with strong generalization abilities.           we find that they may still face three challenges:
     Our code is available at https://github.com/hhy-           (1) Noise Information. Real-world documents
     huang/GraphJudge.
                                                             are not only voluminous but also rife with noise,
                                                             which poses a significant challenge for LLMs ex-
1    Introduction
                                                             tracting valuable structured information. The sheer
The transition from non-structured text to struc-            volume of data can lead to the extraction of ex-
tured Knowledge Graphs (KGs) is a pivotal step               cessive and irrelevant information, overshadowing
in the evolution of data management and informa-             the critical insights that LLMs are meant to un-
tion retrieval systems. The task of automatic KG             cover (Liu et al., 2024b; Shi et al., 2023a). For ex-
construction aims to develop a structured represen-          ample, as shown in Figure 1, the triple <Protein X,
tation of knowledge from various data sources with-          is on, a 50% Discount>is incorrectly constructed
out the need for manual intervention. KGs usually            due to the irrelevant advertisement with red lines
serve as the backbone of numerous data science               in the document, which is the noise information
applications, including GraphRAG systems (Edge               that makes the LLM incorrectly believed that the
et al., 2024; Peng et al., 2024; Huang et al., 2025)         Protein X is on sale with discounts.
and recommendation systems (Wang et al., 2019;                  (2) Domain-Specific Knowledge. Naive LLMs
Jiang et al., 2024; Chen et al., 2025). Exploring a          often generate inaccurate triples with domain-
way to construct high-quality KGs from unstruc-              specific documents, which require a deep un-
                                                          10929
      Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 10929–10948
                           November 4-9, 2025 ©2025 Association for Computational Linguistics
     ...
     <body>                                                                                                    Dr. Johnson
         <p>The recent advancements in oncology have shown that specific proteins can
     inhibit tumor growth.Dr. Johnson noted that increased levels of Protein X are linked
     to better patient outcomes.</p>                                                                         mentioned mentioned
         <p>Meanwhile, the keynote speech covered advancements in AI technology. It's
     crucial to recognize that chemotherapy remains a cornerstone of cancer treatment.
     </p>                                                                                              Protein X
         <div style=.. >                                                                                                  AI Technology
           <h2>Special Offer: Save 50% on Your Next Purchase!</h2>
           <p>Don't miss out on our limited-time discount! Visit our website today and get           cured     is on
                                                                                                                              replaces
     50% off on all products. Click <a href="https://example.com">here</a> to shop
     now!</p>
                                                                                                                    a 50%
         </div>                                                                                  Patient
                                                                                                                   Discount     Chemotherapy
     </body>
     ...

                                 Original Document                                                     Extracted Knowledge Graph

Figure 1: An demonstration of the challenges for constructing KGs with LLMs. The original document shown in
the left part, while the constructed KG with some failure cases is displayed on the right side. The triple highlighted
in red is wrongly formulated due to the presence of noisy information, the one in blue lacks domain knowledge, and
the green-highlighted triple is a result of hallucinations by LLMs.

derstanding of specialized terminology and con-                                     To overcome the second challenge, we suggest the
text (Zhong et al., 2023; Zhu et al., 2024). And this                               module of Knowledge Aware Supervised Fine-
kind of error is hard to be observed by naive LLMs.                                 Tuning (KASFT). We introduce the graph judge-
For example, in Figure 1, the triple <Protein X,                                    ment task from the triple classification task. To
cured, Patient>is inaccurately extracted due to a                                   verify the accuracy of the triples generated by the
lack of medical domain-specific knowledge. While                                    closed-source LLM, we conduct supervised fine-
the document marks a reference with blue lines,                                     tuning (SFT) on an open-source LLM, which can
note that the original text only suggests a link be-                                make it achieve over 90% accuracy on graph judge-
tween Protein X and better patient results, not that                                ment tasks with strong generalization abilities. To
it can cure patients in medical fields.                                             settle the third challenge, the Graph Judgement
   (3) Hallucinations of LLMs. When LLMs are                                        (GJ) module is introduced. We utilize the fine-
directly used to build KGs, they are prone to gen-                                  tuned open-source LLM to conduct judgement on
erating false or distorted information, which is a                                  the generated triples in the first module and filter
phenomenon called hallucinations (Zhang et al.,                                     out the wrong items to finally improve the quality
2023; Ji et al., 2023). This can lead to the incorpo-                               of generated KGs.
ration of inaccurate or fabricated facts into the KG,                                  In summary, the main contributions made in this
undermining the reliability of the KG. For example,                                 work are as follows.
as shown in Figure 1, the triple marked in green                                         • Addressing challenges such as information
<AI Technology, replaces, Chemotherapy>is in-                                              noise, domain knowledge gaps and halluci-
correctly generated without any reference in the                                           nations in LLMs represents a critical step to-
original document, even in the entity-related text                                         wards improving the quality of constructed
highlighted with a green line.                                                             KGs with real-world documents. To the best
    To this end, we propose a new method called                                            of our knowledge, we are the first to leverage
GraphJudge, which utilizes a fine-tuned open                                               both open- and closed-source LLMs to tackle
source LLM (e.g., LLaMA-2 (Touvron et al.,                                                 these problems.
2023)) as an expert to judge the correctness of the
                                                                                         • We propose a new framework named Graph-
triples generated by another closed-source LLM
                                                                                           Judge to leverage their capability as a graph
(e.g., GPT-4o-mini). To address the first challenge,
                                                                                           judge and enhance the performance of LLMs
we introduce the Entity-Centric Text Denoising
                                                                                           in KG construction tasks. We design an entity-
(ECTD) module. We clean up the original docu-
                                                                                           centric strategy to eliminate the irrelevant
ments by eliminating redundant words and irrele-
                                                                                           and messy information in original documents.
vant information not pertinent to the entities iden-
                                                                                           And we introduce graph judgment as the SFT
tified by the LLM. This module also leverages the
                                                                                           task to enhance the quality of generated KGs.
robust zero-shot generation capabilities of LLMs
to ensure the recall of a sufficient number of triple                                    • Experiments on two general and one domain-
candidates (Wei et al., 2023; Carta et al., 2023a).                                        specific text-graph pair datasets demonstrate
                                                                            10930
     that GraphJudge achieves state-of-the-art per-          Definition 1: (Knowledge Graph Construc-
     formance against various baseline methods            tion Task) We define the KG construction task as
     with strong generalization abilities.                a problem of how to extract entities E and relations
                                                          R from a document D, which is also called the text-
2   Related Work                                          to-graph generation (T2G) task. The constructed
                                                          KG, is defined as G = {(h, r, t)|h, t ∈ E, r ∈ R},
In this section, we will introduce recent LLM-
                                                          where E is the set of entities and R is the set of
based OpenIE and KG construction methods. Some
                                                          relations in the graph G. In other words, each KG
work(Agrawal et al., 2022; Wei et al., 2023) has
                                                          G has a corresponding original text D. Our goal is
demonstrated that LLMs have remarkable zero-
                                                          to get a better KG G from a document D.
shot and few-shot information extraction abilities.
                                                             We also define a set of KGs SG =
However, they face difficulties when it comes to
                                                          {G1 , G2 , .., GN } and a set of documents SD =
more intricate tasks such as relation extraction and
                                                          {D1 , D2 , .., DN }. In our implementation, we have
event extraction (Carta et al., 2023a). To address
                                                          a set of graph-text pairs SP = {P1 , P2 , .., PN },
that, Kumar et al. (Kumar et al., 2020) propose
                                                          where Pi = {(Gi , Di )|Gi ∈ SG , Di ∈ SD }. And
a unified approach to construct KGs from unpro-
                                                          N = |SP | is the number of graph-text pairs.
cessed text. They initially fine-tuned a pre-trained
                                                             Definition 2: (Graph Judgement Task) We
language model (PLM) for NER. Subsequently,
                                                          introduce the task of graph judgement to classify
they introduced a ‘2-model BERT’ architecture to
                                                          each triple in generated graphs is correct or not.
extract relations. GPT-RE (Wan et al., 2023) in-
                                                             Here we define the KG we constructed from a
troduces the in-context learning method and task-
                                                          corresponding document as Ĝ and SĜ representing
aware representations in demonstration retrieval
and aims to enhance the connections between ex-           the set of graphs we constructed. And T̂ in Equa-
amples and triples. PiVe (Han et al., 2023) de-           tion (1) represents the triples on which we need
signs a paradigm that fine-tuning a PLM as the            to make judgements. Our goal in the graph judge-
verifier to predict the missing triples. With iterative   ment task is to predict the label of each triple in T̂ ,
verifications, the graph-based generative capabil-        represented as ŷ ∈ {0, 1}|T̂ | .
ity of LLMs can be improved. VicunaNER (Ji,                                 [
2023) utilizes the open-source LLM Vicuna to do                     T̂ =       {(h, r, t)|(h, r, t) ∈ Ĝ}.   (1)
zero-shot or few-shot NER. Similarly, it also per-                      Ĝ∈SĜ
forms recognition to identify entities that were
not recognized in the previous phase. Carta et            4     Methodology
al. (Carta et al., 2023a) develops an iterative LLM       4.1    Overview
prompting-based pipeline to generate KGs with-
                                                          As shown in Figure 2, the proposed model Graph-
out requiring predefined sets or external ontologies.
                                                          Judge consists of three modules. In the first mod-
iText2KG (Lairgi et al., 2024) proposes a zero-shot
                                                          ule, which is Entity-Centric Text Denoising, we
method to construct consistent KGs from docu-
                                                          extract entities and relations separately following
ments with LLMs. It restructures the unprocessed
                                                          results described in (Carta et al., 2023b). In the
documents using a preset template and identifies
                                                          phrase of entity extraction, we generate entities
distinct entities and connections in a semantic man-
                                                          with the denoised document. In the phrase of re-
ner. SAC-KG (Chen et al., 2024) exploits LLMs as
                                                          lation extraction, we generate relations with the
skilled automatic constructors for domain KGs and
                                                          entities and the denoised document as many as pos-
employs a naive LLM to predict the correctness
                                                          sible. Then, in the module of Knowledge Aware
of constructed triples. KGGen (Mo et al., 2025)
                                                          Supervised Fine-Tuning, we perform SFT to let
clusters related entities to reduce sparsity in the
                                                          the LLM become an expert in graph judgement by
KGs constructed by LLMs.
                                                          enhancing their abilities to check facts from docu-
3   Preliminary and Definition                            ments with the triple structure and deepening their
                                                          comprehension of domain-specific knowledge con-
In this section, we first formulate the task of KG        tained in the text-graph pairs. After that, in the final
construction and introduce the definitions we may         module we conduct the Graph Judgement. With
use throughout the paper. Then we detail the defi-        the denoised documents as contexts, we employ
nition of the graph judgement task.                       the fine-tuned LLM as the graph judge to ascertain
                                                     10931
                                      Instructions
                                      According to the original text:

                                      "Insulin, produced by beta cells in the pancreas, regulates
                                      blood glucose levels. When insulin binds to cell receptors, it
                                      facilitates glucose uptake into cells, thereby lowering blood
                                      sugar"
            Ground Truth
             KG Triples               Is this true:

                                                      Insulin
                                                      Insulin
                                                                       Produced by
                                                                       Facilitates
                                                                                              Beta cells
                                                                                         Glucose uptake
                                                                                                           🔥         No, this is not true.
                                                                                                                     Yes, this is true.



                                                                                                           LLaMA
                                                      Glucose uptake       Lowers            Blood sugar             No, this is not true.
                                      (b) Knowledge Aware Supervised Fine Tuning




             Raw Text
                                         ❄️                               ❄️                                         Yes, ..

                                         GPT                               GPT
                                                                                                                     No, ..


       ❄️
                                                                                                                     No, ..

                                       Denoising                        Extraction
       🔥
            Frozen
                                                    Refined Text                                                     Yes, ..
            Trainable
                                                                                             Draft KG                                Predicted KG
            Head/Tail Entity
            Relation                   (a) Entity-Centric Text Denoising                                   (c) Graph Judgement



Figure 2: The overall architecture of our proposed GraphJudge framework for knowledge graph construction. It
consists of three modules: (a) is the Entity-Centric Text Denoising module, (b) is the Knowledge Aware Supervised
Fine Tuning module and (c) is the Graph Judgement module. The only component requiring training across the
entire architecture is the open-source LLM utilized in the second module.

the accuracy of each triple within the graphs we                                        tion retrieval systems are consist of considerable
generate. And then with the predicted results, we                                       noise information. And that may influence the qual-
can filter out the triples that are judged as wrong.                                    ity of relations extracted by LLM (Shi et al., 2023b;
Finally, we can get high quality KGs.                                                   Liu et al., 2024c). So we design an iterative denois-
                                                                                        ing method to remove messy information from the
4.2 Entity-Centric Text Denoising                                                       original text.
                                                                                           Specifically, we extract entities from the origi-
                                                                                        nal document using LLM. And as verified in Ap-
                                                                                        pendix E, the entities extracted by closed-source
                                                                                        LLMs have a high coverage rate and provide a good
    Raw Document           Entities                Denoised Document                    foundation for the following denoising, relation ex-
                                                                                        traction and triple filtering processes. Subsequently,
                                                                                        we input these entities and the original document
                                                                                        into LLM to generate the denoised document. In
                                        Draft KG                                        this way, we can achieve two goals: (1) The noise
                                                                                        information that is not related to the topic of the
Figure 3: Illustrations of Entity-Centric Text Denoising.                               document can be removed. (2) The content of the
   In this module, a two-phrase extraction paradigm                                     documents can be reorganized in an entity-centric
is designed to extract the entities and relations re-                                   way, which is friendly to the triple extraction in the
spectively. In phrase 1, we extract entities first and                                  next phrase. Finally, for each raw document D we
then denoise the original documents with extracted                                      will get the extracted entity set Ê and the denoised
entities. In phrase 2, we conduct relation extraction                                   document D∗ . Note that important information can
and then we obtain the draft KGs. And Figure 3                                          be well preserved in D∗ as verified in Appendix G.
is an overview of this module. In both of the two
                                                                                        4.2.2 Relation extraction
phrases we utilize a closed-source LLM to do the
extraction and denoising.                                                               In phrase 2, we aim to extract relationships (triples)
                                                                                        as many as feasible with the denoised document
4.2.1 Text denoising and entity extraction                                              D∗ and the entity set Ê obtained in phrase 1 uti-
In phrase 1, we consider that a substantial portion                                     lizing LLMs as shown in Equation (2). We create
of real-world documents retrieved from informa-                                         numerous relationships between entities to ensure
                                                                                     10932
a sufficient number of suitable candidate triples for         Similarly, we sample negative triple set T − from
filtering with LLM judgment in the Graph Judge-            the KGs in training set as described in Equation (5),
ment module. Then we can construct a draft KG              where E represents the entity set of the graph G.
G ∗ for each original document D, as illustrated in        (h, r, t− ) is a negative triple of the graph G, where
Equation (3), where R∗ is the draft relation set we        t− is a negative entity. We replace the positive tail
generate.                                                  entity t+ in each positive triple with a randomly
                R∗ = LLM(Ê, D∗ ),                (2)      selected negative tail entity t− . Note that if the
                                                           selected negative entity is the same as or similar
        G ∗ = {(h, r, t)|h, t ∈ Ê, r ∈ R∗ }.       (3)    to the original one, we will skip that because they
                                                           may not construct a triple reflecting a false fact.
4.3 Knowledge Aware Supervised                                                [
    Fine-Tuning                                                    T−=               {(h, r, t− )|
                                                                           G∈SGtrain                          (5)
In this module, inspired by KG-LLaMA (Yao et al.,
2023), we propose the method of treating triples                        (h, r, t+ ) ∈ G, t− ∈ E \ {t+ }}.
in the draft KG G ∗ as textual sequences and model           Then we merge the positive triple set T + and
graph judgement task as a sequence-to-sequence             negative triple set T − constructed from KGs
problem. We construct instruction data from the            SGtrain . Then we can obtain all the triples Ttrain
training set and fine-tune an open-source LLM to           we need to construct instructions.
achieve the goal of both excelling at checking facts
from documents with the triple structure and ac-                           Ttrain = T + ∪ T − .                       (6)
knowledgment of domain-specific knowledge. The                Furthermore, we transfer the triples in Ttrain to
LLM can also learn how to verify the consistency           natural language sentences to construct the instruc-
between the document and the extracted triples.            tion data with paired documents D as contexts fol-
Checking facts with the triple structure refers to         lowing the prompt templates shown in Appendix I.
the general structure of triples is often analogous        The triple sentences either represent a real fact or a
to a grammatical subject, predicate, and object or a       fake fact. Then let the LLM make judgements with
subject with a relational attribute. LLMs are antic-       these instructions. Mathematically, with tokenized
ipated to have the ability to identify their correct-      sentences XTtrain transferred from triples Ttrain
ness from the give documents. Domain-specific              and paired documents D, and tokenized instruction
knowledge refers to the knowledge in the docu-             XI , for a sequence of length L, we compute the
ments could be a new domain (Zhong et al., 2023),          probability of generating the target output XO as
which is typically not part of pre-training data of        follows:
LLMs. By employing SFT, the domain-specific                                         L
                                                                                    Y
knowledge from the documents can be incorporated              p(XO |Xt , XI ) =           pθ (xi |Xt , XI,<i , XO,<i ),
into the LLM, thus enhances its graph judgment                                      i=1
performance. And only if LLMs are fine-tuned as                                                             (7)
graph judges, these types of knowledge can be well         where Xt ∈ XTtrain . And θ are the learnable param-
learned, as justified in Figure 5 and Appendix H.          eters within the open-source LLM to be fine-tuned.
   Before we conduct SFT on the LLM, we con-               4.4   Graph Judgement
struct instructions for the graph judgement task
                                                           The KGs created in the first module are preliminary
with text-graph pair data. Because we need to en-
                                                           and that is also why we call that draft KGs. In this
sure that the LLM not only excels at verifying cor-
                                                           module, we will judge the factual correctness of the
rect triples but also skilled at telling the incorrect
                                                           triples in these draft KGs using our fine-tuned LLM
triples with the paired documents as contexts, we
                                                           in the second module and filter out the incorrect
employ negative sampling to construct instruction
                                                           triples.
data for training. In detail, we first sample the posi-
                                                              In detail, we let LLM do the graph judgement
tive triple set T + from the KGs of training set as
                                                           task on the draft KGs G ∗ . Here we define draft KG
described in Equation (4), where SGtrain is the set
                                                           set as SG ∗ , and the triples in all draft KGs can be
of all KGs in the training set.
                                                           symbolized as
             [                                                              [
  T+=                 {(h, r, t+ )|(h, r, t+ ) ∈ G}. (4)          T∗=            {(h, r, t)|(h, r, t) ∈ G ∗ }. (8)
          G∈SGtrain                                                     G ∗ ∈SG ∗

                                                      10933
   Then, the LLM needs to assess the correctness of          the training data for validation purposes during the
each triple in T ∗ by considering whether it aligns          fine-tuning of the LLM.
with the knowledge in paired documents and avoids
conflicting with both domain-specific knowledge              5.1.2      Baselines
as it learned. We obtain the predictions of all              In our performance comparison, we consider six
the triples in T ∗ with the learned parameters θ             baselines for comprehensive evaluation: GPT-4o-
as shown in Equation (9). And Pred(·) is a func-             mini: We conduct experiments on GPT-4o with
tion that transforms the outputs of LLM into the             one-shot learning method. The instructions we
                              ∗
binary results ŷ ∈ {0, 1}|T | . Based on the judg-          have developed are identical to those outlined in
ments made by LLM, we filter the triples T ∗ in              our method. GPT-4o (Hurst et al., 2024): The
draft KGs to obtain high-quality triples T̂ as de-           same settings as GPT-4o-mini. RAKG (Zhang
scribed in Equation (10), which form the final KGs           et al., 2025), iText2KG (Lairgi et al., 2024), and
we seek. ŷ(h,r,t) is the predicted result of a triple       KGGen (Mo et al., 2025): We follow the default
(h, r, t).                                                   settings of them and their official implementations
                ŷ = Pred(pθ (XT ∗ )),             (9)       with GPT-4o-mini as the LLM. PiVe (Han et al.,
                                                             2023): We follow the default parameter settings of
          T̂ = {(h, r, t) ∈ T ∗ |ŷ(h,r,t) = 1}.     (10)    PiVe. We use the largest verifier module in PiVe,
   Similarly, the refined relation set R̂ =                  Flan-T5-XXL (Chung et al., 2024). We employ the
{r|(h, r, t) ∈ G ∗ , ŷ(h,r,t) = 1} can also be ob-          LoRA adapter checkpoint1 , which has been well
tained. Lastly, for each draft KG G ∗ ∈ SG ∗ we              trained. And the LLM we use in this model is GPT-
can get the refined KG Ĝ that we desire as shown            4o-mini. We implement an iterative prompting
in Equation (11). The implementation details of              approach with three rounds, which represents the
the graph judgment procedure are demonstrated in             optimal number of iteration rounds as outlined in
Appendix D.                                                  their study.

                                                             5.1.3      Implementation Details
    Ĝ = {(h, r, t)|h, t ∈ Ê, r ∈ R̂, (h, r, t) ∈ T̂ }.
                                                      (11)   Large Language Model: The LLMs we employed
                                                             in this research are various in different modules.
5     Experiments                                            In the ECTD module, we utilize the closed-source
                                                             LLM GPT-4o-mini to denoise the original docu-
In this section, we will conduct experiments to ad-
                                                             ments and extract triples from documents. In the
dress the following key research questions: RQ1:
                                                             KASFT module, an open-source LLM LLaMA-
How well does GraphJudge perform on both gen-
                                                             2-7B (Touvron et al., 2023) is used as our base
eral knowledge data and domain-specific knowl-
                                                             model.
edge data? RQ2: How do the different key com-
                                                                Supervised Fine-Tuning: We employ LLaMA-
ponents in our proposed method GraphJudge con-
                                                             2-7B as the base model to carry out SFT with
tribute to its overall performance? RQ3: How
                                                             LoRA (Hu et al., 2021). The instructions are con-
about the generalization capability of GraphJudge
                                                             structed with the documents, query sentences, and
when applied across different datasets?
                                                             the triple sentences. We perform SFT on autore-
5.1 Experimental Settings                                    gression generation tasks, which is a common ap-
                                                             proach to fine-tune LLMs (Black et al., 2022). The
5.1.1 Dataset
                                                             expected responses (labels) are either ‘Yes, that is
In our study, we conduct experiments on two gen-             true.’ or ‘No, that is not true.’. Training settings are
eral datasets (REBEL-Sub (Huguet Cabot and                   illustrated in Appendix C. The training was done
Navigli, 2021) and GenWiki (Jin et al., 2020))               using a single L20 GPU with 48GB of RAM.
and two domain-specific datasets (SCIERC (Luan
et al., 2018) and the Windows-centric subset of              5.1.4      Evaluation Metrics
Re-DocRED (Tan et al., 2022) used by Sun et al.              We acknowledge that conventional evaluation tech-
(2025)) with golden ground truth KGs. We demon-              niques are rule-based. They assess the resemblance
strate the detailed information and statistics of each       between predictions and ground-truth KGs through
dataset in Appendix A. For each dataset we ran-
domly select a sample of 2000 data points from                  1
                                                                    https://huggingface.co/Jiuzhouh/flan-t5-xxl-lora-verifier

                                                        10934
   Dataset   Method        G-BS-Acc↑   G-BS-Recall↑   G-BS-F1↑   G-BL-Acc↑    G-BL-Recall↑   G-BL-F1↑   G-RO-Acc↑   G-RO-Recall↑   G-RO-F1↑
             GPT-4o-mini    0.3571        0.9024       0.4289     0.2343         0.6687       0.3018      0.2095       0.6266       0.2779
             GPT-4o         0.3131        0.9432       0.4163     0.2345         0.7284       0.3158      0.2201       0.6851       0.2966
             RAKG           0.1196        0.9571       0.2127     0.1078         0.8625       0.1917      0.1012       0.8095       0.1799
 REBEL-Sub   iText2KG       0.3847        0.9342       0.4937     0.2704         0.6579       0.3504      0.2180       0.5475       0.2864
             PiVe           0.3082        0.9378       0.4090     0.2217         0.7089       0.3010      0.2068       0.6693       0.2823
             KGGen          0.4190        0.8937       0.4995     0.2587         0.5794       0.3146      0.2233       0.5037       0.2719
             GraphJudge     0.4868        0.9144       0.5796     0.3391         0.6490       0.4057      0.3032       0.5878       0.3571
             GPT-4o-mini    0.7825        0.9334       0.8368     0.6136         0.7353       0.6577      0.5451       0.6568       0.5857
             GPT-4o         0.7871        0.9393       0.8428     0.6318         0.7561       0.6774      0.5614       0.6742       0.6028
             RAKG           0.4695        0.9521       0.6058     0.3804         0.7991       0.5035      0.3397       0.7196       0.4507
   GenWiki   iText2KG       0.8984        0.7611       0.7986     0.7193         0.6007       0.6310      0.6042       0.5227       0.5432
             PiVe           0.7463        0.9485       0.8230     0.5884         0.7516       0.6503      0.5251       0.6746       0.5817
             KGGen          0.8578        0.8230       0.8169     0.5799         0.4700       0.5542      0.4845       0.5598       0.4641
             GraphJudge     0.7936        0.9375       0.8457     0.6407         0.7591       0.6836      0.5714       0.6796       0.6106
             GPT-4o-mini    0.5974        0.9183       0.6882     0.4368         0.6725       0.5040      0.3876       0.6065       0.4490
             GPT-4o         0.6272        0.9079       0.7035     0.4469         0.6530       0.5032      0.3914       0.5807       0.4425
             RAKG           0.2137        0.9528       0.3474     0.1647         0.7334       0.2678      0.1556       0.6864       0.2524
   SCIERC    iText2KG       0.8100        0.6674       0.6724     0.5747         0.4732       0.4772      0.4836       0.3968       0.3999
             PiVe           0.5738        0.9225       0.6725     0.4192         0.6757       0.4924      0.3719       0.6092       0.4385
             KGGen          0.8394        0.6500       0.6635     0.6045         0.4594       0.4725      0.5426       0.4100       0.4211
             GraphJudge     0.6847        0.8775       0.7283     0.4898         0.6273       0.5216      0.4321       0.5591       0.4603
             GPT-4o-mini    0.8036        0.7460       0.6807     0.4840         0.4659       0.4254      0.3337       0.3338       0.2938
             GPT-4o         0.7508        0.7588       0.6864     0.4438         0.4682       0.4066      0.3076       0.3383       0.2833
             RAKG           0.3121        0.9243       0.4422     0.2144         0.6377       0.3044      0.1613       0.4922       0.2306
 Re-DocRED   iText2KG       0.7957        0.5519       0.5502     0.4930         0.3741       0.3522      0.3704       0.2958       0.2678
             PiVe           0.7963        0.6769       0.6289     0.4545         0.4193       0.3729      0.2848       0.2955       0.2506
             KGGen          0.8422        0.3914       0.4372     0.5284         0.2594       0.2768      0.3662       0.1937       0.1959
             GraphJudge     0.7801        0.7579       0.7051     0.4776         0.4830       0.4322      0.3350       0.3532       0.3048


Table 1: Comparisons of GraphJudge with six baseline methods across four datasets. The cells marked with
 red color hold the worst performance in each column of Acc and Recall. The best and second-best results are also
highlighted in each column of F1 scores.

strict string matching, potentially overlooking se-                        GraphJudge leverages the ECTD module based
mantic similarities. Therefore, to better evaluate                         on a closed-source LLM to ensure recall ability,
the quality of the produced KGs against the ground-                        while the KASFT and GJ modules with a fine-tuned
truth KGs, similar to PiVe (Han et al., 2023), we                          open-source LLM guarantee accuracy, enabling its
utilize one semantic level and two soft string match-                      F1 score to surpass those of other baseline mod-
ing evaluation metrics to calculate the Accuracy,                          els. We can also observe that GraphJudge excels
Recall, and F1 scores: G-BERTScore (G-BS),                                 not only with domain-specific documents, but also
G-BLEU (G-BL) and G-ROUGE (G-RO). We                                       demonstrates superior performance with general
elaborate on them in Appendix B.                                           documents.
                                                                              GraphJudge is cost-effective. Remarkably,
5.2 Overall Performance Comparison (RQ1)                                   GraphJudge achieves state-of-the-art performance
We demonstrate the evaluation results of our                               by fine-tuning only a 7B LLM, which is signifi-
method GraphJudge with GPT-4o-mini and other                               cantly more efficient and cost-effective compared
baseline methods across three datasets in Table 1.                         to the 70B LLM employed in PiVe. In addition,
We have the following insights:                                            GraphJudge can even outperform GPT-4o with
                                                                           GPT-4o-mini, which is a small model with lower
   GraphJudge’s superior performance. Graph-
                                                                           token cost. However, other baseline methods fail
Judge outperforms other baselines in most of the
                                                                           to achieve that.
cases. The superiority of GraphJudge’s F1 scores
(marked with gray color) demonstrates that, while
                                                                           5.3   Module Ablation Study (RQ2)
maintaining a reasonable level of recall for triples,
it also achieves improvement in accuracy. For ex-                          We perform an ablation study to explore the specific
ample, as the results marked with red color show,                          impacts of various modules within GraphJudge,
although RAKG and PiVe exhibit stronger recall                             and the results are reported in Table 2. The insights
ability, they overlook triple accuracy. KGGen ex-                          are outlined below:
cels in accuracy but fails at recall. In contrast,                            Effect of Entity-Centric Text Denoising. We
                                                                  10935
                   Dataset                   Method                     G-BS-F1↑             G-BL-F1↑               G-RO-F1↑                         Tuning. We conduct graph judgement on the triples
                                             GraphJudge                       0.5796                 0.4057          0.3571                          without fine-tuning the open-source LLM, which is
   REBEL-Sub
                                             w/o ECTD                         0.4548                 0.3343          0.3094                          denoted as ‘w/o KASFT’. The result in Table 2 in-
                                             w/o GJ                           0.4203                 0.3052          0.2820                          dicates that without SFT, the naive LLM has weak
                                             w/o KASFT                        0.4506                 0.3219          0.2935
                                                                                                                                                     graph judgement abilities. And with a fine-tuned
                                             GraphJudge                       0.7283                 0.5216          0.4603                          LLM as a graph judge, the performance can be im-
                                             w/o ECTD                         0.6818                 0.5029          0.4509
                 SCIERC                                                                                                                              proved a lot. Because KASFT enables the LLM to
                                             w/o GJ                           0.7172                 0.5146          0.4552
                                             w/o KASFT                        0.6700                 0.4644          0.4084                          acquire both fact-checking capabilities and domain-
                                                                                                                                                     specific knowledge within the triples in our instruc-
Table 2: The results of ablation study on REBEL-Sub                                                                                                  tion training data.
dataset and SCIERC dataset.                                                                                                                             Furthermore, we apply negative sampling to con-
          Original Text         Refined Text
                           1                                              1                                                                          struct instructions on the test set like what we did



                                                                                                                                 Cosine Similarity
                                                                                                                                                     on the training set. We randomly select 500 sam-
                                                                                                                           0.8
                           6                                              6


 Chunks                                                              Chunks
                           11                                             11                                                                         ples and perform graph judgement to compare the
                           16                                             16                                               0.6                       capabilities of different models. As shown in Fig-
                                                                                                                                                     ure 5 and Appendix H, both fine-tuned small mod-
                                1             6                 11              1                6             11
                                            Triples                                         Triples
                                                                                                                                                     els like BERT and naive powerful LLMs like GPT-
                                                                                                                                                     4o show poor performance on the graph judgement
Figure 4: (a) The left map is the semantic similarity                                                                                                task even with documents as contexts. However,
between the original document and paired KG triples.                                                                                                 GraphJudge can achieve over 90% judgement
(b) The right map is the semantic similarity between the                                                                                             accuracy on REBEL-Sub and GenWiki, which
denoised document and paired KG triples.                                                                                                             demonstrates the KASFT module can indeed en-
investigate the benefit of introducing entity-centric                                                                                                hance the effectiveness of LLMs as a graph judge.
denoising paradigm using the variant ‘w/o ECTD’,                                                                                                        Effect of Graph Judgement. We compare the
where we do not conduct document denoising and                                                                                                       performances of our full model and the model with-
directly extract entities and relations from original                                                                                                out GJ module denoted as ‘w/o GJ’. The result
documents. The results show that our full model                                                                                                      suggests that GJ module plays a very important
performs significantly better than this ablated ver-                                                                                                 role in GraphJudge. It can significantly enhance
sion. It suggests that ECTD module can avoid                                                                                                         the quality of KGs generated by the closed-source
LLMs extract wrong structured information from                                                                                                       LLM and reduce the effects of the inaccuracies
irrelevance or not well-formatted corpus.                                                                                                            or hallucinations that may arise from LLMs. The
   Furthermore, to showcase the noise reduction                                                                                                      closed-source LLM excels in zero-shot generation,
capability of ECTD, we visualize the semantic                                                                                                        boosting recall but suffering accuracy due to hallu-
correlation of the triples in a known KG with                                                                                                        cinations or knowledge inadequacy. The GJ mod-
the denoised and original document, respectively.                                                                                                    ule relieves this by filtering inaccurate triples, en-
As shown in Figure 4, deeper color in the heat                                                                                                       hancing the quality of constructed KGs.
maps suggests a stronger relevance. The refined                                                                                                      5.4   Generalization Capabilities of
document exhibits greater relevance to the triples,                                                                                                        GraphJudge (RQ3)
demonstrating the effectiveness of ECTD. Imple-
                                                                                                                                                     To demonstrate the generalization abilities of
mentation details are described in Appendix F.
                       1.0                               0.92                             0.91                       BERT(SFT)
                                                                                                                                                     GraphJudge, we conduct experiments in cross-
                                                                                                                                                     dataset scenarios, which are training the LLM on



Graph Judgement Accuracy
                                                                                                                     LLaMA2-7B
                       0.8                        0.74                                                               GPT-4o
                                                                                                                     GraphJudge
                                           0.64                               0.61 0.64                                   0.63                       GenWiki and then evaluate it on REBEL-Sub, train-
                       0.6                                                                                    0.56 0.61
                                    0.44
                                                                      0.50                             0.49                                          ing the LLM on REBEL-Sub and then evaluate it
                       0.4                                                                                                                           on GenWiki and SCIERC, respectively. As shown
                       0.2                                                                                                                           in Table 3, our method can still outperform baseline
                       0.0                                                                                                                           methods, which indicates GraphJudge has great ca-
                                       REBEL-Sub                              GenWiki                         SCIERC
                                                                                                                                                     pabilities of generalization across various corpus.
Figure 5: A comparison of the capabilities of bert-                                                                                                  This is because the ability to check facts with triple
base-uncased (SFT) (Devlin et al., 2019), LLaMA-2-7B,                                                                                                structure learned from graph judgement tasks can
GPT-4o, and our GraphJudge in graph judgment tasks.                                                                                                  be generalized. It also suggests that GraphJudge
                   Effect of Knowledge Aware Supervised Fine-                                                                                        once well trained on a general dataset, can be
                                                                                                                                                10936
                    GenWiki @ REBEL-Sub        REBEL-Sub @ GenWiki        REBEL-Sub @ SCIERC
    Method
                 G-BS-F1↑ G-BL-F1↑ G-RO-F1↑ G-BS-F1↑ G-BL-F1↑ G-RO-F1↑ G-BS-F1↑ G-BL-F1↑ G-RO-F1↑
    GPT-4o        0.4163   0.3158   0.2966     0.8428    0.6774     0.6028      0.7035     0.5032     0.4425
    PiVe          0.4090   0.3010   0.2823     0.8230    0.6503     0.5817      0.6725     0.4924     0.4385
    KGGen         0.4995   0.3146   0.2719     0.8169    0.5542     0.4641      0.6635     0.4725     0.4211
    GraphJudge    0.5814   0.4055   0.3649     0.8587    0.6792     0.5911      0.7431     0.5156     0.4572

Table 3: Results of generalization study on REBEL-Sub, GenWiki, and SCIERC with the LLM fine-tuned on
GenWiki (GenWiki @ REBEL-Sub) and REBEL-Sub (REBEL-Sub @ GenWiki, REBEL-Sub @ SCIERC),
respectively.

readily applied to diverse datasets with common          References
knowledge. We also note that there is a gentle per-      Monica Agrawal, Stefan Hegselmann, Hunter Lang,
formance drop on SCIERC dataset, which is rea-            Yoon Kim, and David Sontag. 2022. Large language
sonable because there is less in-domain knowledge         models are few-shot clinical information extractors.
in the REBEL-Sub dataset than that in SCIERC              arXiv preprint arXiv:2205.12689.
dataset. And this result can further demonstrate         Gabor Angeli, Melvin Jose Johnson Premkumar, and
the domain-specific knowledge within the training          Christopher D. Manning. 2015. Leveraging linguis-
corpora is usefull in GraphJudge.                          tic structure for open domain information extraction.
                                                           In Proceedings of the 53rd Annual Meeting of the As-
                                                           sociation for Computational Linguistics and the 7th
6     Conclusions                                          International Joint Conference on Natural Language
In this paper, we introduce a new method called            Processing (Volume 1: Long Papers), pages 344–354,
                                                           Beijing, China. Association for Computational Lin-
GraphJudge for automatically constructing KGs,             guistics.
which leverages the potential of LLMs to act as
graph judges. In GraphJudge, we propose ECTD,            Isabelle Augenstein, Mrinal Das, Sebastian Riedel,
                                                            Lakshmi Vikraman, and Andrew McCallum. 2017.
KASFT and GJ modules to mitigate the impact of              SemEval 2017 task 10: ScienceIE - extracting
irrelevant information from documents and exploit           keyphrases and relations from scientific publications.
the benefits of trainable open-source LLMs and              In Proceedings of the 11th International Workshop
harnessing the strong zero-shot generation capa-            on Semantic Evaluation (SemEval-2017), pages 546–
                                                            555, Vancouver, Canada. Association for Computa-
bilities of closed-source LLMs. The experiments             tional Linguistics.
conducted on two general and one domain-specific
datasets demonstrate GraphJudge’s consistent su-         Sid Black, Stella Biderman, Eric Hallahan, Quentin
periority against various baseline methods.                Anthony, Leo Gao, Laurence Golding, Horace He,
                                                           Connor Leahy, Kyle McDonell, Jason Phang, et al.
                                                           2022. Gpt-neox-20b: An open-source autoregressive
7     Limitations                                          language model. arXiv preprint arXiv:2204.06745.
GraphJudge has the following limitations. First,         Salvatore Carta, Alessandro Giuliani, Leonardo Piano,
even though we employ the LLMs act as both the             Alessandro Sebastian Podda, Livio Pompianu, and
extractor and the judge to improve the quality of          Sandro Gabriele Tiddia. 2023a. Iterative zero-shot
                                                           llm prompting for knowledge graph construction.
constructed KGs, we still use the entity-level triples     arXiv preprint arXiv:2307.01128.
to construct KGs and there could be better knowl-
edge units to form a better KG. Second, a more           Salvatore M. Carta, Alessandro Giuliani, Lee Cecilia
                                                           piano, Alessandro Sebastian Podda, Livio Pompianu,
reasonable benchmark to evaluate the quality of
                                                           and Sandro Gabriele Tiddia. 2023b. Iterative zero-
constructed KGs should be proposed in the future.          shot llm prompting for knowledge graph construction.
Currently, most of the work just utilize the ‘ground       ArXiv, abs/2307.01128.
truth’ KGs to calculate the correctness and com-
                                                         Hanzhu Chen, Xu Shen, Qitan Lv, Jie Wang, Xiaoqi
prehensiveness of constructed KGs, However, the            Ni, and Jieping Ye. 2024. Sac-kg: Exploiting
quality of ‘ground truth’ KGs may still deserve            large language models as skilled automatic construc-
suspicion. So using a self-supervised approach to          tors for domain knowledge graphs. arXiv preprint
evaluate the KGs is in demands. We will research           arXiv:2410.02811.
for more KG constructing and evaluating method to        Wei Chen, Haoyu Huang, Zhiyu Zhang, Tianyi Wang,
improve the performance of knowledge extraction.          Youfang Lin, Liang Chang, and Huaiyu Wan.
                                                    10937
  2025. Next-poi recommendation via spatial-temporal           exploration. In 2024 IEEE 40th International Confer-
  knowledge graph contrastive learning and trajectory          ence on Data Engineering (ICDE), pages 3462–3475.
  prompt. IEEE Transactions on Knowledge and Data              IEEE.
  Engineering, 37(6):3570–3582.
                                                            Pere-Lluı́s Huguet Cabot and Roberto Navigli. 2021.
Hyung Won Chung, Le Hou, Shayne Longpre, Barret               REBEL: Relation extraction by end-to-end language
  Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi             generation. In Findings of the Association for Com-
  Wang, Mostafa Dehghani, Siddhartha Brahma, et al.           putational Linguistics: EMNLP 2021, pages 2370–
  2024. Scaling instruction-finetuned language models.        2381, Punta Cana, Dominican Republic. Association
  Journal of Machine Learning Research, 25(70):1–53.          for Computational Linguistics.
John Dagdelen, Alexander Dunn, Sanghoon Lee,                Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
  Nicholas Walker, Andrew S Rosen, Gerbrand Ceder,            Perelman, Aditya Ramesh, Aidan Clark, AJ Os-
  Kristin A Persson, and Anubhav Jain. 2024. Struc-           trow, Akila Welihinda, Alan Hayes, Alec Radford,
  tured information extraction from scientific text with      et al. 2024. Gpt-4o system card. arXiv preprint
  large language models. Nature Communications,               arXiv:2410.21276.
  15(1):1418.
                                                            Bin Ji. 2023. Vicunaner: Zero/few-shot named
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and                 entity recognition using vicuna. arXiv preprint
   Kristina Toutanova. 2019. Bert: Pre-training of deep       arXiv:2305.03253.
   bidirectional transformers for language understand-
   ing. In Proceedings of the 2019 conference of the        Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
  North American chapter of the association for com-          Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
   putational linguistics: human language technologies,       Madotto, and Pascale Fung. 2023. Survey of halluci-
   volume 1 (long and short papers), pages 4171–4186.         nation in natural language generation. ACM Comput-
                                                              ing Surveys, 55(12):1–38.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
  Bradley, Alex Chao, Apurva Mody, Steven Truitt,           Yangqin Jiang, Yuhao Yang, Lianghao Xia, and Chao
  and Jonathan Larson. 2024. From local to global: A          Huang. 2024. Diffkg: Knowledge graph diffusion
  graph rag approach to query-focused summarization.          model for recommendation. In Proceedings of the
  arXiv preprint arXiv:2404.16130.                            17th ACM International Conference on Web Search
                                                              and Data Mining, pages 313–321.
Kata Gábor, Davide Buscaldi, Anne-Kathrin Schu-
  mann, Behrang QasemiZadeh, Haı̈fa Zargayouna,             Zhijing Jin, Qipeng Guo, Xipeng Qiu, and Zheng Zhang.
  and Thierry Charnois. 2018. SemEval-2018 task               2020. GenWiki: A dataset of 1.3 million content-
  7: Semantic relation extraction and classification in       sharing text and graphs for unsupervised graph-to-
  scientific papers. In Proceedings of the 12th Inter-        text generation. In Proceedings of the 28th Inter-
  national Workshop on Semantic Evaluation, pages             national Conference on Computational Linguistics,
  679–688, New Orleans, Louisiana. Association for            pages 2398–2409, Barcelona, Spain (Online). Inter-
  Computational Linguistics.                                  national Committee on Computational Linguistics.

Congcong Ge, Xiaoze Liu, Lu Chen, Baihua Zheng,             Abhijeet Kumar, Abhishek Pandey, Rohit Gadia, and
  and Yunjun Gao. 2021. Largeea: Aligning entities            Mridul Mishra. 2020. Building knowledge graph
  for large-scale knowledge graphs. arXiv preprint            using pre-trained language model for learning entity-
  arXiv:2108.05211.                                           aware relationships. In 2020 IEEE International Con-
                                                              ference on Computing, Power and Communication
Jiuzhou Han, Nigel Collier, Wray Buntine, and Ehsan           Technologies (GUCON), pages 310–315. IEEE.
   Shareghi. 2023. Pive: Prompting with iterative verifi-
   cation improving graph-based generative capability       Yassir Lairgi, Ludovic Moncla, Rémy Cazabet, Khalid
   of llms. arXiv preprint arXiv:2305.12392.                  Benabdeslem, and Pierre Cléau. 2024. itext2kg: In-
                                                              cremental knowledge graphs construction using large
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan              language models. arXiv preprint arXiv:2409.03284.
  Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
  and Weizhu Chen. 2021. Lora: Low-rank adap-               Junyi Li, Tianyi Tang, Wayne Xin Zhao, Jian-Yun Nie,
  tation of large language models. arXiv preprint             and Ji-Rong Wen. 2024. Pre-trained language mod-
  arXiv:2106.09685.                                           els for text generation: A survey. ACM Computing
                                                              Surveys, 56(9):1–39.
Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu
  Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, and          Chin-Yew Lin. 2004. ROUGE: A package for auto-
  James Cheng. 2025. Retrieval-augmented genera-              matic evaluation of summaries. In Text Summariza-
  tion with hierarchical knowledge. arXiv preprint            tion Branches Out, pages 74–81, Barcelona, Spain.
  arXiv:2503.10150.                                           Association for Computational Linguistics.

Peng Huang, Meihui Zhang, Ziyue Zhong, Chengliang           Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
  Chai, and Ju Fan. 2024. Representation learning for         Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
  entity alignment in knowledge graph: A design space         Deng, Chenyu Zhang, Chong Ruan, et al. 2024a.
                                                       10938
  Deepseek-v3 technical report.          arXiv preprint      Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
  arXiv:2412.19437.                                            Scales, David Dohan, Ed H. Chi, Nathanael Schärli,
                                                               and Denny Zhou. 2023b. Large language models
Jingyu Liu, Jiaen Lin, and Yong Liu. 2024b. How much           can be easily distracted by irrelevant context. In
   can rag help the reasoning of llm? arXiv preprint           Proceedings of the 40th International Conference
   arXiv:2410.02338.                                           on Machine Learning, volume 202 of Proceedings
                                                               of Machine Learning Research, pages 31210–31227.
Jingyu Liu, Jiaen Lin, and Yong Liu. 2024c. How                PMLR.
   much can rag help the reasoning of llm? Preprint,
   arXiv:2410.02338.                                         Jiaqi Sun, Shiyou Qian, Zhangchi Han, Wei Li, Zelin
                                                                Qian, Dingyu Yang, Jian Cao, and Guangtao Xue.
Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh                2025. Lkd-kgc: Domain-specific kg construction via
  Hajishirzi. 2018. Multi-task identification of entities,      llm-driven knowledge dependency parsing. arXiv
  relations, and coreferencefor scientific knowledge            preprint arXiv:2505.24163.
  graph construction. In Proc. Conf. Empirical Meth-
  ods Natural Language Process. (EMNLP).                     Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and
                                                               Sharifah Mahani Aljunied. 2022. Revisiting docred–
Belinda Mo, Kyssen Yu, Joshua Kazdan, Proud Mpala,             addressing the false negative problem in relation ex-
  Lisa Yu, Chris Cundy, Charilaos Kanatsoulis, and             traction. arXiv preprint arXiv:2205.12696.
  Sanmi Koyejo. 2025. Kggen: Extracting knowledge
  graphs from plain text with language models. arXiv         Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
  preprint arXiv:2502.09956.                                   bert, Amjad Almahairi, Yasmine Babaei, Nikolay
                                                               Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
                                                               Bhosale, et al. 2023. Llama 2: Open founda-
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-
                                                               tion and fine-tuned chat models. arXiv preprint
  apu Wang, and Xindong Wu. 2024. Unifying large
                                                               arXiv:2307.09288.
  language models and knowledge graphs: A roadmap.
  IEEE Transactions on Knowledge and Data Engi-
                                                             Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying
  neering.
                                                               Liu, Haiyue Song, Jiwei Li, and Sadao Kurohashi.
                                                               2023. Gpt-re: In-context learning for relation ex-
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-            traction using large language models. arXiv preprint
  Jing Zhu. 2002. Bleu: a method for automatic evalu-          arXiv:2305.02105.
  ation of machine translation. In Proceedings of the
  40th annual meeting of the Association for Computa-
                                                             Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and
  tional Linguistics, pages 311–318.
                                                               Tat-Seng Chua. 2019. Kgat: Knowledge graph atten-
                                                               tion network for recommendation. In Proceedings of
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,                   the 25th ACM SIGKDD international conference on
  Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang            knowledge discovery & data mining, pages 950–958.
  Tang. 2024. Graph retrieval-augmented generation:
  A survey. arXiv preprint arXiv:2408.08921.                 Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang,
                                                               Xin Zhang, Shen Huang, Pengjun Xie, Jinan Xu,
Kashif Rabbani, Matteo Lissandrini, and Katja Hose.            Yufeng Chen, Meishan Zhang, et al. 2023. ‘chatie:
  2023. Extraction of validating shapes from very large        Zero-shot information extraction via chatting with
  knowledge graphs. Proceedings of the VLDB Endow-             chatgpt. arXiv preprint arXiv:2302.10205.
  ment, 16(5):1023–1032.
                                                             Yuyang Wei, Wei Chen, Xiaofang Zhang, Pengpeng
Swarnadeep Saha, Prateek Yadav, Lisa Bauer, and Mo-            Zhao, Jianfeng Qu, and Lei Zhao. 2024. Multi-modal
  hit Bansal. 2021. Explagraphs: An explanation graph          siamese network for few-shot knowledge graph com-
  generation task for structured commonsense reason-           pletion. In 2024 IEEE 40th International Conference
  ing. arXiv preprint arXiv:2104.07644.                        on Data Engineering (ICDE), pages 719–732. IEEE.

Christoph Schuhmann, Gollam Rabby, Ameya Prabhu,             Liang Yao, Jiazhen Peng, Chengsheng Mao, and
  Tawsif Ahmed, Andreas Hochlehnert, Huu Nguyen,               Yuan Luo. 2023. Exploring large language mod-
  Nick Akinci Heidrich, Ludwig Schmidt, Robert Kacz-           els for knowledge graph completion. arXiv preprint
  marczyk, Sören Auer, et al. 2025. Project alexandria:       arXiv:2308.13916.
  Towards freeing scientific knowledge from copyright
  burdens via llms. arXiv preprint arXiv:2502.19413.         Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin,
                                                               Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou,
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan                 and Maosong Sun. 2019. DocRED: A large-scale
  Scales, David Dohan, Ed H Chi, Nathanael Schärli,           document-level relation extraction dataset. In Pro-
  and Denny Zhou. 2023a. Large language models                 ceedings of the 57th Annual Meeting of the Associa-
  can be easily distracted by irrelevant context. In In-       tion for Computational Linguistics, pages 764–777,
  ternational Conference on Machine Learning, pages            Florence, Italy. Association for Computational Lin-
  31210–31227. PMLR.                                           guistics.
                                                        10939
Hairong Zhang, Jiaheng Si, Guohang Yan, Boyuan
  Qi, Pinlong Cai, Song Mao, Ding Wang, and Bo-
  tian Shi. 2025. Rakg: Document-level retrieval                          0.7                                                  REBEL-Sub
  augmented knowledge graph construction. arXiv                                                                                GenWiKi
                                                                          0.6                                                  SCIERC
  preprint arXiv:2504.09823.
                                                                          0.5

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q
                                                                  Ratio
                                                                          0.4
  Weinberger, and Yoav Artzi. 2019. Bertscore: Eval-                      0.3
  uating text generation with bert. arXiv preprint
                                                                          0.2
  arXiv:1904.09675.
                                                                          0.1
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,                      0.0
  Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,                              1     2     3     4     5      6     7     8     9     10 >10
  Yulong Chen, et al. 2023. Siren’s song in the ai ocean:                                             # of Triples
  a survey on hallucination in large language models.                 0.40
  arXiv preprint arXiv:2309.01219.                                                                                                   REBEL-Sub
                                                                      0.35                                                           GenWiKi
                                                                      0.30                                                           SCIERC
Lingfeng Zhong, Jia Wu, Qian Li, Hao Peng, and Xin-
  dong Wu. 2023. A comprehensive survey on auto-                      0.25
  matic knowledge graph construction. ACM Comput-
  ing Surveys, 56(4):1–62.                                    Ratio   0.20
                                                                      0.15

Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao,                      0.10
  Yixin Ou, Yunzhi Yao, Shumin Deng, Huajun Chen,                     0.05
  and Ningyu Zhang. 2024. Llms for knowledge graph                    0.00
  construction and reasoning: Recent capabilities and
                                                                                0-10 30-40 60-70 90-100120-130150-160180-190210-220240-250270-280300-310
  future opportunities. World Wide Web, 27(5):58.
                                                                                                   Interval of Length
Appendix
                                                             Figure 6: The figure above is the normalized distribution
A    Datasets                                                of the number of triplets in each dataset and the figure
                                                             below is the normalized distribution of the length of
                                                             documents in each dataset.
    Dataset              REBEL-Sub   GenWiki     SCIERC
    # of Train KGs       45,791      69,788      350
                                                                 GenWiki. GenWiki (Jin et al., 2020) is an exten-
    # of Test KGs        1,799       1,000       100
                                                             sive dataset sourced from general Wikipedia, com-
    # of Train Triples   268,864     588,642     6,429
    # of Test Triples    5,595       3,915       974
                                                             prising 1.3 million non-parallel texts and graphics
                                                             that share content. In our study, for efficient valida-
              Table 4: Statistics of datasets.               tion, we use a subset of GenWikiFINE as our training
                                                             set and employ another subset of GenWikiFINE for
   We conduct experiments on the following three             testing. And the documents in original testing data
datasets, And we also calculate the percentages of           are too short for us to validate our method. To en-
KGs with different numbers of triples and different          hance the quality of training data lacking human
lengths of original documents in each dataset in             annotations, we also exclude the triples with incor-
Figure 6. The statistics of each dataset are shown           rect formats.
in Table 4.                                                      SCIERC. SCIERC (Luan et al., 2018) is a scien-
   REBEL-Sub. REBEL (Huguet Cabot and Nav-                   tific domain-specific dataset comprises annotations
igli, 2021) dataset comes from Wikipedia text be-            for scientific entities, their relations, and corefer-
fore the table of contents, as well as Wikidata for          ence clusters within 500 scientific abstracts. It ex-
the triplets annotation. The dataset is collected by         pands upon the datasets from SemEval 2017 Task
the extraction pipeline cRocoDiLe (Huguet Cabot              10 (Augenstein et al., 2017) and SemEval 2018
and Navigli, 2021). The original REBEL dataset               Task 7 (Gábor et al., 2018) by introducing addi-
is a large-scale corpus. We utilize a subset of              tional entity types, relation types, broader relation
REBEL referred to as REBEL-sub, consisting of                coverage, and incorporating cross-sentence rela-
50,000/2,000/2,000 samples for the training, vali-           tions through coreference links. In addition, we
dation, and test set respectively, randomly chosen           filter out the samples with empty ground truth KG.
from the original dataset. Moreover, we filter out               Re-DocRED. Re-DocRED (Tan et al., 2022) is a
the samples with empty ground truth KG.                      well annotated domain-specific dataset that is used
                                                          10940
in LKD-KGC (Sun et al., 2025). It is a refined                  Total(n) =
                                                                    X      X
version of DocRED (Yao et al., 2019), enhanced                                         Count(gram(n), Xt̂ ), (13)
through reannotation of omitted relation triples.                 Xt ∈XT gram(n)∈Xt̂
We follow the settings in LKD-KGC to use the                             
Windows-centric subset containing domain knowl-                          1               if |Xt̂ | > |Xt |
edge about Windows Operation Systems. And be-                     BP =          |X |
                                                                                 t                             (14)
                                                                          (1− |X | )
                                                                           e      t̂      if |Xt̂ | ≤ |Xt |,
cause there is no enough samples in this dataset
for us to train our model, we use the model trained
                                                                                        Match(n)
on GenWiki to evaluate the performance of Graph-                               wn =              ,             (15)
                                                                                        Total(n)
Judge on this dataset.
                                                                                              N
                                                                                              Y
B        Experimental Metrics                                         G-BLEU = BP × (
                                                                                                         1
                                                                                                    wn ) N .   (16)
                                                                                              n=1
In this section, we explain the details of our evalua-
tion metrics.                                               G-ROUGE (G-RO): ROUGE (Recall-Oriented
   G-BERTScore (G-BS): Here we use a match-                 Understudy for Gisting Evaluation)(Lin, 2004) is
ing metric that evaluate the degree of similarity be-       a set of metrics for evaluating automatic summa-
tween the ground-truth and predicted graphs, which          rization and machine translation systems. And here
is called G-BERTScore(Saha et al., 2021). And               we utilize ROUGE to compare the similarities be-
it is designed as an extension of the text genera-          tween the triple sentences in the ground-truth and
tion metric BERTScore(Zhang et al., 2019). In G-            predicted KGs, which is G-ROUGE. Here our G-
BERTScore, each triple within knowledge graphs              ROUGE score is based on the notion of n-gram
is treated as a sentence, and subsequently, the sim-        co-occurrence statistics and we set n = 2. For
ilarity score between sentences of triples in the           G-ROUGE-N, which focuses on the overlap of
ground-truth and predicted knowledge graphs is              n-grams between the ground-truth triple sentence
computed. And we compute the accuracy, recall,              and the predicted triple sentence, the formulas are
and F1 score of each constructed KG against the             shown in (17), (18), (19). Unlike G-BLEU, G-
ground-truth using G-BERTScore, denoted as G-               ROUGE is computed using recall as a metric. And
BS-Acc, G-BS-Recall, and G-BS-F1, respectively.             Count(gram(n), Xt̂ ) is the number of times the n-
   G-BLEU (G-BL): BLEU (Bilingual Evalua-                   gram appears in the predicted triple sentence Xt̂ .
tion Understudy)(Papineni et al., 2002) is a metric         And we compute the accuracy, recall, and F1 score
for evaluating the quality of text which has been           of each constructed KG against the ground-truth
machine-translated from one natural language to             using G-ROUGE, denoted as G-RO-Acc, G-RO-
another. Here we use this approach to determine             Recall, and G-RO-F1, respectively.
the resemblance between the triple sentences in the
ground-truth and predicted KGs, which is called G-              Match(n) =
                                                                   X       X
BLEU. The formulas are shown in (16), (15), (12),                                      Count(gram(n), Xt̂ ), (17)
(13), (14). N is the maximum order of n-grams                     Xt ∈XT gram(n)∈Xt
considered in the evaluation and we set N = 4,
which is a default number in the Python package2 .              Total(n) =
BP is the brevity penalty, which is used to avoid giv-              X      X
                                                                                       Count(gram(n), Xt ), (18)
ing too much credit to short translations. And we
                                                                  Xt ∈XT gram(n)∈Xt
compute the accuracy, recall, and F1 score of each
constructed KG against the ground-truth using G-                                           Match(n)
BLEU, denoted as G-BL-Acc. G-BL-Recall, and                              G-ROUGE =                  .          (19)
                                                                                           Total(n)
G-BL-F1, respectively.
                                                            C    Experimental Settings
    Match(n) =                                              During the Knowledge Aware Supervised Fine-
       X       X                                            Tuning module, we follow the parameter settings
                               Count(gram(n), Xt̂ ), (12)
                                                            in Table 5 referring to the tuning process used for
          Xt ∈XT gram(n)∈Xt
                                                            triple classification tasks in the KG-LLaMA (Yao
    2
        https://pypi.org/project/bert-score/                et al., 2023).
                                                       10941
                                                                                                     Model
       Hyper-parameter       Experimental Setting             Dataset        Context
      Micro Batch Size                 8                                                   LLaMA-3-8B LLaMA-3-70B

          Batch Size                 128                               [Lower-Upper]       47.33-99.33    76.67-100.0
 Gradient Accumulation Steps          16                     REBEL-Sub      D∗                94.67          96.00
                                                                            Ĝ                85.33          90.67
        Training Steps               500
        Learning Rate               3e-4                                   [Lower-Upper]   53.33-100.0    68.00-100.0
                                                              GenWiki           D∗            98.00          96.00
  Lora Attention Dimension             8
                                                                                Ĝ            93.00          93.00
       Alpha Parameter                16
       Target Modules           q proj, v proj                             [Lower-Upper]   65.00-99.67    77.00-99.67
                                                              SCIERC            D∗            95.67          96.33
        Warmup Steps                 100
                                                                                Ĝ            90.33          93.67
          Optimizer                AdamW
                                                         Table 6: MCQ performance across datasets. Each row
Table 5: Implementation detail of SFT in GraphJudge.
                                                         displays the lower-upper bound performance (no con-
                                                         text vs. original document), denoised document perfor-
                                                         mance, and our KG performance for different models.
Algorithm 1 The Graph Judgement procedure of             Using D∗ and Ĝ preserves most information for an-
GraphJudge                                               swering MCQs, perform close to the using the original
                                                         document (upper bound) across datasets and models.
Input: The fine-tuned expert LLM pθ ; Candidate
    triples T ∗ in the draft KGs SG ∗ ; Paired refined           Dataset               Method    Recall      Acc
    text D∗ of the candidate triples;                                                  w/o GJ   0.9784    0.3709
                                                                 REBEL-Sub
Output: The predicted KGs SĜ with refined                                             w/ GJ    0.9411    0.5495
    triples T̂ ;                                                 GenWiki-Hard
                                                                                       w/o GJ   0.9365    0.7932
                                                                                       w/ GJ    0.9018    0.8343
 1: SĜ ← {};
                                                                                       w/o GJ   0.9392    0.6916
 2: T̂ ← {};                                                     SCIERC
                                                                                       w/ GJ    0.9187    0.7176
 3: for each G ∗ in SG ∗ ,
    each denoised document d∗ in D∗ do                   Table 7: Entity coverage and accuracy across datasets.
 4:    R̂ ← {};
 5:    Ê ← {};                                          E Entity Coverage of the LLM Extraction
 6:    for each triple t∗ =< h, r, t >∈ G ∗ do
                                                         To verify the comprehensiveness of extracted enti-
 7:       /*Transform the triple and refined text
                                                         ties by the LLM, we evaluate the entity coverage
          into a sentence*/
                                                         as well as accuracy across three datasets. In detail,
 8:       Xt∗ = Sentence(< h, r, t >, d∗ );
                                                         we use the semantic metric of Recall and Accuracy
 9:       /*Verify the correctness of the current
                                                         based on BertScore to measure the entity coverage
          triple with fine-tuned LLM*/
                                                         and accuracy, similar to the metric G-BS described
10:       ŷt∗ = Pred(pθ (Xt∗ ));
                                                         in the paper. As shown in Table 7, we report the
11:       if ŷt∗ is not ‘False’ then
                                                         coverage as well as accuracy of the extracted en-
12:           T̂ ← t∗ ;
                                                         tities before (w/o GJ) and after (w/ GJ) the GJ
13:           R̂ ← R̂ ∪ {r};
                                                         module. And here are the insights: (1) The entities
14:           Ê ← Ê ∪ {h, t};
                                                         extracted from the original text have high coverage
15:       end if
                                                         and low accuracy, which verifies that the extracted
16:    end for
                                                         entities can provide good foundation for both de-
17:    Ĝ = {< h, r, t > |h, t ∈ Ê, r ∈ R̂, <
                                                         noising and relation extraction. (2) The entities
       h, r, t >∈ T̂ };
                                                         within the triples filtered by GJ module still keep
18:    SĜ ← SĜ ∪ {Ĝ};
                                                         a high coverage and have a significantly higher ac-
19: end for
                                                         curacy. It further validates the effectiveness of our
                                                         GJ module.

                                                         F     Effect of ECTD Module
D    Graph Judgement Algorithm
                                                          To validate that ECTD module can lead to a cleaner
                                                         refined text, we sample a text-graph pair from the
   The detailed procedure of the graph judgement         REBEL-Sub dataset. Then we split the original and
algorithm is demonstrated in Algorithm 1.                refined document into the same number of chunks,
                                                     10942
                                    1.0                                           0.92                                       0.91




             Graph Judgement Accuracy
                                    0.8                                    0.74
                                                               0.68 0.68
                                                        0.64                                                          0.64
                                                                                                0.60 0.61 0.62 0.61                                              0.61 0.63
                                    0.6          0.59
                                                                                                                                           0.55 0.56 0.58 0.56
                                                                                         0.50                                       0.49
                                          0.44
                                    0.4     BERT(SFT)
                                            DeepSeek-V3
                                            LLaMA2-7B
                                    0.2     LLaMA3-8B
                                            LLaMA3-70B
                                            GPT-4o
                                            GraphJudge
                                    0.0            REBEL-Sub                                        GenWiki                                     SCIERC
Figure 7: A comparison of the capabilities of bert-base-uncased (SFT) (Devlin et al., 2019), DeepSeek-V3 (Liu
et al., 2024a), LLaMA-2-7B, LLaMA-3-8B, LLaMA-3-70B, GPT-4o, and our GraphJudge in graph judgment tasks.

which we set 20 here. And we use a PLM BERT                                                               that MCQs performance with D∗ or Ĝ remains far
(bert-base-uncased3 ) (Devlin et al., 2019) to pro-                                                       above the lower bound baseline and approaches
cess these chunks and get the embedding of each                                                           the original-document upper bound. It proofs that
chunk. And we calculate the cosine similarities be-                                                       important information is well preserved in both
tween these document chunks and triple sentences,                                                         our denoised documents and constructed KG.
as shown in Figure 4. Deeper color in the heat
maps suggests a stronger relevance between the                                                            H       Effect of KASFT Module
specific triple and document chunk.                                                                       In this section, we extend the baseline models to
                                                                                                          explore their abilities to be a graph judge with
G Knowledge Retention of ECTD Module
                                                                                                          the same experimental settings in Section 5.3. As
  and KG
                                                                                                          shown in Figure 7, We extend baseline models
While the ECTD module has the ability to remove                                                           to fine-tuned BERT, DeepSeek-V3, LLaMA-2-7B,
irelevant information contained in the original doc-                                                      LLaMA-3-8B, LLaMA-3-70B, and GPT-4o. Com-
uments, it is also necessary to verify that the impor-                                                    pared with them, our proposed GraphJudge demon-
tant knowledge is well preserved in the documents                                                         strates consistent superiority in graph judgement
denoised by ECTD. We test how well multiple-                                                              tasks, which further proofs that the KASFT module
choice question (MCQ) performance is preserved                                                            can improve the capabilities of open-source LLMs
after we refined the original documents.                                                                  as a graph judge. And neither fine-tuning a PLM
   In detail, similar to the existing work (Schuh-                                                        with a smaller parameter size nor directly employ-
mann et al., 2025), we generate various MCQs                                                              ing a powerful closed-source LLM can achieve a
with LLaMA-3-70B for each original document.                                                              high accuracy on graph judgement tasks, which
For REBEL-Sub, we randomly sample 500 docu-                                                               suggests the necessity to introduce our proposed
ments and generate 3 MCQs for each document.                                                              GraphJudge.
For SCIERC, because the test set of that is very
small, we used the full test set of SCIERC with                                                           I     Prompt Templates
3 MCQs for each document. For GenWiki, be-                                                                As shown in Figure 8 and Figure 11, we demon-
cause the average lengths of the documents are                                                            strate the prompt templates for the closed-source
very short, we generate only 1 MCQ for each docu-                                                         LLM to conduct relation extraction and for the
ment. Then we ask LLaMA-3-8B to answer them                                                               open-source LLM to perform graph judgements
with no context (denoted as lower bound), then ask                                                        on the results generated from the closed-source
them again with the original passage (denoted as                                                          LLM. We also provide the prompt templates used
upper bound) for sanity check. Finally, we con-                                                           to generate and answer MCQs in Appendix G.
duct tests using denoised documents (denoted as
D∗ ) and KG triples constructed by GraphJudge (de-                                                        J      Case Study
noted as Ĝ). The results in Table 6 demonstrates                                                         In this section, we present an instance of construct-
   3
       https://huggingface.co/google-bert/bert-base-uncased                                               ing a KG from a document, achieved through the
                                                                                                  10943
   Prompt Template for Graph Judgement.

   Goal:
   You need to do the graph judgement task, which means you need to clarify the correctness of the
   given triples with the given original document.
   Here is the question:
   According to the original document: {text}
   Is this true: {head entity} {relation} {tail entity}?

   Output:
   No, it is not true./ Yes, it is true.

  Figure 8: The prompt template for the open-source LLM LLaMA to construct graph judgement instructions.

   Prompt Template for MCQ Answering.

   Given the context or evidence:
   {context}
   Here is a multiple-choice question:
   Question:
   {question}
   Options:
   A. {option A}
   B. {option B}
   C. {option C}
   D. {option D}
   Please select the correct answer by choosing A, B, C, or D. Respond with only the letter of your
   choice.

   Output:
   A/B/C/D

                             Figure 9: The prompt template for the MCQ answering.


integration of a naive LLM (GPT-4o-mini) and our        does not have the capability to determine whether
GraphJudge. We select a text-graph pair from the        they are useful. However, there are no such triples
SCIERC dataset and contrast the results yielded by      in the KG constructed by our GraphJudge. On the
our approach with that of GPT-4o-mini. As shown         one hand, this is because the triples without any
in Table 8, the KG constructed by GPT-4o-mini           useful information will be clarified as wrong triples
with the given original document includes lots of       by our fine-tuned LLM in graph judgement mod-
meaningless triples. For example, <We, suggest,         ule. On the other hand, as demonstrated in the case,
goal>, <We, suggest, evaluation criterion>, <We,        the document refined by ECTD module exhibits
present, measure>, <We, present, selection func-        enhanced standardization and a reduction in irrele-
tion>, etc. It is obvious that these triples do not     vant terms, for instance, terms such as ‘-LRB-’ and
convey any beneficial information that could be         ‘-RRB-’ have been excluded as they are irrelevant
applied to subsequent tasks. And the triple <eval-      to the document’s subject matter.
uation criterion, new, goal>does not even follow
the general structure of triples, which means that
the adjective word ‘new’ is generally not employed
as a relational term within triples. The naive LLM
have strong zero-shot ability to generate them but it

                                                    10944
Prompt Template for Entity Extraction.

Goal:
Transform the text into a list of entities. Please ensure the comprehensiveness and accuracy of the
extracted entities, which should be related to the topic of the text.

Here are two examples:
Example#1:
Text: ”Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife
Trust.”
List of entities: [”Shotgate Thickets”, ”Nature reserve”, ”United Kingdom”, ”Essex Wildlife
Trust”]
Example#2:
Text: ”Garczynski Nunatak is a cone-shaped nunatak, the highest in a cluster of nunataks close
west of Mount Brecher, lying at the north flank of Quonset Glacier in the Wisconsin Range of the
Horlick Mountains of Antarctica.”
List of entities: [”Garczynski Nunatak”, ”nunatak”, ”Wisconsin Range”, ”Mount Brecher”,
”Quonset Glacier”, ”Horlick Mountains”, ”Antarctica”]

Refer to the examples and here is the question:
Text: {text}
List of entities:


Output:
     [{ entity_1 }, { entity_2 }, ... , { entity_n }]



      Figure 10: The prompt template for the closed-source LLM GPT-4o-mini to extract entities.




                                                        10945
Prompt Template for Relation Extraction.

Goal:
Transform the text into a semantic graph(a list of triples) with the given text and entities. Extract
subject-predicate-object triples from the assistant message. A predicate (1-3 words) defines the
relationship between the subject and object. Relationship may be fact or sentiment based on
assistant’s message. Subject and object are entities. Entities provided are from the assistant
message and prior conversation history, though you may not need all of them. This is for an
extraction task, please be thorough, accurate, and faithful to the reference text.

Note:
1.Generate triples as many as possible.
2.Make sure each item in the list is a triple with strictly three items.
Here are two examples:
Example#1:
Text: “Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife
Trust.”
Entity List: [“Shotgate Thickets”, ... , “Essex Wildlife Trust”]
Semantic Graph: [[S̈hotgate Thickets,̈ “instance of”, “Nature reserve”], ...]
Example#2:
Text: ..
Semantic Graph: ..
Refer to the examples and here is the question:
Text: {text}
Entity List: {entities}
Semantic graph:


Output:
     ```
     [
           { triple_1 },
           { triple_2 },
           ...
           { triple_n }
     ]
     ```



      Figure 11: The prompt template for the closed-source LLM GPT-4o-mini to construct KGs.




                                               10946
Prompt Template for MCQ Generation.

You are an expert in generating multiple-choice questions (MCQs) from scientific text. Your task
is to generate {n} MCQs based on the following document:
Each question should:
- Focus on the factual claims, numerical data, definitions, or relational knowledge from the
document.
- Have 4 options (one correct and three plausible distractors).
- Clearly indicate the correct answer.

The output should be in JSON format, with each question as a dictionary containing:
- “question”: The MCQ question.
- “options”: A list of 4 options (e.g., [“A: ...”, “B: ...”, “C: ...”, “D: ...”]).
- “answer”: The correct answer (e.g., “A”).

Output Example:
     ```
     [
           {
                 " question ": " What is the primary role of a catalyst in a chemical reaction ?" ,
                 " options ": [
                      "A ": "A catalyst is a substance that increases the rate of a chemical reaction
                      without being consumed in the reaction ." ,
                      "B ": "A catalyst is a substance that decreases the rate of a chemical reaction
                      without being consumed in the reaction ." ,
                      "C ": "A catalyst is a substance that is consumed in the reaction ." ,
                      "D ": "A catalyst is a substance that is not consumed in the reaction ."
                 ],
                 " answer ": "A"
           },
           ...
     ]
     ```

Passage:
{passage}

Output:
     ```
     [
           { MCQ_1 },
           { MCQ_2 },
           ...
           { MCQ_n }
     ]
     ```



                          Figure 12: The prompt template for the MCQ generations.




                                                    10947
Original Document: We suggest a new goal and evaluation criterion for word similarity measures .The new criterion
– meaning entailing substitutability – fits the needs of semantic-oriented NLP applications and can be evaluated directly
-LRB-independent of an application -RRB- at a good level of human agreement. Motivated by this semantic criterion we
analyze the empirical quality of distributional word feature vectors and its impact on word similarity results, proposing an
objective measure for evaluating feature vector quality. Finally, a novel feature weighting and selection function is presented ,
which yields superior feature vectors and better word similarity performance.
Ground-Truth Knowledge Graph:
                                                 [[”feature weighting and selection function”, ”used for”, ”word similar-
                                                 ity”], [”measure”, ”evaluate for”, ”feature vector quality”], [”feature vec-
                                                 tors”, ”conjunction”, ”word similarity”], [”evaluation criterion”, ”used for”,
                                                 ”word similarity measures”], [”meaning-entailing substitutability”, ”used
                                                 for”, ”semantic-oriented NLP applications”], [”human agreement”, ”evaluate
                                                 for”, ”meaning-entailing substitutability”], [”semantic criterion”, ”evaluate
                                                 for”, ”distributional word feature vectors”], [”distributional word feature
                                                 vectors”, ”used for”, ”word similarity”], [”feature weighting and selection
                                                 function”, ”used for”, ”feature vectors”]]




GPT-4o-mini:
                                                 [[”We”, ”suggest”, ”goal”], [”We”, ”suggest”, ”evaluation criterion”], [”We”,
                                                 ”propose”, ”measure”], [”evaluation criterion”, ”new”, ”goal”], [”evalua-
                                                 tion criterion”, ”fits”, ”needs”], [”evaluation criterion”, ”evaluated directly”,
                                                 ”good level of human agreement”], [”We”, ”analyze”, ”quality”], [”qual-
                                                 ity”, ”impact on”, ”word similarity results”], [”measure”, ”evaluating”,
                                                 ”feature vector quality”], [”feature vector quality”, ”yield”, ”superior fea-
                                                 ture vectors”], [”feature vector quality”, ”yield”, ”better word similarity
                                                 performance”], [”We”, ”present”, ”feature weighting function”], [”feature
                                                 weighting function”, ”yields”, ”superior feature vectors”], [”We”, ”present”,
                                                 ”selection function”], [”selection function”, ”yields”, ”superior feature vec-
                                                 tors”]]




GraphJudge:
                                                  [[’evaluation criterion’, ’related to’, ’word similarity measures’], [’evaluation
                                                  criterion’, ’assessed by’, ’human agreement’], [’evaluation criterion’, ’related
                                                  to’, ’semantic-oriented NLP applications’], [’evaluation criterion’, ’related
                                                  to’, ’meaning-entailing substitutability’], [’word similarity measures’, ’qual-
                                                  ity assessed by’, ’distributional word feature vectors’], [’distributional word
                                                  feature vectors’, ’impact on’, ’word similarity results’], [’distributional word
                                                  feature vectors’, ’quality evaluated by’, ’objective measure’], [’feature vec-
                                                  tors’, ’generated by’, ’feature weighting and selection function’], [’feature
                                                  vectors’, ’enhances’, ’word similarity performance’]]

                                                  Denoised Document: We propose a new evaluation criterion for word sim-
                                                  ilarity measures, called meaning-entailing substitutability, which is suitable
                                                  for semantic-oriented NLP applications. This criterion can be assessed inde-
                                                  pendently of any specific application with a high level of human agreement.
                                                  We examine the quality of distributional word feature vectors and their impact
                                                  on word similarity results, introducing an objective measure for evaluating
the quality of these feature vectors. Additionally, we present a novel feature weighting and selection function that generates
superior feature vectors and enhances word similarity performance.

            Table 8: Comparison of Construction Results between our GraphJudge and GPT-4o-mini.


                                                             10948
