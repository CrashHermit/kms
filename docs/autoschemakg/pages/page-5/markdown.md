|  Knowledge Graph | Triple Type | Precision | Recall | F1  |
| --- | --- | --- | --- | --- |
|  ATLAS-Wiki | Entity-Entity | 99.13 | 90.10 | 94.09  |
|   |  Event-Entity | 100.0 | 92.59 | 95.60  |
|   |  Event-Event | 99.60 | 93.59 | 96.01  |
|  ATLAS-Pes2o | Entity-Entity | 97.66 | 89.89 | 93.03  |
|   |  Event-Entity | 100.0 | 94.29 | 96.83  |
|   |  Event-Event | 99.54 | 91.31 | 94.94  |
|  ATLAS-CC | Entity-Entity | 95.65 | 84.64 | 88.82  |
|   |  Event-Entity | 99.93 | 87.92 | 92.72  |
|   |  Event-Event | 99.86 | 93.20 | 96.16  |

Table 2: Triple precision, recall and F1 score across datasets. Each row displays the performance of a type of extracted triples.

original text and the triples extracted by Llama-3-8B-Instruct; (2) DeepSeek-V3 identifies triples that are incorrectly extracted (false positives); (3) DeepSeek-V3 lists facts present in the original text but missing from the extracted triples (false negatives); This methodology allows us to calculate precise metrics: (1) Precision: proportion of correctly extracted triples out of all extracted triples; (2) Recall: proportion of correctly extracted triples among all ground-truth triples in the text; (3) F1 score: the harmonic mean of precision and recall. As shown in Table 2, our approach demonstrates exceptional extraction quality across all datasets, with particularly strong performance on Wikipedia content. The precision, recall, and F1 scores of the triples in our KG exceed 90% in most cases, demonstrating the high quality and reliability of our extracted triples.

|  Dataset | Context | Model  |   |
| --- | --- | --- | --- |
|   |   |  LLaMA-3-8B | LLaMA-3-70B  |
|  ATLAS-Wiki | [Lower-Upper] | 46.29-99.29 | 65.69-99.70  |
|   |  Entity | 65.08 | 70.96  |
|   |  Event | 92.69 | 94.82  |
|   |  Event + Entity | 93.30 | 95.13  |
|  ATLAS-Pes2o | [Lower-Upper] | 62.32-98.99 | 75.05-99.49  |
|   |  Entity | 80.00 | 83.33  |
|   |  Event | 96.97 | 97.78  |
|   |  Event + Entity | 97.37 | 97.98  |
|  ATLAS-CC | [Lower-Upper] | 56.08-97.29 | 70.25-99.10  |
|   |  Entity | 76.78 | 81.01  |
|   |  Event | 94.87 | 96.78  |
|   |  Event + Entity | 96.28 | 96.98  |

Table 3: KG performance across datasets, showing bounds (no context to full passage) and results with different knowledge representations. Entity, Event, and combined representations preserve most information for MCQs, approaching full-passage performance across all datasets and models.

Measuring Information Preservation in Knowledge Graphs We evaluate the effectiveness of

the entity-level triples and event-level triples of our constructed KG in preserving information from original passages. We test how well multiple-choice question (MCQ) performance is preserved when we convert the original passage into KG data. Following the evaluation protocol from the existing work (Schuhmann et al., 2025), we generate five MCQs with LLaMA-3-70B-Instruct for each original passage, and the prompts are shown in Figure 8. We sample 200 original passages and 1,000 MCQs are obtained for each dataset. We ask LLMs to answer them with no context (denoted as lower bound), then ask them again with the original passage (denoted as upper bound) for sanity check. Finally, we conduct tests using entity-level triples (denoted as Entity), event-level triples (denoted as Event), and a combination of both entity-level and event-level triples (denoted as Event + Entity). We evaluate on three pre-training datasets with our constructed KG in Table 1: En-Wiki, Pes2o-Abstract and Common Crawl. According to the results shown in Table 3, we have the following insights: (1) Information is well preserved in our constructed KG. MCQs performance with Entity, Event or Event + Entity remains far above the lower bound baseline and approaches the original-passage upper bound. It suggests that the information in the original passages is well preserved in our constructed KG; (2) Events are more effective than entities. The MCQs performance with Event or Event + Entity is much closer to the upper bound than that with Entity, which accuracy is more than \(95\%\) in most of the cases. It demonstrates that the event-level triples can preserve richer and more precious information than entity-level triples.

|   | Task | Dataset | BS-R | BS-C  |
| --- | --- | --- | --- | --- |
|  LLaMA-3-8B | Entity Typing | FB15kET | 88.57 | 86.54  |
|   |   |  YAGO43kET | 80.67 | 58.86  |
|   |  Event Typing | wikiHow | 99.18 | 99.26  |
|   |  Relation Typing | FB15kET | 88.75 | 88.41  |
|  LLaMA-3-13B | Entity Typing | FB15kET | 89.25 | 88.59  |
|   |   |  YAGO43kET | 94.26 | 90.56  |
|   |  Event Typing | wikiHow | 98.97 | 99.33  |
|   |  Relation Typing | FB15kET | 88.58 | 88.66  |
|  LLaMA-3-70B | Entity Typing | FB15kET | 89.49 | 87.30  |
|   |   |  YAGO43kET | 94.61 | 92.64  |
|   |  Event Typing | wikiHow | 99.41 | 99.15  |
|   |  Relation Typing | FB15kET | 88.70 | 90.33  |

Table 4: Results of schema induction with various LLaMA family LLMs across three kinds of typing tasks.