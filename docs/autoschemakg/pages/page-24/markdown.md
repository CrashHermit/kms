|  Subject | Task  |
| --- | --- |
|  History | high school european history, high school us history, high school world history, prehistory  |
|  Formal Logic | formal logic, logical fallacies  |
|  Law | international law, jurisprudence, professional law  |
|  Philosophy and Ethics | philosophy, moral disputes, moral scenarios, business ethics  |
|  Religion | world religions  |
|  Medicine and Health | clinical knowledge, college medicine, medical genetics, professional medicine, virology, human aging, nutrition, anatomy  |
|  Social Sciences | high school geography, high school government and politics, high school psychology, professional psychology, sociology, human sexuality, us foreign policy, security studies  |
|  Economics | high school macroeconomics, high school microeconomics, econometrics  |
|  Business and Management | management, marketing, professional accounting, public relations  |
|  Math | abstract algebra, college mathematics, elementary mathematics, high school mathematics, high school statistics  |
|  Natural Sciences | astronomy, college biology, college chemistry, college physics, conceptual physics, high school biology, high school chemistry, high school physics  |
|  Computer Science and Engineering | college computer science, high school computer science, computer security, electrical engineering, machine learning  |
|  Global Facts | global facts, miscellaneous  |

Table 11: The correspondence between subjects and tasks.

|  Corpus | MMLU  |   |   |   |   |   |   |   |   |   |   |   |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |  overall | History | Law | Religion | PaE | MaH | GF | BaM | SS | Logic | Econ | Math | NS | CSaE  |
|  None | 69.18 | 76.59 | 66.86 | 83.04 | 63.55 | 70.38 | 66.72 | 72.20 | 79.74 | 64.35 | 68.35 | 57.31 | 65.27 | 66.70  |
|  Freebase-ToG | **70.36** | **78.42** | 69.00 | 75.44 | 65.67 | 72.65 | 67.27 | **73.67** | 76.00 | 66.03 | 67.34 | 60.56 | 69.39 | **68.23**  |
|  *Random Baseline*  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
|  Wikipedia | 68.06 | 76.64 | 66.82 | 79.53 | 59.26 | 70.34 | 66.46 | 67.34 | 77.78 | 59.21 | 65.35 | 60.91 | 67.52 | 61.87  |
|  Common Crawl | 67.93 | 74.89 | 66.52 | 79.53 | 61.74 | 69.82 | 68.11 | 67.67 | 77.52 | 59.30 | 64.20 | 59.80 | 67.42 | 62.22  |
|  Pes2o-Abstract | 68.07 | 76.24 | 64.16 | 80.70 | 62.01 | 70.62 | 66.59 | 69.27 | 77.16 | 62.39 | 64.70 | 60.07 | 66.41 | 62.18  |
|  *Text Corpora + DBM25*  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
|  Wikipedia | 68.99 | 76.67 | 67.35 | 78.36 | 63.34 | 69.35 | 61.98 | 71.39 | 76.99 | 62.30 | 65.56 | 61.67 | 69.31 | 65.60  |
|  Common Crawl | 68.33 | 76.15 | 66.36 | 80.12 | 60.43 | 69.58 | 64.67 | 69.47 | 76.71 | 63.18 | 68.22 | 62.26 | 65.55 | 65.04  |
|  Pes2o-Abstract | 69.04 | 78.01 | 65.89 | 78.95 | 63.83 | 71.01 | 65.78 | 68.29 | 77.07 | 59.34 | **68.87** | 61.20 | 67.73 | 65.74  |
|  *Text Corpora + Dense Retrieval*  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
|  Wikipedia | 69.37 | 73.59 | 69.60 | 79.53 | 63.58 | 70.82 | 62.41 | 72.57 | 76.83 | 62.21 | 67.35 | 61.79 | **69.81** | 65.39  |
|  Common Crawl | 67.03 | 74.47 | 68.98 | 79.53 | 60.46 | 69.29 | 64.09 | 68.88 | 75.21 | 61.86 | 62.27 | 57.13 | 64.54 | 64.47  |
|  Pes2o-Abstract | 69.07 | 75.79 | 61.82 | 78.36 | 65.15 | 69.72 | 66.77 | 69.02 | 76.47 | 63.05 | 63.07 | **63.92** | 69.53 | 67.86  |
|  *ATLAS + HippoRAG2*  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
|  ATLAS-Wiki | 68.22 | 76.73 | 67.38 | 84.21 | **66.01** | 70.82 | 68.36 | 72.35 | 79.16 | 63.65 | 64.53 | 50.09 | 65.45 | 62.10  |
|  ATLAS-CC | 68.26 | 78.16 | **70.85** | 83.04 | 65.60 | 71.28 | 63.95 | 68.84 | 78.16 | 65.42 | 67.51 | 52.87 | 63.32 | 63.40  |
|  ATLAS-Pes2o | 69.19 | 77.13 | 68.41 | 81.29 | 65.05 | **72.75** | 65.67 | 72.32 | 81.19 | 62.98 | 65.29 | 54.24 | 65.41 | 64.04  |
|  *ATLAS + ToG*  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
|  ATLAS-Wiki | 68.29 | 77.91 | 66.60 | 84.21 | 65.10 | 70.69 | 63.85 | 70.49 | 78.31 | 67.08 | 66.24 | 54.41 | 63.65 | 64.18  |
|  ATLAS-CC | 68.40 | 77.07 | 68.18 | 83.63 | 65.24 | 72.03 | 66.87 | 71.21 | 79.72 | 66.59 | 66.42 | 48.74 | 63.97 | 64.22  |
|  ATLAS-Pes2o | 68.97 | 77.52 | 66.95 | **84.80** | 63.44 | 71.15 | **68.92** | 69.98 | **81.59** | **67.87** | 66.17 | 55.46 | 63.35 | 64.75  |

Table 12: Performance comparison of our knowledge graph (KG) integrated with HippoRAG2 and ToG against baseline retrieval methods (Random, BM25, Dense Retrieval) across Wikipedia, Common Crawl, and Pes2o-Abstract corpora on MMLU benchmarks. Tasks are classified according to subjects, with bold and underline indicating the highest and the second highest performance. PaE, MaH, GF, BaM, SS, Econ, NS and CSaE represent Philosophy_and_Ethics, Medicine_and_Health, Global_Facts, Business_and_Management, Social_Sciences, Economics, Natural_Sciences and Computer_Science_and_Engineering respectively.