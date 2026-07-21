Table B3 The attributes stored in entities of AutoMathKG.

|  Attribute | Storage type | Storage structure | Entity  |
| --- | --- | --- | --- |
|  Id | Int | Number | Thm, Def, Prob  |
|  Type | String | Theorem/definition/problem | Thm, Def, Prob  |
|  Label | String | Entity label | Thm, Def, Prob  |
|  Title | String | Entity title | Thm, Def, Prob  |
|  Field | String | Algebra/geometry/analysis/probability and statistics/applied mathematics/... | Thm, Def, Prob  |
|  Contents | List | [content1, content2, ...] | Thm, Def, Prob  |
|  Bodylist | List nested dictionary | [{"description": content1,"action": action1},...] | Thm, Def  |
|  Refs | List | [reference1, reference2, ...] | Thm, Def, Prob  |
|  References_tactics | Dictionary | {"reference1": tactic1,"reference2": tactic2,...} | Thm, Def, Prob  |
|  Source | String | ProofWiki/textbook/arXiv/TheoremQA | Thm, Def, Prob  |
|  Proofs | List nested dictionary | [{"contents": [...],"refs": [...],"bodylist": [...],"references_tactics": {...}},...] | Thm  |
|  Solutions | List nested dictionary | [{"contents": [...],"refs": [...],"bodylist": [],"references_tactics": {...}},...] | Prob  |
|  In_refs | Dictionary | {"in_reference1": tactic1,"in_reference2": tactic2,...} | Thm, Def, Prob  |
|  In_ref_ids | List | [in_id1, in_id2,...] | Thm, Def, Prob  |
|  Out_refs | Dictionary | {"out_reference1": tactic1,"out_reference2": tactic2,...} | Thm, Def, Prob  |
|  Out_ref_ids | List | [out_id1, out_id2,...] | Thm, Def, Prob  |

## Appendix D Math LLM training details

### D.1 Training datasets

The gemma-7b-it model was chosen as the base model for the construction of Math-LLM. Firstly, the model was fine-tuned using LORA on the MathInstruct  \( [15] \)  dataset, which contains various types of mathematical problem sets, enabling the model to acquire foundational problem-solving capabilities. Furthermore, the application adapter was trained on the “Microsoft/orca-math-word-problems-200k”  \( [57] \)  dataset, which includes a diverse range of application problems. The sympy adapter was trained on the GSM8K-sympy-v2  \( [58] \)  dataset, enhancing its ability to solve problems using the SymPy package. The proof adapter was trained on the NaturalProofs  \( [12] \)  dataset, which provides a comprehensive collection of definitions, theories, and their corresponding proofs.