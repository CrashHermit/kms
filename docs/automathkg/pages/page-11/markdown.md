Table 2 Samples of different types of entities storage.

|  Definition: Symmetric Group  |   |
| --- | --- |
|  Contents | Let S be a set. Let Γ(S) denote the set of permutations on S. Let (Γ, o) be the algebraic structure such that o denotes the composition of mappings. Then (Γ, o) is called the symmetric group on S. If S has n elements, then (Γ, o) is often denoted Sn.  |
|  Bodylist | [{"description": "Let S be a set.", "action": "premise"}, {"description": "Let Γ(S) denote the set of permutations on S. Let (Γ, o) be the algebraic structure such that o denotes the composition of mappings. Then (Γ, o) is called the symmetric group on S. If S has n elements, then (Γ, o) is often denoted Sn.", "action": "definition"}]  |
|  References_tactics | {"Definition:Composition of Mappings": "definition", "Definition:Algebraic Structure": "definition", "Definition:Element": "definition", "Definition:Permutation": "definition", "Definition:Set": "premise"}  |
|  Theorem: Center of Symmetric Group is Trivial  |   |
|  Contents | Let n ∈ N be a natural number. Let Sn denote the symmetric group of order n. Let n ≥ 3. Then the center Z(Sn) of Sn is trivial.  |
|  Bodylist | [{"description": "Let n ∈ N be a natural number.", "action": "premise"}, {"description": "Let Sn denote the symmetric group of order n. Let n ≥ 3.", "action": "assumption"}, {"description": "Let n ≥ 3. Then the center Z(Sn) of Sn is trivial.", "action": "conclusion"}]  |
|  References_tactics | {"Definition:Order of Structure": "premise", "Definition:Center (Abstract Algebra)/Group": "premise", "Definition:Natural Numbers": "premise", "Definition:Trivial Group": "premise", "Definition:Symmetric Group": "premise"}  |
|  Problem: Positive Definiteness of a Matrix  |   |
|  Contents | Consider the matrix of A = [[1,4], [4,1]], is this a positive definite matrix?  |
|  References_tactics | {"definition:positive definite matrix": "deduction"}  |

process. Edges can point to any type of entity, but they always originate from a Definition or Theorem entity. These directed edges form logical or inferential chains among mathematical knowledge, demonstrating the dependency relationships and deduction processes between different mathematical entities.

All references between entities form a directed graph, which may contain cycles. For example, Pythagoras's Theorem and Sum of Squares of Sine and Cosine are mutually referenced in their proofs. Figure 5 illustrates all possible relationships between different types of entities. Additionally, to clarify the tactic role played by entities when referenced, we design nine tactic labels: premise, assumption, lemma, proposition, corollary, calculation, enumeration, definition, and conclusion. Each edge includes tactic label information, stored in the "in_refs" and "out_refs" fields of the entity JSON structure.

### 4.3 Information extraction and augmentation

This subsection will detail the instruction on how to extract and augment information to construct an Input KG according to the input text from a technical perspective.