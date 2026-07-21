improving the schema's relevance and specificity.

#### Abstract Entity Phrase Generation

I will give you an ENTITY. You need to give several phrases containing 1-2 words for the ABSTRACT ENTITY of this ENTITY. You must return your answer in the following format: phrases1, phrases2, phrases3,... You can't return anything other than answers. These abstract intention words should fulfill the following requirements:

1. The ABSTRACT ENTITY phrases can well represent the ENTITY, and it could be the type of the ENTITY or the related concepts of the ENTITY.
2. Strictly follow the provided format, do not add extra characters or words.
3. Write at least 3 or more phrases at different abstract level if possible.
4. Do not repeat the same word and the input in the answer.
5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

##### Examples:

ENTITY: Soul CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release Your answer: movie, film

ENTITY: Thinkpad X60 CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60 Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand

ENTITY: Harry Callahan CONTEXT: bluffs another robber, tortures Scorpio Your answer: person, American, character, police officer, detective

ENTITY: Black Mountain College CONTEXT: was started by John Andrew Rice, attracted faculty Your answer: college, university, school, liberal arts college

ENTITY: 1st April CONTEXT: Utkal Dibas celebrates Your answer: date, day, time, festival

ENTITY: [ENTITY] CONTEXT: [CONTEXT] Your answer:

Figure 6: This figure shows the conceptualization prompts for entities enhanced with context.

Events and relations, as shown in Figure 5 and Figure 7, by contrast, rely solely on their textual descriptions without additional context, as their abstraction focuses on inherent semantics rather than graph connectivity. This distinction reflects the differing roles of nodes and edges in the knowledge graph structure.

#### B.2.3 Implementation Details

The schema induction pipeline processes a graph G serialized from the triple extraction phase, typically stored in a binary format and loaded into memory. The elements are partitioned into batches, with the option to apply slicing (dividing the workload into  \( S_{total} \)  slices and processing the  \( S_{slice} \) -th portion) for distributed computation. If a sample size  \( N_{sample} \)  is specified, a random subset of batches is

#### Abstract Relation Phrase Generation

I will give you a RELATION. You need to give several phrases containing 1-2 words for the ABSTRACT RELATION of this RELATION. You must return your answer in the following format: phrases1, phrases2, phrases3,... You can't return anything other than answers. These abstract intention words should fulfill the following requirements:

1. The ABSTRACT RELATION phrases can well represent the RELATION, and it could be the type of the RELATION or the simplest concepts of the RELATION.
2. Strictly follow the provided format, do not add extra characters or words.
3. Write at least 3 or more phrases at different abstract level if possible.
4. Do not repeat the same word and the input in the answer.
5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

##### Examples:

RELATION: participated in Your answer: become part of, attend, take part in, engage in, involve in

RELATION: be included in Your answer: join, be a part of, be a member of, be a component of

RELATION: [RELATION] Your answer:

Figure 7: This figure shows the conceptualization prompts for relations enhanced with context.

selected to reduce processing time during experimentation.

The LLM, configured with a precision setting (e.g., float16) and optimized with acceleration techniques, operates on a GPU to handle the batched inference efficiently. Prompts are formatted using a model-specific chat template,  \( T_{chat} \) , ensuring compatibility with the LLM's input-output conventions. The generated phrases are written to a CSV file, with each row recording the original element, its abstracted phrases, and its type (event, entity, or relation). Post-processing aggregates these phrases to compute the unique concepts in C, providing statistics on the schema's coverage, such as the number of distinct event types, entity types, and relation types.

This approach yields a flexible and automated schema, mapping each node  \( v \in V \)  and relation  \( r \in R \)  to a subset of concepts in C via  \( \phi \)  and  \( \psi \) . By abstracting specific instances into general types, the induced schema enhances the knowledge graph's adaptability, supporting downstream applications across varied domains without requiring manual curation.