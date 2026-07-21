ual schema design. The induced schema aligns with the formal definition of a knowledge graph  \( G = (V, E, C, \phi, \psi) \) , where C denotes the set of concepts, and  \( \phi \)  and  \( \psi \)  map nodes and relations to subsets of C, respectively.

Our schema induction pipeline processes the triples extracted from the Dolma corpus (Soldaini et al., 2024), organizing them into batches and employing a generative approach to derive abstract representations. The process targets three components—events  \( (V_{E}) \) , entities  \( (V_{N}) \) , and relations  \( (R) \) —producing a set of conceptual phrases for each, which collectively form the concept set C. This section outlines the abstraction methodology, the role of context in entity conceptualization, and key implementation details.

#### Abstract Event Phrase Generation

I will give you an EVENT. You need to give several phrases containing 1-2 words for the ABSTRACT EVENT of this EVENT. You must return your answer in the following format: phrases1, phrases2, phrases3,... You can't return anything other than answers. These abstract event words should fulfill the following requirements:

1. The ABSTRACT EVENT phrases can well represent the EVENT, and it could be the type of the EVENT or the related concepts of the EVENT.
2. Strictly follow the provided format, do not add extra characters or words.
3. Write at least 3 or more phrases at different abstract level if possible.
4. Do not repeat the same word and the input in the answer.
5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

Examples:

EVENT: A man retreats to mountains and forests
Your answer: retreat, relaxation, escape, nature, solitude

EVENT: A cat chased a prey into its shelter Your answer: hunting, escape, predation, hiding, stalking
EVENT: Sam playing with his dog Your answer: relaxing event, petting, playing, bonding, friendship
EVENT: [EVENT] Your answer:

Figure 5: This figure shows the prompt used for generating the concepts for an event.

#### B.2.1 Abstraction Methodology

The schema induction begins by categorizing the nodes and edges of the knowledge graph G into events, entities, and relations. For each category, we process the elements in batches of size  \( B_{s} \)  to optimize computational efficiency and scalability. The LLM is prompted with tailored instructions to generate a list of phrases, each containing one to

two words, that abstractly represent the input element. These phrases must satisfy several criteria: they should encapsulate the element's type or related concepts, vary in abstraction level, and avoid repetition or inclusion of the original input term. For each element, a minimum of three phrases is targeted, though more may be generated depending on the LLM's output.

For events  \( (v \in V_{E}) \) , the prompt directs the LLM to identify abstract event types or related notions. For example, an event such as "Sam playing with his dog" might yield phrases like "playing," "bonding," and "relaxing event," reflecting different levels of generality. For entities  \( (e \in V_{N}) \) , the prompt similarly elicits abstract entity types, augmented by contextual information derived from the graph structure, as detailed below. Relations  \( (r \in R) \)  are abstracted into phrases that capture their semantic essence, such as transforming "participated in" into "engage in," "attend," and "involve in." The LLM generates these outputs in a structured format, which we parse into lists of phrases, forming the mappings  \( \phi(v) \) ,  \( \phi(e) \) , and  \( \psi(r) \)  to the concept set C.

The abstraction process operates in batches, processing  \( B_{s} \)  elements simultaneously. The input prompts are tokenized to a maximum length  \( L_{tok} \) , and the LLM generates responses under controlled parameters (e.g., temperature  \( \tau \)  and top-p sampling with probability p) to balance creativity and coherence. The resulting phrases are stored alongside their corresponding elements, ensuring traceability and enabling subsequent analysis.

#### B.2.2 Contextual Enhancement for Entities

As shown in Figure 6, to enhance the accuracy of entity abstraction, we incorporate contextual information extracted from the knowledge graph. For each entity \( e \in V_N \), we examine its neighboring nodes—predecessors and successors—along with their associated relations. A subset of these neighbors, limited to \( N_{ctx} \) (e.g., one predecessor and one successor), is randomly sampled to construct a context string. This string concatenates the neighbor's identity and relation (e.g., "neighbor1 relation1, relation2 neighbor2"), providing the LLM with additional semantic cues. For instance, an entity "Black Mountain College" with context "started by John Andrew Rice" might yield phrases like "college," "school," and "liberal arts college." This contextual enrichment ensures that the abstracted types are grounded in the entity's role within the graph,