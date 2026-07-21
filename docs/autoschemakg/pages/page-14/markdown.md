## A Prompts for Triple Extractions

### Entity Relationship Extraction

Given a passage, summarize all the important entities and the relations between them in a concise manner. Relations should briefly capture the connections between entities, without repeating information from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from being considered as entities. The output should strictly adhere to the following JSON format:

[
    {
    "Head": "{a noun}",
    "Relation": "{a verb}",
    "Tail": "{a noun}",
    }
    ...
]
Here is the passage:

Figure 2: The figure demonstrates the prompts we use to generate the text triples describing relations between entities.

The prompts used for extracting entity-entity, entity-event, and event-event triples are given in Figure 2, Figure 3, and Figure 4 respectively.

## B Implementation Details of Knowledge Graph Construction Framework

In this section, we elaborate on the process of fully automating knowledge graph construction. Given a collection of n documents  \( D = \{D_{1}, D_{2}, D_{3}, \ldots, D_{n}\} \) , our method systematically builds the graph.

### B.1 Triple Extraction

Our approach to triple extraction employs a multiphase pipeline that utilizes the generative power of Large Language Models (LLMs) to convert unstructured text into structured knowledge triples, drawing from the Dolma corpus (Soldaini et al., 2024). This pipeline systematically extracts three categories of relationships—Entity-Entity, Entity-Event, and Event-Event—forming the foundation of a comprehensive knowledge graph. Designed for scalability and resilience, the method incorporates batch processing, text segmentation, and robust output parsing to efficiently process large-scale datasets.

The extraction unfolds across three sequential stages, each tailored to a specific relationship type, leveraging a single LLM guided by distinct prompts to produce structured outputs in JSON

### Event and Entity Triple Extraction

Please analyze and summarize the participation relations between the events and entities in the given paragraph. Each event is a single independent sentence. Additionally, identify all the entities that participated in the events. Do not use ellipses. Please strictly output in the following JSON format:

[
    {
    "Event": "{a simple sentence describing an event}",
    "Entity": ["{entity 1}", "{entity 2}", "..."]
    }
    ...
]

Figure 3: The figure demonstrates the prompts we use to generate the text triples describing relations between entities and events.

format. To manage the constraints of LLM input capacity, denoted as  \( L_{max} \)  tokens, we preprocess the text corpus to ensure compatibility, segmenting documents as needed and organizing them into batches for efficient processing. This section outlines the preprocessing strategy, the staged extraction process, and key implementation details.

#### B.1.1 Text Preprocessing and Data Organization

Given a corpus  \( D = \{D_{1}, D_{2}, \ldots, D_{n}\} \)  of n documents, we first filter the dataset to include only English-language texts, identified through metadata or assumed if unspecified, to match the linguistic capabilities of our LLMs. To adhere to the token limit  \( L_{max} \) , we account for an instructional prompt length,  \( L_{inst} \)  derived from empirical observations. The maximum token length per text segment,  \( C_{max} \) , is calculated as:  \( C_{max} = (L_{max} - L_{inst}) \)

Documents exceeding \( C_{max} \) are divided into smaller chunks, each tagged with a unique identifier and metadata to maintain traceability. This segmentation ensures that inputs remain within the LLM's token capacity, preserving contextual integrity without truncation.

The preprocessed text is then grouped into batches of size B, utilizing a custom data management framework that integrates with standard dataset loading tools. Tokenization is applied to each batch, adjusting for padding and truncation to produce consistent input representations suitable for LLM processing.