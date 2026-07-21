Algorithm 2 HippoRAG2 (Gutiérrez et al., 2025)

General algorithm follows the original implementation, while we modify the initialization of graph and embeddings.

1: function INIT(graph_type)
2:    if graph_type is entity then
3:    graph, embedding ← Graph(entity), Embeddings(entity)
4:    else if graph_type is entity+event then
5:    graph, embedding ← Graph(entity, event), Embeddings(entity, event)
6:    else if graph_type is entity+event+concept then
7:    graph, embedding ← Graph(entity, event, concept), Embeddings(entity, event, concept)
8:    end if
9: end function
10: function QUERY2EDGE(query, topN)
11:    \( Q_{emb} \leftarrow \) Retriever(query)
12:    \( S = Q \cdot W_e \)   ▷ Calculate similarity scores with precomputed edge embeddings
13:    \( E = \text{argsort}_i(S)[: N] \)   ▷ Select topN edges based on scores
14:    filtered_edges ← LLM_filter(E)   ▷ Filter edges using Large Language Model
15:    mapped_edges ← Map_edges(filtered_edges)   ▷ Map filtered edges to original edges
16:    return_node_scores ← Calculate_node_scores(mapped_edges)
17:    return return_node_scores
18: end function
19: function QUERY2PASSAGE(query, weight_adjust)
20:    \( Q_{pass} \leftarrow \) Encode(query)   ▷ Encode query into passage representation
21:    \( S_{text} \leftarrow \) Similarity_Scores(\( Q_{pass} \), text_embeddings)
22:    return Scores_Dictionary(\( S_{text} \))
23: end function
24: function RETRIEVE_PERSONALIZATION_DICT(query, topN)
25:    node_dict ← query2edge(query, topN)
26:    text_dict ← query2passage(query, weight_adjust)
27:    return node_dict, text_dict
28: end function
29: function RETRIEVE_PASSAGES(query, topN)
30:    node_dict, text_dict ← retrieve_personalization_dict(query, topN)
31:    if node_dict is empty then
32:    return TopN_Text_Passages(text_dict)
33:    else
34:    personalization_dict ← {node_dict, text_dict}
35:    page_rank_scores ← PageRank(personalization_dict)
36:    return TopN_Passages(page_rank_scores)
37:    end if
38: end function