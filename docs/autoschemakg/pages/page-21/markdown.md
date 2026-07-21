Algorithm 3 LargeKGRetriever

A variant of HippoRAG2 (Gutiérrez et al., 2025), optimized with dynamic graph sampling and common word filtering

1: function INIT(graph_type)
2:    keyword ← Default_graph_keyword ▷ Keyword can be cc, pes2o, wiki
3:    Initialize_resources(keyword)
4:    Load_node_and_edge_indexes()
5: end function
6: function RETRIEVE_TOPK_NODES(query, top_k_nodes)
7:    entities ← LLM_NER(query)
8:    KG_entities ← Encode_and_Search(entities, FAISS_index)
9:    filtered_keywords ← LLM_filter(KG_entities)
10:    return filtered_keywords
11: end function
12: function RETRIEVE_PERSONALIZATION_DICT(query, number_of_source_nodes)
13:    topk_nodes ← retrieve_topk_nodes(query, number_of_source_nodes)
14:    if topk_nodes == {} then
15:    return {}
16:    end if
17:    Update personalization dictionary with topk_nodes
18:    return Personalization dictionary
19: end function
20: function PAGERANK(personalization_dict, topN, sampling_area)
21:    \( G_{Sample} \leftarrow \) Random Walk with Restart Sampling
22:    \( Scores = \text{PageRank}(G_{Sample}, \text{personalization\_dict}) \)
23:    topN_nodes = argsort\( _i \)(Scores)[: N]
24:    for node in topN nodes do
25:    Connected_Passage += node.score
26:    end for
27:    return TopN_Ranked_Passages
28: end function
29: function RETRIEVE_PASSAGES(query, topN, number_of_source_nodes, sampling_area)
30:    personalization_dict ← retrieve_personalization_dict(query, number_of_source_nodes)
31:    if personalization_dict is empty then
32:    return {}, [0]
33:    end if
34:    topN_passages ← pagerank(personalization_dict, topN, sampling_area)
35:    return topN_passages
36: end function