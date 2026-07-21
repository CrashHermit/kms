Algorithm 1 Think on Graph (ToG) (Sun et al., 2024a) for Question Answering

Require: Knowledge Graph G, Query q, Top-N parameter, Maximum depth  \( D_{max} \) 

Ensure: Answer to query q

1: Extract entities from query q using NER
2: Retrieve top-k initial nodes from G based on entity similarity
3: Let  \( P \leftarrow \)  set of paths, each containing a single initial node
4:  \( D \leftarrow 0 \)  ▷ Current search depth
5: while  \( D \leq D_{max} \)  do
6:    \( P \leftarrow \text{Search}(q, P, G) \)  ▷ Expand paths by one hop
7:    \( P \leftarrow \text{Prune}(q, P, N) \)  ▷ Keep top-N most relevant paths
8:    if Reasoning(q, P) determines paths sufficient then
9:    return Generate(q, P) ▷ Generate answer using paths
10:    end if
11:    \( D \leftarrow D + 1 \) 
12: end while
13: return Generate(q, P) ▷ Generate answer using best available paths
14: procedure SEARCH(q, P, G)
15:    \( P_{new} \leftarrow \emptyset \) 
16:    for each path  \( p \in P \)  do
17:    \( e_{tail} \leftarrow \)  last entity in path p
18:    \( S \leftarrow \)  successors of  \( e_{tail} \)  in G not already in p
19:    \( R \leftarrow \)  predecessors of  \( e_{tail} \)  in G not already in p
20:    if  \( S = \emptyset \)  and  \( R = \emptyset \)  then
21:    \( P_{new} \leftarrow P_{new} \cup \{p\} \)  ▷ Keep dead-end paths
22:    else
23:    for each node  \( n \in S \)  do
24:    \( r \leftarrow \)  relation from  \( e_{tail} \)  to n in G
25:    \( P_{new} \leftarrow P_{new} \cup \{p + [r, n]\} \)  ▷ Extend path forward
26:    end for
27:    for each node  \( n \in R \)  do
28:    \( r \leftarrow \)  relation from n to  \( e_{tail} \)  in G
29:    \( P_{new} \leftarrow P_{new} \cup \{p + [r, n]\} \)  ▷ Extend path backward
30:    end for
31:    end if
32:    end for
33:    return  \( P_{new} \) 
34: end procedure
35: procedure PRUNE(q, P, N)
36:    Score each path in P using LLM relevance assessment (1-5 scale)
37:    Sort paths by decreasing score
38:    return top-N highest scoring paths
39: end procedure
40: procedure REASONING(q, P)
41:    Extract triples from paths in P
42:    Ask LLM if triples are sufficient to answer q (Yes/No)
43:    return True if answer is "Yes", False otherwise
44: end procedure
45: procedure GENERATE(q, P)
46:    Extract triples from paths in P
47:    Prompt LLM with triples and query q to generate answer
48:    return generated answer
49: end procedure