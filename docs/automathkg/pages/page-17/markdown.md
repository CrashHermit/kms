original problem and used as a prompt to initiate another cycle of answer revision until it passes the self-calibration module.

## 6.2 Automatic knowledge fusion

Considering that mathematical corpora from different sources may have different statements of the same knowledge, we design a mechanism to automatically achieve knowledge fusion. For each new math entity from other sources, we utilize VD to search for similar candidate entities of Existing KG and employ LLM to determine whether to merge with a candidate or add the new entity. Given new input text, the following steps are taken for automatic knowledge fusion.

### *Step 1: Construction of Input KG from new input*

Information extraction and augmentation are conducted on the new input text as described in Section 4.3, for entity recognition preprocessing and knowledge enhancement, generating Input KG from the current input.

### *Step 2: Construction of VDs from Input KG and Existing KG*

Entities of Input KG and Existing KG are embedded using SBERT as described in Section 5, constructing two VDs, named Input VD and Existing VD, respectively.

### *Step 3: Fuzzy search for similar candidate entities in VD*

For each input entity $v_{input}$, a fuzzy search for its similar entities is conducted by comparing the cosine similarity between their entity vectors. Specifically, let $e_{input}$ be the vector embedding of $v_{input}$ in Input VD. Then, for each entity vector in Existing VD, calculate the cosine similarity with $e_{input}$. Finally, sort them in descending order and select the top $n$ entity vectors as the similar candidate entities for the input entity $v_{input}$.

### *Step 4: Determination of entity consistency by LLM*

After obtaining $n$ similar candidate entities, LLM is used to determine whether $v_{input}$ is consistent with any of the candidates. Specifically, we employ LLM through ICL to determine the consistency (see Table C5 in Appendix C for more details). For each similar candidate entity $v_i, i = 1, \dots, n$, LLM is asked to output “yes” if it is consistent to $v_{input}$; otherwise, LLM outputs “no”.

### *Step 5: Selection of update approach by LLM*

Finally, we analyze the consistency results and select the appropriate update approach for each input entity. If LLM outputs “no” for all similar candidate entities, $v_{input}$ will be considered as a new entity and added to the Existing KG. If LLM outputs “yes” for only one candidate, $v_{input}$ will be merged with it, with reference relationships between $v_{input}$ and other entities in Input KG being added. If LLM outputs “yes” for multiple candidates, LLM will perform a secondary judgment through ICL on all the candidates with “yes” output, selecting the most similar candidate entity (see Table C5 in Appendix C for more details). Then, $v_{input}$ will be merged with that candidate, with relationships being added.