Table C4 The templates of entity augmentation.

|  Attribute | Entity | Template  |
| --- | --- | --- |
|  Title | Def | Your task is to generate a title that can summarize the content of the given math definition/theorem/problem.  |
|   |  Thm  |   |
|   |  Prob  |   |
|  Field | Def | Your task is to identify the most relevant mathematical field for the given math definition/theorem/problem from the following choices: “algebra”, “geometry”, “analysis”, “logic”, “probability and statistics”, “applied mathematics”, “foundations of mathematics”. Choose one from them.  |
|   |  Thm  |   |
|   |  Prob  |   |
|  Bodylist | Def | Your task is to label each element in the given content list of a math definition/theorem/theorem proof, in order to determine the role of each element. The labels of mathematical roles are: “premise”, “assumption”, “lemma”, “corollary”, “definition”, “conclusion”, “deduction”, “calculation”, “enumeration”. Choose only the most relative one from them when you label.  |
|   |  Thm (including proof)  |   |
|  Refs | Prob (including solution) | Your task is to identify the mathematical definitions or theorems referenced in the given math problem/problem solution. When there is at least one reference, output each reference in the following format: “definition:” or “theorem:” along with the original reference. When there is no reference, output an empty list.  |
|  References_tactics | Def | Your task is to label each reference shown in the given content list of a math definition/theorem/theorem proof/problem/problem solution, in order to determine the roles of each reference. The labels of mathematical roles are: “premise”, “assumption”, “lemma”, “corollary”, “definition”, “conclusion”, “deduction”, “calculation”, “enumeration”. Choose only the most relative one from them when you label.  |
|   |  Thm (including proof)  |   |
|   |  Prob (including solution)  |   |

## D.2 Training hyperparameters

For all training processes, including base model fine-tuning and task adapter training, we employed fully shared data parallelism (FSDP) with the configuration parameters: node=1 and proc_per_node=4 in the PyTorch framework. The optimizer used was paged_adaw_32bit. The gemma-7b-it model was loaded in torch.bf16 format. The maximum sequence length was set to 1024. The LoRA alpha was configured to 16, LoRA rank to 8, and LoRA dropout to 0.1. The learning rate was set to 2e-4, with a weight decay of 0.001. The training batch size was set to 4, with fine-tuning performed for 50 epochs and adapters trained for 25 epochs.