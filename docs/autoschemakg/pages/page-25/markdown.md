### Multiple-Choice Question Generation and Answering

#### MCQ Generation Prompt:

You are an expert in generating multiple-choice questions (MCQs) from scientific texts. Your task is to generate 5 multiple-choice questions based on the following passage.

Each question should:

- Focus on factual claims, numerical data, definitions, or relational knowledge from the passage.

- Have 4 options (one correct answer and three plausible distractors).

- Clearly indicate the correct answer.

The output should be in JSON format, with each question as a dictionary containing:

- "question": The MCQ question.
- "options": A list of 4 options (e.g., ["A: ..", "B: ..", "C: ..", "D: .."]).

- "answer": The correct answer (e.g., "A").

Output Example:

[
    {
    "question": "What is the primary role of a catalyst in a chemical reaction?",
    "options": [

"options": [
    "A: To make a thermodynamically unfavorable reaction proceed",
    "B: To provide a lower energy pathway between reactants and products",
    "C: To decrease the rate of a chemical reaction",
    "D: To change the overall reaction itself"
    ],
    "answer": "B"
]

"B: To provide a lower energy pathway between reactants and products",

"C: To decrease the rate of a chemical reaction",

"D: To change the overall reaction itself"
],
"answer": "B"
}
Passage: {passage}

#### MCQ Answering Prompt:

Given the contexts or evidences: {contexts}

Here is a multiple-choice question:

Question: {question}

Options: A. {options_0} B. {options_1} C. {options_2} D. {options_3}

Please select the correct answer by choosing A, B, C, or D. Respond with only the letter of your choice.

Figure 8: The prompts for generating and answering MCQ questions for evaluating knowledge retention in knowledge graph.