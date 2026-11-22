**Tool Used:** Cursor AI (AI-powered coding assistant integrated into the development environment), using its available language models.

Cursor AI was used throughout the assignment as a learning, coding, debugging, and documentation assistant. The assignment was approached in multiple stages rather than asking AI to generate the complete solution in one step. I first used AI to understand the assignment requirements and theoretical concepts, followed by discussions about possible approaches, implementation assistance, numerical verification, debugging, and report preparation.

## 1. Understanding the Assignment and Requirements

I started by providing the assignment PDF to Cursor AI and asking it to explain the overall structure of the assignment, its objectives, and the requirements of each question.

**Example prompts:**

- "Read the complete assignment PDF and explain what each question is asking me to do. Start with the overall objective and then explain each question and its subparts in simple language."
- "Explain the difference between spatial modelling and temporal modelling in the context of this assignment."
- "Break down the requirements of Q1 to Q7 and identify which parts require mathematical derivations, coding, experiments, plots, and written explanations."
- "Help me create a checklist so I can track the assignment requirements and avoid missing any subpart."

**AI assistance:** Cursor AI helped interpret the assignment instructions, organise the requirements, and establish a step-by-step implementation plan.

## 2. Understanding Theoretical Concepts and Exploring Approaches

Before implementing the models, I used Cursor AI to understand the mathematical and machine learning concepts involved. I asked follow-up questions to clarify unfamiliar concepts and compare alternative approaches.

**Example prompts:**

- "Explain how message aggregation works in Graph Neural Networks using a simple example."
- "What is the difference between Sum, Mean, and Max aggregation? Explain when each one can preserve information."
- "Explain adjacency matrices, one-hop and two-hop neighbours, and graph convolution with a small numerical example."
- "What is over-smoothing in GCNs? Why can adding more graph convolution layers become problematic?"
- "Explain fixed adjacency, adaptive adjacency, and hybrid adjacency. What are their differences?"
- "Why do we need both temporal and spatial information for traffic forecasting?"

**AI assistance:** Cursor AI provided explanations, mathematical intuition, examples, and comparisons to help me understand the concepts before applying them.

## 3. Mathematical Derivations and Numerical Verification

I used AI to discuss the mathematical questions, understand the derivations, and generate small numerical experiments to verify the results.

**Example prompts:**

- "Walk me through the derivation step by step and explain why the aggregation produces the same output for these two neighbourhoods."
- "Can you give me a small numerical example to understand how the sum uniquely represents different neighbour counts?"
- "Help me verify this derivation using NumPy rather than assuming the expected answer is correct."
- "Explain the relationship between the previous and current snapshots and how temporal information helps identify the direction of movement."
- "Generate a small test case and explain what each line of the verification code is doing."

**AI assistance:** Cursor AI assisted with mathematical explanations and NumPy verification scripts. I reviewed the calculations and used numerical checks to validate the conclusions.

## 4. Dataset Exploration and Preprocessing

For the LargeST San Diego dataset, I used Cursor AI to understand the dataset structure, explore the available files, and develop the preprocessing pipeline.

**Example prompts:**

- "Help me understand the structure of the LargeST dataset and what information is available in its metadata and historical traffic files."
- "How should I identify the 716 San Diego sensors and ensure that their ordering matches the adjacency matrix?"
- "Explain whether SUM or MEAN aggregation is appropriate when converting five-minute traffic observations into fifteen-minute intervals."
- "Help me identify missing values, suspicious zero values, and flat-line sensors in the dataset."
- "Explain how chronological training, validation, and testing splits should be implemented without data leakage."

**AI assistance:** Cursor AI helped explore the dataset, develop preprocessing code, investigate data quality, and verify sensor ordering and temporal splits.

## 5. Model Development and Implementation

I used Cursor AI within the development environment to discuss model architecture, implement individual components, understand code, and resolve implementation issues.

**Example prompts:**

- "Help me implement a lightweight temporal encoder using Conv1d. Explain the expected tensor dimensions at each stage."
- "How can I implement graph convolution using a normalised adjacency matrix and PyTorch?"
- "Help me understand how the temporal encoder, spatial GCN, and forecasting head work together."
- "Show me how to implement adaptive adjacency using trainable embedding matrices without relying on a predefined PyG adaptive graph layer."
- "Help me implement the hybrid adjacency approach by combining fixed and learned adjacency matrices."
- "Review this implementation and identify any tensor dimension mismatches or potential implementation errors."

**AI assistance:** Cursor AI provided implementation suggestions, code examples, debugging assistance, and explanations of the model components. The generated implementations were tested and adjusted during development.

## 6. Experiments, Debugging, and Performance Evaluation

During experimentation, I used AI to investigate errors, understand unexpected results, and improve the experimental workflow.

**Example prompts:**

- "The validation performance is not improving. Help me inspect the normalisation strategy and identify possible sources of the problem."
- "Review this training loop for potential data leakage or incorrect handling of the validation and test sets."
- "Help me implement early stopping and save the best-performing model checkpoint."
- "Explain how to calculate MAE and RMSE after reversing the normalisation."
- "Help me compare the temporal-only, fixed graph, adaptive graph, and hybrid graph models using the same experimental configuration."
- "The results do not match our initial expectations. Analyse the actual measurements and suggest possible explanations without changing or fabricating the results."

**AI assistance:** Cursor AI helped debug implementation issues, review experimental configurations, generate evaluation scripts, and interpret measured results.

## 7. Visualisation and Result Analysis

I also used AI assistance to generate and refine visualisations and understand the patterns observed in the experimental results.

**Example prompts:**

- "Help me generate a heatmap of the adjacency matrix with readable labels and an appropriate colour scale."
- "Create a plot showing the actual versus predicted traffic flow for selected sensors."
- "Help me visualise the learned adaptive adjacency matrix and compare it with the original road-network graph."
- "Explain what the learned graph tells us and how to interpret its strongest connections."
- "Help me calculate the overlap between the strongest learned edges and the physical road-network edges."

**AI assistance:** Cursor AI assisted with plotting code, quantitative comparisons, and interpretation of the resulting visualisations.

## 8. Report Preparation and Final Review

After completing the experiments, I used Cursor AI to organise the findings, prepare technical explanations, and review the report against the assignment requirements.

**Example prompts:**

- "Help me organise the results of Q1 to Q7 into a structured technical report."
- "Convert these actual experimental results into clear explanations while preserving the original values."
- "Review this section for mathematical correctness and consistency with the computed results."
- "Check whether every sub-question in the assignment has been addressed in the report."
- "Identify missing figures, unclear explanations, or incomplete requirements."
- "Improve the readability of the report while keeping the technical meaning and results unchanged."

**AI assistance:** Cursor AI helped structure the report, improve explanations, prepare tables and figures, and perform a final requirements review.

## 9. Student Involvement and Verification

AI was used as an interactive assistant throughout the development process, not as a replacement for understanding and reviewing the assignment. I guided the workflow, selected the implementation approaches, provided the development environment and dataset access, reviewed the generated code and explanations, and examined the experimental results.

I also requested numerical verification of mathematical claims and reviewed unexpected results instead of accepting initial AI-generated assumptions. Several initial explanations and implementation choices were revised after examining the actual measurements and identifying inconsistencies.

The final report and implementation were reviewed against the assignment requirements, and the GitHub repository was maintained separately by me. Cursor AI was not instructed to push code or perform GitHub operations.
