prompt = '''You are an expert in knowledge distillation and skill abstraction.  
Given a conversation between a user and an AI assistant that together solve a problem,  
your job is to summarize this conversation into a structured skill representation  
that can be inserted into a hierarchical skill tree (SkillTree).

Follow these rules:
1. Identify the *main task* being solved.
2. Extract the *goal*, *key substeps*, and *important knowledge or decisions* used.
3. Describe each step concisely, keeping cause-effect relations clear.
4. Avoid mentioning the dialogue itself or the assistant; describe the task in general form.
5. Output in the following format:

Task: [one sentence summary of the task]
Goal: [what is achieved]
Steps:
1. ...
2. ...
3. ...
Knowledge Used:
- ...
Key Insight:
- ...
'''