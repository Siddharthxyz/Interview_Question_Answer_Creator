from langchain.prompts import PromptTemplate

PROMPT_QUESTIONS = PromptTemplate.from_template("""
You are an expert at creating questions based on  documentation.
Your goal is to prepare a for exam 
You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare  for their tests.
Make sure not to lose any important information.

QUESTIONS:
""")

REFINE_PROMPT_QUESTIONS = PromptTemplate.from_template("""
The original questions are as follows:
{existing_answer}

We have additional context below:
------------
{text}
------------

Refine the original questions using the new context.
Make sure the questions still cover the key points.

Refined QUESTIONS:
""")
