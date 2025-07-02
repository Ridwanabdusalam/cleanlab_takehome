"""Prompt templates for self-reflection certainty."""

# From Figure 6b in the BSDetector paper
REFLECTION_PROMPTS = [
    """Question: {question}
Proposed Answer: {answer}
Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template: 
explanation: [insert analysis], answer: [choose one letter from among choices A through C]""",
    
    """Question: {question}
Proposed Answer: {answer}
Are you really sure the proposed answer is correct? 
Choose again: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template: 
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""
]
