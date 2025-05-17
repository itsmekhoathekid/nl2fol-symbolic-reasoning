
def extract_predicate_prompt(sentence):
    return f"""You are a logic extraction assistant.  

Extract **complete predicate phrases** (including actions and properties) by these rules:  
1. Remove **ONLY leading auxiliaries/negations**:  
   - Delete: am/is/are/was/were/be/do/does/did + "not"  
   - Keep main verbs (e.g., improve, collaborate, submit)  
2. Preserve full verb phrases including adverbs/prepositions  
3. Split compound phrases with "and"/"or" into separate entries  

Examples:  
- "does not attend class" → "attend class"  
- "improve cultural awareness and adaptability" → "improve cultural awareness", "improve adaptability"  
- "should have submitted reports" → "submitted reports"  

Respond strictly in this format:  

| Predicate                    |  
|------------------------------|  
| ...                          |  

### Sentence:  
{sentence}"""


def MAKE_CONCLUSION_FROM_OPTION_QUESTION():
    return '''
        ### Instruction:
        Combine the question and the provided option into a fully **factual statement** that fully reflects the meaning of all clause in question and option. Your goal is to rephrase the question and each option into a **factual statement** that represents the assumption being evaluated for option. The **factual statement** must be logically consistent with the content of the original question and the specific option. You are given:
            - **question**: The question might be include clause or entity which related to option.
            - **option**: The option that might be the answer of the problem, might be contain entity of the question but written as alias (he, she, it, they, them, etc.), or a clause that is related to the clause in the question.

        ### Examples:
        Question: "Which of the following is the capital of France?"
        Options: "Berlin"
        Factual statement: Berlin is the capital of France
        -----

        Question: "Which option that Helly like"
        Options: "Doing homework and not playing sport"
        Factual statement: Helly like doing homework and not playing sport
        -----

        Question: "If a men go to school, then what he like to ride"
        Options: "a motorbike but not a bicycle"
        Factual statement: If a men go to school, then what he like to ride a motorbike but not a bicycle
        -----

        ### Output Requirements:
        Factual statement note:
            - Include all parts of the the option, not singular part.
            - Factual statement must include **all** parts of the option.
            - Factual statement must be **a direct declarative restatement** of the core meaning of the original question for the corresponding option.
            
        
        Recheck:
            - Check if factual statement include **all** part of the option.
            - Check if logical meaning is correct according to the question and option.


        There is some note about the output that must to follow:
            - **Do not** give explanation.
            - **Do not** include any extra information, such as reasoning steps or explanations.
            - Only give the correct conclusion based on provided question and option.


        ### Output:
            - "Factual Statement": <the correct conclusion based on the option question>
    '''