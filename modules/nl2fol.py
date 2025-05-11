from .base import chatAgent

class nl_to_fol(chatAgent):
    def __init__(self, pipeline):
        super().__init__(pipeline)
    

    def convert(self, list_predicates):
        dic = {}
        for predicate in list_predicates:
            convert_prompt = self.get_predicate_fol_prompt(predicate)
            response = self.get_response(convert_prompt)
            dic[predicate] = response

        return dic

    def get_predicate_fol_prompt(self, predicate):
        prompt = f"""You are a symbolic logic assistant.

        Your task is to convert the following natural language predicate into its equivalent First-Order Logic (FOL) expression.

        Strict instructions:
        - Return only **one line**.
        - Format exactly as: [natural language predicate] ::: [FOL expression]
        - Do NOT add any explanation, prefix (like "Solution:"), or newline.
        - Do NOT include any additional words.

        Guidelines:
        1. Use variable `x` as the subject.
        2. Use variable `y` only if the predicate involves two entities.
        3. Use concise snake_case for predicate names.

        Examples:
        is eligible for graduation ::: eligible(x)  
        is the advisor of ::: advisor_of(x, y)  
        receives a scholarship from ::: receives_scholarship_from(x, y)  
        is enrolled in course ::: enrolled_in(x, y)

        Now convert the following:

        ## Predicate:
        {predicate.strip()}
        ## FOL Expression:

        """
        return prompt

    def construct_logic_program(self, lps):
        logic_program = [lp for lp in lps]
        return logic_program

    def premise_to_fol_prompt(self, logic_program, nl_premise, subject):
        return f"""
        ### Task: Convert the given *natural language premise* into a complete *First-Order Logic (FOL) expression* using only the provided predicate definitions.

        You are given:
        - A *natural language premise sentence* that either states a general rule (e.g., about "a student") or a specific fact (e.g., about "John").
        - A *list of logic program predicates*, where each predicate is paired with a natural language definition.
        - A *subject* variable which tells you who or what the premise is about.

        ### Your Goal:
        Write exactly **one complete FOL expression** that represents the meaning of the input premise using the correct predicate(s) and argument(s).

        ### Rules:
        1. Only use predicate names from the Logic Program list. **Do not invent new predicates.**
        2. Use the predicate name exactly as written in the logic program.
        3. Replace arguments as follows:
        - Use variable `x` if the premise is a general rule about an unspecified subject.
        - Use the given *subject* (e.g., John) if the premise is a specific fact.
        4. If multiple predicates are involved, combine them using logical connectives such as `∧`, `∨`, `→`.
        5. Your output must be a **single valid FOL expression** on one line.

        ### Input:
        - Natural Language Premise: {nl_premise}
        - Subject: {subject}
        - Logic Program: {logic_program}

        ### Output:
        """

    def convert_premise_to_fol(self, premise_nl, premise_nl_pred, dic_predicates, premise_nl_subject):
        # Construct the prompt
        # premise nl pred = [pred 1, pred2, ...]

        

        logic_program = [ dic_predicates[predicate] for predicate in premise_nl_pred ]
        subject = premise_nl_subject[premise_nl]
        logic_program = self.construct_logic_program(logic_program)

        prompt = self.premise_to_fol_prompt(logic_program, premise_nl, subject)
        # Get the response
        response = self.get_response(prompt)
        return response
    
