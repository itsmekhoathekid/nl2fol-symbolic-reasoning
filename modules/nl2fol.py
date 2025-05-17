from .base import chatAgent

class nl_to_fol(chatAgent):
    def __init__(self, pipeline):
        super().__init__(pipeline)


    def convert(self, list_predicates):
        dic = {}
        
        prompt = self.get_predicate_fol_prompt(list_predicates)
        response = self.get_response(prompt).split('\n')
        for idx, pred in enumerate(list_predicates):
            dic[pred] = pred + ':::' + response[idx].split(':::')[1].strip()

        
        # for idx, pred in enumerate(list_predicates):
        #     dic[pred] = pred + ':::' + response[idx].split(':::')[1].strip()
        return dic

    def get_predicate_fol_prompt(self, predicate_nl):
        PROMPT_LP = """<s>[INST]
            ### Task: For each given **natural language predicate**, generate a corresponding **First-Order Logic (FOL) predicate** that faithfully captures its logical meaning.

            You are given:
            - A **premise** that provides the contextual background or knowledge base from which the logical predicates are derived.
            - Each sentence describes a logical relationship or property involving one or more entities.
            - Each definition is a **complete English sentence** and clearly specifies the **roles of all arguments** (e.g., who performs an action, who receives it).

            #### Your Goal:
            For every natural language predicate, produce a well-formed **FOL predicate expression** that:
            - Accurately reflects the **logical meaning** conveyed by the sentence.
            - Identifies the **correct number of arguments** (e.g., x, y, z) based on the entities involved.
            - Creates a **clear and self-explanatory predicate name** (in English) that summarizes the core relationship or property described.
            - Assigns each argument to its correct role (e.g., subject, object, action recipient).
            - Uses **logical arguments (e.g., x, y, z)** to represent the entities involved.
            - Reflects the **correct argument structure** (e.g., number of arguments, their logical roles).
            - Uses logical notation: `PredicateName(x)`, `PredicateName(x, y)`, etc.

            ### Instructions:
            1. **Analyze each natural language predicate carefully** to determine:
                - The number of **entities** involved (i.e., how many arguments the predicate should take).
                - The **core relationship or property** being described (used to form the predicate name).
                - The **roles** of the arguments (e.g., agent, theme, location, object).
                - If the predicate involves a **fixed object like a degree, title, award, job, etc.**, represent it as a **constant**, not part of the predicate name.

            2. **Rules for constructing the FOL predicate**:
                - The **predicate name** must summarize the core action/relation/property using meaningful English terms.
                - The number of arguments in the predicate must match the number of entities involved.
                - Maintain the **semantic roles** of the arguments
                - Avoid copying the sentence structure directly — instead, generate a compact and expressive predicate name.
                - Use consistent naming style: either `CamelCase` or `snake_case`.
                - **Do NOT merely paraphrase or copy** the sentence — you must **analyze** and convert it into a **logical representation**.

            3. **Output Format**:
                - Each output must be exactly **one line per definition**.
                - Each line must follow this format:
                    Natural Language Predicate ::: FOL-Predicate
                - Use **exactly three colons** (` ::: `) as the separator.
                - Example:
                    + is a Teacher ::: Teacher(x)
                    + do test on y subject ::: DoTestSubject(x, y)

            4. **Final Requirements**:
                - You must define **every single natural language predicate** provided in the list.
                - Each FOL predicate must be **logically sound, syntactically correct, and match the right number of arguments and roles**.
                - **Do not skip any natural language predicates**.
                - No extra commentary or explanation — only the list of converted predicates.
                - The number of output lines must **exactly match** the number of input definitions.

            ### Input:
            - List of Natural Language Predicate: {predicate_nl}
            [/INST]
            Output: </s>"""
        return PROMPT_LP.format(predicate_nl=predicate_nl)

    def construct_logic_program(self, lps):
        logic_program = [lp for lp in lps]
        return logic_program

    def premise_to_fol_prompt(self, logic_program, nl_premise, subject):
        prompt = f"""
                [INST]
        You are a formal-logic expert.
        Your task: convert each natural-language statement (Premise-NL)
        into one first-order-logic formula (Premise-FOL) in *logic-program style*.

        ╭─ OUTPUT RULES (mandatory) ─────────────────────────────────────────╮
        │ • Return **one line per premise**, nothing else.                  │
        │ • Line = pure FOL: no bullet numbers, no “:::”, no code-fences.   │
        │ • Keep antecedent ⇒ consequent order exactly as in NL.            │
        │ • Allowed symbols:  ∀  ∃ / Exists  →  ∧  ∨  ¬                    │
        ╰────────────────────────────────────────────────────────────────────╯

        GENERAL CONVENTIONS  
        1. **Predicates** (section ❶) are declared as  
            <Predicate>(<vars>) ::: <NL description>  
        → *Never* insert constants there; always use a variable (x, y…).  

        2. **Arity guide** — when to choose 1-place vs 2-place  
        • `P(x)`  for an intrinsic property or an action toward a fixed object.  
        • `R(x,y)` for relations where y can vary / be queried later.  
        • If unsure, prefer 2-place; you can always constrain later.  

        3. **Subject handling**  
        • *Class name* (Student, FacultyMember…): add that predicate in the
            antecedent → `FacultyMember(x) → …`.  
        • *Generic word* (person, someone, individual…): omit any `Person(x)`
            unless NL states it.

        4. **Named-individual rule**  
        • If NL refers to a **specific proper name** (e.g. “Professor John”),
            treat it as a **constant**  `John`.  
        • Do **not** quantify over a constant; simply use it in the formula:  
            `CompletedTraining(John) → CanTeachUndergrad(John)`  

        5. Named-individual fact  
        • If a clause simply asserts **one property** of a specific, named
            individual (John, Course101…), output **exactly one
            predicate(constant)**—no quantifier, no arrow, no condition.  
            NL: “Professor John holds a PhD.”  
            FOL: HoldsPhD(John)

        WORKFLOW FOR EACH SENTENCE  
        S1  Detect quantifier pattern: universal, existential, none (named individual).  
        S2  Map NL phrases to declared predicates; apply arity & subject rules.  
        S3  Build the formula, adding ∀x / ∃x only when the sentence is about
            unspecified individuals.  
        S4  Output the single-line formula.

        [/INST]
        [INPUT]
        "premises_nl": {nl_premise}
        "subject": {subject}
        "logic_program": {logic_program}
        [/INPUT]
        [OUTPUT]

            """
        return prompt

    # def convert_premise_to_fol(self, premise_nl, premise_nl_pred, dic_predicates, premise_nl_subject):
    #     # Construct the prompt
    #     # premise nl pred = [pred 1, pred2, ...]

        

    #     logic_program = [ dic_predicates[predicate] for predicate in premise_nl_pred ]
    #     subject = premise_nl_subject[premise_nl]
    #     logic_program = self.construct_logic_program(logic_program)

    #     prompt = self.premise_to_fol_prompt(logic_program, premise_nl, subject)
    #     # Get the response
    #     response = self.get_response(prompt)
    #     return response

    def convert_premise_to_fol(self, premise_nl_list, premise_nl_pred, dic_predicates, premise_nl_subject):
        # print(premise_nl_pred)
        # print premise_nl_pred type
        # print("DEBUG type premise_nl_pred:", type(premise_nl_pred))
        # print("DEBUG example keys:", premise_nl_pred[:5] if isinstance(premise_nl_pred, list) else list(premise_nl_pred.keys()))
       

        prompt_list = []

        for premise_nl in premise_nl_list:
            premise_pred  = premise_nl_pred[premise_nl]
            logic_program = [ dic_predicates[predicate] for predicate in premise_pred ]
            subject = premise_nl_subject[premise_nl]
            logic_program = self.construct_logic_program(logic_program)

            prompt = self.premise_to_fol_prompt(logic_program, premise_nl, subject)
            prompt_list.append(prompt)
        
        responses = self.pipeline(prompt_list, batch_size = 8 )

        res = {}
        for idx, response in enumerate(responses):
            res[premise_nl_list[idx]] = response[0]["generated_text"].split('[OUTPUT]')[-1].strip()
            # responses[idx] = response[0]["generated_text"].split('[OUTPUT]')[-1].strip()

        # # Get the response
        # response = self.get_response(prompt)
        
        return res


    
