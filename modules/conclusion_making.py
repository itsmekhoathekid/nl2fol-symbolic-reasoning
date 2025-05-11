
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains.llm import LLMChain
from icecream import ic
import re
from langchain_community.llms import HuggingFacePipeline

class Prompt():
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        self.prompt_template = None


    def create_prompt_template(self):
        self.prompt_template = PromptTemplate(
            input_variables=self.input_variables,
            template=self.template,
        )
    
    def create_fewshot_template(self, examples: list, suffix="", prefix=""):
        self.create_prompt_template()
        self.prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.prompt_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=[],
        )
    
    def get_prompt(self, input_keys_values: dict):
        return self.prompt_template.format(**input_keys_values)

class ChatAgent():
    def __init__(self, model, config):
        self.config = config
        self.llm_model = HuggingFacePipeline(pipeline=model)


    def batch_inference(self, prompt, questions):
        """
            - Variables depend on the input_variables of your llama_prompt
            llama_prompt = PromptTemplate(
                input_variables=["q_question", "q_premises"],
                template=prompt,
            )
            
            questions = [{
                "q_question": q_question,
                "q_premises": q_premises,
            }, ....]
        """
        llama_prompt = PromptTemplate(
            input_variables=list(questions[0].keys()),
            template=prompt,
        )

        qa_chains = LLMChain(
            llm=self.llm_model,
            prompt=llama_prompt,
        )

        qa_chains.batch_inference(questions, return_source_documents=False)


    def inference(self, prompt, input_values: dict):
        """
            - Variables depend on the input_variables of your llama_prompt
            llama_prompt = PromptTemplate(
                input_variables=["q_question", "q_premises"],
                template=prompt,
            )
            
            input_values = {
                "q_question": q_question,
                "q_premises": q_premises,
            }
        """
        llama_prompt = PromptTemplate(
            input_variables=list(input_values.keys()),
            template=prompt,
        )

        prompt_text = llama_prompt.format(**input_values)
        # ic(prompt_text)
        ic(len(prompt_text.split()))

        qa_chains = LLMChain(
            llm=self.llm_model,
            prompt=llama_prompt,
        )
        results = qa_chains.invoke(input_values, return_source_documents=False,)
        return results
    
    def inference_direct(self, prompt):
        prompt_template = PromptTemplate.from_template(prompt)
        qa_chains = LLMChain(
            llm=self.llm_model,
            prompt=prompt_template,
        )
        results = qa_chains.invoke({}, return_source_documents=False,)
        return results

    def make_prompt(self):
        NotImplemented

class ChatAgentMakeConclusion(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, question, option, INSTRUCTION_PROMPT):
        # PROMPT TEMPLATE
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question} [/INST]
        """

        # FINAL PROMPT
        # INSTRUCTION_PROMPT = MAKE_CONCLUSION_FROM_OPTION_QUESTION_WITH_REG_DETAIL()
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt', 'user_question']
        )
        final_prompt_obj.create_prompt_template()

        # final_prompt = final_prompt_obj.get_prompt({
        #     'instruct_prompt': INSTRUCTION_PROMPT, 
        #     'user_question': f"{pair_predicate_premise_prompt}\n{pair_predicate_question_prompt}",
        # })

        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': INSTRUCTION_PROMPT, 
            'user_question': f'Question: {question} \nOption: {option}',
        })
        return final_prompt


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

def parse_factual_statement(response):
    text = response.split("<</SYS>>")[-1]
    pattern_factual_statement = r"Factual Statement: .*$"

    match_statement = re.findall(pattern_factual_statement, text, re.MULTILINE)
    statement = match_statement[0].replace("Factual Statement:", "").strip()
    return statement


def make_conclusion(model, question, config):
    print("Start Making Conclusion")
    chat_agent_make_conclusion = ChatAgentMakeConclusion(model, config)  

    def parse_options(text):
        parts = re.split(r'\n(?=[A-D]\.)', text.strip())
        parts = [re.sub(r'^[A-D]\.\s*', '', opt.strip()) for opt in parts]
        question = parts[0]
        options = parts[1:]
        return question, options

    options = []
    # Check multiple choice
    if len([True for option in ["\nA.", "\nB.", "\nC.", "\nD."] if option in question]) >= 2:
        question, options = parse_options(question)
    
    if len(options) == 0:
        print(question)
        raise Exception("There is not Multiple Choice Question")


    # Input question
    new_options = []
    option_labels = ["\n1 ", "\n1 ", "\n1 ", "\n1 "]
    for label, option in zip(option_labels, options):
        conclusion_prompt = chat_agent_make_conclusion.make_prompt(
            question=question.strip(),
            option=option.strip(),
            INSTRUCTION_PROMPT=MAKE_CONCLUSION_FROM_OPTION_QUESTION(),
        )

        # ic(conclusion_prompt)
        make_conclusion_results = chat_agent_make_conclusion.inference(
            prompt=conclusion_prompt,
            input_values={},
        )
        response = make_conclusion_results['text']
        factual_statement = parse_factual_statement(response)
        new_options.append(f"{label} {factual_statement}")
    return " ".join(new_options)


PROMPT_CREATE_HYPOTHESIS_ANOTHER = """<s>[INST]
### Task: 
Your task is to transform this question into a single declarative sentence called a hypothesis, which expresses the core meaning or assumption of the question in statement form.

You are given:
- A Natural language Question: It may be in the form of WH-questions (e.g., What, Why, How, When, Where), Yes/No/Uncertain question, or other interrogative forms. The question typically contains a focus or target of inquiry and can often be rephrased into a statement (hypothesis) to be evaluated based on relevant information.
- This question typically seeks information or clarification and may be rephrased into a hypothesis to be evaluated as true, false, or uncertain given certain context (premises).

### What is a Hypothesis?
A hypothesis is a **single declarative sentence** that expresses what would be true **if the answer to the question is "Yes"**. It must:
- Be **grammatically complete**.
- Be **logically faithful** to the question's meaning.
- Be **evaluatable** as true, false, or uncertain given external context or premises (not included in this prompt).
- **Avoid including phrases** like “according to the premises”, “based on the context”, “as per the passage”, etc., as these are implied.

- Example for Yes/No question type and its hypothesis:
  + Question: "Does Sophia qualify for the university scholarship, according to the premises?"
  + Hypothesis: Sophia qualify for the university scholarship.

### Output Requirements:
- The output **must start with** `Hypothesis:` followed by exactly **one declarative sentence**.
- The hypothesis must:
  - Be **grammatically complete**.
  - Be a **faithful and concise restatement** of the question’s intent.
  - Be evaluable as **true**, **false**, or **uncertain** based on supporting context.
- Do **not** return a question.
- Do **not** include explanations, reasoning, or invented content.
- Do **not** use markdown, bullet points, or formatting symbols.
- The output must be a **single line** only.

### Input:
Natural language question: {question_NL}
[/INST]
Output: </s>"""

import re

class Extract_Hypothesis:
    def __init__(self, mistral_pipeline):
        self.mistral = mistral_pipeline

    def generate_hypothesis(self, question):
        # Nếu có sẵn câu giả thuyết trong chuỗi
        _match = re.search(r"Statement:\s*['\"]?([^'\"]+)['\"]?", question)
        if _match:
            hypothesis = _match.group(1)
        else:
            # Tạo prompt đầu vào
            prompt = PROMPT_CREATE_HYPOTHESIS_ANOTHER.format(question_NL=question)
            
            # Sinh câu trả lời từ pipeline
            output = self.mistral(prompt, return_full_text=False)[0]['generated_text']
            
            # Tách phần "Hypothesis: ..."
            match1 = re.search(r"Hypothesis:\s*(.*)", output)
            if match1:
                hypothesis = match1.group(1).strip()
            else:
                hypothesis = output.strip()

        return hypothesis