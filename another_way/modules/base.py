

class chatAgent:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_response(self, prompt):
        response = self.pipeline(prompt)
        return response[0]['generated_text'][len(prompt):].strip()
    
    