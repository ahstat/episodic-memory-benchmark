class ModelsWrapper:
    def __init__(self, model_name = "gpt-4o-mini-2024-07-18", config = {}):           
        assert model_name is not None, f"model_name is required, got: {model_name}"
        self.model_name = model_name
        if ("gpt-4o" in model_name) or ("o1-" in model_name):
            from openai import OpenAI
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        elif "claude-3" in model_name:
            from anthropic import Anthropic, DefaultHttpxClient
            self.client = Anthropic(
                api_key=config.ANTHROPIC_API_KEY,
                http_client=DefaultHttpxClient(
                    proxies=config.PROXY['http'],
                    verify=False
                ),
            )
        else:
            raise ValueError("Wrapper for this model name has not been coded, see ModelsWrapper class")

    def generate(self, user_prompt: str = "Who are you?", system_prompt: str = "You are a content event generator assistant.", 
                 full_outputs = False, max_new_tokens: int = 256, temperature: float = 1.0):

        if "gpt-4o" in self.model_name:            
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt
                }
                ],
                max_tokens = max_new_tokens,
                temperature = temperature
            )

            if not full_outputs:
                outputs = outputs.choices[0].message.content
        
        elif "o1-" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": user_prompt
                }
                ]
            )
            # max_completion_tokens = max_new_tokens,
            # temperature = temperature

            if not full_outputs:
                outputs = outputs.choices[0].message.content
                #print(outputs)
 
        elif "claude-3" in self.model_name:
            outputs = self.client.messages.create(
                model=self.model_name,
                system = system_prompt, # different syntax compared to openai
                messages=[
                    {"role": "user", "content": user_prompt
                }
                ],
                max_tokens = max_new_tokens,
                temperature = temperature
            )

            if not full_outputs:
                outputs = outputs.content[0].text

        else:
            raise ValueError("there is no generate function for this model name")

        return outputs 
