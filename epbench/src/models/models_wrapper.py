import requests
import json

class ModelsWrapper:
    def __init__(self, model_name = "gpt-4o-mini-2024-07-18", config = {}):           
        assert model_name is not None, f"model_name is required, got: {model_name}"
        self.model_name = model_name
        if ("gpt-4o" in model_name) or ("o1" in model_name) or ("o3" in model_name):
            from openai import OpenAI
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        elif ("deepseek" in model_name):
            from openai import OpenAI
            self.client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
            # also prepare openrouter for longer context (deepseek currently limited at 65k)
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            self.client = None # instead using the OpenRouter API directly
            self.key = config.OPENROUTER_API_KEY
        elif "claude-3" in model_name:
            from anthropic import Anthropic, DefaultHttpxClient
            self.client = Anthropic(
                api_key=config.ANTHROPIC_API_KEY,
                http_client=DefaultHttpxClient(
                    proxies=config.PROXY['http'],
                    verify=False
                ),
            )
        elif "gemini" in model_name:
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            from google import genai
            self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        elif ("llama" in model_name):
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            self.client = None # instead using the OpenRouter API directly
            self.key = config.OPENROUTER_API_KEY
        else:
            raise ValueError("Wrapper for this model name has not been coded, see ModelsWrapper class")

    def generate(self, user_prompt: str = "Who are you?", system_prompt: str = "You are a content event generator assistant.", 
                 full_outputs = False, max_new_tokens: int = 256, temperature: float = 1.0, keep_reasoning = False):

        reasoning = None

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
        
        elif "o1" in self.model_name:
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
        elif "o3" in self.model_name:
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

        elif "gemini" in self.model_name:
            outputs = self.client.models.generate_content(
                model=self.model_name,
                # system prompt omitted
                contents=user_prompt
            )

            if not full_outputs:
                # note: reasoning steps not available, according to documentation (Feb 2025)
                # "The Flash Thinking model is an experimental model and has the following limitations:
                # Thoughts are only shown in Google AI Studio"
                outputs = outputs.text
        
        elif "llama" in self.model_name:
            outputs = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                data=json.dumps({
                    "model": "meta-llama/" + self.model_name,
                    "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ],
                    "provider": {"order": ["Fireworks"]} # working with large context, contrary to the others
                })
                )
            
            if not full_outputs:
                raw_string = outputs.text
                cleaned_string = raw_string.strip()
                print(cleaned_string)
                parsed_dict = json.loads(cleaned_string)
                if not 'choices' in parsed_dict:
                    print(parsed_dict)
                outputs = parsed_dict['choices'][0]['message']['content']

            ## below (replicate):
            #outputs = self.client.run(
            #    "meta/" + self.model_name, # "meta/llama-2-13b",
            #    input={
            #        #"top_k": 0,
            #        #"top_p": 0.95,
            #        "prompt": user_prompt,
            #        #"max_tokens": 512,
            #        # "temperature": temperature, # take the default temperature
            #        "system_prompt": system_prompt,
            #        #"length_penalty": 1,
            #        "max_new_tokens": max_new_tokens
            #        #"stop_sequences": "<|end_of_text|>,<|eot_id|>",
            #        #"prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            #        #"presence_penalty": 0,
            #        #"log_performance_metrics": False
            #    })
            #
            #if not full_outputs:
            #    outputs = "".join(outputs)
        elif "deepseek" in self.model_name:

            from epbench.src.generation.generate_3_secondary_entities import count_tokens
            nb_tokens_user_prompt = count_tokens(user_prompt)
            using_deepseek_api = True
            if nb_tokens_user_prompt > 60000:
                print(f"The question has length {nb_tokens_user_prompt}; using openrouter instead of deepseek api")
                using_deepseek_api = False

            if using_deepseek_api:
                outputs = self.client.chat.completions.create(
                    model=self.model_name, # deepseek-reasoner # deepseek-chat
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt
                    }
                    ],
                    stream=False
                )
            else:
                if self.model_name == "deepseek-reasoner":
                    model_name_here = "deepseek-r1"
                    print(f"model name changed to {model_name_here}")
                else:
                    model_name_here = self.model_name
                outputs = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"},
                    data=json.dumps({
                        "model": "deepseek/" + model_name_here,
                        "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                        ],
                        "provider": {"order": ["Fireworks"], 
                                    "ignore": ["Avian",  "Novita", "DeepInfra", "Featherless", "DeepSeek", "Kluster", "Nebius", "Together"]},
                        "include_reasoning": True # Include the parameter for downstream processing
                    })
                )

            if not full_outputs:
                if using_deepseek_api:
                    # specific for deepseek, with objects, not list
                    reasoning = outputs.choices[0].message.reasoning_content
                    outputs = outputs.choices[0].message.content
                else:
                    raw_string = outputs.text
                    cleaned_string = raw_string.strip()
                    print(cleaned_string)
                    parsed_dict = json.loads(cleaned_string)
                    if not 'choices' in parsed_dict:
                        print(parsed_dict)
                    outputs = parsed_dict['choices'][0]['message']['content']
                    reasoning = parsed_dict['choices'][0]['message']['reasoning']

        else:
            raise ValueError("there is no generate function for this model name")
        
        if keep_reasoning:
            return outputs, reasoning

        return outputs 
