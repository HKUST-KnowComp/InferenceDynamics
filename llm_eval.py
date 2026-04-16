from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import argparse
import pandas as pd
import os
import re
from tqdm import tqdm
import time
import json
from openai import AzureOpenAI, OpenAI
import openai
import torch
import requests
from together import Together
from zhipuai import ZhipuAI
import tiktoken
import shortuuid
# from mistralai import Mistral
from utils import _is_number, _normalize_answer, compute_price, extract_numbers_from_string, computeChrF, PATH, MODEL_FROM_API, MODELS_FROM_OPENROUTER, MODELS_WITH_REASONING
import random
import traceback
import evaluate
from thirdpartyeval.instruction_following_eval import evaluation_lib
from thirdpartyeval.natural_plan import evaluate_meeting_planning, evaluate_trip_planning, evaluate_calendar_scheduling
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
OPENAI_KEY = ""
SF_KEY = ""
HS_KEY = ""
XAI_KEY = ""
MISTRAL_KEY = ''
ALI_KEY = ''
SAMBANOVA_KEY = ''
LINGYI_KEY = ''
ZHIPU_KEY = ""
STEP_KEY = ""
TOGETHER_KEY = ""
TAOBAO_KEY = ''
OPENROUTER_KEY = ''
HUNYUAN_KEY = ""





def promptLLMWithAPI(prompt, max_new_tokens, client, model_id, temperature=1.0):
    stream = False
    if "Qwen/Qwen2.5-72B-Instruct"!=model_id and model_id not in MODELS_FROM_OPENROUTER:
        model_id = model_id.split('/')[1]

    if not stream and model_id not in MODELS_WITH_REASONING:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    else:
        reasoning_content = ""
        answer_content = ""
        is_answering = False
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_id,
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                reasoning_content += delta.reasoning_content
            else:
                if delta.content != "" and is_answering is False:
                    is_answering = True
                answer_content += delta.content
        return answer_content

def promptLRMWithAPI(prompt, max_new_tokens, client, model_id, exclude_reasoning=True):
    encoding = tiktoken.get_encoding('cl100k_base')
    if 'grok' in model_id:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_id.split('/')[1],
            max_tokens=max_new_tokens+8000+len(encoding.encode(prompt)),
            reasoning_effort="high"
        )
        return response.choices[0].message.content, len(encoding.encode(response.choices[0].message.content+response.choices[0].message.reasoning_content))
    elif 'gemini' in model_id:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "reasoning": {
                "max_tokens": 2048
            },
            "max_tokens": max_new_tokens+2048+len(encoding.encode(prompt))
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # print(response.json())
        return response.json()['choices'][0]['message']['content'], response.json()['usage']['completion_tokens']
    elif model_id == 'deepseek/deepseek-r1':
        url = "https://openrouter.ai/api/v1/chat/completions"
        provider = {
            'sort': 'throughput',
            'require_parameters': True,
            'ignore': ['Lambda', 'Featherless', 'SambaNova', 'Cent-ML', 'Fireworks', \
                       'Friendli', 'Together', 'Parasail', 'Azure']
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "reasoning": {
                "max_tokens": 2048
            },
            "max_tokens": max_new_tokens+8000+len(encoding.encode(prompt)),
            "response_format": {"type": "json_object"},
            "provider": provider
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # print(response.json())
        return response.json()['choices'][0]['message']['content'], response.json()['usage']['completion_tokens']
    elif model_id == 'deepseek-ai/DeepSeek-R1':
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": "Pro/deepseek-ai/DeepSeek-R1",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_new_tokens+8000+len(encoding.encode(prompt)),
            "response_format": {"type": "text"},
        }
        headers = {
            "Authorization": f"Bearer {SF_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        return response.json()['choices'][0]['message']['content'], response.json()['usage']['completion_tokens']
    elif model_id == 'deepseek/deepseek-r1-250120':
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_id.split('/')[1],
            max_tokens=max_new_tokens+8000+len(encoding.encode(prompt))
        )
        return response.choices[0].message.content, response.usage.completion_tokens
    elif 'zhipu' in model_id:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_id.split('/')[1],
            max_tokens=max_new_tokens+12000+len(encoding.encode(prompt))
        )
        return response.choices[0].message.content, response.usage.completion_tokens
    else: 
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_id,
            max_tokens=max_new_tokens+8000+len(encoding.encode(prompt))
        )
        # print(response)
        return response.choices[0].message.content, response.usage.completion_tokens

class InstanceReader():
    def __init__(self, filepath, tokenizer, shot=1, test_mode=False, enable_cot = False, pass_k = 1):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.df = pd.read_json(filepath, lines=True)
        self.test_mode = test_mode
        if self.test_mode:
            self.df = self.df.head(10)
        self.shot = shot
        self.cot = enable_cot
        self.k = pass_k

class MCQAInstanceReader(InstanceReader):
    def __init__(self, filepath, tokenizer, shot=1, test_mode=False, enable_cot = False, pass_k = 1):
        super().__init__(filepath, tokenizer, shot, test_mode, enable_cot, pass_k)
        if self.cot:
            self.prompt = "Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: \"Answer: <your answer>\".  You must always give an answer at the end.  You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."
        else:
            self.prompt = "\n\nOnly output the correct choice such as 'A', 'B', 'C' without any explanation. Full answer not needed. \nAnswer:"
            # self.prompt = "Only output the choice in the following format: \"Answer: <your answer>\" You must always give an answer. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."
        self.label = []

    def compute_accuracy(self, predictions):
        def remove_special_characters(word):
            # replace all the special characters with ""
            word = re.sub(r'[^a-zA-Z0-9]', '', word)
            return word
        result = []
        for i in range(len(predictions)):
            if f"answer{self.label[i].lower()}" in remove_special_characters(predictions[i].lower()):
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)

class GSM8KInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        if self.test_mode:
            self.df = self.df.sample(n=10)
        for i in range(len(self.df)):
            if self.cot:
                prompt = 'Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final integer that solves the problem.\n\nProblem: '
            else:
                prompt = 'Solve the following math problem efficiently and clearly. Output the final answer as: $\\boxed{answer}$. \n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: '
            # if self.shot == 1:
            #     prompt = prompt+'**Problem**: '+self.df['question'][0]+'\n**Solution**: '+self.df['answer'][0]+'\n**Answer**: '+self.df['answer'][0].split('####')[1].strip().replace(',', '')+'\n\nGive the final numerical answer at the end **WITH OUT ANY EXPLANATION**.\n**Problem**: '+self.df['question'][i]+'\n**Answer**: '
            #     self.data.append(prompt)
            prompt += self.df['question'][i]
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            prediction = predictions[i].split('\\boxed{')[-1].strip().replace(' ', '').replace(',','').lower()
            answer = self.df['answer'][i].split('####')[1].strip().replace(',', '').replace(' ', '').lower()
            if answer in prediction:
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)
    
class MATHInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = 'Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\nWhere [answer] is just the final number or expression that solves the problem.\nProblem: '+self.df['problem'][i]
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        result = []
        left = '\\boxed{'
        for i in range(len(predictions)):
            prediction = predictions[i].split(left)[-1].strip().replace(' ', '').lower()
            ground_truth = self.df['solution'][i].split(left)[1].split('}')[0].replace(' ', '').lower()
            if ground_truth in prediction:
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)

class FinQAInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            if self.cot:
                prompt = 'Solve the following financial problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: \\boxed{answer}. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem. Keep the answer to five decimal places if it is a number, and do not use percentages; keep the decimal format.\nProblem: '+self.df['question'][i]
            else:
                prompt = 'Solve the following financial problem efficiently and clearly. Output the final answer as: \\boxed{answer}. \n\nWhere [answer] is just the final number or expression that solves the problem. Keep the answer to five decimal places if it is a number, and do not use percentages; keep the decimal format.\nProblem: '+self.df['question'][i]
                # prompt = "Given the financial problem, please output your final answer within \\boxed{}.\nProblem: "+self.df[0][i]['question']
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            prediction = predictions[i].split('\\boxed{')[-1].strip()
            number = extract_numbers_from_string(prediction)
            if type(self.df['answer'][i]) == str:
                if self.df['answer'][i] in prediction:
                    result.append(1.0)
                else:
                    result.append(0.0)
            elif len(number)!=0:
                potential_answer = [float(number[0]), float(number[0])/100, float(number[0])*100, float(number[0])*1000, float(number[0])/1000]
                found = False
                for answer in potential_answer:
                    if (abs(answer - float(self.df['answer'][i])) < 0.001):
                        result.append(1.0)
                        found = True
                        break
                if not found:
                    result.append(0.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)

class ARCInstanceReader(MCQAInstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = ""
            prompt += self.df['question'][i]['stem']
            for j in range(len(self.df['question'][i]['choices'])):
                prompt += '\n'+chr(65+j)+'. '+self.df['question'][i]['choices'][j]['text']
            prompt += '\n'+self.prompt
            # prompt += "\nOnly output 'A', 'B', 'C', or 'D' without any explanation. Full answer not needed. \nAnswer:"
            self.data.append(prompt)
            self.label.append(self.df['answerKey'][i])
        return self.data
    
    # def compute_accuracy(self, predictions):
    #     result = []
    #     for i in range(len(predictions)):
    #         predicion = predictions[i].split('Answer:')[-1].strip()
    #         if predicion == "":
    #             result.append(0.0)
    #         elif predicion[0] == self.df['answerKey'][i]:
    #             result.append(1.0)
    #         else:
    #             result.append(0.0)
    #     return result, sum(result)/len(predictions)


class SIQAInstanceReader(MCQAInstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        answer = {1: 'A', 2: 'B', 3: 'C'}
        for i in range(len(self.df)):
            prompt = self.df['context'][i]+self.df['question'][i]
            prompt += '\nA. '+self.df['answerA'][i]+'\nB. '+self.df['answerB'][i]+'\nC. '+self.df['answerC'][i]
            prompt += "\n"+self.prompt
            self.data.append(prompt)
            self.label.append(answer[int(self.df['label'][i])])
        return self.data
    
    # def compute_accuracy(self, predictions):
    #     result = []
    #     answer = {1: 'A', 2: 'B', 3: 'C'}
    #     for i in range(len(predictions)):
    #         prediction = predictions[i].split('Answer:')[-1].strip()
    #         if prediction == "":
    #             result.append(0.0)
    #         elif prediction[0] == answer[int(self.df['label'][i])]:
    #             result.append(1.0)
    #         else:
    #             result.append(0.0)
    #     return result, sum(result)/len(predictions)

class LogicGameBQAInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['context'][i]+self.df['question'][i]
            prompt += "\nOnly answer 'yes' or 'no' without any explanation. Full answer not needed. \nAnswer:"
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            if predictions[i].split('Answer:')[-1].strip()[:3].lower() == self.df['answer'][i].lower() or predictions[i].split('Answer:')[-1].strip()[:2].lower() == self.df['answer'][i].lower():
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)
        
class RuleTakerInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        for i in range(len(self.df)):
            prompt = self.df['context'][i]+"\nDoes the context above entail the following statement?\n"+self.df['question'][i]
            prompt += "\nOnly output 'yes' or 'no' without any explanation. Full answer not needed. \nAnswer:"
            self.data.append(prompt)
            self.label.append('yes') if self.df['label'][i] == "entailment" else self.label.append('no')
        return self.data

    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            if predictions[i].split('Answer:')[-1].strip()[:3].lower() == self.label[i] or predictions[i].split('Answer:')[-1].strip()[:2].lower() == self.label[i]:
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)
    
class MMLUInstanceReader(MCQAInstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        for i in range(len(self.df)):
            prompt = self.df[0][i]
            prompt +='\nA. '+self.df[1][i]+'\nB. '+self.df[2][i]+'\nC. '+self.df[3][i]+'\nD. '+self.df[4][i]
            prompt += "\n"+self.prompt
            self.label.append(self.df[5][i])
            self.data.append(prompt)
        return self.data
    
    # def compute_accuracy(self, predictions):
    #     result = []
    #     for i in range(len(predictions)):
    #         prediction = predictions[i].split('Answer:')[-1].strip()
    #         if prediction == "":  
    #             result.append(0.0)
    #         elif prediction[0] == self.df[5][i]:
    #             result.append(1.0)
    #         else:
    #             result.append(0.0)
    #     return result, sum(result)/len(predictions)

class PubMedQAInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = " ".join(self.df['context'][i])+"\n"+self.df['question'][i]
            prompt += "\nOnly answer 'yes' or 'no' without any explanation. Full answer not needed. \nAnswer:"
            self.data.append(prompt)
        return self.data
    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            if predictions[i].split('Answer:')[-1].strip()[:3].lower() == self.df['answer'][i].lower() or predictions[i].split('Answer:')[-1].strip()[:2].lower() == self.df['answer'][i].lower():
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)

class MedQAInstanceReader(MCQAInstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        for i in range(len(self.df)):
            prompt = self.df['question'][i]
            for j in range(len(self.df['options'][i])):
                prompt += '\n'+chr(65+j)+'. '+self.df['options'][i][chr(65+j)]
            prompt += "\n"+self.prompt
            self.data.append(prompt)
            self.label.append(self.df['answer_idx'][i])
        return self.data
    
    # def compute_accuracy(self, predictions):
    #     result = []
    #     for i in range(len(predictions)):
    #         prediction = predictions[i].split('Answer:')[-1].strip()
    #         if prediction == "":
    #             result.append(0.0)
    #         elif prediction[0] == self.df['answer_idx'][i]:
    #             result.append(1.0)
    #         else:
    #             result.append(0.0)
    #     return result, sum(result)/len(predictions)

class LegalBenchInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['prompt'][i]
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            if self.df['label'][i].strip().lower() in predictions[i].split(self.data[i])[-1].strip().lower():
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)

class ScienceQAInstanceReader(MCQAInstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        for i in range(len(self.df)):
            prompt = self.df['hint'][i]+self.df['question'][i]
            for j in range(len(self.df['choices'][i])):
                prompt += '\n'+chr(65+j)+'. '+self.df['choices'][i][j]
            prompt += "\n"+self.prompt
            self.data.append(prompt)
            self.label.append(answer[self.df['answer'][i]])
        return self.data
    
    # def compute_accuracy(self, predictions):
    #     result = []
    #     choice_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    #     for i in range(len(predictions)):
    #         prediction = predictions[i].split('Answer:')[1].strip()
    #         if prediction == "":
    #             result.append(0.0)
    #         elif prediction[0] == choice_dict[self.df['answer'][i]]:
    #             result.append(1.0)
    #         else:
    #             result.append(0.0)
    #     return result, sum(result)/len(predictions)

class GPQAInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        for i in range(len(self.df)):
            prompt = self.df['Question'][i]
            incorrect_options = [self.df['Incorrect Answer 1'][i], self.df['Incorrect Answer 2'][i], self.df['Incorrect Answer 3'][i]]
            correct_answer_added = False
            for j in range(4):
                if chr(65+j) != self.df['label'][i] and not correct_answer_added:
                    prompt += '\n'+chr(65+j)+'. '+incorrect_options[j]
                elif correct_answer_added:
                    prompt += '\n'+chr(65+j)+'. '+incorrect_options[j-1]
                else:
                    prompt += '\n'+chr(65+j)+'. '+self.df['Correct Answer'][i]
                    correct_answer_added = True
                
            prompt += "\n"+"Think step by step before giving your final answer to the question. To think step-by-step, state the facts or premises you are using along with their deductions that yield the correct answer (even if those facts or premises are commonsense knowledge).  When you are ready to answer write the answer in the format: \"Answer: <your answer>\".  Failure to follow the answer format will result in no credit. You must always give an answer at the end.  You may only pick one answer choice. You must pick an answer and that answer must be one of the multiple choice options.  Let\'s think step by step."
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            # find the label of correct answer
            correct_answer = self.df['Correct Answer'][i]
            input_question = predictions[i].split("Think step by step before giving your final answer")[0]
            if input_question.find("A. "+correct_answer) != -1:
                correct_answer = 'A'
            elif input_question.find("B. "+correct_answer) != -1:
                correct_answer = 'B'
            elif input_question.find("C. "+correct_answer) != -1:
                correct_answer = 'C'
            elif input_question.find("D. "+correct_answer) != -1:
                correct_answer = 'D'
            prediction = predictions[i]
            if f"answer {correct_answer.lower()}" in _normalize_answer(prediction.lower()):
                result.append(1.0)
            else:
                result.append(0.0)
        return result, sum(result)/len(predictions)
    
class BIGBenchInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        for i in range(len(self.df)):
            prompt, target = self.setup_ex(self.df['input'][i], self.df['target'][i], self.df['task'][i])
            self.data.append(prompt)
            self.label.append(target)
        return self.data
    
    def setup_ex(self, input, target, task):
        question = None
        choices = None

        if task == 'boolean_expressions':
            choices = {'text': ['True', 'False'], 'label': ['A', 'B']}
            question = input
            if target == 'True':
                target = 'A'
            else:
                target = 'B'
        elif task == 'causal_judgement' or task == 'navigate' or task == 'sports_understanding' or task=='web_of_lies':
            choices = {'text': ['Yes', 'No'], 'label': ['A', 'B']}
            question = input.split("Options:")[0].strip()
            if target == 'Yes':
                target = 'A'
            else:
                target = 'B'
        elif task == 'formal_fallacies':
            choices = {'text': ['valid', 'invalid'], 'label': ['A', 'B']}
            question = input.split("Options:")[0].strip()
            if target == 'valid':
                target = 'A'
            else:
                target = 'B'
        elif task in [
            'date_understanding',
            'disambiguation_qa',
            'geometric_shapes',
            'hyperbaton',
            'logical_deduction_five_objects',
            'logical_deduction_seven_objects',
            'logical_deduction_three_objects',
            'movie_recommendation',
            'penguins_in_a_table',
            'reasoning_about_colored_objects',
            'ruin_names',
            'salient_translation_error_detection',
            'snarks',
            'temporal_sequences',
            'tracking_shuffled_objects_five_objects',
            'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_three_objects',
        ]:
            raw_question = input
            question, choices = raw_question.split('Options:')
            question = question.strip()
            choice_texts = [x.split(')')[1].strip() for x in choices.split('(') if x.strip() != '']
            choices = {'text': choice_texts, 'label': [chr(65 + i) for i in range(len(choice_texts))]}
            target = target.replace('(','').replace(')','')
        elif task == 'dyck_languages' or task == 'multistep_arithmetic_two' or task == 'object_counting' or task=='word_sorting':
            question = input
            target = target
            choices = None
        
        if choices is not None:
            choice_texts = ''
            for i in range(len(choices['text'])):
                choice_texts += f"{choices['label'][i]}. {choices['text'][i]}\t"
            prompt = question + "\n"
            prompt += choice_texts + '\n'
            prompt += f"Think step by step before giving your final answer to the question.  When you are ready to answer write the answer in the format: \"Answer: <your answer>\". You must always give an answer at the end. You may only pick one answer choice, if you think multiple are correct only pick the one you think is best."
            return prompt, target
        else:
            prompt = f'Question: {question}\n\nAnswer the question above using this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nAlways conclude with:\nThe best answer is <your answer>.\nwhere the <your answer> is a short answer response to the question\n\nLet\'s think step by step.'
            return prompt, target
        
    def compute_accuracy(self, predictions):
        result = []
        for i in range(len(predictions)):
            if self.df['task'][i] in ['dyck_languages', 'multistep_arithmetic_two', 'object_counting', 'word_sorting']:
                answer = predictions[i].split("The best answer is")[-1].strip()
                if self.label[i].lower() in answer.lower():
                    result.append(1.0)
                else:
                    result.append(0.0)
            else:
                prediction = predictions[i].split('Answer:')[-1].strip()
                if prediction[0].lower() == self.label[i].lower():
                    result.append(1.0)
                else:
                    result.append(0.0)
        return result, sum(result)/len(predictions)


class NaturalPlanMeetingPlanningInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            if self.shot == 0:
                prompt = self.df['prompt_0shot'][i]+'Directly output the plan without any explanation.\nSOULUTION:'
            else:
                prompt = self.df['prompt_5shot'][i]+'Directly output the plan without any explanation.\nSOULUTION:'
            self.data.append(prompt)
        return self.data

    def compute_accuracy(self, predictions):
        responses = []
        for i in range(len(predictions)):
            responses.append(predictions[i].split(self.data[i])[-1].strip())
        output_list, acc_for_all = evaluate_meeting_planning.compute_score(self.df['num_people'].tolist(), self.df['constraints'].tolist(), self.df['dist_matrix'].tolist(), responses, self.df['golden_plan'].tolist())
        return output_list, acc_for_all

class NaturalPlanTripPlanningInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            if self.shot == 0:
                prompt = self.df['prompt_0shot'][i]+'Directly output the plan without any explanation.\nSOULUTION:'
            else:
                prompt = self.df['prompt_5shot'][i]+'Directly output the plan without any explanation.\nSOULUTION:'
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        responses = []
        for i in range(len(predictions)):
            responses.append(predictions[i].split(self.data[i])[-1].strip())
        output_list, acc_for_all = evaluate_trip_planning.compute_score(self.df['cities'].tolist(), self.df['durations'].tolist(), responses)
        return output_list, acc_for_all

class NaturalPlanCalendarSchedulingInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            if self.shot == 0:
                prompt = self.df['prompt_0shot'][i]+'Directly output the plan without any explanation.\nSOULUTION:'
            else:
                prompt = self.df['prompt_5shot'][i]+'Directly output the plan without any explanation.\nSOULUTION:'
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        responses = []
        for i in range(len(predictions)):
            responses.append(predictions[i].split(self.data[i])[-1].strip())
        output_list, acc_for_all = evaluate_calendar_scheduling.compute_solve_rate(responses, self.df['golden_plan'].tolist())
        return output_list, acc_for_all

class SciTLDRInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = 'Given the abstract, introduction and conclusion of a paper, summarize the paper in 1-2 short sentences with 40 words.\n'
            # prompt = 'Given part of a scientific paper, produce a concise, one to two sentence summary within 30 wordsthat clearly and accurately captures the primary contribution, finding, or purpose of the paper in a neutral and accessible manner, avoiding technical jargon where possible and focusing on the core insight or result.\n'
            prompt += ' '.join(self.df['source'][i])
            prompt += '\nSummary:'
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        results = []
        rogue1 = []
        rogue_metrics = evaluate.load('rouge')
        for i in range(len(predictions)):
            result = rogue_metrics.compute(predictions=[predictions[i].split(self.data[i])[-1].strip()], references=[self.df['target'][i]])
            rogue1.append(result['rouge1'])
            results.append(result)
        return results, sum(rogue1)/len(rogue1)
    
class XSumInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = "Given a news article, generate a 1-2 sentence summary within 40 words.\n"
            # prompt = 'Given a news article, generate a concise, one-sentence summary that captures the main point of the article in a clear and neutral manner. Ensure the summary is abstract, avoiding direct quotes or overly specific details, and focuses on the core event or finding of the article.\n'
            prompt += 'Article: '+self.df['document'][i]+'\n'
            prompt += 'Summary:'
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        results = []
        rogue1 = []
        rogue_metrics = evaluate.load('rouge')
        for i in range(len(predictions)):
            result = rogue_metrics.compute(predictions=[predictions[i].split(self.data[i])[-1].strip()], references=[self.df['summary'][i]])
            rogue1.append(result['rouge1'])
            results.append(result)
        return results, sum(rogue1)/len(rogue1)
    
class BigCodeBenchInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['complete_prompt'][i]
            self.data.append(prompt)
        return self.data

class BigGenBenchInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['system_prompt'][i]+'\n\n'+self.df['input'][i]
            self.data.append(prompt)
        return self.data

class LiveBenchInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['turns'][i][0]
            self.data.append(prompt)
        return self.data


class PlanBenchInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['query'][i]+'Directly output the plan without any explanation.\nPLAN:'
            self.data.append(prompt)
        return self.data
    
class HiToMInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        for i in range(len(self.df)):
            label = self.df['answer'][i]
            prompt = "Read the following story and answer the multiple-choice question. Think step-by-step. Provide the answer in the exact format: \"Answer: <your answer>\", and then explain it.\n"
            prompt += self.df['story'][i]
            prompt += 'Question: '+self.df['question'][i]+'\n'
            prompt += 'Choices' + self.df['choices'][i]+'\n\n'+'Note: You should assume the following. (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B\'s mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn\'t affect his actual belief. An agent tend to trust a agent that exited the room later than himself. The exit order is known to all agents. (4) Agents in private communications know that others won\'t hear them, but they know that anyone can hear any public claims.'
            self.data.append(prompt)
            self.label.append(self.df['choices'][i][self.df['choices'][i].find(label)-3])
        return self.data
    
    def compute_accuracy(self, predictions):
        results = []
        for i in range(len(predictions)):
            prediction = predictions[i].split(self.data[i])[-1].strip()
            # print(self.df['answer'][i])
            # print(self.label[i])
            # print(_normalize_answer(prediction.lower()))
            if f"answer {self.label[i].lower()}" in _normalize_answer(prediction.lower()):
                results.append(1.0)
            else:
                results.append(0.0)
            # if i > 10:
            #     break
        return results, sum(results)/len(results)
    
class RACEInstanceReader(MCQAInstanceReader):
    def to_input(self):
        self.data = []
        self.label = []
        for i in range(len(self.df)):
            prompt = self.df['article'][i]
            prompt += 'Question: '+self.df['question'][i]+'\n'
            for j in range(4):
                prompt += chr(65+j)+'. '+self.df['options'][i][j]+'\n'
            prompt += self.prompt
            self.data.append(prompt)
            self.label.append(self.df['answer'][i])
        return self.data
    
class FloresInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = "Translate the following text from English to {}\n".format(self.df['language'][i])
            prompt += self.df['input'][i]
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        results = []
        chrf = evaluate.load('chrf')
        for i in range(len(predictions)):
            prediction = predictions[i].split(self.data[i])[-1].strip()
            result = chrf.compute(predictions=[prediction], references=[[self.df['target'][i]]], word_order=2)
            results.append(result['score'])
        return results, sum(results)/len(results)
    
class MMMLUInstanceReader(InstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['Question'][i]
            for j in range(4):
                prompt += '\n'+chr(65+j)+'. '+self.df[chr(65+j)][i]
            prompt += self.df['prompt'][i]
            self.data.append(prompt)
        return self.data
    
    def compute_accuracy(self, predictions):
        results = []
        for i in range(len(predictions)):
            prediction = predictions[i].split(self.data[i])[-1].strip()
            if prediction == '':
                results.append(0.0)
            elif prediction[0].lower() == self.df['Answer'][i].lower():
                results.append(1.0)
            else:
                results.append(0.0)
        return results, sum(results)/len(results)
    
class MMLUProInstanceReader(MCQAInstanceReader):
    def to_input(self):
        self.data = []
        for i in range(len(self.df)):
            prompt = self.df['question'][i]
            for j in range(len(self.df['options'][i])):
                prompt += '\n'+chr(65+j)+'. '+self.df['options'][i][j]
            prompt += self.prompt
            self.data.append(prompt)
            self.label.append(self.df['answer'][i])
        return self.data
            
INSTANCE_READER = {
        'SciTLDR': SciTLDRInstanceReader,
        'XSum': XSumInstanceReader,
        'PlanBench': PlanBenchInstanceReader,
        'HiToM': HiToMInstanceReader,
        'RACE': RACEInstanceReader,
        'MMLU': MMLUInstanceReader,
        'BigCodeBench': BigCodeBenchInstanceReader,
        'GSM8K': GSM8KInstanceReader,
        'MATH': MATHInstanceReader,
        'FinQA': FinQAInstanceReader,
        'LegalBench': LegalBenchInstanceReader,
        'ScienceQA': ScienceQAInstanceReader,
        'NaturalPlanMeetingPlanning': NaturalPlanMeetingPlanningInstanceReader,
        'NaturalPlanTripPlanning': NaturalPlanTripPlanningInstanceReader,
        'NaturalPlanCalendarScheduling': NaturalPlanCalendarSchedulingInstanceReader,
        'RuleTaker': RuleTakerInstanceReader,
        'MedQA': MedQAInstanceReader,
        'PubMedQA': PubMedQAInstanceReader,
        'ARC': ARCInstanceReader,
        'GPQA': GPQAInstanceReader,
        'BIGBench': BIGBenchInstanceReader,
        'BigGenBench': BigGenBenchInstanceReader,
        'LiveBench': LiveBenchInstanceReader,
        'Flores': FloresInstanceReader,
        'MMMLU': MMMLUInstanceReader,
        'MMLUPro': MMLUProInstanceReader
}

def test_model(model, tokenizer, args, dataset:InstanceReader, task:str, test_mode=False, max_new_tokens=128, tempature=1.0):
    prediction = []
    idx = []
    used_time = []
    price_list = []
    i=0
    if task == 'HumanEval' or task == 'MBPP':
        completion = []
    # Create the client
    if args.model_id not in MODEL_FROM_API and args.model_id not in MODELS_FROM_OPENROUTER:
        client = None
    elif args.model_id in MODELS_FROM_OPENROUTER:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY,
            )
    elif "Qwen/Qwen2.5-72B-Instruct"==args.model_id or args.model_id == 'deepseek-ai/DeepSeek-R1':
        client = OpenAI(api_key=SF_KEY, base_url="https://api.siliconflow.cn/v1")
    elif args.model_id == 'deepseek/deepseek-r1-250120':
        print('deepseek-testing')
        client = OpenAI(api_key=HS_KEY, base_url="https://ark.cn-beijing.volces.com/api/v3")
    elif "grok" in args.model_id:
        client = OpenAI(api_key=XAI_KEY, base_url="https://api.x.ai/v1")
    elif 'qwen' in args.model_id or 'qwq' in args.model_id:
        client = OpenAI(api_key=ALI_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    else:
        client = None

    for data in tqdm(dataset.to_input()):
        for temp in range(dataset.k):
            if client is not None:
                retry_attempt = 5
                retry_num = 0
                generation_success = False
                while retry_num < retry_attempt and not generation_success:
                    try:
                        start = time.time_ns()
                        # if "Qwen/Qwen2.5-72B-Instruct"==args.model_id or args.model_id in MODELS_FROM_OPENROUTER:
                        #     output = promptLLMWithAPI(data, max_new_tokens, client, args.model_id)
                        # elif "qwq" in args.model_id:
                        #     output = promptLLMWithAPI(data, max_new_tokens, client, args.model_id.split('/')[1], stream=True)
                        # else:
                        #     output = promptLLMWithAPI(data, max_new_tokens, client, args.model_id.split('/')[1])
                        if args.model_id in MODELS_WITH_REASONING:
                            output, tokens = promptLRMWithAPI(data, max_new_tokens, client, args.model_id)
                            price = compute_price(data, tokens, args.model_id)
                        else:
                            encoding = tiktoken.get_encoding('cl100k_base')
                            output = promptLLMWithAPI(data, max_new_tokens, client, args.model_id)
                            price = compute_price(data, len(encoding.encode(output)), args.model_id)
                        end = time.time_ns()
                        used_time.append(end-start)
                        prediction.append(data+output)
                        price_list.append(price)
                        if task == 'HumanEval' or task == 'MBPP':
                            completion.append(output)
                        generation_success = True
                        break
                    except openai.APIError as e:  # TRIGGERED OpenAI API ERROR, Could be network issue
                        print(e)
                        retry_num += 1
                        generation_success = False
                        time.sleep(10)
                    except openai.RateLimitError as e:  # TRIGGERED OpenAI RATE LIMIT    
                        print(e)
                        retry_num += 1
                        generation_success = False
                        time.sleep(60)
                    except openai.BadRequestError as e:  # TRIGGERED OpenAI CONTENT SAFETY FILTER
                        print(e)
                        retry_num += 1
                        generation_success = False
                        # time.sleep(1)
                    except openai.OpenAIError as e:
                        print(e)
                        retry_num += 1
                        generation_success = False
                        time.sleep(10)
                    except openai.APIConnectionError as e:
                        print(e)
                        retry_num += 1
                        generation_success = False
                        time.sleep(10)
                    except openai.AuthenticationError as e:
                        print(e)
                        retry_num += 1
                        generation_success = False
                        time.sleep(10) 
                    except Exception as e:
                        traceback.print_exc()
                        retry_num += 1
                        generation_success = False
                        time.sleep(10)
                    except:
                        retry_num += 1
                        generation_success = False
                        time.sleep(10)
                if not generation_success:
                    prediction.append(data+'[ERROR]')
                    price_list.append(-1)
                    used_time.append(-1)
            elif 'chatglm' in args.model_id:
                start = time.time_ns()
                response, _ = model.chat(tokenizer, data, history=[], max_length=1024, max_new_tokens=max_new_tokens)
                end = time.time_ns()
                used_time.append(end-start)
                prediction.append(response)
            else:
                inputs = tokenizer(data, return_tensors='pt', truncation=False, max_length=1048)
                inputs.to(args.device)
                start = time.time_ns()
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                end = time.time_ns()
                used_time.append(end-start)
                prediction.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
                if task == 'HumanEval' or task == 'MBPP':
                    completion.append(tokenizer.decode(outputs[0], skip_special_tokens=True).split(data)[1])
            if task == 'HumanEval' or task == 'MBPP':
                idx.append(dataset.df['task_id'][i])
            elif 'LiveBench' in task:
                idx.append(shortuuid.uuid())
            else:
                idx.append(i)
        i+=1
    if task!='IFEval' and task!='PlanBench' and task!='BigCodeBench' and task!='BigGenBench' and 'LiveBench' not in task and 'TableBench' != task and 'Flores' != task:
        df = pd.DataFrame(columns = ['idx', 'predictions', 'label', 'used_time'])
        # catch all runtime error generated by the compute_accuracy function
        try:
            label, acc = dataset.compute_accuracy(prediction)
            df['predictions'] = prediction
            df['label'] = label
            df['idx'] = idx
            df['used_time'] = used_time
            df['price'] = price_list
        except Exception as e:
            print("Error happened in compute_accuracy: ", e)
            traceback.print_exc()
            acc = None
            df['predictions'] = prediction
            df['label'] = None
            df['idx'] = idx
            df['used_time'] = used_time
            df['price'] = price_list
    elif task == 'IFEval':
        df = pd.DataFrame(columns = ['idx', 'predictions', 'label', 'used_time', 'follow_list'])
        try:
            follow_all, follow_list, acc = dataset.compute_accuracy(prediction)
            df['predictions'] = prediction
            df['idx'] = idx
            df['used_time'] = used_time
            df['label'] = follow_all
            df['follow_list'] = follow_list
            df['price'] = price_list
        except Exception as e:
            print(e)
            df['predictions'] = prediction
            df['label'] = None
            df['idx'] = idx
            df['used_time'] = used_time
            df['follow_list'] = None
            df['price'] = price_list
    elif task == 'PlanBench':
        # output_dict = {}
        # output_dict['task'] = 't1'
        # output_dict['engine'] = 'gpt-4_chat'
        # output_dict['prompt_type'] = 'oneshot'
        # output_dict['domain'] = 'blocksworld'
        # output_instance_dict = []
        # for i in range(len(prediction)):
        #     output_instance_dict.append({
        #         'instance_id': dataset.df['instance_id'][i],
        #         'example_instance_id': dataset.df['example_instance_id'][i],
        #         'query': dataset.df['query'][i],
        #         'ground_truth_plan': dataset.df['ground_truth_plan'][i],
        #         'response': prediction[i].split(dataset.data[i])[-1].strip()
        #     })
        # output_dict['instances'] = output_instance_dict
        # path = os.path.join(args.result_dir, task)+'/response'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # print(output_dict)
        # with open(os.path.join(path, args.model_id.split('/')[-1]+'.json'), 'w') as f:
        #     json.dump(output_dict, f)
        df = pd.DataFrame(columns = ['instance_id', 'response','used_time'])
        df['instance_id'] = dataset.df['instance_id']
        temp = []
        for i in range(len(prediction)):
            temp.append(prediction[i].split(dataset.data[i])[-1].strip())
        df['response'] = temp
        df['used_time'] = used_time
        df['price'] = price_list
        path = os.path.join(args.result_dir, task)+'/response'
        if not os.path.exists(path):
            os.makedirs(path)
        path = path+'/'+args.model_id.split('/')[-1]+'.jsonl'
        df.to_json(path, orient='records', lines=True, index=False)
        acc = None
    elif task == 'BigCodeBench':
        df = pd.DataFrame(columns = ['task_id', 'solution','used_time'])
        df['solution'] = prediction
        df['task_id'] = dataset.df['task_id']
        df['used_time'] = used_time
        df['price'] = price_list
        acc = None
    elif task == 'BigGenBench':
        df = pd.DataFrame(columns = ['id', 'solution','used_time', 'price'])
        temp = []
        for i in range(len(prediction)):
            temp.append(prediction[i].split(dataset.data[i])[-1].strip())
        df['solution'] = temp
        df['id'] = dataset.df['id']
        df['used_time'] = used_time
        df['price'] = price_list
        acc = None
    elif 'LiveBench' in task:
        df = pd.DataFrame(columns = ['question_id', 'answer_id', 'solution','used_time', 'price'])
        temp = []
        for i in range(len(prediction)):
            temp.append(prediction[i].split(dataset.data[i])[-1].strip())
        df['solution'] = temp
        df['question_id'] = dataset.df['question_id']
        df['answer_id'] = idx
        df['used_time'] = used_time
        df['price'] = price_list
        acc = None
    elif 'TableBench' == task:
        df = pd.DataFrame(columns = ['id', 'response', 'used_time', 'price'])
        df['id'] = dataset.df['id']
        temp = []
        for i in range(len(prediction)):
            temp.append(prediction[i].split(dataset.data[i])[-1].strip())
        df['response'] = temp
        df['used_time'] = used_time
        df['price'] = price_list
        acc = None
    elif 'Flores' == task:
        df = pd.DataFrame(columns = ['input', 'target','response', 'used_time', 'price'])
        df['input'] = dataset.df['input']
        df['target'] = dataset.df['target']
        temp = []
        for i in range(len(prediction)):
            temp.append(prediction[i].split(dataset.data[i])[-1].strip())
        df['response'] = temp
        df['used_time'] = used_time
        df['price'] = price_list
        acc= None

    if task != 'PlanBench':
        path = os.path.join(args.result_dir, task)
        if not os.path.exists(path):
            os.makedirs(path)
        path = path+'/'+args.model_id.split('/')[-1]+'.jsonl'
        df.to_json(path, orient='records', lines=True, index=False)
        time.sleep(10)
        if task == 'BigCodeBench':
            try:
                import subprocess
                result = subprocess.run(['bigcodebench.sanitize', '--samples', path, '--calibrate'])
                if result.returncode != 0:
                    print(f"Error occurred while sanitizing BigCodeBench samples: {result.stderr}")
            except Exception as e:
                print(f"Error occurred while sanitizing BigCodeBench samples: {e}")
    return acc
def detect_unfinished_response(response, task):
    if response == '':
        return True
    if '[ERROR]' in response:
        return True
    if '<think>' in response and '</think>' not in response:
        return True
    if task == 'BigCodeBench':
        if response.endswith('.'):
            return False
        if 'return' in response:
            return False
        encoding = tiktoken.get_encoding('cl100k_base')
        if len(encoding.encode(response)) < 100:
            return True
    if task in ['SciTLDR', 'XSum']:
        if not response.endswith('.'):
            return True
    if task in ['GSM8K', 'MATH', 'FinQA']:
        if "\\boxed{" not in response:
            return True
    if task in ['BIGBench', 'GPQA', 'HiToM']:
        if  'answer' not in _normalize_answer(response).lower():
            return True
        if task == 'GPQA':
            if 'answer a' not in _normalize_answer(response).lower() and 'answer b' not in _normalize_answer(response).lower() and 'answer c' not in _normalize_answer(response).lower() and 'answer d' not in _normalize_answer(response).lower():
                return True
    return False

def regenration_for_LRM(model, tokenizer, args):
    for dir in sorted(os.listdir(args.result_dir)):
        if dir.endswith(".json") or dir == 'LiveBench' or dir == 'BigCodeBench':
            continue
        else:
            if dir != 'GPQA':
                continue
            if dir == 'PlanBench':
                path = args.result_dir+'/'+dir+'/response/'+args.model_id.split('/')[1]+'.jsonl'
            else:
                path = args.result_dir+'/'+dir+'/'+args.model_id.split('/')[1]+'.jsonl'
            if dir == 'Flores':
                continue
            if not os.path.exists(path):
                continue
            df = pd.read_json(path, lines=True)
            if dir == 'NaturalPlanMeetingPlanning':
                task_path = 'NaturalPlan'
                filepath = os.path.join(args.data_root, PATH[task_path], 'meeting_planning.jsonl')
            elif dir == 'NaturalPlanTripPlanning':
                task_path = 'NaturalPlan'
                filepath = os.path.join(args.data_root, PATH[task_path], 'trip_planning.jsonl')
            elif dir == 'NaturalPlanCalendarScheduling':
                task_path = 'NaturalPlan'
                filepath = os.path.join(args.data_root, PATH[task_path], 'calendar_scheduling.jsonl')
            else:
                task_path = dir
                filepath = os.path.join(args.data_root, PATH[task_path], 'data.jsonl')
                
                
            print(filepath)
            instance_reader = INSTANCE_READER[dir](filepath=filepath, tokenizer=tokenizer, test_mode=args.test_mode)
            prompt = instance_reader.to_input()
            # check whether the df of instance_reader has the column 'response'
            if 'price' not in instance_reader.df.columns:
                price_computed = False
            else:
                price_computed = True
            prediction = []
            idx = []
            used_time = []
            price_list = []
            i=0
            for i in tqdm(range(len(df))):
                if dir == 'PlanBench':
                    model_response = df['response'][i]
                elif dir == 'BigCodeBench':
                    model_response = df['solution'][i].split(prompt[i])[-1].strip()
                elif dir == 'BigGenBench':
                    model_response = df['solution'][i]
                else:
                    model_response = df['predictions'][i].split(prompt[i])[-1].strip()
                if detect_unfinished_response(model_response, dir):
                    # regenerate the response
                    if args.model_id not in MODEL_FROM_API and args.model_id not in MODELS_FROM_OPENROUTER:
                        client = None
                    elif args.model_id in MODELS_FROM_OPENROUTER:
                        client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=OPENROUTER_KEY,
                            )
                    elif 'zhiyun' in args.model_id:
                        client = OpenAI(
                            base_url="https://api.zhiyunai168.com/v1",
                            api_key=TAOBAO_KEY
                            )
                    elif "Qwen/Qwen2.5-72B-Instruct"==args.model_id or args.model_id == 'deepseek-ai/DeepSeek-R1':
                        client = OpenAI(api_key=SF_KEY, base_url="https://api.siliconflow.cn/v1")
                    elif args.model_id == 'deepseek/deepseek-r1-250120':
                        print('deepseek-testing')
                        client = OpenAI(api_key=HS_KEY, base_url="https://ark.cn-beijing.volces.com/api/v3")
                    elif "grok" in args.model_id:
                        client = OpenAI(api_key=XAI_KEY, base_url="https://api.x.ai/v1")
                    elif 'qwen' in args.model_id or 'qwq' in args.model_id:
                        client = OpenAI(api_key=ALI_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
                    else:
                        client = None
                    retry_attempt = 5
                    retry_num = 0
                    generation_success = False
                    while retry_num < retry_attempt and not generation_success:
                        try:
                            start = time.time_ns()
                            # if "Qwen/Qwen2.5-72B-Instruct"==args.model_id or args.model_id in MODELS_FROM_OPENROUTER:
                            #     output = promptLLMWithAPI(data, max_new_tokens, client, args.model_id)
                            # elif "qwq" in args.model_id:
                            #     output = promptLLMWithAPI(data, max_new_tokens, client, args.model_id.split('/')[1], stream=True)
                            # else:
                            #     output = promptLLMWithAPI(data, max_new_tokens, client, args.model_id.split('/')[1])
                            if args.model_id in MODELS_WITH_REASONING:
                                output, tokens = promptLRMWithAPI(prompt[i], 8000, client, args.model_id)
                                price = compute_price(prompt[i], tokens, args.model_id)
                            else:
                                encoding = tiktoken.get_encoding('cl100k_base')
                                output = promptLLMWithAPI(prompt[i], 8000, client, args.model_id)
                                price = compute_price(prompt[i], len(encoding.encode(output)), args.model_id)
                            end = time.time_ns()
                            used_time.append(end-start)
                            prediction.append(prompt[i]+output)
                            price_list.append(price)
                            generation_success = True
                            break
                        except openai.APIError as e:  # TRIGGERED OpenAI API ERROR, Could be network issue
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except openai.RateLimitError as e:  # TRIGGERED OpenAI RATE LIMIT    
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(60)
                        except openai.BadRequestError as e:  # TRIGGERED OpenAI CONTENT SAFETY FILTER
                            print(e)
                            retry_num += 1
                            generation_success = False
                            # time.sleep(1)
                        except openai.OpenAIError as e:
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except openai.APIConnectionError as e:
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except openai.AuthenticationError as e:
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10) 
                        except Exception as e:
                            traceback.print_exc()
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except:
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                    if not generation_success:
                        prediction.append(prompt[i]+'[ERROR]')
                        price_list.append(-1)
                        used_time.append(-1)
                    idx.append(i)
                else:
                    if dir != 'BigCodeBench' and dir != 'PlanBench':
                        prediction.append(prompt[i]+model_response)
                    else:
                        prediction.append(model_response)
                    if price_computed:
                        price_list.append(df['price'][i])
                    else:
                        price_list.append(-1)
                    used_time.append(df['used_time'][i])
                    idx.append(i)
            if dir != 'BigCodeBench' and dir != 'PlanBench':
                df['predictions'] = prediction
            elif dir == 'BigCodeBench':
                df['solution'] = prediction
            elif dir == 'PlanBench':
                df['response'] = prediction
            if price_computed:
                df['price'] = price_list
            df['used_time'] = used_time
            df.to_json(path, orient='records', lines=True, index=False)

def load_result_from_file(args, tokenizer):
    final_result = {}
    for dir in sorted(os.listdir(args.result_dir)):
        if dir.endswith(".json") or dir == "PlanBench" or dir == "BigCodeBench" or dir == "BigGenBench" or dir == "LiveBench" or dir == 'TableBench' or dir == 'Flores':
            continue
        elif dir == 'GPQA':
            path = args.result_dir+'/'+dir+'/'+args.model_id.split('/')[1]+'.jsonl'
            if not os.path.exists(path):
                continue
            df = pd.read_json(path, lines=True)
            if 'glm-z1' in args.model_id:
                predictions = df['predictions'].tolist()
                for i in range(len(predictions)):
                    predictions[i] = predictions[i].split('<think>')[0]+predictions[i].split('</think>')[-1]
                df['predictions'] = predictions
            instance_reader = INSTANCE_READER[dir](filepath=os.path.join(args.data_root, PATH[dir], 'data.jsonl'), tokenizer=tokenizer, test_mode=args.test_mode)
            acc_list, acc = instance_reader.compute_accuracy(df['predictions'].tolist())
            final_result[dir] = acc
            df['label'] = acc_list
            df.to_json(path, orient='records', lines=True, index=False)
        else:
            if dir != 'MMLUPro':
                continue
            print('testing '+dir)
            if dir == 'PlanBench':
                path = args.result_dir+'/'+dir+'/response/'+args.model_id.split('/')[1]+'.jsonl'
            else:
                path = args.result_dir+'/'+dir+'/'+args.model_id.split('/')[1]+'.jsonl'
            if not os.path.exists(path):
                continue
            df = pd.read_json(path, lines=True)
            if dir == 'NaturalPlanMeetingPlanning':
                task_path = 'NaturalPlan'
                filepath = os.path.join(args.data_root, PATH[task_path], 'meeting_planning.jsonl')
            elif dir == 'NaturalPlanTripPlanning':
                task_path = 'NaturalPlan'
                filepath = os.path.join(args.data_root, PATH[task_path], 'trip_planning.jsonl')
            elif dir == 'NaturalPlanCalendarScheduling':
                task_path = 'NaturalPlan'
                filepath = os.path.join(args.data_root, PATH[task_path], 'calendar_scheduling.jsonl')
            else:
                task_path = dir
                filepath = os.path.join(args.data_root, PATH[task_path], 'data.jsonl')

            print(filepath)
            instance_reader = INSTANCE_READER[dir](filepath=filepath, tokenizer=tokenizer, test_mode=args.test_mode)
            prompts = instance_reader.to_input()
            encoding = tiktoken.get_encoding('cl100k_base')
            if dir != 'IFEval' and dir != 'Flores':
                if 'glm-z1' in args.model_id:
                    predictions = df['predictions'].tolist()
                    for i in range(len(predictions)):
                        predictions[i] = predictions[i].split('<think>')[0]+predictions[i].split('</think>')[-1]
                    df['predictions'] = predictions
                acc_list, acc = instance_reader.compute_accuracy(df['predictions'].tolist())
                final_result[dir] = acc
                df['label'] = acc_list
            elif dir == 'Flores':
                if 'glm-z1' in args.model_id:
                    predictions = df['response'].tolist()
                    for i in range(len(predictions)):
                        predictions[i] = predictions[i].split('<think>')[0]+predictions[i].split('</think>')[-1]
                    df['response'] = predictions
                acc_list, acc = instance_reader.compute_accuracy(df['response'].tolist())
                final_result[dir] = acc
                df['label'] = acc_list
            else:
                if 'glm-z1' in args.model_id:
                    predictions = df['predictions'].tolist()
                    for i in range(len(predictions)):
                        predictions[i] = prompts[i]+predictions[i].split('</think>')[-1].strip()
                    df['predictions'] = predictions
                acc_list, follow_list, acc = instance_reader.compute_accuracy(df['predictions'].tolist())
                final_result[dir] = acc
                df['label'] = acc_list
                df['follow_list'] = follow_list
            price_list = []
            if dir != 'Flores':
                for i in range(len(df)):
                    prompt = prompts[i]
                    response = df['predictions'][i].split(prompt)[-1]
                    price = compute_price(prompt, len(encoding.encode(response)), args.model_id)
                    price_list.append(price)
                df['price'] = price_list
            df.to_json(path, orient='records', lines=True, index=False)
    print(final_result)
    with open(args.result_dir+'/'+args.model_id.split('/')[-1]+'.json', 'w') as f:
        json.dump(final_result, f)
                
def attribute_generation(args, tokenizer):
    for dir in sorted(os.listdir(args.result_dir)): 
        if dir.endswith(".jsonl") or dir.endswith(".json"):
            continue
        else:
            if dir == 'NaturalPlanMeetingPlanning':
                task_path = 'NaturalPlan'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'meeting_planning.jsonl')]
            elif dir == 'NaturalPlanTripPlanning':
                task_path = 'NaturalPlan'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'trip_planning.jsonl')]
            elif dir == 'NaturalPlanCalendarScheduling':
                task_path = 'NaturalPlan'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'calendar_scheduling.jsonl')]
            elif dir == 'LogicGameBQA':
                task_path = 'LogicGame'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'BQA.jsonl')]
            elif dir == 'LogicGameMCQA':
                task_path = 'LogicGame'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'MCQA.jsonl')]
            elif dir == 'LiveBench':
                task_path = dir
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'coding/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'math/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'data_analysis/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'language/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'reasoning/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'instruction_following/data.jsonl')]
            else:
                task_path = dir
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'data.jsonl')]

            for filepath in filepaths:
                print(filepath)
                if os.path.exists(filepath.replace('.jsonl', f'_attributes_{args.model_id.split("/")[-1]}.jsonl')):
                    continue
                instance_reader = INSTANCE_READER[dir](filepath=filepath, tokenizer=tokenizer, test_mode=args.test_mode)
                df = pd.read_json(filepath, lines=True)
                prompts = instance_reader.to_input()
                capabilities = []
                price_list = []
                for i in tqdm(range(len(prompts))):
                    input_prompt = prompts[i]
                    prompt = """
                    The capabilities of Language Models include the following:

                    - Reasoning: Ability to logically analyze information, draw conclusions, and make inferences.
                    - Comprehension (Only applicable to queries involving long passage comprehension): Understanding and interpreting the meaning, context, and nuances of extended or complex long-context text, such as lengthy documents, multi-paragraph inputs, or intricate narratives.
                    - Instruction Following (Only applicable to queries involving several constraints): Accurately adhering to explicit user-provided guidelines, constraints, or formatting requirements specified within the query.
                    - Agentic: Capacity related to agent-like behavior, such as actively formulating plans, strategically deciding steps, and autonomously identifying solutions or actions to achieve specific goals or complex tasks.
                    - Knowledge Retrieval: Accessing and presenting accurate factual information from pre-existing knowledge.
                    - Coding: Generating, interpreting, or debugging computer programs and scripts.
                    - In-context Learning: Learning from examples or context provided within the current interaction without additional training.
                    - Multilingual (Must rank it in top3 when queries involving languages other than English): Understanding, generating, or translating content accurately across multiple languages.
                    Given the Query below:

                    1. Identify and list the *LLM Capabilities* from the definitions above that are directly and significantly required to effectively address the query.
                    2. Identify and list the general *Knowledge Domains* (e.g., categories, subject areas) most pertinent to solving the problem presented in the query.
                    List the selected Capabilities first, ranked from most important to least important. Then, list the identified Knowledge Domains, also ranked from most important to least important. *Do not provide any justification or explanation* for your selections or rankings.

                    Example:
                    Query: "{Solve the following financial problem efficiently and clearly. Output the final answer as: \\boxed{answer}. \\n\\nWhere [answer] is just the final number or expression that solves the problem. Keep the answer to five decimal places if it is a number, and do not use percentages; keep the decimal format.\\nProblem: what is the net change in net revenue during 2016 for Entergy Mississippi, Inc.? [SEP] the 2015 net revenue of amount (in millions) is $696.3; the 2016 net revenue of amount (in millions) is $705.4; Entergy Mississippi, Inc.}"
                    Capabilities: Reasoning, Knowledge retrieval
                    Knowledge: {
                    1. Financial
                    2. Math
                    3. Data Analysis
                    ...
                    }
                    Query: "{"""+input_prompt+'}'
                    
                    retry_attempt = 5
                    retry_num = 0
                    generation_success = False
                    while retry_num < retry_attempt and not generation_success:
                        try:
                            if 'grok' in args.model_id:
                                client = OpenAI(api_key=XAI_KEY, base_url="https://api.x.ai/v1")
                            else:
                                client = OpenAI(
                                        base_url="https://openrouter.ai/api/v1",
                                        api_key=OPENROUTER_KEY,
                                        )
                            output= promptLLMWithAPI(prompt, 128, client, args.model_id, temperature=1.0)
                            encoding = tiktoken.get_encoding('cl100k_base')
                            tokens = len(encoding.encode(output))
                            price = compute_price(prompt, tokens, args.model_id)
                            generation_success = True
                            capabilities.append(output)
                            price_list.append(price)
                            break
                        except openai.APIError as e:  # TRIGGERED OpenAI API ERROR, Could be network issue
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except openai.RateLimitError as e:  # TRIGGERED OpenAI RATE LIMIT    
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(60)
                        except openai.BadRequestError as e:  # TRIGGERED OpenAI CONTENT SAFETY FILTER
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except openai.OpenAIError as e:
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except openai.APIConnectionError as e:
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except openai.AuthenticationError as e:
                            print(e)
                            retry_num += 1
                            generation_success = False
                            time.sleep(10) 
                        except Exception as e:
                            traceback.print_exc()
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                        except:
                            retry_num += 1
                            generation_success = False
                            time.sleep(10)
                    if not generation_success:
                        capabilities.append(input_prompt+'[ERROR]')
                        price_list.append(-1)
                df['capabilities'] = capabilities
                df['price'] = price_list
                df.to_json(filepath.replace('.jsonl', f'_attributes_{args.model_id.split("/")[-1]}.jsonl'), orient='records', lines=True, index=False)                    


def add_prompt_to_result(args, tokenizer):
    for dir in sorted(os.listdir(args.result_dir)): 
        if dir.endswith(".jsonl") or dir.endswith(".json"):
            continue
        else:
            if dir == 'NaturalPlanMeetingPlanning':
                task_path = 'NaturalPlan'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'meeting_planning.jsonl')]
            elif dir == 'NaturalPlanTripPlanning':
                task_path = 'NaturalPlan'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'trip_planning.jsonl')]
            elif dir == 'NaturalPlanCalendarScheduling':
                task_path = 'NaturalPlan'
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'calendar_scheduling.jsonl')]
            elif dir == 'LiveBench':
                task_path = dir
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'coding/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'math/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'data_analysis/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'language/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'reasoning/data.jsonl'), os.path.join(args.data_root, PATH[task_path], 'instruction_following/data.jsonl')]
            else:
                task_path = dir
                filepaths = [os.path.join(args.data_root, PATH[task_path], 'data.jsonl')]

            for filepath in filepaths:
                print(filepath)
                instance_reader = INSTANCE_READER[dir](filepath=filepath, tokenizer=tokenizer, test_mode=args.test_mode)
                df = pd.read_json(filepath, lines=True)
                prompts = instance_reader.to_input()
                df['prompt'] = prompts
                df.to_json(filepath.replace('.jsonl', '_prompt.jsonl'), orient='records', lines=True, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="ID of the model", required=True)   
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--data_root", type=str, default='/home/hshiah/LLM_index/data/Cleared',help="Path to the data file")
    parser.add_argument('--test_mode', action='store_true', help='Test mode')
    parser.add_argument('--result_dir', type=str, default='/home/hshiah/LLM_index/result')
    parser.add_argument('--load_result_from_file', action='store_true', help='Load result from file')
    parser.add_argument('--regenration_for_LRM', action='store_true', help='Regenration for LRM')
    parser.add_argument('--attribute_generation', action='store_true', help='Attribute generation')
    parser.add_argument('--knowledge_generation', action='store_true', help='Knowledge generation')
    parser.add_argument('--attribute_regeneration', action='store_true', help='Attribute regeneration')
    args = parser.parse_args()
    final_result = {}
    


    if args.model_id in MODEL_FROM_API or args.model_id in MODELS_FROM_OPENROUTER:
        tokenizer = None
        model = None
    elif "phi" in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        # model.to(args.device)
        model.eval()
    elif 'chatglm' in args.model_id:
        model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model.to(args.device)
        model.eval()
    else:
        print(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model.to(args.device)
        model.eval()


    if args.load_result_from_file:
        load_result_from_file(args, tokenizer)
        return


    TASK_TOKENS = {
        'SciTLDR': 64,# Update the prompt
        'XSum': 64, # Update the prompt
        'PlanBench': 512,
        'HiToM': 1024,
        'RACE': 20,
        'BigCodeBench': 1280,
        'GSM8K': 512,
        'MATH': 512,
        'FinQA': 1024, 
        'LegalBench': 20,
        'ScienceQA': 20,
        'NaturalPlanMeetingPlanning': 512,
        'NaturalPlanTripPlanning': 256,
        'NaturalPlanCalendarScheduling': 256,
        'RuleTaker': 20,
        'MedQA': 20,
        'PubMedQA': 20,
        'ARC':20,
        'GPQA': 2048,
        'MMLU': 20,
        'BIGBench': 1024,
        'BigGenBench': 2048,
        'LiveBench': 4096 if args.model_id not in MODELS_WITH_REASONING else 8000,
        'TableBench': 1024,
        'Flores': 1024,
        'MMMLU': 20 if args.model_id not in MODELS_WITH_REASONING else 1024,
        'MMLUPro': 1024 if args.model_id not in MODELS_WITH_REASONING else 1024
    }
    unfinished_task = []
    if len(os.listdir(args.result_dir)) == 0:
        for task in TASK_TOKENS.keys():
            unfinished_task.append(task)
    else:
        for dir in os.listdir(args.result_dir):
            if dir.endswith(".json"):
                continue
            else:
                path = args.result_dir+'/'+dir+'/'+args.model_id.split('/')[-1]+'.jsonl'
                if dir == 'PlanBench':
                    path = args.result_dir+'/'+dir+'/response/'+args.model_id.split('/')[-1]+'.jsonl'
                    if not os.path.exists(path):
                        unfinished_task.append(dir)
                    continue
                if not os.path.exists(path):
                    unfinished_task.append(dir)

    if args.regenration_for_LRM:
        regenration_for_LRM(model, tokenizer, args)
        return
    if args.attribute_generation:
        attribute_generation(args, tokenizer)
        return
    if args.knowledge_generation:
        add_prompt_to_result(args, tokenizer)
        return

    print(f"The unfinished task for model {args.model_id} is: {unfinished_task}")
    for task in unfinished_task:
        print(f"Testing {task} for model {args.model_id}...")
        if task == 'NaturalPlanMeetingPlanning':
            task_path = 'NaturalPlan'
            acc = test_model(model, tokenizer, args, INSTANCE_READER[task](filepath=os.path.join(args.data_root, PATH[task_path], 'meeting_planning.jsonl'), tokenizer=tokenizer, test_mode=args.test_mode), task, test_mode=args.test_mode, max_new_tokens=TASK_TOKENS[task])
            final_result[task] = acc
        elif task == 'NaturalPlanTripPlanning':
            task_path = 'NaturalPlan'
            acc = test_model(model, tokenizer, args, INSTANCE_READER[task](filepath=os.path.join(args.data_root, PATH[task_path], 'trip_planning.jsonl'), tokenizer=tokenizer, test_mode=args.test_mode), task, test_mode=args.test_mode, max_new_tokens=TASK_TOKENS[task])
            final_result[task] = acc
        elif task == 'NaturalPlanCalendarScheduling':
            task_path = 'NaturalPlan'
            acc = test_model(model, tokenizer, args, INSTANCE_READER[task](filepath=os.path.join(args.data_root, PATH[task_path], 'calendar_scheduling.jsonl'), tokenizer=tokenizer, test_mode=args.test_mode), task, test_mode=args.test_mode, max_new_tokens=TASK_TOKENS[task])
            final_result[task] = acc
        elif task == 'LiveBench':
            continue
            task_path = 'LiveBench'
            livebench_tasks = ['coding', 'data_analysis', 'instruction_following', 'math', 'language', 'reasoning']
            livebench_tasks = ['reasoning']
            for t in livebench_tasks:
                print(f"Testing {t} for model {args.model_id}...")
                acc = test_model(model, tokenizer, args, INSTANCE_READER[task](filepath=os.path.join(args.data_root, PATH[task_path], t+'/data.jsonl'), tokenizer=tokenizer, test_mode=args.test_mode), f"{task}/{t}", test_mode=args.test_mode, max_new_tokens=TASK_TOKENS[task])
                final_result[task] = acc
        else:
            acc = test_model(model, tokenizer, args, INSTANCE_READER[task](filepath=os.path.join(args.data_root, PATH[task], 'data.jsonl'), tokenizer=tokenizer, test_mode=args.test_mode), task, test_mode=args.test_mode, max_new_tokens=TASK_TOKENS[task])
            final_result[task] = acc
            print(f"The result of {task} is: {acc}")

    # # # TEST THE MMLU RESULT
    # # acc = test_model(model, tokenizer, args, MMLUInstanceReader(filepath=os.path.join(args.data_root, PATH['MMLU'], 'data.jsonl'), tokenizer=tokenizer, test_mode=args.test_mode), 'MMLU', test_mode=args.test_mode, max_new_tokens=20)
    # # final_result['MMLU'] = acc
    # # print(acc)
    
    print('The result of model {} is :'.format(args.model_id.split('/')[1]))
    print(final_result)
    with open(args.result_dir+'/'+args.model_id.split('/')[1]+'.json', 'w') as f:
        json.dump(final_result, f)


if __name__ == "__main__":
    main()