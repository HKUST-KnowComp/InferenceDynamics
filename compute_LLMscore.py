import pandas as pd
import os
import json
import numpy as np
import time
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_list(length, alpha):
    return sum([alpha**i for i in range(length)])

if __name__ == "__main__":
    alpha = 0.5
    beta = 0
    index_path = {'/home/hshiah/LLM_index/data/Cleared/ARC/data_capabilities_knowledge.jsonl':'ARC', 
                '/home/hshiah/LLM_index/data/Cleared/BIGBench/data_capabilities_knowledge.jsonl':'BIGBench',
                '/home/hshiah/LLM_index/data/Cleared/BigCodeBench/data_capabilities_knowledge.jsonl':'BigCodeBench',
                '/home/hshiah/LLM_index/data/Cleared/FinQA/data_capabilities_knowledge.jsonl':'FinQA',
                '/home/hshiah/LLM_index/data/Cleared/Flores/data_capabilities_knowledge.jsonl':'Flores',
                '/home/hshiah/LLM_index/data/Cleared/GSM8K/data_capabilities_knowledge.jsonl':'GSM8K',
                '/home/hshiah/LLM_index/data/Cleared/HiToM/data_capabilities_knowledge.jsonl':'HiToM',
                '/home/hshiah/LLM_index/data/Cleared/LegalBench/data_capabilities_knowledge.jsonl':'LegalBench',
                '/home/hshiah/LLM_index/data/Cleared/MATH-500/data_capabilities_knowledge.jsonl':'MATH',
                '/home/hshiah/LLM_index/data/Cleared/MedQA/data_capabilities_knowledge.jsonl':'MedQA',
                '/home/hshiah/LLM_index/data/Cleared/MMLU/data_capabilities_knowledge.jsonl':'MMLU',
                '/home/hshiah/LLM_index/data/Cleared/MMMLU/data_capabilities_knowledge.jsonl':'MMMLU',
                '/home/hshiah/LLM_index/data/Cleared/NaturalPlan/calendar_scheduling_capabilities_knowledge.jsonl':'NaturalPlanCalendarScheduling',
                '/home/hshiah/LLM_index/data/Cleared/NaturalPlan/meeting_planning_capabilities_knowledge.jsonl':'NaturalPlanMeetingPlanning',
                '/home/hshiah/LLM_index/data/Cleared/NaturalPlan/trip_planning_capabilities_knowledge.jsonl':'NaturalPlanTripPlanning',
                '/home/hshiah/LLM_index/data/Cleared/PlanBench/data_capabilities_knowledge.jsonl':'PlanBench',
                '/home/hshiah/LLM_index/data/Cleared/PubMedQA/data_capabilities_knowledge.jsonl':'PubMedQA',
                '/home/hshiah/LLM_index/data/Cleared/RACE/data_capabilities_knowledge.jsonl':'RACE',
                '/home/hshiah/LLM_index/data/Cleared/RuleTaker/data_capabilities_knowledge.jsonl':'RuleTaker',
                '/home/hshiah/LLM_index/data/Cleared/ScienceQA/data_capabilities_knowledge.jsonl':'ScienceQA',
                '/home/hshiah/LLM_index/data/Cleared/SciTLDR/data_capabilities_knowledge.jsonl':'SciTLDR',
                '/home/hshiah/LLM_index/data/Cleared/XSum/data_capabilities_knowledge.jsonl':'XSum'}
    test_path = {
        '/home/hshiah/LLM_index/data/Cleared/MMLUPro/data_capabilities_knowledge.jsonl': 'MMLUPro',
             '/home/hshiah/LLM_index/data/Cleared/BigGenBench/data_capabilities_knowledge.jsonl': 'BigGenBench',
             '/home/hshiah/LLM_index/data/Cleared/GPQA/data_capabilities_knowledge.jsonl':'GPQA',
            '/home/hshiah/LLM_index/data/Cleared/LiveBench/coding/data.jsonl': 'coding',
             '/home/hshiah/LLM_index/data/Cleared/LiveBench/data_analysis/data.jsonl': 'data_analysis',
            '/home/hshiah/LLM_index/data/Cleared/LiveBench/language/data.jsonl': 'language',
            '/home/hshiah/LLM_index/data/Cleared/LiveBench/math/data.jsonl': 'math',
            '/home/hshiah/LLM_index/data/Cleared/LiveBench/reasoning/data.jsonl': 'reasoning',
             '/home/hshiah/LLM_index/data/Cleared/LiveBench/instruction_following/data.jsonl': 'instruction_following'
             }
    route_method = 'both'
    constraints = 'price'
    all_model_list = [
        'grok-2-latest',
        'nova-pro-v1',
        'qwen-max',
        'glm-4-plus',
        'gpt-4o-2024-11-20',
        'llama-3.3-70b-instruct',
        'gemini-pro-1.5',
        'qwen-2.5-72b-instruct',
    ]
    # replaced_path = '_attributes_gemma-3-12b-it.jsonl'
    replaced_path = '_attributes_gpt-4o-mini.jsonl'
    # replaced_path = '_attributes_qwen-2.5-7b-instruct.jsonl'
    # implement a permutation of the model list
    output_path = f'/home/hshiah/LLM_index/LLM_score.json'
    for i in range(1):
        combination_list = list(combinations(all_model_list, len(all_model_list)-i-1))
        combination_list = [all_model_list]
        for combination in combination_list:
            model_list = list(combination)
            output_dict = {}
            livebench_total_price = 0
            livebench_total_price_for_annotation = 0
            for model in model_list:
                # print(f"Processing {model}")
                knowledge_score = {}
                capability_score = {}
                knowledge_upperbound = {}
                capability_upperbound = {}
                for path,task in index_path.items():
                    if 'LiveBench' not in path:
                        path = path.replace("_capabilities_knowledge.jsonl",replaced_path)
                    else:
                        path = path.replace('.jsonl', replaced_path)
                    df = pd.read_json(path, lines=True)
                    prediction_labels = []
                    price_list = []
                    if task == 'SciTLDR':
                        prediction_df = pd.read_json(f'/home/hshiah/LLM_index/result/{task}/{model}.jsonl', lines=True)
                        for i in range(len(df)):
                            prediction_labels.append(int(prediction_df['label'][i]['rougeL']>0.2))
                        price_list = prediction_df['price'].tolist()
                    elif task == 'XSum':
                        prediction_df = pd.read_json(f'/home/hshiah/LLM_index/result/{task}/{model}.jsonl', lines=True)
                        for i in range(len(df)):
                            prediction_labels.append(int(prediction_df['label'][i]['rougeL']>0.15))
                        price_list = prediction_df['price'].tolist()
                    elif task == 'Flores':
                        prediction_df = pd.read_json(f'/home/hshiah/LLM_index/result/{task}/{model}.jsonl', lines=True)
                        for i in range(len(df)):
                            prediction_labels.append(int(prediction_df['label'][i]>45))
                        price_list = prediction_df['price'].tolist()
                    elif task == 'PlanBench':
                        with open(f'/home/hshiah/LLM_index/result/PlanBench/{model}.jsonl', 'r') as f:
                            prediction_df = json.load(f)
                        price_list = pd.read_json(f'/home/hshiah/LLM_index/result/PlanBench/response/{model}.jsonl', lines=True)['price'].tolist()
                        prediction_df = prediction_df['instances']
                        for i in range(len(df)):
                            try:
                                prediction_labels.append(int(prediction_df[i]['llm_correct']))
                            except:
                                print(task, i)
                    elif task == 'IFEval':
                        prediction_df = pd.read_json(f'/home/hshiah/LLM_index/result/{task}/{model}.jsonl', lines=True)
                        for i in range(len(df)):
                            prediction_labels.append(int(prediction_df['label'][i]['follow_all_instructions']))
                        price_list = prediction_df['price'].tolist()
                    elif task == 'BigCodeBench':
                        prediction_df = pd.read_json(f'/home/hshiah/LLM_index/result/{task}/{model}-hard.jsonl', lines=True)
                        for i in range(len(df)):
                            prediction_labels.append(int(prediction_df['label'][i]))
                        price_list = prediction_df['price'].tolist()
                    else:
                        try:
                            prediction_df = pd.read_json(f'/home/hshiah/LLM_index/result/{task}/{model}.jsonl', lines=True)
                        except:
                            print(path, model)
                        for i in range(len(df)):
                            try:
                                prediction_labels.append(prediction_df['label'][i])
                            except:
                                print(task, i)
                        try:
                            price_list = prediction_df['price'].tolist()
                        except:
                            print(task, model)
                    for i in range(len(df)):
                        capability_list = df['capabilities_list'][i]
                        knowledge_list = df['main_knowledge'][i]
                        for j in range(len(capability_list)):
                            capability_score[capability_list[j]] = capability_score.get(capability_list[j], 0) + prediction_labels[i]*(alpha**j)/compute_list(len(capability_list), alpha)
                            capability_upperbound[capability_list[j]] = capability_upperbound.get(capability_list[j], 0) + 1#(alpha**j)/compute_list(len(capability_list), alpha)
                            if constraints == 'price':
                                capability_score[capability_list[j]] -= price_list[i]*beta*(alpha**j)/compute_list(len(capability_list), alpha)
                        for j in range(len(knowledge_list)):
                            knowledge_score[knowledge_list[j]] = knowledge_score.get(knowledge_list[j], 0) + prediction_labels[i]*(alpha**j)/compute_list(len(knowledge_list), alpha)
                            knowledge_upperbound[knowledge_list[j]] = knowledge_upperbound.get(knowledge_list[j], 0) + 1#(alpha**j)/compute_list(len(knowledge_list), alpha)
                            if constraints == 'price':
                                knowledge_score[knowledge_list[j]] -= price_list[i]*beta*(alpha**j)/compute_list(len(knowledge_list), alpha)
                # scale the score to 0-1
                # print(capability_upperbound)
                for capability in capability_score:
                    capability_score[capability] = round(capability_score[capability]/capability_upperbound[capability], 4)
                for knowledge in knowledge_score:
                    knowledge_score[knowledge] = round(knowledge_score[knowledge]/knowledge_upperbound[knowledge], 4)
                output_dict[model] = {}
                output_dict[model]['capability_score'] = capability_score
                output_dict[model]['knowledge_score'] = knowledge_score
            with open(output_path, 'w') as f:
                json.dump(output_dict, f, indent=4)
            time.sleep(10)

            print('start route')
            
            with open('/home/hshiah/LLM_index/LLM_score.json', 'r') as f:
                LLM_score = json.load(f)
            livebench_score = {}
            livebench_price = {}
            for path, task in test_path.items():
                if 'LiveBench' not in path:
                    path = path.replace("_capabilities_knowledge.jsonl",replaced_path)
                else:
                    path = path.replace('.jsonl', replaced_path)
                try:
                    df = pd.read_json(path, lines=True)
                except:
                    print(path)
                chosen_models = []
                for i in range(len(df)):
                    capability_list = df['capabilities_list'][i]
                    knowledge_list = df['main_knowledge'][i]
                    knowledge_score_list = [0.0]*len(LLM_score)
                    capability_score_list = [0.0]*len(LLM_score)
                    model_name_list = []
                    model_index = 0
                    for model, score in LLM_score.items():
                        for j in range(len(capability_list)):
                            if route_method == 'capability' or route_method == 'both':
                                capability_score_list[model_index] += score['capability_score'][capability_list[j]]*alpha**j/compute_list(len(capability_list), alpha)
                            else:
                                capability_score_list[model_index] = 0.0
                        for j in range(len(knowledge_list)):
                            if route_method == 'knowledge' or route_method == 'both':
                                
                                knowledge_score_list[model_index] += score['knowledge_score'][knowledge_list[j]]*alpha**j/compute_list(len(knowledge_list), alpha)
                            else:
                                knowledge_score_list[model_index] = 0.0
                        model_name_list.append(model)
                        model_index += 1
                    # find the model with the highest score
                    capability_score = np.array(capability_score_list)
                    knowledge_score = np.array(knowledge_score_list)
                    score = capability_score + knowledge_score
                    
                    max_score = max(score)
                    chosen_idx = np.where(max_score == score)[0]
                    chosen_models.append(model_name_list[chosen_idx])
                
                # Load all unique models' predictions at once
                unique_models = list(set(all_model_list))
                model_predictions = {}
                model_score_dict = {}
                price_df = {}
                model_price = {}
                for model in unique_models:
                    if 'LiveBench' in path:
                        try:
                            model_predictions[model] = pd.read_json(f"/home/hshiah/LLM_index/thirdpartyeval/LiveBench/livebench/data/live_bench/{task}/model_judgment/{model}.jsonl", lines=True)
                            price_df[model] = pd.read_json(f"/home/hshiah/LLM_index/result/LiveBench/{task}/{model}.jsonl", lines=True)
                        except:
                            print(model)
                    elif task == 'MMLUPro':
                        model_predictions[model] = pd.read_json(f'/home/hshiah/LLM_index/results/{task}/{model}.jsonl', lines=True)
                        price_df[model] = pd.read_json(f"/home/hshiah/LLM_index/results/{task}/{model}.jsonl", lines=True)
                    else:
                        model_predictions[model] = pd.read_json(f'/home/hshiah/LLM_index/result/{task}/{model}.jsonl', lines=True)
                        price_df[model] = pd.read_json(f"/home/hshiah/LLM_index/result/{task}/{model}.jsonl", lines=True)
                    if task == 'MMLUPro' or task == 'GPQA':
                        model_score_dict[model] = model_predictions[model]['label'].sum()*1.0/len(model_predictions[model])
                        model_price[model] = price_df[model]['price'].sum()*1.0
                    elif task == 'BigGenBench':
                        model_score_dict[model] = len(model_predictions[model][model_predictions[model]['score']>=4])*1.0/len(model_predictions[model])
                        model_price[model] = price_df[model]['price'].sum()*1.0
                    elif "LiveBench" in path:
                        model_score_dict[model] = model_predictions[model]['score'].sum()*1.0/len(model_predictions[model])
                        model_price[model] = price_df[model]['price'].sum()*1.0
                    else:
                        model_score_dict[model] = model_predictions[model]['label'].sum()*1.0/len(model_predictions[model])
                        model_price[model] = price_df[model]['price'].sum()*1.0

                # print the model with the highest score
                max_key = max(model_score_dict, key=model_score_dict.get)
                
                # Extract predictions for each data point
                label = []
                random_selection = []
                price_list = []
                
                for i in range(len(df)):
                    current_model = chosen_models[i]
                    # randomly select one model from unique_model
                    random_model = unique_models[np.random.randint(0, len(unique_models))]
                    
                    if task == 'BigGenBench':
                        label.append(int(model_predictions[current_model]['score'][i]>=4))
                        random_selection.append(int(model_predictions[random_model]['score'][i]>=4))
                    elif task == 'MMLUPro' or task == 'GPQA':
                        label.append(model_predictions[current_model]['label'][i])
                        random_selection.append(model_predictions[random_model]['label'][i])
                    elif 'LiveBench' in path:
                        label.append(model_predictions[current_model]['score'][i])
                        random_selection.append(model_predictions[random_model]['score'][i])
                    else:
                        label.append(model_predictions[current_model]['label'][i])
                        random_selection.append(model_predictions[random_model]['label'][i])
                    price_list.append(price_df[current_model]['price'][i])
                
                df['label'] = label
                # print(f"accuracy: {sum(df['label'])*1.0/len(df)}")
                df['chosen_models'] = chosen_models
                # if sum(df['label'])*1.0/len(df) > model_score_dict[max_key]:
                print("*"*50)
                print(f"Task: {task}")
                print(f"accuracy: {sum(df['label'])*1.0/len(df)}")
                # if 'LiveBench' in path:
                livebench_score[task] = sum(df['label'])*1.0/len(df)
                livebench_price[task] = sum(price_list)+df['price'].sum()
                print(f"Model list: {model_list}")
                # print 
                print(f"Model with the highest score: {max_key} {model_score_dict[max_key]}")
                print(f"random selection: {sum(random_selection)*1.0/len(random_selection)}")
                print(f"price: {sum(price_list)+df['price'].sum()}")
                print(f"price for annotation: {df['price'].sum()}")
                print(f"price for annotation: {df['price'].sum()/(sum(price_list)+df['price'].sum())}")
                livebench_total_price += sum(price_list)+df['price'].sum()
                livebench_total_price_for_annotation += df['price'].sum()
                if 'LiveBench' in path:
                    print(f"Model price with the highest score: 'gemini-pro-1.5' {model_price['gemini-pro-1.5']}")
                else:   
                    print(f"model price with the highest score: {model_price[max_key]}")
                print("*"*50)
                df.to_json(f"/home/hshiah/LLM_index/routed_result_models/{task}_route_LLM_{route_method}_{beta}_{replaced_path}", orient='records', lines=True, index=False)
            # compute the average score of livebench_score
            print(sum(livebench_score.values())*1.0/len(livebench_score))
            print(sum(livebench_price.values())*1.0)
            print(f"livebench_total_price: {livebench_total_price}")
            print(f"livebench_total_price_for_annotation: {livebench_total_price_for_annotation}")
            print(f"livebench_total_price_for_annotation/livebench_total_price: {livebench_total_price_for_annotation/livebench_total_price}")

    
