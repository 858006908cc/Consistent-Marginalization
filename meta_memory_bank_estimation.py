import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Union
import aiohttp
import nest_asyncio
import numpy as np
import openai
import torch
from datasets import load_from_disk
from gensim.models import Word2Vec
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your saved model
custom_model_path = "/your/custom/path/llama3-8b-model"

#Test loading the model and tokenizer
# Load tokenizer and model from the saved path
tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
model11 = AutoModelForCausalLM.from_pretrained(
    custom_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Now continue with the rest of your code
messages = [
    {"role": "system", "content": "Hello! Who are you?"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model11.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False) # https://github.com/Lightning-AI/litgpt/issues/327


def to_single_nested_list(data):
    """Convert any format (including stringified lists) to a clean single-nested list like ['intent.txt']"""
    cleaned = []

    if not isinstance(data, list):
        data = [data]

    for item in data:
        if isinstance(item, list):
            cleaned.extend(to_single_nested_list(item))  # flatten nested list
        elif isinstance(item, str):
            # Extract actual filename if it's embedded in a stringified list or set
            match = re.search(r"[a-zA-Z0-9_.-]+\.txt", item)
            if match:
                cleaned.append(match.group(0))
            else:
                cleaned.append("Unknown")
        else:
            cleaned.append("Unknown")

    return cleaned

def extract_list_of_singleton_sets(raw_text: str) -> list:
    """
    Extract a list of singleton values from a raw text string containing [{'...'}, {...}, ...].
    """
    # Step 1: Find the first [...]-enclosed part using regex
    match = re.search(r'\[(.*?)\]', raw_text, re.DOTALL)
    if not match:
        return []

    list_content = "[" + match.group(1) + "]"  # Re-wrap it as full list
    
    try:
        parsed_list = ast.literal_eval(list_content)
    except Exception as e:
        print(f"[Warning] Parsing failed: {e}")
        return []

    # Step 2: Flatten each singleton set into string
    extracted = []
    for item in parsed_list:
        if isinstance(item, set):
            if len(item) == 1:
                extracted.append(next(iter(item)))
            else:
                extracted.append(list(item)[0])
        else:
            extracted.append(str(item))

    return extracted


def intent_to_index(intent_list, intent_to_index_mapping):
    return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]


def prompting1(prompts, true_intentions,intention_set, temperature, formats,formats1,formats2,formats3):

    combination=[]
    for prompt,true_label in zip(prompts,true_intentions):
        prompt1=f'For the sentence:"{prompt}"'
        # prompt1=f'For the sentence:"{prompt}"'
        combination.append(prompt1)

    respon = ''
    respon += f'Please ensure there are a total of {len(prompts)} responses. For these "{len(prompts)}" queries "{combination}", please identify the expressed intentions for each sentence based on the provided intention set \n{intention_set}, using the provided intents for guidance:'
    respon += f'\n\n Please strictly only provide identified intents and do not add any additional information !!! .'
    respon += f'\n\n Please do not show any additional information !!! .'
    respon += f'\n\n Concatenate reponse together in a single array.'
    respon += f'\n\n Please ensure there are a total of {len(prompts)} responses.'
    messages = [
        {"role": "system", "content": respon},
    ]


    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model11.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    # https://github.com/Lightning-AI/litgpt/issues/327

    outputs = model11.generate(
        input_ids,
        max_new_tokens=60*len(prompts),
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response11 = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response11, skip_special_tokens=True)
    return response
    
def prompting1(prompts, true_intentions,intention_set, temperature, formats,formats1,formats2,formats3):

    instruction_blocks = []

    for i, prompt in enumerate(prompts):
        instruction_blocks.append(f'''
    Query {i + 1}: "{prompt}"

    Identify the SINGLE most likely intent from the following set:
    {intention_set}

    Strictly follow the required format: {formats}
    ''')

    # Combine all queries into one block
    full_instruction = "\n".join(instruction_blocks)

    # Append unified strict instructions
    full_prompt = full_instruction + f'''

    IMPORTANT INSTRUCTIONS:

    - Only return the identified single intent from the provided set {intention_set}.
    - Do NOT provide any explanation, reasoning, or extra text.
    - Only respond with the intent string — nothing else.
    - Concatenate all responses into a single array.
    - Do not include any headings, comments, or formatting outside the array.
    '''

    messages = [
        {"role": "system", "content": full_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model11.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    # https://github.com/Lightning-AI/litgpt/issues/327

    outputs = model11.generate(
        input_ids,
        max_new_tokens=30,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    response11 = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response11, skip_special_tokens=True)
    return response
    
    
def load_and_clean_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Load the JSON data
        data_json = json.load(file)
        
        # Parse and extract intent names from the nested dictionary strings
        cleaned_data = []
        for key, value in data_json.items():
            # Assuming each value is a dictionary in string format
            inner_dict = eval(value)
            intent = list(inner_dict.keys())[0]  # Get the key from the dictionary which is the intent
            cleaned_data.append(intent)
    return cleaned_data



def load_and_convert_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire file content, which seems to be a single JSON array in text form
        data_str = file.read()
        # Convert the string representation of the list into an actual Python list
        data = json.loads(data_str)
    return data


def clean_intents(intent_list: List[str]) -> List[str]:
    cleaned = []
    intent=intent_list
    # for intent in intent_list:
    # If it came as a list or nested, flatten
    if isinstance(intent, list):
        intent = intent[0]
    # Remove common unwanted characters
    intent = intent.strip().strip("{}").strip("[]").strip("'").strip('"')
    cleaned.append(intent)
    return cleaned

def process_responses1(lines, true_intentions,temperature,dataset_name,intention_set, step):
    if dataset_name=='clinc':
        formats = {'caring'}
        formats1 = {'surprise'}
        formats2 = {'admiration'}
        formats3 = {'application status'}
        
    if dataset_name=='stackexchange':
        formats = {'engineering.stackexchange.com.txt'}
        formats1 = {'writers.stackexchange.com.txt'}
        formats2 = {'wordpress.stackexchange.com.txt'}
        formats3 = {'law.stackexchange.com.txt'}
        def intent_to_index(intent_list, intent_to_index_mapping):
            return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]

    if dataset_name=='banking77':
        formats = {'Refund not showing up'}
        formats1 = {'card not working'}
        formats2 = {'visa or mastercard'}
        formats3 = {'card not working'}
        def intent_to_index(intent_list, intent_to_index_mapping):
            return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]

    if dataset_name=='mtop_intent':

        formats = {'delete playlist music'}

        formats1 = {'update reminder todo'}

        formats2 = {'update reminder location'}

        formats3 = {'get track info music'}


    if dataset_name=='massive_scenario':

        formats = {'weather'}

        formats1 = {'alarm'}

        formats2 = {'customer service'}

        formats3 = {'currency exchange'}


    if dataset_name=='reddit':

        formats = {'baltimore.txt'}

        formats1 = {'brasil.txt'}

        formats2 = {'coins.txt'}

        formats3 = {'collapse.txt'}

    if dataset_name=='go_emotion':

        formats = {'admiration'}

        formats1 = {'amusement'}

        formats2 = {'anger'}

        formats3 = {'annoyance'}

    if dataset_name=='few_rel_nat':

        formats = {'composer'}

        formats1 = {'military branch'}

        formats2 = {'screenwriter'}

        formats3 = {'sibling'}

    if dataset_name=='few_nerd_nat':

        formats = {'astronomy'}

        formats1 = {'athlete'}

        formats2 = {'mountain'}

        formats3 = {'soldier'}



    if dataset_name=='massive_intent':

        formats = {'alarm query'}

        formats1 = {'audio volume down'}

        formats2 = {'weather query'}

        formats3 = {'transport traffic'}


    all_responses = defaultdict(dict)
    predicted_intentions_list = defaultdict(list)
    valid_predictions = defaultdict(list)
    error_lines = []
    intent_arrays = []
    all_pred_indices = {}
    all_true_indices = {}
    total_line_index=0
    subset_accuracy_index=0
    true_counter_index=0
    true_counter=0
    subset_counter=0
    overall_index = 0
    subset_accuracy_index=0
    subset_counter_index=0
    subset_ratios = []
    matching_ratios = []
    all_good=0
    subset_number=[]
    matching_number =[]
    for line_index in range(0, len(lines), step):
        print("line_index:",line_index)
        actual_step = step if line_index + step <= len(lines) else len(lines) - line_index
        print("actual_step:",actual_step+line_index)
        selected_lines = lines[total_line_index:total_line_index+step]
        true_intentions1 = true_intentions[total_line_index:total_line_index+step]
        total_line_index += step
        attempt_counter = 0
        max_attempts = 10
        attempt_counter = 0
        predicted_intentions = []  # Ensure it's defined outside the loop

        while attempt_counter < max_attempts:
            try:
                response = prompting1(selected_lines, true_intentions1, intention_set,
                                    temperature, formats, formats1, formats2, formats3)

                all_responses[line_index] = response

                try:
                    response = clean_intents(response)
                    predicted_intentions = [[sublist] for sublist in response]

                except Exception:

                    predicted_intentions = re.findall(r"{(.+?)}", response)
                    predicted_intentions = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions]
                    predicted_intentions = [list(intent_dict) for intent_dict in predicted_intentions]
                    predicted_intentions = [[item for item in sublist] for sublist in predicted_intentions]
                print("Predicted_annotation", predicted_intentions)
                print("True_annotation", true_intentions1)

                attempt_counter += 1

                if len(predicted_intentions) == step:
                    break
                elif len(predicted_intentions[0]) == step:
                    break

            except Exception as exc:
                attempt_counter += 1
                error_lines.extend([f'Error on line {overall_index + i}: {exc}' for i in range(len(selected_lines))])
                predicted_intentions = [['unknown']]  # Ensure it's in valid triple-nested format

        # After retries, ensure length and format are correct
        if not predicted_intentions or len(predicted_intentions) == 0:
            predicted_intentions = [clean_intents(response)] * step
        elif len(predicted_intentions) > step:
            predicted_intentions = predicted_intentions[:step]
        elif len(predicted_intentions) < step:
            predicted_intentions.extend([clean_intents(response)] * (step - len(predicted_intentions)))

        # Final check
        if len(predicted_intentions) == step:
            print("all_good")
            all_good += 5
        else:
            print("not_good")

        print("Predicted_annotation",predicted_intentions)
        print('True_annotation',true_intentions1)

        clean_data = predicted_intentions
        true_intentions1 = [[item] for item in true_intentions1]

        def intent_to_index(intent_list, intent_to_index_mapping):
            return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]

        try:
            pred_indices = intent_to_index(clean_data, intent_to_index_mapping)
        except Exception:
            pred_indices = intent_to_index(clean_data, intent_to_index_mapping)

        true_indices = intent_to_index(true_intentions1, intent_to_index_mapping)

        for i, clean in enumerate(clean_data):
            predicted_intentions_list[overall_index] = clean
            if "Error" not in clean:  # If no error in the intention
                all_pred_indices[overall_index] = pred_indices[i]
                valid_predictions[overall_index] = clean
                all_true_indices[overall_index] = true_indices[i]
            else:  # If there is an error
                all_pred_indices[overall_index] = None  # Or any other value that signifies an error
                valid_predictions[overall_index] = None  # Or any other value that signifies an error
                all_true_indices[overall_index] =None
            overall_index += 1

        for i in range(len(true_intentions1)):
            is_subset = set(true_intentions1[i]).issubset(set(clean_data[i]))
            if is_subset:
                subset_counter += 1
            is_matching = set(true_intentions1[i]) == set(clean_data[i])
            if is_matching:
                true_counter += 1

        if all_good==total_line_index:
            print("all_good_so_far")
        if len(predicted_intentions_list) != total_line_index:
            print(f"Warning: total_line_index ({total_line_index}) and the length of predicted_intentions_list ({len(predicted_intentions_list)}) do not match!")
        else:
            print(f"Success: total_line_index ({total_line_index}) and the length of predicted_intentions_list ({len(predicted_intentions_list)}) match as expected.")

        if (total_line_index) % 20 == 0:
            file_name = f".../{dataset_name}/Clustering_{dataset_name}_{line_index}_lower_embeddings_rand_randoom1.txt"
            with open(file_name, "w") as file:
                for index, labels in predicted_intentions_list.items():
                    file.write(f"{index}: {labels}\n")
        
        if (total_line_index) % 20 == 0:
            file_name = f".../{dataset_name}/Clustering_{dataset_name}_{line_index}_lower_embeddings_rand_randoom4.txt"
            with open(file_name, "w") as file:
                for index, labels in all_responses.items():
                    file.write(f"{index}: {labels}\n")

        if (total_line_index) % 5 == 0:
            print("Ratio of subset predicted intents: ", subset_counter/total_line_index)
            print("Ratio of correctly matched intents: ", true_counter/total_line_index)
            print("Total subsets: ", subset_counter)
            print("Number of subset predicted intents_index: ", subset_counter_index)
            print("Number of correctly matched intents_index: ", subset_accuracy_index)

        subset_number.append(subset_counter)
        matching_number.append(true_counter)
        subset_ratios.append(subset_counter/total_line_index)
        matching_ratios.append(true_counter/total_line_index)        

    print("Ratio of correctly predicted intents: ", subset_counter_index/total_line_index)
    print("Number of correctly predicted intents: ", subset_counter_index)
    print("Total subsets: ", subset_counter)
    print("Total matching: ", true_counter)
    print("Total subsets ratio: ", subset_counter/len(lines))
    print("Total lines processed: ", len(lines))
    print("Total subsets: ", subset_counter)
    print("Error lines: ", len(error_lines))
    
    return true_counter,all_responses, predicted_intentions_list, valid_predictions, error_lines


top_10_sentences_indices = {}

def extract_inputs_and_labels(file_path):
    inputs = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            inputs.append(entry['input'])
            labels.append(entry['label'])
    return inputs, labels


dataset_names=['stackexchange','banking77','massive_scenario','clinc','mtop_intent']

#please change the proportions to 0.01,0.05
proportions=[0.01, 0.05]
#please change the temperature to 0.1,0.5, 0.7
for temperature in [0.1, 0.5, 0.7]:
    for proportion in proportions:
        for dataset_name in dataset_names:
            print("dataset_name:",dataset_name)

            large_inputs, large_labels = extract_inputs_and_labels(f'.../datasets/{dataset_name}/large.jsonl')


            selected_labels11 = large_labels
            selected_lines11 =large_inputs
            labels = np.unique(selected_labels11)

            # Generate the intent_to_index_mapping dictionary
            intent_to_index_mapping = {label: index for index, label in enumerate(labels)}


            intents = np.unique(np.array(large_labels)).tolist()
            intent_labeltoid = {intents[i]: i  for i in range(len(intents))}
            label_candidate_index=intent_labeltoid

            np.unique(large_labels)

            lines=large_inputs
            intent_set=np.unique(large_labels)
            intention_set=np.unique(large_labels)

            labels = np.unique(large_labels)

            # Generate the intent_to_index_mapping dictionary
            intent_to_index_mapping = {label: index for index, label in enumerate(labels)}

            # Print the generated dictionary
            print(intent_to_index_mapping)


            large_inputs, large_labels = extract_inputs_and_labels(f'.../datasets/{dataset_name}/large.jsonl')

            # Set the random seed for reproducibility
            random.seed(40)  # You can choose any integer value as the seed

            # Define the maximum possible index
            max_index = len(large_inputs)  # Adjust this number based on your data


            # Generate a list of all possible indices excluding those in Whole_indices
            all_possible_indices = set(range(max_index))

            # Check if there are at least 400 indices to sample from
            if len(all_possible_indices) >= 100:
                random_samples = random.sample(all_possible_indices, int(proportion * len(large_inputs)))
            else:
                print("Not enough elements to sample 400 unique indices outside Whole_indices.")

            print("400 Random Samples outside Whole_indices:", random_samples)
            print("400 Random Samples outside Whole_indices:", len(random_samples))

            selected_labels = [large_labels[index] for index in random_samples]
            
            selected_lines =  [large_inputs[index] for index in random_samples]

            true_counter,de_2000_all_responses_2_10_least, de_2000_predicted_intentions_list_2_10_least, de_2000_valid_predictions_2_10_least, de_2000_error_lines_2_10_least = process_responses1(selected_lines, selected_labels,temperature,dataset_name,intention_set, step=1)

            print("Total_exact_matching:",true_counter/(len(selected_lines)))


            file_name1 = f".../{dataset_name}/CM_{proportion}_{temperature}.txt"
            with open(file_name1, 'w') as file:
                file.write(json.dumps(selected_labels, indent=4))


            # Specify the path to your ground truth file
            ground_truth_path = f'.../{dataset_name}/CM_{proportion}_{temperature}.txt'
            # Load and convert the data
            ground_truth_data = load_and_convert_data(ground_truth_path)
            # Print the first 10 entries to verify the format
            print(ground_truth_data[:10])
            print(len(ground_truth_data))
            file_name1 = f'.../{dataset_name}/CM_{proportion}_{temperature}.txt'
            with open(file_name1, 'w') as file:
                file.write(json.dumps(de_2000_all_responses_2_10_least, indent=4))
                
            # Specify the path to your predictions file
            predictions_path = f'.../{dataset_name}/CM_{proportion}_{temperature}.txt'

            def load_and_clean_data(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data_json = json.load(file)

                    cleaned_data = []
                    for key, values in data_json.items():
                        if isinstance(values, list):
                            cleaned_data.extend(values)  # Extend with all intents in the list
                        elif isinstance(values, str):
                            cleaned_data.append(values)

                return cleaned_data


            def normalize_one_to_one(intent_data):
                cleaned_intents = []

                for item in intent_data:
                    if not isinstance(item, str):
                        cleaned_intents.append('unknown')
                        continue

                    item = item.strip()

                    # Decode escaped characters (e.g., "{\"intent here\"}")
                    item = item.encode('utf-8').decode('unicode_escape')

                    # Extract first quoted value (single or double)
                    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", item)
                    if matches:
                        g1, g2 = matches[0]
                        cleaned = (g1 or g2).strip()
                    else:
                        # Fallback: remove brackets/braces
                        cleaned = re.sub(r"[\[\]\{\}]", "", item).strip()

                    cleaned_intents.append(cleaned.lower())

                return cleaned_intents




            intent_data = load_and_clean_data(predictions_path)

            intent_data = [item.strip("{''}") for item in intent_data]
            print("intent_data;",intent_data)

            intent_data = normalize_one_to_one(intent_data)


            def normalize_intents_preserve_all(intent_data):
                cleaned_intents = []

                for item in intent_data:
                    if not isinstance(item, str):
                        cleaned_intents.append("unknown")
                        continue

                    item = item.strip()

                    # 1. Unescape escaped quotes (e.g., {\"delete something\"})
                    item = item.encode('utf-8').decode('unicode_escape')

                    # 2. Split if multiple intents are separated by comma
                    parts = re.split(r"[,\n]", item)

                    for part in parts:
                        part = part.strip()

                        # 3. Extract quoted content (single or double quotes)
                        matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", part)
                        if matches:
                            for g1, g2 in matches:
                                value = (g1 or g2).strip()
                                if value:
                                    cleaned_intents.append(value)
                        else:
                            # 4. Fallback: remove brackets/braces
                            fallback = re.sub(r"[\[\]\{\}]", "", part).strip()
                            if fallback:
                                cleaned_intents.append(fallback)

                return cleaned_intents


            # intent_data = normalize_intents_preserve_all(intent_data)

            import re

            def clean_intent_list(intent_list):
                cleaned_intents = []

                for intent in intent_list:
                    if not isinstance(intent, str):
                        cleaned_intents.append("unknown")
                        continue

                    # 1. Handle broken strings with embedded quotes or commas
                    # e.g., "improve credit score', 'credit score" → split
                    split_candidates = re.split(r"[,\n]", intent)
                    for candidate in split_candidates:
                        # 2. Extract quoted content if any
                        matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", candidate)
                        if matches:
                            for g1, g2 in matches:
                                value = g1 or g2
                                cleaned_intents.append(value.strip().lower())
                        else:
                            # 3. Fallback: clean up brackets or braces like "[{'calendar'}]"
                            stripped = re.sub(r"[\[\]\{\}]", "", candidate).strip().lower()
                            if stripped:
                                cleaned_intents.append(stripped)

                # Optional: remove duplicates while preserving order
                seen = set()
                deduped = []
                for intent in cleaned_intents:
                    if intent not in seen:
                        deduped.append(intent)
                        seen.add(intent)

                return deduped

            ground_truth=ground_truth_data
            # Observed predictions
            predictions=intent_data
            # Unique labels
            labels = list(set(ground_truth + predictions))

            # Initialize transition matrix with zeros
            transition_matrix = np.zeros((len(labels), len(labels)))

            # Iterate over ground truth and predictions
            for gt, pred in zip(ground_truth, predictions):
                # Get indices of ground truth and predicted labels
                gt_idx = labels.index(gt)
                pred_idx = labels.index(pred)
                
                # Increment count in transition matrix
                transition_matrix[gt_idx, pred_idx] += 1

            # Print transition matrix with labels
            print("Transition Matrix:")
            print("Ground Truth \\ Predicted\t" + "\t".join(labels))
            for i, label in enumerate(labels):
                print(f"{label}\t" + "\t".join(str(int(x)) for x in transition_matrix[i]))

            ground_truth = intent_data

            predictions = ground_truth_data

            # Unique labels
            labels = list(set(ground_truth + predictions))

            # Initialize transition matrix with zeros
            transition_matrix = np.zeros((len(labels), len(labels)))

            # Iterate over ground truth and predictions
            for gt, pred in zip(ground_truth, predictions):
                # Get indices of ground truth and predicted labels
                gt_idx = labels.index(gt)
                pred_idx = labels.index(pred)

                # Increment count in transition matrix
                transition_matrix[gt_idx, pred_idx] += 1

            import numpy as np
            import matplotlib.pyplot as plt

            import numpy as np

            def compute_transition_matrix_mse(ground_truth, predictions):
                # Unique label list (consistent ordering)
                labels = sorted(list(set(ground_truth + predictions)))
                label_to_index = {label: idx for idx, label in enumerate(labels)}
                n = len(labels)

                # Initialize predicted transition matrix
                transition_matrix = np.zeros((n, n))
                for gt, pred in zip(ground_truth, predictions):
                    gt_idx = label_to_index[gt]
                    pred_idx = label_to_index[pred]
                    transition_matrix[gt_idx, pred_idx] += 1

                # Initialize true transition matrix (perfect diagonal)
                true_transition_matrix = np.zeros((n, n))
                for gt in ground_truth:
                    gt_idx = label_to_index[gt]
                    true_transition_matrix[gt_idx, gt_idx] += 1

                # Normalize both matrices row-wise (convert to probability distributions)
                def normalize_rows(mat):
                    row_sums = mat.sum(axis=1, keepdims=True)
                    return np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

                transition_matrix_prob = normalize_rows(transition_matrix)
                true_transition_matrix_prob = normalize_rows(true_transition_matrix)

                # Compute Mean Squared Error (MSE)
                mse = np.mean((transition_matrix_prob - true_transition_matrix_prob) ** 2)

                return mse, transition_matrix_prob, true_transition_matrix_prob, labels

            mse, trans_prob, true_prob, labels = compute_transition_matrix_mse(ground_truth, predictions)
            print("Transition Matrix MSE:", mse)
            print("Labels:", labels)
            print("Predicted Transition (Probabilities):\n", trans_prob)
            print("True Transition (Probabilities):\n", true_prob)
            # Plotting the heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(transition_matrix, cmap='Reds', interpolation='nearest')
            plt.colorbar()
            plt.title(f'Response Discrepancies Masking for {dataset_name}', fontsize=25)
            plt.ylabel('Predicted Label Index', fontsize=25)
            plt.xlabel('Ground Truth Label Index', fontsize=25)

            plt.show()


            plt.figure(figsize=(10, 8))
            plt.imshow(trans_prob, cmap='Reds', interpolation='nearest')
            plt.colorbar()
            plt.title(f'Response Discrepancies Masking for {dataset_name}', fontsize=25)
            plt.ylabel('Predicted Label Index', fontsize=25)
            plt.xlabel('Ground Truth Label Index', fontsize=25)

            plt.show()

            # Create a dictionary to store the ground truth and predicted intents
            intent_table = {}

            # Iterate over the ground truth and predictions
            for gt, pred in zip(ground_truth, predictions):
                # If the ground truth intent is not in the dictionary, add it
                if gt not in intent_table:
                    intent_table[gt] = []
                # Add the predicted intent to the list of predicted intents for the ground truth intent
                intent_table[gt].append(pred)

            # Deduplicate the predicted intents for each ground truth intent
            for gt in intent_table:
                intent_table[gt] = list(set(intent_table[gt]))

            # Save the intent frequency table to a JSON file
            with open(f'.../Memory_Masking_{dataset_name}_{proportion}_{temperature}_1.json', 'w') as file:
                json.dump(intent_table, file, indent=4)

            # print("The intent frequency table has been saved to 'intent_frequency_table.json'.")
            with open(f'.../Memory_Masking_{dataset_name}_{proportion}_{temperature}_1.json', 'r') as file:
                loaded_intent_table = json.load(file)

            print("loaded_intent_table:",intent_table)

            print("loaded_intent_table:",len(intent_table))