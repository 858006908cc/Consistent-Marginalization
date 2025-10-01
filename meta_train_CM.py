import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Union
import numpy as np
import openai
import torch
from gensim.models import Word2Vec
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import openai
import json
from typing import List, Dict, Union
import ast

login(token = "your_api_key")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
custom_model_path = "/your/custom/path/llama3-8b-model"

# Path to your saved model
custom_model_path = "/your/custom/path/llama3-8b-model"

# Load tokenizer and model from the saved path
tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
model11 = AutoModelForCausalLM.from_pretrained(
    custom_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Now continue with the rest of your code
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
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

outputs = model11.generate(
    input_ids,
    max_new_tokens=128,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))


def get_most_common_response(responses):
    """
    Takes a list of responses and returns the most common response.

    Args:
    responses (list of str): List of response strings.

    Returns:
    str: The response that appears most frequently.
    """
    if not responses:
        return None

    response_counter = Counter(responses)
    most_common_response, _ = response_counter.most_common(1)[0]
    return most_common_response


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

def extract_inputs_and_labels(file_path):
    inputs = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            inputs.append(entry['input'])
            labels.append(entry['label'])
    return inputs, labels

def intent_to_index(intent_list, intent_to_index_mapping):
    return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]


def process_responses(line_index, response, c_answer1, all_responses):
    try:
        # If the response list contains identical elements, simplify it
        if len(set(response)) == 1:
            all_responses[line_index] = response[0]
        else:
            all_responses[line_index] = response
        
        # Extract and parse the intentions from c_answer1
        parsed_intentions = parse_intentions(c_answer1)
    except Exception:
        all_responses[line_index] = response[0]
        parsed_intentions = parse_intentions(response[0])
    
    print("predicted_intentions:", parsed_intentions)
    return parsed_intentions

def process_answers(overall_index, c_answer1, all_answers):
    try:
        # Store the first element if all elements in c_answer1 are the same
        parsed_answers = parse_intentions(c_answer1)
        all_answers[overall_index] = parsed_answers[0] if len(set(parsed_answers)) == 1 else c_answer1
    except Exception as e:
        print(f"Error processing answer1: {e}")
        all_answers[overall_index] = c_answer1[0]
        parsed_answers = parse_intentions(c_answer1[0])
    return parsed_answers

def parse_intentions(text):
    matched_intentions = re.findall(r"{(.+?)}", text)
    parsed_intentions = [ast.literal_eval("{" + intent_str + "}") for intent_str in matched_intentions]
    return [list(intent_dict.keys())[0] for intent_dict in parsed_intentions] if parsed_intentions else []

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)


args = parser.parse_args()

dataset_name0 = args.dataset_name
print("Running on dataset:", dataset_name0)

dataset_names1 = [dataset_name0]

proportion = 0.01

for dataset_name in dataset_names1:

    print("dataset_name:", dataset_name)
    small_inputs, small_labels = extract_inputs_and_labels(f'.../datasets/{dataset_name}/small.jsonl')
    large_inputs, large_labels = extract_inputs_and_labels(f'.../datasets/{dataset_name}/large.jsonl')

    np.unique(large_labels)
    # Return a sample to show the split data
    (large_inputs[:3], large_labels[:3]), 
    # (small_inputs[:3], small_labels[:3])
    lines=large_inputs
    intent_set=np.unique(large_labels)
    intention_set=np.unique(large_labels)

    labels = np.unique(large_labels)

    # Generate the intent_to_index_mapping dictionary
    intent_to_index_mapping = {label: index for index, label in enumerate(labels)}
    intent_normalization = {intent: intent for intent in intent_to_index_mapping}

    if dataset_name=='clinc':

        formats = {'pending transfer'}
        formats1 = {'transfer timing'}
        formats2 = {'verify top up'}
        formats3 = {'card payment wrong exchange rate'}

        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")

        import json
        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

    if dataset_name=='stackexchange':


        formats = {'engineering.stackexchange.com.txt'}
        formats1 = {'writers.stackexchange.com.txt'}
        formats2 = {'wordpress.stackexchange.com.txt'}
        formats3 = {'law.stackexchange.com.txt'}
        
        def intent_to_index(intent_list, intent_to_index_mapping):
            return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]

        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")


    if dataset_name=='banking77':

        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")

        formats = {
            'Refund not showing up'
        }
        formats1 = {'declined card payment'}
        formats2 = {'card not working'}
        formats3 = {
        'card not working'}

        formats3 = {
            'visa or mastercard'}

        def intent_to_index(intent_list, intent_to_index_mapping):
            return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]


    if dataset_name=='mtop_intent':


        formats = {'delete playlist music'}

        formats1 = {'update reminder todo'}

        formats2 = {'update reminder location'}

        formats3 = {'get track info music'}
        def intent_to_index(intent_list, intent_to_index_mapping):
            return [[intent_to_index_mapping[intent] for intent in intent_set if intent in intent_to_index_mapping] for intent_set in intent_list]


        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")


    if dataset_name=='massive_scenario':


        formats = {'weather'}

        formats1 = {'alarm'}

        formats2 = {'customer service'}

        formats3 = {'currency exchange'}


        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")

    if dataset_name=='reddit':

        formats = {'baltimore.txt'}

        formats1 = {'brasil.txt'}

        formats2 = {'coins.txt'}

        formats3 = {'collapse.txt'}


        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")


    if dataset_name=='go_emotion':

        formats = {'admiration'}

        formats1 = {'amusement'}

        formats2 = {'anger'}

        formats3 = {'annoyance'}

        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")


    if dataset_name=='few_rel_nat':

        formats = {'composer'}

        formats1 = {'military branch'}

        formats2 = {'screenwriter'}

        formats3 = {'sibling'}


        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")

    if dataset_name=='few_nerd_nat':

        formats = {'astronomy'}

        formats1 = {'athlete'}

        formats2 = {'mountain'}

        formats3 = {'soldier'}


        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")



    if dataset_name=='massive_intent':

        formats = {'alarm query'}

        formats1 = {'audio volume down'}

        formats2 = {'weather query'}

        formats3 = {'transport traffic'}

        with open(f'.../meta_memorisation_estimated/Memory_Masking_{dataset_name}_{proportion}_{0.7}_1.json', 'r') as file:
            loaded_intent_table_inverse = json.load(file)

        with open(f'.../Transition_Matrix/output/selected_data_{dataset_name}.json') as f:
            examples_new = json.load(f)
            
        loaded_model = Word2Vec.load(f".../meta/sentence_embedding_models/{dataset_name}_word2vec_model/word2vec_model.bin")

    model = loaded_model
    examples=examples_new
    example_vectors = {example: np.mean([model.wv[word] for word in example.split() if word in model.wv], axis=0) for example in examples.keys()}

    def get_intent_chain_of_thought1(sentence):
        sentence_words = sentence.lower().split()
        
        # Build sentence embedding
        sentence_vector = np.mean(
            [model.wv[word] for word in sentence_words if word in model.wv],
            axis=0
        )

        similarities = {}
        for example in example_vectors:
            denom = np.linalg.norm(example_vectors[example]) * np.linalg.norm(sentence_vector)
            if denom == 0:
                similarities[example] = 0.0
            else:
                sim = np.dot(example_vectors[example], sentence_vector) / denom
                if isinstance(sim, np.ndarray):
                    sim = sim.flatten()[0]  # Extract scalar safely
                else:
                    sim = float(sim)
                similarities[example] = sim

        top_similar_examples = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:1]


        pp = ''
        for example, similarity in top_similar_examples:
            intent = examples[example]["intent"]
            chain_of_thoughts_expressed = examples[example]["chain_of_thoughts"]["expressed"]
            pp += (
                f'\n\nThe sentence: "{sentence}"\n'
                f'Example: "{example}"\n'
                f'Intent: {intent}\n'
                f'Similarity: {similarity:.4f}\n'
                f'Chain of thoughts (expressed) of example: {chain_of_thoughts_expressed}'
            )
        return pp


    def get_intent_chain_of_thought2(sentence):
        sentence_words = sentence.lower().split()
        
        # Build the sentence embedding using word vectors
        sentence_vector = np.mean(
            [model.wv[word] for word in sentence_words if word in model.wv],
            axis=0
        )

        similarities = {}
        for example in example_vectors:
            denom = np.linalg.norm(example_vectors[example]) * np.linalg.norm(sentence_vector)
            if denom == 0:
                similarities[example] = 0.0
            else:
                sim = np.dot(example_vectors[example], sentence_vector) / denom
                if isinstance(sim, np.ndarray):
                    sim = sim.flatten()[0]  # Extract scalar safely
                else:
                    sim = float(sim)
                similarities[example] = sim

        top_similar_examples = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:5]

        pp = ''
        for example, similarity in top_similar_examples:
            intent = examples[example]["intent"]
            chain_of_thoughts_expressed = examples[example]["chain_of_thoughts"]["expressed"]
            pp += (
                f'\n\nThe sentence: "{sentence}"\n'
                f'Example: "{example}"\n'
                f'Intent: {intent}\n'
                f'Similarity: {similarity:.4f}\n'
                f'Chain of thoughts (expressed) of example: {chain_of_thoughts_expressed}'
            )
        return pp


    response_file11 = []

    def agent_cot_0(prompts, intention_set, true_intentions1, temperatures, formats, formats1, formats2, formats3):


        instruction_blocks = []

        for i, prompt in enumerate(prompts):
            instruction_blocks.append(f'''
            Query {i+1}: "{prompt}"

            Identify the SINGLE most likely intention from this set:
            {intention_set}

            Strictly follow the required format like {formats},{formats1}.
            ''')


        full_instruction = "\n".join(instruction_blocks)

        full_prompt = full_instruction


        full_prompt += f'\n\n Please strictly identified the SINGLE most likely intention from the set {intention_set} and do not add any additional information !!! .'
        full_prompt += f'\n\n Please strictly only provide identified intents and do not add any additional information !!! .'
        full_prompt += f'\n\n Please ensure that only show the intents and do not add any words or additional response please.'
        full_prompt += f'\n\n Concatenate reponse together in a single array.'
        full_prompt += f'\n\n Please do not show anything except responses.'
        full_prompt += f'\n\n Please ensure there are a total of {len(prompts)} responses.'
        full_prompt += f'\n\n Please ensure there are a total of {len(prompts)} responses.'


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
            max_new_tokens=60,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperatures,
            top_p=0.9,
        )

        response11 = outputs[0][input_ids.shape[-1]:]
        # print(tokenizer.decode(response11, skip_special_tokens=True))
        response = tokenizer.decode(response11, skip_special_tokens=True)

        return response
    
    response_file11 = []

    def agent_cot(prompts, intention_set, true_intentions1, temperatures, formats, formats1, formats2, formats3):


        instruction_blocks = []

        for i, prompt in enumerate(prompts):
            cot_reference = get_intent_chain_of_thought1(prompt)
            instruction_blocks.append(f'''
            Query {i+1}: "{prompt}"
            Reference (semantically similar example):\n{cot_reference}

            Identify the SINGLE most likely intention from this set:
            {intention_set}

            Strictly follow the required format like {formats}.
            ''')


        full_instruction = "\n".join(instruction_blocks)

        full_prompt = full_instruction
        full_prompt += f'\n\n Please strictly identified the SINGLE most likely intention from the set and do not add any additional information !!! .'
        full_prompt += f'\n\n Please strictly only provide identified intents and do not add any additional information !!! .'
        full_prompt += f'\n\n Please ensure that only show the intents and do not add any words or additional response please.'
        full_prompt += f'\n\n Concatenate reponse together in a single array.'
        full_prompt += f'\n\n Please do not show anything except responses.'
        full_prompt += f'\n\n Please ensure there are a total of {len(prompts)} responses.'


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
            max_new_tokens=60,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperatures,
            top_p=0.9,
        )

        response11 = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response11, skip_special_tokens=True)
        return response



    response_file = []

    def agent_fot(prompts, intention_set, true_intentions1, temperatures, formats, formats1, formats2, formats3, dataset_name):
        """
        Generates few-shot intent predictions with CoT examples for multiple prompts.
        """

        instruction_blocks = []

        for i, prompt in enumerate(prompts):
            cot_reference = get_intent_chain_of_thought2(prompt)
            instruction_blocks.append(f'''
            Query {i+1}: "{prompt}"
            Reference (semantically similar example):\n{cot_reference}

            Identify the SINGLE most likely intention from this set:
            {intention_set}

            Strictly follow the required format like {formats},{formats1}.
            ''')

        full_instruction = "\n".join(instruction_blocks)


        full_prompt = full_instruction
        full_prompt += f'\n\n Please strictly identified the SINGLE most likely intention from the set and do not add any additional information !!! .'
        full_prompt += f'\n\n Please strictly only provide identified intents and do not add any additional information !!! .'
        full_prompt += f'\n\n Please ensure that only show the intents and do not add any words or additional response please.'
        full_prompt += f'\n\n Concatenate reponse together in a single array.'
        full_prompt += f'\n\n Please do not show anything except responses.'
        full_prompt += f'\n\n Please ensure there are a total of {len(prompts)} responses.'

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
            max_new_tokens=60,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperatures,
            top_p=0.9,
        )

        response11 = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response11, skip_special_tokens=True)

        return response


    def agent_fot(prompts, intention_set, true_intentions1, temperatures, formats, formats1, formats2, formats3, dataset_name):

        instruction_blocks = []

        for i, prompt in enumerate(prompts):
            cot_reference = get_intent_chain_of_thought2(prompt)
            instruction_blocks.append(f'''
            Query {i+1}: "{prompt}"
            Reference (semantically similar example):\n{cot_reference}

            Identify the SINGLE most likely intention from this set:
            {intention_set}

            Strictly follow the required format like {formats}.
            ''')


        full_instruction = "\n".join(instruction_blocks)

        full_prompt = full_instruction


        full_prompt += f'\n\n Please strictly identified the SINGLE most likely intention from the set and do not add any additional information !!! .'
        full_prompt += f'\n\n Please strictly only provide identified intents and do not add any additional information !!! .'
        full_prompt += f'\n\n Please ensure that only show the intents and do not add any words or additional response please.'
        full_prompt += f'\n\n Concatenate reponse together in a single array.'
        full_prompt += f'\n\n Please do not show anything except responses.'

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
            max_new_tokens=60,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperatures,
            top_p=0.9,
        )

        response11 = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response11, skip_special_tokens=True)

        return response

    response_file_o=[]
    def agent1(prompts, intention_set,true_intentions1, temperatures, formats):
        for prompt in prompts:
            combined_response = f'{prompt}'
            instruction= f' Considering the following query "{combined_response}", must identifing the most likely single intention only from the intention set \n{intention_set}. Ensure each response strictly adheres to the prescribed format.'
        combined_response = instruction
        combined_response += f'\n\nProvide each response in the specified format: {formats}. Do not include multiple outputs and input or explanatory text.'


        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": combined_response}
            ]
        )
        
        return response.choices[0].message.content

    import json

    response_file_o = []

    def agent11(prompts, intention_set, true_intentions1, temperatures, formats):
        combined_response_base = f' Considering the following {len(prompts)} queries "{prompts}", must identify the most likely single intention for each input only from the intention set \n{intention_set}. Ensure each response strictly adheres to the prescribed format.'
        combined_response_base += f'\n\nProvide each response in the specified format as {formats}. Do not include multiple outputs and input or explanatory text.'
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": combined_response_base}
            ]
        )


        response_file_o.append({
            "prompts": prompts,
            "responses": [choice.message.content for choice in response.choices]
        })

        return response.choices[0].message.content


    def save_responses_to_file(data, file_name):
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)


    def call_openai(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0.3, max_tokens: int = 350) -> str:
        """A helper function to call OpenAI API with error handling."""
        try:


            full_prompt = messages[0]['content'] 
        

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
                max_new_tokens=60,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            response_tokens = outputs[0][input_ids.shape[-1]:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)

            return response

        except Exception as e:
            print(f"[Warning] OpenAI API call failed: {e}")
            return ""

    def clean_response_list(raw_list: List[str]) -> List[str]:
        """Clean up model outputs: remove '-', quotes, and whitespace."""
        cleaned = []
        for line in raw_list:
            line = line.strip()
            if line.startswith("- "):
                line = line[2:]
            line = line.strip("'").strip('"')
            cleaned.append(line)
        return cleaned

    def generate_batch_feedback(prompts: List[str], responses: List[str],intention_set: List[str]) -> List[str]:
        """Generate correctness feedback for each prompt-response pair."""
        batch_prompt = (
            f"Evaluate whether each identified intent from {intention_set} correctly matches its sentence.\n"
            "Respond ONLY with:\n- 'Correct' if it matches\n- 'Incorrect' if it doesn't\n\n"
            "Evaluation pairs:\n" + "\n".join(
                f"{i+1}. Sentence: {p}\n   Intent: {r}" for i, (p, r) in enumerate(zip(prompts, responses))
            )
        )

        feedback_text = call_openai(messages=[{"role": "user", "content": batch_prompt}])
        

        feedback_text = []
        max_attempts = 10
        attempt = 0

        # Retry loop for valid response
        while (not feedback_text or not feedback_text[0].strip()) and attempt < max_attempts:
            try:
                feedback_text = call_openai(messages=[{"role": "user", "content": batch_prompt}])
            except Exception as e:
                print(f"Error during OpenAI call: {e}")
                feedback_text = []
            attempt += 1

        # Parse and normalize
        if not feedback_text:
            feedback_lines = ["Unknown"] * len(prompts)
        else:            
            feedback_lines = [line.strip() for line in feedback_text.split("\n") if line.strip()]
            feedback_lines = feedback_lines[:len(prompts)]
        return feedback_lines



    def batch_self_refine(prompts: List[str], answers: List[str], batch_feedback: List[str], intention_set: List[str], max_iterations: int = 1) -> List[str]:
        """Self-refine answers based on feedback, selecting only from intention_set."""
        refined = answers.copy()

        formatted_intentions = intention_set


        for _ in range(max_iterations):
            feedback_lines = batch_feedback[:len(prompts)]
            if len(feedback_lines) < len(prompts):
                feedback_lines += ["No changes needed"] * (len(prompts) - len(feedback_lines))

            if all("Correct" in fb for fb in feedback_lines):
                break


            refinement_prompt = (
                f"Refine the current intents {refined} for the following {len(prompts)} queries based on feedback {feedback_lines}.\n"
                f"Select the final single intent **only** from: [{formatted_intentions}].\n"
                f"Only modify an intent if the feedback suggests it is incorrect; otherwise, retain the original.\n\n"
                + "\n".join(
                    f"{i+1}. Query: {p}\n   Feedback: {f}\n   Current Intent: {r}"
                    for i, (p, f, r) in enumerate(zip(prompts, feedback_lines, refined))
                ) +
                f"\n\nReturn exactly {len(prompts)} outputs.\n"
                f"Format requirements:\n"
                f"1. One single intent per query in the format: {formats}, {formats1}, {formats2}\n"
                f"2. One intent per line, in order\n"
                f"3. Select only from: [{formatted_intentions}]\n"
                f"4. No explanations or extra text"
            )

            
            
            refined_text = call_openai(messages=[{"role": "user", "content": refinement_prompt}])


            max_attempts = 10
            attempt = 0

            while (not refined_text or not refined_text.strip()) and attempt < max_attempts:
                refined_text = call_openai(messages=[{"role": "user", "content": refinement_prompt}])
                attempt += 1
            
            feedback_lines=refined_text
            if not feedback_lines:
                break


        return feedback_lines
    
    
    def batch_self_refine(
        prompts: List[str],
        answers: List[str],
        batch_feedback: List[str],
        intention_set: List[str],
        max_iterations: int = 1
    ) -> List[str]:
        """Self-refine answers based on feedback, selecting only from intention_set. Always returns a non-empty list."""

        refined = answers.copy()
        formatted_intentions = intention_set

        for _ in range(max_iterations):
            # Ensure feedback is aligned with prompts
            feedback_lines = batch_feedback[:len(prompts)]
            if len(feedback_lines) < len(prompts):
                feedback_lines += ["Correct"] * (len(prompts) - len(feedback_lines))

            # Exit early if all feedback indicates no change
            if all("Correct" in fb for fb in feedback_lines):
                return refined

            # Build the refinement prompt
            refinement_prompt = (
                f"Refine the current intents {refined} for the following {len(prompts)} queries based on feedback {feedback_lines}.\n"
                f"Select the final Single intent **only** from: [{formatted_intentions}].\n"
                f"Only modify an intent if the feedback suggests it is incorrect; otherwise, retain the original.\n\n"
                + "\n".join(
                    f"{i+1}. Query: {p}\n   Feedback: {f}\n   Current Intent: {r}"
                    for i, (p, f, r) in enumerate(zip(prompts, feedback_lines, refined))
                ) +
                f"\n\nReturn exactly {len(prompts)} outputs.\n"
                f"Format requirements:\n"
                f"1. One single intent per query in the format: intent, intent1, intent2\n"
                f"2. One intent per line, in order\n"
                f"3. Select single intent only from: [{formatted_intentions}]\n"
                f"4. No explanations or extra text"
            )

            # Retry loop
            refined_text = [""]
            max_attempts = 5
            attempt = 0

            while (not refined_text or not refined_text[0].strip()) and attempt < max_attempts:
                refined_text = call_openai(messages=[{"role": "user", "content": refinement_prompt}])
                attempt += 1

            # Extract and clean response lines
            if refined_text and isinstance(refined_text, list):
                refined_lines = [line.strip() for line in refined_text[0].split("\n") if line.strip()]
            else:
                refined_lines = []
                # refined_lines = call_openai(messages=[{"role": "user", "content": refinement_prompt}])

            # Final fallback to ensure output is non-empty and aligned
            if not refined_lines:
                refined_lines = refined
            elif len(refined_lines) < len(prompts):
                refined_lines += ["Unknown"] * (len(prompts) - len(refined_lines))
            elif len(refined_lines) > len(prompts):
                refined_lines = refined_lines[:len(prompts)]

            # Update refined for next iteration
            refined = refined_lines

        return refined

    
    
    import ast

    def clean_singleton_sets(list_of_strings: list) -> list:
        """
        Convert ["{'intent'}", ...] into ['intent', ...]
        """
        cleaned = []
        for item in list_of_strings:
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, set):
                    if len(parsed) == 1:
                        cleaned.append(list(parsed)[0])
                    else:
                        # Unexpected multiple items in set
                        cleaned.append(list(parsed)[0])  # take first
                else:
                    cleaned.append(str(parsed))
            except Exception as e:
                print(f"[Warning] Skipping invalid item: {item} due to {e}")
                cleaned.append(item)
        return cleaned
                        
    def clean_intents(intent_list: List[str]) -> List[str]:
        cleaned = []
        intent=intent_list
        if isinstance(intent, list):
            intent = intent[0]
        # Remove common unwanted characters
        intent = intent.strip().strip("{}").strip("[]").strip("'").strip('"')
        cleaned.append(intent)
        return cleaned
    def clean_numbered_singleton_sets(list_of_strings: list) -> list:
        """
        Convert ["1. {'intent'}", ...] into ['intent', ...] safely.
        """
        cleaned = []
        for item in list_of_strings:
            # Step 1: Remove leading number and dot if exists
            if ". " in item:
                item = item.split(". ", 1)[-1].strip()
            
            # Step 2: Try parsing as Python set
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, set) and len(parsed) == 1:
                    cleaned.append(next(iter(parsed)))  # extract the single element
                else:
                    cleaned.append(str(parsed))
            except Exception:
                # If parsing fails, fallback: clean manually
                item = item.strip().lstrip("- ").strip("{").strip("}").strip("'").strip('"')
                cleaned.append(item)
        return cleaned

    def batch_agent_feedback(
        prompts: List[str],
        intention_set: List[str],
        true_intentions: List[str],
        temperature: float,
        output_format: Dict[str, str],
        batch_size: int = 1
    ) -> Dict[str, Union[List[str], str]]:
        """Process prompts through prediction, evaluation, and refinement."""
        all_results = []
        save_dir = f".../refinement_responses_meta_{dataset_name}_0.5"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_true = true_intentions[i:i+batch_size]

            responses=true_intentions

            if len(responses) < len(batch_prompts):
                responses += ["Unknown"] * (len(batch_prompts) - len(responses))

            # Step 2: Evaluate predictions
            batch_feedback = generate_batch_feedback(batch_prompts, responses)

            # Step 3: Self-refine the batch
            refined_responses = batch_self_refine(batch_prompts, responses, batch_feedback,intention_set)

            try:
                refined_responses = clean_intents(refined_responses)
            except:
                refined_responses = clean_intents(refined_responses)
            # print(cleaned)

            if len(refined_responses) < len(batch_prompts):
                # refined_responses += ["Unknown"] * (len(batch_prompts) - len(refined_responses))
                refined_responses=responses


            batch_feedback1 = generate_batch_feedback(batch_prompts, refined_responses)

            # Step 3: Self-refine the batch
            refined_responses2 = batch_self_refine(batch_prompts, refined_responses, batch_feedback1, intention_set)

            # Step 4: Clean singleton sets (if model outputs '{...}' format)
            # refined_responses = clean_singleton_sets(refined_responses)
            
            try:
                # refined_responses2 = clean_numbered_singleton_sets(refined_responses2)
                refined_responses2 = clean_intents(refined_responses2)

            except:
                # refined_responses2 = clean_singleton_sets(refined_responses2)
                refined_responses2 = clean_intents(refined_responses2)

            if len(refined_responses2) < len(batch_prompts):
                # refined_responses2 += ["Unknown"] * (len(batch_prompts) - len(refined_responses2))
                refined_responses2=refined_responses

            # Store results
            for p, pred, fb, true, refined in zip(batch_prompts, responses, batch_feedback1, batch_true, refined_responses2):
                all_results.append({
                    "prompt": p,
                    "predicted_intention": pred,
                    "true_intention": true,
                    "feedback": fb,
                    "temperature": temperature,
                    "refined_response": refined
                })

        # Save all results
        file_name = os.path.join(save_dir, f"response_batch_{temperature}_feedback.json")
        with open(file_name, "w") as f:
            json.dump(all_results, f, indent=4)

        return {
            "responses": [[r["refined_response"]] for r in all_results],
            "file_path": file_name
        }



    def batch_agent_feedback(
        prompts: List[str],
        intention_set: List[str],
        true_intentions: List[str],
        temperature: float,
        output_format: Dict[str, str],
        batch_size: int = 1
    ) -> Dict[str, Union[List[str], str]]:
        """Process prompts through prediction, evaluation, and refinement. Ensures no empty set returned."""
        all_results = []
        save_dir = f".../refinement_responses_meta_{dataset_name}_0.5"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_true = true_intentions[i:i + batch_size]

            # Initial prediction: use true_intentions as placeholder
            responses = batch_true.copy()
            if len(responses) < len(batch_prompts):
                responses += ["Unknown"] * (len(batch_prompts) - len(responses))

            # Step 1: Generate feedback
            batch_feedback = generate_batch_feedback(batch_prompts, responses,intention_set)
            if not batch_feedback or len(batch_feedback) < len(batch_prompts):
                batch_feedback = ["Unknown"] * len(batch_prompts)

            # Step 2: First refinement
            refined_responses = batch_self_refine(batch_prompts, responses, batch_feedback, intention_set)
            if not refined_responses or len(refined_responses) < len(batch_prompts):
                refined_responses = responses

            try:
                refined_responses = clean_intents(refined_responses)
            except:
                refined_responses = responses

            if not refined_responses or len(refined_responses) < len(batch_prompts):
                refined_responses = responses

            # Step 3: Second round of feedback and refinement
            batch_feedback1 = generate_batch_feedback(batch_prompts, refined_responses,intention_set)
            if not batch_feedback1 or len(batch_feedback1) < len(batch_prompts):
                batch_feedback1 = ["Unknown"] * len(batch_prompts)

            refined_responses2 = batch_self_refine(batch_prompts, refined_responses, batch_feedback1, intention_set)
            
            
            try:
                refined_responses2 = clean_intents(refined_responses2)
            except:
                refined_responses2 = refined_responses

            if not refined_responses2 or len(refined_responses2) < len(batch_prompts):
                refined_responses2 = refined_responses

            # Final fallback: Ensure no empty values
            if not refined_responses2 or any(not r for r in refined_responses2):
                refined_responses2 = ["Unknown"] * len(batch_prompts)

            # Store results
            for p, pred, fb, true, refined in zip(batch_prompts, responses, batch_feedback1, batch_true, refined_responses2):
                all_results.append({
                    "prompt": p,
                    "predicted_intention": pred,
                    "true_intention": true,
                    "feedback": fb,
                    "temperature": temperature,
                    "refined_response": refined
                })



        file_name = os.path.join(save_dir, f"response_batch_{temperature}_{dataset_name}_feedback.json")
        with open(file_name, "w") as f:
            json.dump(all_results, f, indent=4)

        return {
            "responses": [[r["refined_response"]] for r in all_results],
            "file_path": file_name
        }


    response_file_o=[]

    def agent111(prompts, intention_set,true_intentions1, temperatures, formats):
        for prompt in prompts:
            combined_response = f'{prompt}'
            instruction= f' Considering the following query "{combined_response}", must identifing the most likely single intention only from the intention set \n{true_intentions1}. Ensure each response strictly adheres to the prescribed format.'
        combined_response = instruction
        combined_response += f'\n\nProvide each response in the specified format: {formats}. Do not include multiple outputs and input or explanatory text.'
        response = openai.ChatCompletion.create(
                model="deepseek-chat",
            messages=[{"role": "user", "content": combined_response}],
            temperature=temperatures,
            max_tokens=30*len(prompts)
        )
        file_name = f".../Transition_Matrix/stackexchange/response_one_shot_{temperatures}_consistentssss.json"
        response_file_o.append({"prompt": combined_response, "response": response})
        save_responses_to_file(response_file_o, file_name)
        return response.choices[0].message.content


    response_file_o=[]
    def agent1111(prompts, intention_set,most_common_response,true_intentions1, temperatures, formats):
        for prompt in prompts:
            combined_response = f'{prompt}'
            instruction= f' Considering the following query "{combined_response}" and its most confident intent "{most_common_response}", must identifing the most likely single intention only from the intention set \n{true_intentions1}. Ensure each response strictly adheres to the prescribed format.'
        combined_response = instruction
        combined_response += f'\n\nProvide each response in the specified format: {formats}. Do not include multiple outputs and input or explanatory text.'
        response = openai.ChatCompletion.create(
                model="deepseek-chat",
            messages=[{"role": "user", "content": combined_response}],
            temperature=temperatures,
            max_tokens=30*len(prompts)
        )
        file_name = f".../Transition_Matrix/stackexchange/response_one_shot_{temperatures}_consistentssss.json"
        response_file_o.append({"prompt": combined_response, "response": response})
        save_responses_to_file(response_file_o, file_name)
        return response.choices[0].message.content


    response_file_self_correction=[]
    def agent2222(prompts, intention_set,true_intentions1, temperatures, formats):
        for prompt in prompts:
            combined_response = f'{prompt}'
            instruction= f' Considering the following query "{combined_response}" and its most confident intent "{true_intentions1}", must identifing the most likely single intention only from the intention set \n{intention_set}. Ensure each response strictly adheres to the prescribed format.'
        combined_response = instruction
        combined_response += f'\n\nProvide each response in the specified format: {formats}, {formats1},{formats2}. Do not include multiple outputs and input or explanatory text.'
        response = openai.ChatCompletion.create(
                model="deepseek-chat",
            messages=[{"role": "user", "content": combined_response}],
            temperature=temperatures,
            max_tokens=30*len(prompts)
        )
        file_name = f".../Transition_Matrix/stackexchange/response_one_shot_{temperatures}_self_correction.json"
        response_file_self_correction.append({"prompt": combined_response, "response": response})
        save_responses_to_file(response_file_self_correction, file_name)
        return response.choices[0].message.content

    response_file_o=[]
    def agent111111(prompts, intention_set,true_intentions1, temperatures, formats):
        for prompt in prompts:
            combined_response = f'{prompt}'
            instruction= f' Considering the following query "{combined_response}", must identifing the most likely single intention only from the intention set \n{true_intentions1}. Ensure each response strictly adheres to the prescribed format.'
        combined_response = instruction
        combined_response += f'\n\nProvide each response in the specified format: {formats}. Do not include multiple outputs and input or explanatory text.'
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": combined_response}],
            temperature=temperatures,
            max_tokens=30*len(prompts)
        )
        file_name = f".../Transition_Matrix/stackexchange/response_one_shot_{temperatures}_consistentssss.json"
        response_file_o.append({"prompt": combined_response, "response": response})
        save_responses_to_file(response_file_o, file_name)
        return response.choices[0].message.content



    response_file_o=[]
    def agent1111111(prompts, intention_set,true_intentions1, temperatures, formats):
        for prompt in prompts:
            combined_response = f'{prompt}'
            instruction= f' Considering the following query "{combined_response}", must identifing the most likely single intention only from the intention set \n{true_intentions1}. Ensure each response strictly adheres to the prescribed format.'
        combined_response = instruction
        combined_response += f'\n\nProvide each response in the specified format: {formats}. Do not include multiple outputs and input or explanatory text.'
        response = openai.ChatCompletion.create(
                model="deepseek-chat",
            messages=[{"role": "user", "content": combined_response}],
            temperature=temperatures,
            max_tokens=30*len(prompts)
        )
        file_name = f".../Transition_Matrix/stackexchange/response_one_shot_{temperatures}_consistentssss.json"
        response_file_o.append({"prompt": combined_response, "response": response})
        save_responses_to_file(response_file_o, file_name)
        return response.choices[0].message.content


    def batch_map_intents(predicted_intentions_list: List[List[str]], 
                        loaded_intent_table_inverse: Dict[str, str]) -> List[List[str]]:
        """
        Batch maps predicted intentions using inverse intent table with error handling
        
        Args:
            predicted_intentions_list: List of predicted intention lists (each sublist contains multiple intents)
            loaded_intent_table_inverse: Mapping dictionary for intent conversion
            
        Returns:
            List of mapped intention lists (preserves original structure)
        """
        batch_mapped = []
        
        for predicted_intentions in predicted_intentions_list:
            # Handle each sublist of predicted intentions
            mapped_sublist = []
            for intent in predicted_intentions:  # Process all intents in the sublist
                try:
                    mapped = loaded_intent_table_inverse[intent]
                except KeyError:
                    mapped = loaded_intent_table_inverse.get(intent, intent)
                mapped_sublist.append(mapped)
            batch_mapped.append(mapped_sublist)
        
        return batch_mapped


    import json
    from typing import List, Dict, Tuple  # Added Tuple import

    def agent11_batch(
        prompts: List[str],
        intention_set: List[str],
        true_intentions: List[List[str]],  # List of intent lists for each prompt
        temperatures: float,
        formats: str,
        # dataset_name: str,
        batch_size: int = 10
        ) -> Tuple[List[str], str]:
        """
        Batch-processes multiple prompts to identify intents from a set
        
        Args:
            prompts: List of input queries
            intention_set: Available intent labels
            true_intentions: Reference intents for each prompt
            temperatures: LLM temperature parameter
            formats: Required output format
            dataset_name: For saving results
            batch_size: Number of prompts per API call
            
        Returns:
            Tuple of (list of predicted intents, file path of saved results)
        """


        for prompt in prompts:
            combined_response = f'{prompt}'
            instruction= f' Considering the following query "{combined_response}", must identifing the most likely single intention only from the intention set \n{intention_set}. Ensure each response strictly adheres to the prescribed format.'
        full_prompt = instruction
        full_prompt += f'\n\nProvide each response in the specified format: {formats},{formats1},{formats2}. Do not include multiple outputs and input or explanatory text.'
        full_prompt += f'\n\n Please ensure there are a total of one responses.'

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
            max_new_tokens=60,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperatures,
            top_p=0.9,
        )

        response11 = outputs[0][input_ids.shape[-1]:]
        # print(tokenizer.decode(response11, skip_special_tokens=True))
        response = tokenizer.decode(response11, skip_special_tokens=True)

        return response


    def agent11_batch_feedback(
        prompts: List[str],
        intention_set: List[str],
        true_intentions: List[List[str]],  # List of intent lists for each prompt
        temperatures: float,
        formats: str,
        # dataset_name: str,
        batch_size: int = 1
        ) -> Tuple[List[str], str]:
        """
        Batch-processes multiple prompts to identify intents from a set
        
        Args:
            prompts: List of input queries
            intention_set: Available intent labels
            true_intentions: Reference intents for each prompt
            temperatures: LLM temperature parameter
            formats: Required output format
            dataset_name: For saving results
            batch_size: Number of prompts per API call
            
        Returns:
            Tuple of (list of predicted intents, file path of saved results)
        """
        response_file_o = []
        all_responses = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_intents = true_intentions[i:i+batch_size]
            
            # Create batch instruction
            batch_instruction = f"""
            For each of these {len(batch_prompts)} queries, identify the SINGLE most likely intention 
            from: {intention_set}
            
            Queries:
            {chr(10).join(f'{j+1}. {p}' for j, p in enumerate(batch_prompts))}
            
            Requirements:
            1. Provide JUST the intent for each query in this format: {formats}
            2. Return exactly {len(batch_prompts)} responses, one per line
            """
            
            # API call
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                # model="deepseek-chat",
                messages=[{"role": "user", "content": batch_instruction}],
                temperature=temperatures,
                max_tokens=20
            )
            
            # Process responses
            batch_responses = [
                choice.message.content.strip() 
                for choice in response.choices
            ]
            
            # Handle cases where response combines multiple intents
            if len(batch_responses) == 1:
                batch_responses = [
                    r.strip() for r in 
                    batch_responses[0].split('\n')[:len(batch_prompts)]
                    if r.strip()
                ]
            
            all_responses.extend(batch_responses)
            
            # Save batch results
            response_file_o.append({
                "prompts": batch_prompts,
                "true_intentions": batch_intents,
                "responses": batch_responses,
                "temperature": temperatures
            })
        
        # Save all results
        # file_name = f"./Transition_Matrix/{dataset_name}/{dataset_name}_response_few_shot_{temperatures}_agent11.json"
        # save_responses_to_file(response_file_o, file_name)
        
        return all_responses

    def agent111_batch(
        prompts: List[str],
        intention_set: List[str],
        true_intentions: List[List[str]],  # List of intent lists for each prompt
        temperatures: float,
        formats: str,
        # dataset_name: str,
        batch_size: int = 10
        ) -> Tuple[List[str], str]:
        """
        Batch-processes multiple prompts to identify intents from a set
        
        Args:
            prompts: List of input queries
            intention_set: Available intent labels
            true_intentions: Reference intents for each prompt
            temperatures: LLM temperature parameter
            formats: Required output format
            dataset_name: For saving results
            batch_size: Number of prompts per API call
            
        Returns:
            Tuple of (list of predicted intents, file path of saved results)
        """
        response_file_o = []
        all_responses = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_intents = true_intentions[i:i+batch_size]
            
            # Create batch instruction
            batch_instruction = f"""
            For each of these {len(batch_prompts)} queries, identify the SINGLE most likely intention 
            from: {batch_intents}
            
            Queries:
            {chr(10).join(f'{j+1}. {p}' for j, p in enumerate(batch_prompts))}
            
            Requirements:
            1. Strictly use only these intentions: {batch_intents}
            2. Provide JUST the intent for each query in this format: {formats}
            3. Return exactly {len(batch_prompts)} responses, one per line
            """
            
            # API call
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": batch_instruction}],
                temperature=temperatures,
                max_tokens=20
            )
            
            # Process responses
            batch_responses = [
                choice.message.content.strip() 
                for choice in response.choices
            ]
            
            # Handle cases where response combines multiple intents
            if len(batch_responses) == 1:
                batch_responses = [
                    r.strip() for r in 
                    batch_responses[0].split('\n')[:len(batch_prompts)]
                    if r.strip()
                ]
            
            all_responses.extend(batch_responses)
            
            # Save batch results
            response_file_o.append({
                "prompts": batch_prompts,
                "true_intentions": batch_intents,
                "responses": batch_responses,
                "temperature": temperatures
            })
        
        # # Save all results
        # file_name = f"./Transition_Matrix/{dataset_name}/{dataset_name}_response_few_shot_{temperatures}_agent11.json"
        # save_responses_to_file(response_file_o, file_name)
        
        return all_responses
    



    def reprocess_single_intents(prompts, intention_set, temperatures, formats, 
                                c_answer1_vanillas, new_true_intentions5):
        """
        Reprocess prompts where new_true_intentions5 has >1 intent using agent11_batch
        
        Args:
            prompts: List of original prompts
            intention_set: Available intents
            temperatures: Temperature parameter
            formats: Output format
            c_answer1_vanillas: Original responses
            new_true_intentions5: New intent candidates
            
        Returns:
            Updated responses with reprocessed prompts where needed
        """

            
        reprocess_indices = []
        for i, intents in enumerate(new_true_intentions5):
            try:
                if isinstance(intents[0], list) and len(intents[0]) > 1:
                    reprocess_indices.append(i)
                elif isinstance(intents, list) and len(intents) > 1:
                    reprocess_indices.append(i)
            except Exception as e:
                print(f"Index {i} caused error: {e}")

        
        if not reprocess_indices:
            return c_answer1_vanillas  # No changes needed
        

        valid_reprocess_indices = [i for i in reprocess_indices if i < len(prompts) and i < len(new_true_intentions5)]
        reprocess_prompts = [prompts[i] for i in valid_reprocess_indices]
        reprocess_intents = [new_true_intentions5[i] for i in valid_reprocess_indices]

        # Get new responses for these prompts
        new_responses = agent111_batch(
            prompts=reprocess_prompts,
            intention_set=reprocess_intents,
            true_intentions=reprocess_intents,
            temperatures=temperatures,
            formats=formats
        )
        
        # Update the original responses
        updated_responses = c_answer1_vanillas.copy()
        for idx, new_resp in zip(reprocess_indices, new_responses):
            updated_responses[idx] = new_resp
        
        return updated_responses


    def reprocess_single_intents(prompts, intention_set, temperatures, formats, 
                                c_answer1_vanillas, new_true_intentions5):
        """
        Reprocess prompts where new_true_intentions5 has >1 intent using agent11_batch
        
        Args:
            prompts: List of original prompts
            intention_set: Available intents
            temperatures: Temperature parameter
            formats: Output format
            c_answer1_vanillas: Original responses
            new_true_intentions5: New intent candidates
            
        Returns:
            Updated responses with reprocessed prompts where needed (same length as c_answer1_vanillas)
        """
        # Identify which samples have multiple intent candidates
        reprocess_indices = []
        for i, intents in enumerate(new_true_intentions5):
            try:
                if isinstance(intents[0], list) and len(intents[0]) > 1:
                    reprocess_indices.append(i)
                elif isinstance(intents, list) and len(intents) > 1:
                    reprocess_indices.append(i)
            except Exception as e:
                print(f"Index {i} caused error: {e}")


        reprocess_prompts = prompts
        reprocess_intents = new_true_intentions5

        # Call your model or function to get reprocessed answers
        new_responses = agent11_batch(
            prompts=reprocess_prompts,
            intention_set=reprocess_intents,
            true_intentions=reprocess_intents,
            temperatures=temperatures,
            formats=formats
        )

        # Replace only the selected indices in a copied list
        updated_responses = new_responses

        return updated_responses


    def reprocess_single_intents(prompts, intention_set, temperatures, formats, 
                                c_answer1_vanillas, new_true_intentions5):
        """
        Reprocess prompts where new_true_intentions5 has >1 intent using agent11_batch
        
        Args:
            prompts: List of original prompts
            intention_set: Available intents
            temperatures: Temperature parameter
            formats: Output format
            c_answer1_vanillas: Original responses
            new_true_intentions5: New intent candidates
            
        Returns:
            Updated responses with reprocessed prompts where needed (same length as c_answer1_vanillas)
        """

        reprocess_prompts = prompts
        reprocess_intents = new_true_intentions5

        new_responses = agent11_batch(
            prompts=reprocess_prompts,
            intention_set=reprocess_intents,
            true_intentions=reprocess_intents,
            temperatures=0.7,
            formats=formats
        )

        updated_responses=new_responses

        return updated_responses


    import json
    import re

    def clean_agent_responses(responses):
        """
        Extract just the intent values from responses in format "{'intent': 'value'}"
        
        Args:
            responses: List of responses in format ["{'intent': 'value'}", ...]
            
        Returns:
            List of cleaned intent values
        """
        cleaned = []
        for resp in responses:
            # Method 1: Using json.loads (most reliable)
            try:
                intent_dict = json.loads(resp.replace("'", '"'))
                cleaned.append(intent_dict['intent'])
            except:
                # Method 2: Fallback to regex if JSON parsing fails
                match = re.search(r"'intent':\s*'([^']+)'", resp)
                if match:
                    cleaned.append(match.group(1))
                else:
                    # Keep original if no match found
                    cleaned.append(resp)
        return cleaned


    def clean_agent_responses(responses):
        """
        Extract just the intent values from responses in format "{'intent': 'value'}"
        
        Args:
            responses: List of responses in format ["{'intent': 'value'}", ...]
            
        Returns:
            List of cleaned intent values
        """
        try:
            cleaned = []
            for resp in responses:
                # Method 1: Using json.loads (most reliable)
                try:
                    intent_dict = json.loads(resp.replace("'", '"'))
                    cleaned.append(intent_dict['intent'])
                except:
                    # Method 2: Fallback to regex if JSON parsing fails
                    match = re.search(r"'intent':\s*'([^']+)'", resp)
                    if match:
                        cleaned.append(match.group(1))
                    else:
                        # Keep original if no match found
                        cleaned.append(resp)
        
        except:
            cleaned = []
            for resp in responses:
                # Method 1: Using json.loads (most reliable)
                try:
                    intent_dict = json.loads(resp[0].replace("'", '"'))
                    cleaned.append(intent_dict['intent'])
                except:
                    # Method 2: Fallback to regex if JSON parsing fails
                    match = re.search(r"'intent':\s*'([^']+)'", resp[0])
                    if match:
                        cleaned.append(match.group(1))
                    else:
                        # Keep original if no match found
                        cleaned.append(resp[0])
                
        return cleaned

    import re
    import ast

    def normalize_response(item):
        """Convert various formats to list of strings like ['pending transfer', 'verify top up']"""
        if isinstance(item, list):
            results = []
            for i in item:
                if isinstance(i, str):
                    # Handle "{'intent'}" or "1. {'intent'}" cases
                    matches = re.findall(r"{(.+?)}", i)
                    if matches:
                        try:
                            intent_dict = ast.literal_eval("{" + matches[0] + "}")
                            results.extend(list(intent_dict))
                        except:
                            results.append(i.strip())
                    elif i.strip() not in ['{', '}']:
                        results.append(i.strip())
                else:
                    results.append(i)
            return results

        elif isinstance(item, str):
            # Handle multiline string like: "1. {'intent'}\n2. {'intent'}"
            lines = item.strip().split("\n")
            results = []
            for line in lines:
                matches = re.findall(r"{(.+?)}", line)
                if matches:
                    try:
                        intent_dict = ast.literal_eval("{" + matches[0] + "}")
                        results.extend(list(intent_dict))
                    except:
                        results.append(line.strip())
                else:
                    results.append(line.strip())
            return results

        else:
            return []

    def safe_fix_predicted_intentions(predicted_intentions_raw):
        fixed = []
        for intent_str_list in predicted_intentions_raw:
            if intent_str_list:  # non-empty match
                try:
                    parsed = ast.literal_eval("{" + intent_str_list[0] + "}")
                    fixed.append(list(parsed))
                except Exception as e:
                    print(f"[Warning] Failed to parse intent: {intent_str_list[0]} — {e}")
                    fixed.append(["Unknown"])
            else:
                fixed.append(["Unknown"])  # fallback if no match found
        return fixed

    def save_responses_to_file(data: List[Dict], file_path: str) -> None:
        """Saves response data to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def safe_get(pred_list, idx, default="Unknown"):
        try:
            item = pred_list[idx]
            if isinstance(item, list):
                return item[0] if item else default
            elif isinstance(item, str):
                return item
            else:
                return default
        except (IndexError, TypeError):
            return default


    def fix_and_print_list_lengths(lists_dict: Dict[str, list], prompts_key: str = "prompts") -> None:
        """
        Check and fix lists inside lists_dict by padding 'Unknown' if shorter than prompts.
        Print all lengths nicely.
        """
        prompts_length = len(lists_dict[prompts_key])
        
        
        for name, lst in lists_dict.items():
            length = len(lst)
            flag = ""
            if name != prompts_key and length < prompts_length:
                missing = prompts_length - length
                lst += ["Unknown"] * missing
                flag = f"⚠️ Added {missing} 'Unknown'"
            print(f"{name:35} | {len(lst):7} | {flag}")


    def safe_batch_map(preds, inverse_table):
        mapped = batch_map_intents(preds, inverse_table)
        
        # Check if mapped is empty or contains only empty sublists
        if not mapped or all(not sublist for sublist in mapped):
            if not preds or all(not sub for sub in preds):
                return [["Unknown"]]
            return preds

        return mapped


    def ensure_triple_nested(data):
        """Ensure data is in [[[...]]] format."""
        if not isinstance(data, list):
            return [[[data]]]
        if not data or not isinstance(data[0], list):
            return [[data]]
        if not isinstance(data[0][0], list):
            return [data]
        return data
    
    
        
    def ensure_double_nested(data):
        """Ensure data is in [[...]] format."""
        if not isinstance(data, list):
            return [[data]]
        if not data or not isinstance(data[0], list):
            return [data]
        return data

    def normalize_intents(raw_intents):
        normalized_intents = []
        for intent in raw_intents:
            cleaned = intent.strip().lower()
            # First apply normalization mapping
            canonical_intent = intent_normalization.get(cleaned, cleaned)
            # Only keep if it's a known intent
            if canonical_intent in intent_normalization:
                normalized_intents.append(canonical_intent)
            else:
                normalized_intents.append('Unknown')
        return normalized_intents
    def prompting1(prompts,intention_set,true_intentions1, temperatures, formats):
        max_attempts = 3

        c_answer1_vanillas = []
        attempt = 0
        while not c_answer1_vanillas and attempt < max_attempts:
            try:
                c_answer1_vanillas = agent_cot_0(prompts, intention_set, true_intentions1, temperatures, formats, formats1, formats2, formats3)
            except Exception as e:
                print(f"[agent_cot_0] Error on attempt {attempt + 1}: {e}")
                c_answer1_vanillas = []

            attempt += 1

            # After 5 failed attempts, stop retrying and insert "Unknown"
            if not c_answer1_vanillas and attempt == 3:
                print("[agent_cot_0] Still empty after 5 attempts. Appending 'Unknown' and exiting.")
                c_answer1_vanillas = ['Unknown']
                break



        # Assign to tot
        c_answer1_tot = c_answer1_vanillas

        # --- Retry agent_cot ---
        c_answer1_cot = []
        attempt = 0
        while not c_answer1_cot and attempt < max_attempts:
            try:
                c_answer1_cot = agent_cot(prompts, intention_set, true_intentions1, temperatures, formats, formats1, formats2, formats3)
            except Exception as e:
                print(f"[agent_cot] Error on attempt {attempt+1}: {e}")
                c_answer1_cot = []
            attempt += 1

            if not c_answer1_cot and attempt == 3:
                print("[agent_cot_0] Still empty after 5 attempts. Appending 'Unknown' and exiting.")
                c_answer1_cot = ['Unknown']
        # --- Retry agent_fot ---
        c_answer1_fot = []
        attempt = 0
        while not c_answer1_fot and attempt < max_attempts:
            try:
                c_answer1_fot = agent_fot(prompts, intention_set, true_intentions1, temperatures, formats, formats1, formats2, formats3, dataset_name)
            except Exception as e:
                print(f"[agent_fot] Error on attempt {attempt+1}: {e}")
                c_answer1_fot = []
            attempt += 1

            if not c_answer1_fot and attempt == 3:
                print("[agent_cot_0] Still empty after 5 attempts. Appending 'Unknown' and exiting.")
                c_answer1_fot = ['Unknown']

                    
        def clean_intents(intent_list: List[str]) -> List[str]:
            cleaned = []
            intent=intent_list
            if isinstance(intent, list):
                intent = intent[0]
            # Remove common unwanted characters
            intent = intent.strip().strip("{}").strip("[]").strip("'").strip('"')
            cleaned.append(intent)
            return cleaned


        # Chain of thought
        try:
            predicted_intentions_cot = clean_intents(c_answer1_cot)

        except:
            predicted_intentions_cot = re.findall(r"{(.+?)}", c_answer1_cot)                   
            predicted_intentions_cot = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions_cot]
            predicted_intentions_cot = [list(intent_dict) for intent_dict in predicted_intentions_cot]
        # Few shot of thought
        
        try:
            predicted_intentions_fot = clean_intents(c_answer1_fot)
        except:
            # predicted_intentions_fot = clean_intents(c_answer1_fot)
            predicted_intentions_fot = re.findall(r"{(.+?)}", c_answer1_fot)                   
            predicted_intentions_fot = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions_fot]
            predicted_intentions_fot = [list(intent_dict) for intent_dict in predicted_intentions_fot]

        # Zero shot of thought
        try:
            predicted_intentions = clean_intents(c_answer1_vanillas)
        except:
            predicted_intentions = re.findall(r"{(.+?)}", c_answer1_vanillas)                   
            predicted_intentions = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions]
            predicted_intentions = [list(intent_dict) for intent_dict in predicted_intentions]


        # Refined
        c_answer1_feedback = batch_agent_feedback(
            prompts=prompts,
            intention_set=intention_set,
            true_intentions=predicted_intentions,
            temperature=temperatures,
            output_format=formats
        )
                
        c_answer1_feedback11 = c_answer1_feedback['responses']
        predicted_intentions_feedback=c_answer1_feedback11
        
        
        
        

        import re

        def clean_intents(raw_list):
            clean_list = []
            for item in raw_list:
                if not item or item.strip() == "{}":
                    continue

                # Extract text inside curly braces, single quotes, or double quotes
                match = re.findall(r"[{'\"]?([\w\s\.\-]+)[}'\"]?", item)
                if match:
                    clean_list.append(match[0].strip())
            return clean_list
        
        try:
            predicted_intentions_feedback=clean_intents(predicted_intentions_feedback)
        except:
            predicted_intentions_feedback=clean_intents(predicted_intentions_feedback[0])

        # predicted_intentions_feedback = ensure_double_nested(predicted_intentions_feedback)

        print("predicted_intentions_feedback: ", predicted_intentions_feedback)
        def fix_fragmented_intents(nested_intent_list):
            fixed = []
            for intent_group in nested_intent_list:
                if all(isinstance(char, str) and len(char) == 1 for char in intent_group):
                    # Looks like list of characters → join into one word
                    fixed.append([''.join(intent_group).strip()])
                else:
                    fixed.append(intent_group)
            return fixed

        predicted_intentions_feedback = normalize_intents(predicted_intentions_feedback)

        new_true_intentions_feedback = safe_batch_map([predicted_intentions_feedback], loaded_intent_table_inverse)

        new_true_intentions_feedback = fix_fragmented_intents(new_true_intentions_feedback)

        new_true_intentions_feedback = ensure_triple_nested(new_true_intentions_feedback)


        print(f"new_true_intentions_feedback: {new_true_intentions_feedback}")
        print(f"new_true_intentions_feedback [0][0]: {len(new_true_intentions_feedback[0][0])}")
        print(f"new_true_intentions_feedback [0]: {len(new_true_intentions_feedback[0])}")


        if (len(new_true_intentions_feedback[0][0]) > 1):
            c_answer1_our_feedback = reprocess_single_intents(
                prompts=prompts,
                intention_set=intention_set,
                temperatures=temperatures,
                formats=formats,
                c_answer1_vanillas=predicted_intentions_feedback,
                new_true_intentions5=new_true_intentions_feedback[0][0]
            )
        else:
            # fallback if no multiple intents
            c_answer1_our_feedback = predicted_intentions_feedback


        def safe_get(pred_list, idx, default="Unknown"):
            """
            Safely get item at index idx from pred_list.
            If not available, return default.
            """
            try:
                item = pred_list[idx]
                if isinstance(item, list):
                    return item[0] if item else default
                elif isinstance(item, str):
                    return item
                else:
                    return default
            except (IndexError, TypeError):
                return default

        consistent_samples_set = []

        def safe_format(entry):
            """Ensure the entry is a single string, even if it's nested or malformed."""
            if isinstance(entry, list):
                if entry and isinstance(entry[0], str):
                    return entry[0]  # unwrap single-item list
                return "Unknown"
            elif isinstance(entry, str):
                return entry
            else:
                return "Unknown"
        

        predicted_intentions = clean_intents(predicted_intentions)
        predicted_intentions_cot = clean_intents(predicted_intentions_cot)
        predicted_intentions_fot = clean_intents(predicted_intentions_fot)
        # consistent_samples_set = clean_intents(consistent_samples_set)


        # Accumulate consistent samples
        for i in range(len(prompts)):
            all_c_answers = [
                safe_format(safe_get(predicted_intentions, i)),
                # safe_format(safe_get(predicted_intentions_feedback, i)),
                safe_format(safe_get(predicted_intentions_fot, i)),
                safe_format(safe_get(predicted_intentions_cot, i)),
            ]
            most_common_response = get_most_common_response(all_c_answers)
            consistent_samples_set.append(most_common_response)


                
        import re

        # intent_normalization = {intent: intent for intent in intent_to_index_mapping}

        predicted_intentions = normalize_intents(predicted_intentions)
        predicted_intentions_cot = normalize_intents(predicted_intentions_cot)
        predicted_intentions_fot = normalize_intents(predicted_intentions_fot)
        consistent_samples_set = normalize_intents(consistent_samples_set)


        new_true_intentions1 = safe_batch_map([predicted_intentions], loaded_intent_table_inverse)
        new_true_intentions2 = safe_batch_map([predicted_intentions_cot], loaded_intent_table_inverse)
        new_true_intentions3 = safe_batch_map([predicted_intentions_fot], loaded_intent_table_inverse)
        new_true_intentions5 = safe_batch_map([consistent_samples_set], loaded_intent_table_inverse)



        new_true_intentions1 = ensure_triple_nested(new_true_intentions1)
        new_true_intentions2 = ensure_triple_nested(new_true_intentions2)
        new_true_intentions3 = ensure_triple_nested(new_true_intentions3)
        new_true_intentions5 = ensure_triple_nested(new_true_intentions5)



        if len(new_true_intentions1[0][0])>1:

            c_answer1 = reprocess_single_intents(
                prompts=prompts,
                intention_set=intention_set,
                temperatures=temperatures,
                formats=formats,
                c_answer1_vanillas=predicted_intentions,
                new_true_intentions5=new_true_intentions1[0][0]
            )
        else:
            c_answer1 = predicted_intentions
        
        
        # c_answer1 = c_answer1_vanillas
        if len(new_true_intentions2[0][0])>1:
            c_answer1_our_cot = reprocess_single_intents(
                prompts=prompts,
                intention_set=intention_set,
                temperatures=temperatures,
                formats=formats,
                c_answer1_vanillas=predicted_intentions_cot,
                new_true_intentions5=new_true_intentions2[0][0]
            )
        else:
            c_answer1_our_cot = predicted_intentions_cot
            
        # c_answer1_our_cot =predicted_intentions_cot
        
        if len(new_true_intentions3[0][0])>1:
            c_answer1_our_fot = reprocess_single_intents(
                prompts=prompts,
                intention_set=intention_set,
                temperatures=temperatures,
                formats=formats,
                c_answer1_vanillas=predicted_intentions_fot,
                new_true_intentions5=new_true_intentions3[0][0]
            )
        else:
            c_answer1_our_fot = predicted_intentions_fot

        c_answer1_our_tot=c_answer1_our_fot
        c_answer1_our_tot=predicted_intentions_fot
        
        

        if len(new_true_intentions5[0][0])>1:
            consistent_response = reprocess_single_intents(
                prompts=prompts,
                intention_set=intention_set,
                temperatures=temperatures,
                formats=formats,
                c_answer1_vanillas=consistent_samples_set,
                new_true_intentions5=new_true_intentions5[0][0]
            )
        else:   
            consistent_response = consistent_samples_set
            

        based_consistent_response=consistent_samples_set
        
        
        # Example normalization
        c_answer1_our_cot = normalize_response(c_answer1_our_cot)
        c_answer1_cot = normalize_response(c_answer1_cot)
        
        
        c_answer1_our_fot = normalize_response(c_answer1_our_fot)
        c_answer1_fot = normalize_response(c_answer1_fot)
        
        
        c_answer1_our_tot = normalize_response(c_answer1_our_tot)
        c_answer1_tot = normalize_response(c_answer1_tot)
        
        
        c_answer1 = normalize_response(c_answer1)
        c_answer1_vanillas = normalize_response(c_answer1_vanillas)
        
        
        consistent_response = normalize_response(consistent_response)
        based_consistent_response = normalize_response(based_consistent_response)
        
        c_answer1_feedback = normalize_response(c_answer1_feedback11)
        c_answer1_our_feedback = normalize_response(c_answer1_our_feedback)
        

        lists_dict = {
            "c_answer1_our_cot": c_answer1_our_cot,
            "c_answer1_cot": c_answer1_cot,
            "c_answer1_our_fot": c_answer1_our_fot,
            "c_answer1_fot": c_answer1_fot,
            "c_answer1_our_tot": c_answer1_our_tot,
            "c_answer1_tot": c_answer1_tot,
            "c_answer1": c_answer1,
            "c_answer1_vanillas": c_answer1_vanillas,
            "consistent_response": consistent_response,
            "based_consistent_response": based_consistent_response,
            "c_answer1_feedback": c_answer1_feedback,
            "c_answer1_our_feedback": c_answer1_our_feedback,
            "prompts": prompts
        }



        import re

        def to_single_nested_list(data, intent_normalization=None):
            """
            Flatten nested list and normalize intent strings using intent_normalization.
            
            Args:
                data: Input data to flatten and normalize (list or other type)
                intent_normalization (dict, optional): Dictionary for normalizing intent strings.
                    Defaults to None (no normalization).
            
            Returns:
                list: Flattened list with normalized strings
            """
            if intent_normalization is None:
                intent_normalization = {}
            
            cleaned = []

            if not isinstance(data, list):
                data = [data]

            for item in data:
                if isinstance(item, list):
                    # Recursive processing for nested lists
                    cleaned.extend(to_single_nested_list(item, intent_normalization))
                elif isinstance(item, str):
                    item = item.strip()
                    
                    # Try extracting quoted text inside stringified list or dict
                    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", item)
                    if matches:
                        extracted = next((g1 or g2 for g1, g2 in matches), None)
                        intent_candidate = extracted.lower().strip() if extracted else None
                    else:
                        intent_candidate = item.lower().strip()

                    # Normalize using dictionary if provided
                    if intent_normalization:
                        normalized = intent_normalization.get(intent_candidate, 'Unknown')
                        cleaned.append(normalized)
                    else:
                        cleaned.append(intent_candidate if intent_candidate else 'Unknown')
                else:
                    cleaned.append('Unknown')

            return cleaned


        fix_and_print_list_lengths(lists_dict, prompts_key="prompts")


        c_answer1_our_cot = to_single_nested_list(c_answer1_our_cot)
        c_answer1_cot = to_single_nested_list(c_answer1_cot)
        c_answer1_our_fot = to_single_nested_list(c_answer1_our_fot)
        c_answer1_fot = to_single_nested_list(c_answer1_fot)
        c_answer1_our_tot = to_single_nested_list(c_answer1_our_tot)
        c_answer1_tot = to_single_nested_list(c_answer1_tot)
        c_answer1 = to_single_nested_list(c_answer1)
        c_answer1_vanillas = to_single_nested_list(c_answer1_vanillas)
        consistent_response = to_single_nested_list(consistent_response)
        based_consistent_response = to_single_nested_list(based_consistent_response)
        c_answer1_feedback = to_single_nested_list(c_answer1_feedback)
        c_answer1_our_feedback = to_single_nested_list(c_answer1_our_feedback)


        return c_answer1_our_cot,c_answer1_cot,c_answer1_our_fot,c_answer1_fot,c_answer1_our_tot,c_answer1_tot,c_answer1,c_answer1_vanillas,consistent_response,based_consistent_response,c_answer1_feedback,c_answer1_our_feedback


    # ---------------------------------------------------
    # helpers & data-structures (put near top of file)
    # ---------------------------------------------------
    from dataclasses import dataclass
    import re, ast


    def format_nested_list(data):
        # If data is a string, wrap it in a nested list
        if isinstance(data, str):
            return [[data]]
        # If data is a list, check its structure and adjust as necessary
        elif isinstance(data, list):
            # If it's a list of strings, nest each string
            if all(isinstance(item, str) for item in data):
                return [data]
            # If it's already a list of lists, return as is if the sublists contain strings
            elif all(isinstance(item, list) and all(isinstance(subitem, str) for subitem in item) for item in data):
                return data
            # For mixed or deeper nested structures, extract strings and reformat
            else:
                flattened = [subitem for item in data for subitem in (item if isinstance(item, list) else [item])]
                return [flattened]
        # For other types, return an empty nested list
        else:
            return [[]]

    FALLBACK = [["superuser.com.txt"]]          # single place to change

    def ensure_nonempty(value, fallback=FALLBACK):
        """Return *value* if it contains at least one element, else *fallback*."""
        return value if len(value) else fallback


    # ── utilities (add once, e.g. utils.py) ───────────────────────────────
    def normalize_preds(preds: list, step: int) -> list:
        """
        • Returns exactly *step* items.
        • If empty  ➜ fills with 'Error: No predictions'.
        • If short ➜ pads with 'Error: insufficient predictions'.
        • If long  ➜ truncates.
        """
        if not preds:
            return ["Error: No predictions"] * step
        if len(preds) < step:
            preds += ["Error: insufficient predictions"] * (step - len(preds))
        elif len(preds) > step:
            preds = preds[:step]
        return preds

    def clean_bracket_consistency(data):
        cleaned = []
        for item in data:
            # Keep if it's a list with exactly one string
            if isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                cleaned.append(item)
            else:
                # Convert anything else (e.g., [[...]], nested lists) to []
                cleaned.append(item[0])
        return cleaned
    def process_responses1(lines, true_intentions, temperature,step):
        all_responses = defaultdict(dict)


        all_responses_feedback = defaultdict(dict)
        all_responses_self_correction = defaultdict(dict)


        predicted_intentions_list = defaultdict(list)
        valid_predictions = defaultdict(list)
        error_lines = []
        intent_arrays = []
        all_pred_indices = {}
        all_true_indices = {}
        running_master = {}            # survives across iterations

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
        our_all_responses=defaultdict(dict)
        vanilla_all_responses=defaultdict(dict)
        subset_counter1=0
        true_counter1=0
        our_true_counter=0
        vallina_true_counter=0
        

        subset_counter131=0
        our_true_counter1=0
        subset_counter12=0
        subset_counter13=0


        subset_counter1_feedback=0
        true_counter_correction=0



        subset_counter1_feedback_our=0
        true_counter_correction_our=0


        subset_counter1311=0
        our_true_counter11=0
        all_responses1= defaultdict(dict)
        for line_index in range(0, len(lines), step):
            print("total_line_index:",total_line_index)
            print("total_line_index+step:",total_line_index+step)
            print("line_index:",line_index)
            actual_step = step if line_index + step <= len(lines) else len(lines) - line_index
            print("actual_step:",actual_step+line_index)
            selected_lines = lines[total_line_index:total_line_index+step]
            true_intentions1 = true_intentions[total_line_index:total_line_index+step]
            total_line_index += step

            response,c_answer1_cot,response1,c_answer1_fot,our_vanilla_response,c_answer1_tot ,c_answer1_fot11,c_answer1_our_fot_base11,c_answer1_fot111,c_answer1_our_fot_base111,c_answer1_feedback,our_feedback= prompting1(selected_lines,intention_set,true_intentions1,temperature, formats)
            
            c_answer1_our_cot,c_answer1_our_fot,c_answer1_our_tot =response,response1,our_vanilla_response

            c_answer1_our_cot_base,c_answer1_our_fot_base,c_answer1_our_tot_base =c_answer1_cot,c_answer1_fot,c_answer1_tot

            c_answer1_our_fot1,c_answer1_our_fot_base1=c_answer1_fot11,c_answer1_our_fot_base11

            c_answer1_our_con,c_answer1_our_fot_con=c_answer1_fot111,c_answer1_our_fot_base111

            #########################
            our_all_responses[line_index] = c_answer1_our_cot

                
            try:
                our_predicted_intentions_cot = [[intent_dict] for intent_dict in c_answer1_our_cot]


            except:

                our_predicted_intentions_cot = re.findall(r"{(.+?)}", c_answer1_our_cot)
                our_predicted_intentions_cot = [ast.literal_eval("{" + intent_str + "}") for intent_str in our_predicted_intentions_cot]
                our_predicted_intentions_cot = [list(intent_dict) for intent_dict in our_predicted_intentions_cot]
    
    
    
            try:
                our_predicted_intentions_base_cot = [[intent_dict] for intent_dict in c_answer1_our_cot_base]


            except:

                our_predicted_intentions_base_cot = re.findall(r"{(.+?)}", c_answer1_our_cot_base)                   
                our_predicted_intentions_base_cot = [ast.literal_eval("{" + intent_str + "}") for intent_str in our_predicted_intentions_base_cot]
                our_predicted_intentions_base_cot = [list(intent_dict) for intent_dict in our_predicted_intentions_base_cot]



            all_responses[line_index] = c_answer1_our_fot_base1
            our_all_responses[line_index] = c_answer1_our_fot1

            try:
                predicted_intentions_fot1 = [[intent_dict] for intent_dict in c_answer1_our_fot1]

  
            except:
                predicted_intentions_fot1 = re.findall(r"{(.+?)}", c_answer1_our_fot1) 
                predicted_intentions_fot1 = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions_fot1]
                predicted_intentions_fot1 = [list(intent_dict) for intent_dict in predicted_intentions_fot1]



            all_responses[line_index] = c_answer1_our_fot_base1
            
            try:
                predicted_intentions_base_fot1 = [[intent_dict] for intent_dict in c_answer1_our_fot_base1]


            except:

                predicted_intentions_base_fot1 = re.findall(r"{(.+?)}", c_answer1_our_fot_base1)                   
                predicted_intentions_base_fot1 = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions_base_fot1]
                predicted_intentions_base_fot1 = [list(intent_dict) for intent_dict in predicted_intentions_base_fot1]
                

            all_responses[line_index] = c_answer1_our_con

            try:
                
                predicted_intentions_fot11 = [[intent_dict] for intent_dict in c_answer1_our_con]

                
            except:
 
                predicted_intentions_fot11 = [re.findall(r"{(.+?)}", text) for text in c_answer1_our_con]
                
                predicted_intentions_fot11 = [ast.literal_eval("{" + intent_str[0] + "}") for intent_str in predicted_intentions_fot11]
            
                predicted_intentions_fot11 = [list(intent_dict) for intent_dict in predicted_intentions_fot11]
                    



            all_responses[line_index] = c_answer1_our_fot_con
            try:
                predicted_intentions_base_fot11 = [[intent_dict] for intent_dict in c_answer1_our_fot_con]
            except:
                predicted_intentions_base_fot11 = [re.findall(r"{(.+?)}", text) for text in c_answer1_our_fot_con]
                predicted_intentions_base_fot11 = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions_base_fot11]
                predicted_intentions_base_fot11 = [list(intent_dict) for intent_dict in predicted_intentions_base_fot11]
                predicted_intentions_base_fot11 = [c_answer1_our_fot_con]

            
            all_responses[line_index] = c_answer1_our_fot
            
            try:
                predicted_intentions_fot = [[intent_dict] for intent_dict in c_answer1_our_fot]

            except:

                predicted_intentions_fot = [re.findall(r"{(.+?)}", text) for text in c_answer1_our_fot]
                predicted_intentions_fot = [ast.literal_eval("{" + intent_str[0] + "}") for intent_str in predicted_intentions_fot]
                predicted_intentions_fot = [list(intent_dict) for intent_dict in predicted_intentions_fot]
                

            all_responses[line_index] = c_answer1_our_fot_base
            
            try:
                predicted_intentions_base_fot = [[intent_dict] for intent_dict in c_answer1_our_fot_base]

            except:

                predicted_intentions_base_fot = re.findall(r"{(.+?)}", c_answer1_our_fot_base)                   
                predicted_intentions_base_fot = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions_base_fot]
                predicted_intentions_base_fot = [list(intent_dict) for intent_dict in predicted_intentions_base_fot]

            #########################
            all_responses1[line_index] = c_answer1_our_tot
            
            try:
                
                predicted_intentions_tot = [[intent_dict] for intent_dict in c_answer1_our_tot]


            except:
                predicted_intentions_tot = re.findall(r"{(.+?)}", c_answer1_our_tot)                   
                predicted_intentions_tot = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions_tot]
                predicted_intentions_tot = [list(intent_dict) for intent_dict in predicted_intentions_tot]


            all_responses[line_index] = c_answer1_our_tot_base
            
            try:
                predicted_intentions1_base_tot = [[intent_dict] for intent_dict in c_answer1_our_tot_base]


            except:
 
                predicted_intentions1_base_tot = re.findall(r"{(.+?)}", c_answer1_our_tot_base)                   
                predicted_intentions1_base_tot = [ast.literal_eval("{" + intent_str + "}") for intent_str in predicted_intentions1_base_tot]
                predicted_intentions1_base_tot = [list(intent_dict) for intent_dict in predicted_intentions1_base_tot]

            FALLBACK = [["superuser.com.txt"]]          # single place to change

            # --- use the helper ---------------------------------------------------
            predicted_intentions= ensure_nonempty(predicted_intentions_fot)
            predicted_intentions_fot= ensure_nonempty(predicted_intentions_fot)
            predicted_intentions_tot     = ensure_nonempty(predicted_intentions_tot)
            predicted_intentions1_base_tot = ensure_nonempty(predicted_intentions1_base_tot)
            predicted_intentions_base_fot11 = ensure_nonempty(predicted_intentions_base_fot11)
            predicted_intentions_base_fot1  = ensure_nonempty(predicted_intentions_base_fot1)
            predicted_intentions_fot1       = ensure_nonempty(predicted_intentions_fot1)
            our_predicted_intentions_cot    = ensure_nonempty(our_predicted_intentions_cot)
            our_predicted_intentions_base_cot = ensure_nonempty(our_predicted_intentions_base_cot)
            predicted_intentions_base_fot   = ensure_nonempty(predicted_intentions_base_fot)
            predicted_intentions_fot11      = ensure_nonempty(predicted_intentions_fot11)



            def ensure_triple_nested(var):
                if isinstance(var, list):
                    if not var:  # empty list
                        return [[[None]]]
                    elif isinstance(var[0], str):  # ['book flight']
                        return [[var]]
                    elif isinstance(var[0], list) and isinstance(var[0][0], str):  # [['book flight']]
                        return [[var[0]]]
                    elif isinstance(var[0], list) and isinstance(var[0][0], list):  # already [[[...]]]
                        return var
                return [[[var]]]

            # Apply to all variables
            predicted_intentions                 = ensure_triple_nested(predicted_intentions)
            predicted_intentions_fot            = ensure_triple_nested(predicted_intentions_fot)
            predicted_intentions_tot            = ensure_triple_nested(predicted_intentions_tot)
            predicted_intentions1_base_tot      = ensure_triple_nested(predicted_intentions1_base_tot)
            predicted_intentions_base_fot11     = ensure_triple_nested(predicted_intentions_base_fot11)
            predicted_intentions_base_fot1      = ensure_triple_nested(predicted_intentions_base_fot1)
            predicted_intentions_fot1           = ensure_triple_nested(predicted_intentions_fot1)
            our_predicted_intentions_cot        = ensure_triple_nested(our_predicted_intentions_cot)
            our_predicted_intentions_base_cot   = ensure_triple_nested(our_predicted_intentions_base_cot)
            predicted_intentions_base_fot       = ensure_triple_nested(predicted_intentions_base_fot)
            predicted_intentions_fot11          = ensure_triple_nested(predicted_intentions_fot11)


            def save_intent_batches(data: dict,
                                    dataset_name: str,
                                    temperature: float,
                                    total_idx: int,
                                    save_every: int = 5,
                                    master_cache: dict = None):
                """
                • Always updates (and rewrites) a single master JSON.
                • Optionally writes a snapshot file every `save_every` lines.
                """
                base_dir = f".../Transition_Matrix/{dataset_name}"
                os.makedirs(base_dir, exist_ok=True)

                # ----------- 1)  Append to the running master cache ------------
                if master_cache is not None:
                    master_cache[total_idx] = data          # store by global index
                    master_path = f"{base_dir}/all_predicted_intentions_MASTER_{temperature}_{proportion}_meta_new_updated.json"
                    with open(master_path, "w") as f:
                        json.dump(master_cache, f, indent=2)
                    print(f"[✓] Master file updated → {master_path}")
            

            predicted_intentions1=predicted_intentions_fot1
            self_feedback1=c_answer1_feedback

            all_responses_feedback[line_index] = self_feedback1

            prediction_self_feedback1 = self_feedback1

            prediction_self_feedback1 = ensure_triple_nested(prediction_self_feedback1)



            prediction_our=our_feedback
            
        
            all_responses[line_index] = prediction_our

            predicted_intentions_feedback = our_feedback
            predicted_intentions_feedback = ensure_triple_nested(predicted_intentions_feedback)


            intent_batches = {
                "Vanilla"         : ensure_triple_nested(predicted_intentions_base_fot1),
                "Memorable_Vanilla"              : ensure_triple_nested(predicted_intentions_fot1),
                
                "CoT_Baseline"            : ensure_triple_nested(our_predicted_intentions_base_cot),
                "Memorable_CoT"                 : ensure_triple_nested(our_predicted_intentions_cot),

                "FoT_Baseline"            : ensure_triple_nested(predicted_intentions_base_fot),
                "Memorable_FoT"            : ensure_triple_nested(predicted_intentions_fot),

                "Consistent_Baseline"              : ensure_triple_nested(predicted_intentions_fot11),
                "Memorable_Consistent "         : ensure_triple_nested(predicted_intentions_base_fot11),

                "Feedback"                 : ensure_triple_nested(predicted_intentions_feedback),
                "Feedback_Memory"          : ensure_triple_nested(predicted_intentions_feedback),
            }

            save_intent_batches(
                data=intent_batches,       # ← or match_eval_inputs if you prefer
                dataset_name=dataset_name,
                temperature=temperature,
                total_idx=total_line_index,   # global counter
                save_every=1,                # snapshot interval; set to None or 0 to disable snapshots
                master_cache=running_master   # keeps everything
            )



            if len(predicted_intentions) == 0:
                error_messages = ['Error: No predictions'] * step
                predicted_intentions.extend(error_messages)
            
            elif len(predicted_intentions) > step:
                predicted_intentions = predicted_intentions[:step]

            elif len(predicted_intentions) < step:
                error_messages = ['Error: insufficient predictions'] * (step - len(predicted_intentions))
                predicted_intentions.extend(error_messages)

            if len(predicted_intentions) == step:
                print("all_good")
                all_good+=1
            else:
                print("not_good")


            clean_data1 = [predicted_intentions1]
            based_clean_data1 = [predicted_intentions_base_fot1]


            clean_data = [predicted_intentions]
            based_clean_data = [predicted_intentions_base_fot]

            
            our_clean_data_cot=[our_predicted_intentions_cot]

            clean_data_base_cot= [our_predicted_intentions_base_cot] 
            
            consistent_data=[predicted_intentions_fot11]
            based_consistent_data=[predicted_intentions_base_fot11]
            
            cleaned_data = clean_bracket_consistency(clean_data[0])

            true_intentions1 = [[item] for item in true_intentions1]

            pred_indices = intent_to_index(cleaned_data, intent_to_index_mapping)
            true_indices = intent_to_index(true_intentions1, intent_to_index_mapping)

            
            for i, clean in enumerate(clean_data):
                predicted_intentions_list[overall_index] = clean
                try:
                    if "Error" not in clean:  # If no error in the intention
                        all_pred_indices[overall_index] = pred_indices[i]
                        valid_predictions[overall_index] = clean
                        all_true_indices[overall_index] = true_indices[i]
                    else:  # If there is an error
                        all_pred_indices[overall_index] = None  # Or any other value that signifies an error
                        valid_predictions[overall_index] = None  # Or any other value that signifies an error
                        all_true_indices[overall_index] =None
                except:
                    if "Error" not in clean:  # If no error in the intention
                        all_pred_indices[overall_index] =clean
                        valid_predictions[overall_index] = clean
                        all_true_indices[overall_index] = clean
                    else:  # If there is an error
                        all_pred_indices[overall_index] = None  # Or any other value that signifies an error
                        valid_predictions[overall_index] = None  # Or any other value that signifies an error
                        all_true_indices[overall_index] =None
                        
                overall_index += 1

            # ── Evaluation ────────────────────────────────────────────────────────
            def safe_clean(data):
                try:
                    return clean_bracket_consistency(data[0])
                except Exception as e:
                    print(f"[Warning] Failed to clean data: {e}")
                    return [["Unknown"] * len(true_intentions1)]

            def compare_sets(true_intents, predicted):
                return [set(t) == set(p) for t, p in zip(true_intents, predicted)]

            # -- Feedback Comparison --
            prediction_self_feedback1 = safe_clean(prediction_self_feedback1)
            matches = compare_sets(true_intentions1, prediction_self_feedback1)
            subset_counter1_feedback += sum(matches)
            print("prediction_self_feedback1_counts:", subset_counter1_feedback)

            predicted_intentions_feedback2 = safe_clean(predicted_intentions_feedback)
            matches = compare_sets(true_intentions1, predicted_intentions_feedback2)
            subset_counter1_feedback_our += sum(matches)
            print("prediction_memorable_self_feedback1_counts:", subset_counter1_feedback_our)

            # -- CoT Comparison --
            clean_data_base_cot = safe_clean(clean_data_base_cot)
            subset_counter1 += sum(compare_sets(true_intentions1, clean_data_base_cot))

            our_clean_data_cot1 = safe_clean(our_clean_data_cot)
            true_counter += sum(compare_sets(true_intentions1, our_clean_data_cot1))

            print("cot baseline matching_counts:", subset_counter1)
            print("cot memorisable matching_counts:", true_counter)

            # -- FoT Comparison --
            # Flatten based_clean_data if needed
            if len(based_clean_data) == 1 and isinstance(based_clean_data[0], list):
                based_clean_data = based_clean_data[0]
            while len(based_clean_data) < len(true_intentions1):
                based_clean_data.append(["Unknown"])

            based_clean_data = safe_clean(based_clean_data)
            subset_counter13 += sum(compare_sets(true_intentions1, based_clean_data))

            clean_data0 = safe_clean(clean_data)
            our_true_counter += sum(compare_sets(true_intentions1, clean_data0))

            print("fot baseline matching_counts:", subset_counter13)
            print("fot matching_counts:", our_true_counter)

            # -- FoT V1 Comparison --
            based_clean_data1 = safe_clean(based_clean_data1)
            subset_counter131 += sum(compare_sets(true_intentions1, based_clean_data1))

            clean_data1 = safe_clean(clean_data1)
            our_true_counter1 += sum(compare_sets(true_intentions1, clean_data1))

            print("fot baseline matching_counts1:", subset_counter131)
            print("fot matching_counts1:", our_true_counter1)

            # -- Consistency-based Comparison --
            try:
                based_consistent_data = clean_bracket_consistency(based_consistent_data[0])
            except:
                based_consistent_data = based_consistent_data[0]
            subset_counter1311 += sum(compare_sets(true_intentions1, based_consistent_data))

            try:
                consistent_data = clean_bracket_consistency(consistent_data[0])
            except:
                consistent_data = consistent_data[0]
            our_true_counter11 += sum(compare_sets(true_intentions1, consistent_data))

            print("consistency+baseline matching_counts1:", subset_counter1311)
            print("consistency+Memorisable matching_counts1:", our_true_counter11)


            if all_good==total_line_index:
                print("all_good_so_far")
            if len(predicted_intentions_list) != total_line_index:
                print(f"Warning: total_line_index ({total_line_index}) and the length of predicted_intentions_list ({len(predicted_intentions_list)}) do not match!")
            else:
                print(f"Success: total_line_index ({total_line_index}) and the length of predicted_intentions_list ({len(predicted_intentions_list)}) match as expected.")

            if (total_line_index) % 50 == 0:
                file_name = f".../Transition_Matrix/{dataset_name}/CM_{line_index}_{temperature}_{proportion}_{step}_meta11.txt"
                with open(file_name, "w") as file:
                    for index, labels in predicted_intentions_list.items():
                        file.write(f"{index}: {labels}\n")
            
            if (total_line_index ) % 50 == 0:
                    file_name1 = f".../Transition_Matrix/{dataset_name}/CM_{line_index}_{temperature}_{proportion}_{step}_meta21.txt"
                    with open(file_name1, 'w') as file:
                        file.write(json.dumps(predicted_intentions_list, indent=4))

            if (total_line_index ) % 50 == 0:
                    file_name1 = f".../Transition_Matrix/{dataset_name}/CM_{line_index}_{temperature}_{proportion}_{step}_meta31.txt"
                    with open(file_name1, 'w') as file:
                        file.write(json.dumps(all_pred_indices, indent=4))

            if (total_line_index) % 50 == 0:
                file_name = f".../Transition_Matrix/{dataset_name}/CM_{line_index}_{temperature}_{proportion}_{step}_meta41.txt"
                with open(file_name, "w") as file:
                    for index, labels in all_responses.items():
                        file.write(f"{index}: {labels}\n")

            file_path = f".../Transition_Matrix/{dataset_name}/output_statistics_{temperature}_{proportion}_{step}_summary_meta51.txt"

            with open(file_path, 'a') as file:
                if (total_line_index) % 10 == 0:

                    file.write(f"Ratio of CoT base_line: {subset_counter1/total_line_index}\n")
                    file.write(f"Ratio of Memorable CoT: {true_counter/total_line_index}\n")

                    file.write(f"Ratio of FoT base_line: {subset_counter13/total_line_index}\n")
                    file.write(f"Ratio of FoT + Memorable: {our_true_counter/total_line_index}\n")

                    file.write(f"Ratio of Vanilla: {subset_counter131/total_line_index}\n")
                    file.write(f"Ratio of Memorable + Vanilla: {our_true_counter1/total_line_index}\n")

                    file.write(f"Ratio of Consistent base_line1: {subset_counter1311/total_line_index}\n")
                    file.write(f"Ratio of Memorable consistent: {our_true_counter11/total_line_index}\n")

                    file.write(f"Ratio of Feedback base_line1: {subset_counter1_feedback/total_line_index}\n")
                    file.write(f"Ratio of Memorable Feedback base_line1: {subset_counter1_feedback_our/total_line_index}\n")


            if (total_line_index) % 10 == 0:
                print("CoT baseline : ", subset_counter1/total_line_index)
                print("Memorable CoT : ", true_counter/total_line_index)

                print("Fot base_line : ", subset_counter13/total_line_index)
                print("Memorable Fot: ", our_true_counter/total_line_index)

                print("Vanilla : ", subset_counter131/total_line_index)
                print("Memorable + Vanilla: ", our_true_counter1/total_line_index)

                print("consistent base_line1 : ", subset_counter1311/total_line_index)
                print("Memorable consistent: ", our_true_counter11/total_line_index)

                print("Ratio of Feedback base_line1:", subset_counter1_feedback/total_line_index)
                print(f"Ratio of Memorable Feedback base_line1: {subset_counter1_feedback_our/total_line_index}\n")

            subset_number.append(subset_counter)
            matching_number.append(true_counter)
            subset_ratios.append(subset_counter/total_line_index)
            matching_ratios.append(true_counter/total_line_index)        

        print("total cot baseline : ", subset_counter1/total_line_index)
        print("total Memorable+cot: ", true_counter/total_line_index)


        print("Total Vanilla : ", subset_counter131/total_line_index)
        print("Total Memorable + Vanilla: ", our_true_counter1/total_line_index)


        print("total fot base_line: ", subset_counter13/total_line_index)
        print("total Memorable +fot : ", our_true_counter/total_line_index)


        print("consistent base_line1 : ", subset_counter1311/total_line_index)
        print("Memorable consistent: ", our_true_counter11/total_line_index)


        print(f"Ratio of Feedback base_line1: {subset_counter1_feedback/total_line_index}\n")
        # print(f"Ratio of Correction: {true_counter_correction/total_line_index}\n")

        print(f"Ratio of Memorable Feedback base_line1: {subset_counter1_feedback_our/total_line_index}\n")
        # print(f"Ratio of Memorable Correction: {true_counter_correction_our/total_line_index}\n")

        return true_counter,all_responses, predicted_intentions_list, valid_predictions, error_lines


    import random
    random_seeds=[1]
    for random_seed in random_seeds:
        print("dataset_name:",dataset_name)
        print("random_seed:",random_seed)
        # small_inputs, small_labels = extract_inputs_and_labels(f'.../datasets/{dataset_name}/small.jsonl')
        small_inputs, small_labels = extract_inputs_and_labels(f'.../datasets/{dataset_name}/small.jsonl')

        random.seed(40)
        # Define the maximum possible index
        max_index = len(small_inputs)  # Adjust this number based on your data

        # Generate a list of all possible indices excluding those in Whole_indices
        all_possible_indices = set(range(max_index))

        # Check if there are at least 400 indices to sample from
        if len(all_possible_indices) >= 100:
            random_samples1 = random.sample(all_possible_indices,int(0.5*len(small_inputs)))
        else:
            print("Not enough elements to sample 400 unique indices outside Whole_indices.")

        # print("400 Random Samples outside Whole_indices:", random_samples1)
        print("400 Random Samples outside Whole_indices:", len(random_samples1))

        small_inputs, small_labels = extract_inputs_and_labels(f'.../datasets/{dataset_name}/small.jsonl')

        selected_labels = [small_labels[index] for index in random_samples1]
        selected_lines = [small_inputs[index] for index in random_samples1]
        temperatures=[0.7, 0.1]
        for temperature in temperatures:
            true_counter,de_2000_all_responses_2_10_least, de_2000_predicted_intentions_list_2_10_least, de_2000_valid_predictions_2_10_least, de_2000_error_lines_2_10_least = process_responses1(selected_lines, selected_labels,temperature,step=1)
            print("Total_exact_matching:",true_counter/(len(selected_lines)))
