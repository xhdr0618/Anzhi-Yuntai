import os
import re
from tqdm import tqdm
import pandas as pd

import json
import os

import numpy as np
from openai import OpenAI
import argparse
import requests

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str, default='/root/autodl-tmp/WSDM/data')
    parser.add_argument('--model_name',
                        type=str, default='deepseek-chat')
    parser.add_argument('--temperature',
                        type=float, default=0.0)
    parser.add_argument('--save_dir',
                        type=str, default="./v1/")
    return parser.parse_args()
    
def save_prompt_to_file(filename, content): 
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def load_result(filename):
    """Load cached result from json file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None
        
def generate_prompt(prompt, response_a, response_b):
    prompt = f"""You are an AI model evaluating response quality.
    Please rate the quality of each response on a scale of 0-10, where 0 is the lowest quality and 10 is the highest quality.
    You must output your evaluation in the following format:
    Response A Score: [number between 0-10]
    Response B Score: [number between 0-10]
    
    Prompt:
    {prompt}

    Response A:
    {response_a}

    Response B:
    {response_b}
    """
    return prompt

def extract_scores(response_text):
    """Extract numerical scores from model response"""
    try:
        # need to catch
        score_a = float(re.search(r"Response A Score:\s*(\d+\.?\d*)", response_text).group(1))
        score_b = float(re.search(r"Response B Score:\s*(\d+\.?\d*)", response_text).group(1))
        return score_a, score_b
    except:
        return None, None

def query_model(sys_prompt, user_prompt):
    response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **settings,
            stream=False
        )
    return response.choices[0].message.content, int(response.usage.total_tokens)

if __name__ == '__main__':
    args = get_args()

    data_path = args.data_path
    df = pd.read_parquet(f'{data_path}/train_with_fold.parquet')
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    settings = {
        "model": args.model_name,
        "temperature": args.temperature,
        "max_tokens": 8192,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    
    cache_dir = args.save_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    correct_n = 0
    total_tokens = 0
    skipped = 0
    cached = 0
    
    # Create progress bar
    pbar = tqdm(total=len(df), desc="Processing")
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        response_a = row['response_a']
        response_b = row['response_b']
        
        prompt_file = os.path.join(cache_dir, f"prompt_{idx}.txt")
        result_file = os.path.join(cache_dir, f"result_{idx}.json")
    
        # Check if cached result exists
        cached_result = load_result(result_file)
        if cached_result is not None:
            # Use cached results
            reward1 = cached_result.get('reward1')
            reward2 = cached_result.get('reward2')
            total_tokens += cached_result.get('tokens', 0)
            cached += 1
            
            if reward1 is not None and reward2 is not None:
                pred = 0 if reward1 > reward2 else 1
                if pred == row['label']:
                    correct_n += 1
            else:
                skipped += 1
                
            pbar.set_postfix({
                'Tokens': total_tokens,
                'Acc': f'{correct_n/(idx + 1 - skipped):.3f}',
                'Cached': cached,
                'Skipped': skipped
            })
            pbar.update(1)
            continue
    
        sys_prompt = """You are an AI assistant evaluating response quality. 
        You must rate each response on a scale of 0-10 and output scores in the exact format:
        Response A Score: [number]
        Response B Score: [number]"""
        
        user_prompt = generate_prompt(prompt, response_a, response_b)
        save_prompt_to_file(prompt_file, user_prompt)
    
        try:
            response_text, tokens = query_model(sys_prompt, user_prompt)
            total_tokens += tokens
            
            reward1, reward2 = extract_scores(response_text)
            
            if reward1 is None or reward2 is None:
                print(f"Warning: Could not parse scores for index {idx}. Response: {response_text}")
                skipped += 1
                continue
                
            pred = 0 if reward1 > reward2 else 1
            
            if pred == row['label']:
                correct_n += 1
    
            pbar.set_postfix({
                'Tokens': total_tokens,
                'Acc': f'{correct_n/(idx + 1 - skipped):.3f}',
                'Cached': cached,
                'Skipped': skipped
            })
            pbar.update(1)
    
            result = {
                'response_text': response_text,
                'reward1': reward1,
                'reward2': reward2,
                'pred': pred,
                'tokens': tokens
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
    
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            skipped += 1
    
    pbar.close()
    print(f"\nProcessing complete. Final accuracy: {correct_n/(len(df)-skipped):.3f}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Cached results used: {cached}")
    print(f"Skipped examples: {skipped}")