import numpy as np
import pickle
from tqdm import tqdm
from utils import *


# acc under oracle router
def oracle_router(group_data):
    return np.mean(np.max(group_data, axis=0))


# divide data into train, val, test
def split_data(dataset, train_indices, val_indices, test_indices):
    res_dict = {'train_score': None, 'val_score': None, 'test_score': None}
    res_dict['train_score'] = dataset[train_indices, :]
    res_dict['val_score'] = dataset[val_indices, :]
    res_dict['test_score'] = dataset[test_indices, :]
    return res_dict


# For datasets with non-binary scores for individual test cases, we need to process the scores.
# For each test case, select the model with the highest score on that case,
# as well as other models with relatively high scores, as the label.
def process_label(arr):
    n, m = arr.shape
    result = np.zeros_like(arr)
    for i in range(m):
        max_val = np.max(arr[:, i])
        result[:, i] = np.where(arr[:, i] >= 0.95 * max_val, 1, 0)
    return result


# Select candidate LLMs from a pool of numerous LLMs to construct the router dataset.
def select_candidates(dataset, model_list, num_candidates, strength):
    # sort models by avg acc
    model_scores = np.mean(dataset, axis=1)
    sorted_indices = np.argsort(model_scores)
    
    # Find the first model with a score greater than 0.1
    first_index = next((i for i, x in enumerate(model_scores[sorted_indices]) if x > 0.1), None)
    # Find the first model with a score greater than 0.9
    last_index = next((i for i, x in enumerate(model_scores[sorted_indices]) if x > 0.9), dataset.shape[0])
    # print(first_index, last_index)
    # Divide the dataset into the lowest and highest 20% regions
    split_point = max(len(sorted_indices) // 5, num_candidates)
    low_20 = sorted_indices[first_index:first_index+split_point]   # Indices of the lowest 20%
    high_20 = sorted_indices[last_index-split_point:last_index]    # Indices of the highest 20%
    
    if strength == 'all_weak':
        # Slide a window over the lowest 20% region (excluding models with acc less than 0.1) 
        candidate_pool = dataset[low_20]
        candidate_list = model_list[low_20]
        window_size = num_candidates
        max_score = -np.inf
        best_window = None
        best_candidate = None
        
        for start in range(len(low_20) - window_size + 1):
            window = candidate_pool[start:start+window_size]
            current_score = oracle_router(window)
            if current_score > max_score:
                best_candidate = candidate_list[start:start+window_size]
                max_score = current_score
                best_window = window
        return best_window, best_candidate
    
    elif strength == 'all_strong':
        # Slide a window over the highest 20% region to select the optimal window
        candidate_pool = dataset[high_20]
        candidate_list = model_list[high_20]
        window_size = num_candidates
        max_score = -np.inf
        best_window = None
        best_candidate = None
        
        for start in range(len(high_20) - window_size + 1):
            window = candidate_pool[start:start+window_size]
            current_score = oracle_router(window)
            if current_score > max_score:
                best_candidate = candidate_list[start:start+window_size]
                max_score = current_score
                best_window = window
        return best_window, best_candidate
    
    elif strength == 'strong_to_weak':
        # Remove models with scores below 0.1 and then perform strong_to_weak sampling
        valid_indices = sorted_indices[first_index:last_index]
        # Divide all models into num_candidates groups equally
        group_size = len(valid_indices) // num_candidates
        groups = []
        remainder = len(valid_indices) % num_candidates
        
        # Handle the remainder
        start = 0
        for i in range(num_candidates):
            end = start + group_size + (1 if i < remainder else 0)
            groups.append(valid_indices[start:end])
            start = end
        
        # Perform 1000 random samplings
        best_group = None
        best_candidate = None
        max_score = -np.inf
        for _ in range(1000):
            # Randomly select one model from each group
            sampled = [np.random.choice(group) for group in groups]
            current_score = oracle_router(dataset[sampled])
            if current_score > max_score:
                best_candidate = model_list[sampled]
                max_score = current_score
                best_group = dataset[sampled]
        return best_group, best_candidate


if __name__ == '__main__':

    random_state = 42
    np.random.seed(random_state)

    # choose a benchmark
    # 'mmlu' / 'gsm8k' / 'winogrande' / 'arc' / 'hellaswag' / 'harness_truthfulqa_mc_0'
    # 'bbh' / 'gpqa' / 'ifeval' / 'math' / 'musr'  / 'mmlu_pro'
    to_handle_scenario = 'gsm8k'

    acc_ref_dict = {'arc': 0.852, 'hellaswag': 0.953, 'mmlu': 0.864, 'harness_truthfulqa_mc_0': 0.669, 'winogrande': 0.875, 'gsm8k': 0.92,
           'ifeval': 0.7689, 'bbh': 0.8303, 'gpqa': 0.397, 'math': 0.4, 'musr': 0.699, 'mmlu_pro': 0.637}
    
    # choose an embed model
    # longformer / RoBERTa / RoBERTa_last / sentence_bert
    embed_model = 'RoBERTa'   

    if to_handle_scenario == 'mmlu_pro':
        key = 'mmlu_pro'
    elif to_handle_scenario in ['mmlu', 'gsm8k', 'winogrande', 'arc', 'hellaswag', 'harness_truthfulqa_mc_0']:
        key = 'old'
    elif to_handle_scenario in ['bbh', 'gpqa', 'ifeval', 'math', 'musr']:
        key = 'new'
    else:
        raise RuntimeError(f'no such scenario: {to_handle_scenario}')

    scenarios = scenarios_lb[key]

    with open(f'data/leaderboard_score/leaderboard_{key}.pkl', 'rb') as handle:
        score_data = pickle.load(handle)
    with open(f'data/leaderboard_embed/leaderboard_{key}_embed/{embed_model}_result.pkl', 'rb') as handle:
        embed_data = pickle.load(handle)
    with open(f'data/leaderboard_prompt/leaderboard_{key}_prompt.pkl', 'rb') as handle:
        prompt_data = pickle.load(handle)

    scenarios_position, subscenarios_position = prepare_data(scenarios, score_data)
    Y = create_responses(scenarios, score_data)
    embed = create_embeds(scenarios, embed_data)
    prompt = create_prompts(scenarios, prompt_data)


    np.random.seed(random_state)
    np.set_printoptions(threshold=10, edgeitems=5) 
    subset_ranges = list(subscenarios_position[to_handle_scenario].values())
    dataset_start_pos = np.min(subset_ranges[0])   # where the dataset begins

    train_indices = []
    val_indices = []
    test_indices = []

    # Perform stratified random sampling for each subset
    for subset_range in subset_ranges:
        # Get the indices of the current subset
        subset_indices = np.array(list(subset_range)) - dataset_start_pos
        # print(subset_indices)
        
        # Shuffle the indices of the current subset
        np.random.shuffle(subset_indices)
        
        # Calculate the split points for the current subset
        train_split = int(0.8 * len(subset_indices))  # 80% for the train set
        val_split = int(0.9 * len(subset_indices))    # 10% for the val set, 10% for the test set
        
        # Split the indices of the current subset
        train_indices.extend(subset_indices[:train_split])
        val_indices.extend(subset_indices[train_split:val_split])
        test_indices.extend(subset_indices[val_split:])

    print(test_indices[:5])
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)


    dataset = Y[:, scenarios_position[to_handle_scenario]]  # shape: (num_models, num_test_cases)
    dataset_embeddings = embed[scenarios_position[to_handle_scenario], :]  # shape: (num_test_cases, embed_len)
    dataset_prompts = prompt[scenarios_position[to_handle_scenario]]  # shape: (num_test_cases,)
    # print(dataset_prompts.shape)

    # init router dict
    router_dataset = {
        'easy': {
            3: {'all_strong': {'data': None, 'model': None}, 'all_weak': {'data': None, 'model': None}, 'strong_to_weak': {'data': None, 'model': None}},
            5: {'all_strong': {'data': None, 'model': None}, 'all_weak': {'data': None, 'model': None}, 'strong_to_weak': {'data': None, 'model': None}}
        },
        'hard': {
            10: {'all_strong': {'data': None, 'model': None}, 'all_weak': {'data': None, 'model': None}, 'strong_to_weak': {'data': None, 'model': None}},
            100: {'all_strong': {'data': None, 'model': None}, 'all_weak': {'data': None, 'model': None}, 'strong_to_weak': {'data': None, 'model': None}},
            1000: {'all_strong': {'data': None, 'model': None}, 'all_weak': {'data': None, 'model': None}, 'strong_to_weak': {'data': None, 'model': None}}
        },
        'split_index':{
            'train_indices': None, 'val_indices': None, 'test_indices': None
        },
        'embedding': {
            'train_embed': None, 'val_embed': None, 'test_embed': None
        },
        'prompt': {
            'train_prompt': None, 'val_prompt': None, 'test_prompt': None
        }
    }

    router_dataset['split_index']['train_indices'] = train_indices
    router_dataset['split_index']['val_indices'] = val_indices
    router_dataset['split_index']['test_indices'] = test_indices
    router_dataset['embedding']['train_embed'] = dataset_embeddings[train_indices]
    router_dataset['embedding']['val_embed'] = dataset_embeddings[val_indices]
    router_dataset['embedding']['test_embed'] = dataset_embeddings[test_indices]
    router_dataset['prompt']['train_prompt'] = dataset_prompts[train_indices]
    router_dataset['prompt']['val_prompt'] = dataset_prompts[val_indices]
    router_dataset['prompt']['test_prompt'] = dataset_prompts[test_indices]


    print(f"{'number':<8}", f"{'group':<15}", f"{'oracle':<8}", f"{'or/ref':<8}", f"{'or/bsm':<8}", f"{'detail':<8}")
    # fill in the router dataset dict
    for difficulty in ['easy', 'hard']:
        for num_cand in router_dataset[difficulty].keys():
            for config in ['all_strong', 'all_weak', 'strong_to_weak']:
                selected_data, selected_model = select_candidates(
                    dataset=dataset,
                    model_list=np.array(score_data['model']),
                    num_candidates=num_cand,
                    strength=config
                )
                router_dataset[difficulty][num_cand][config]['data'] = split_data(selected_data.T, train_indices, val_indices, test_indices)
                if to_handle_scenario == 'harness_truthfulqa_mc_0':
                    router_dataset[difficulty][num_cand][config]['data']['train_label'] = process_label(router_dataset[difficulty][num_cand][config]['data']['train_score'].T).T
                router_dataset[difficulty][num_cand][config]['model'] = selected_model
                tmp_test_set = router_dataset[difficulty][num_cand][config]['data']['test_score'].T

                oracle_acc = oracle_router(tmp_test_set)
                print(f"{num_cand:<8}", f"{config:<15}", f"{oracle_acc:<8.3f}", f"{oracle_acc/acc_ref_dict[to_handle_scenario]:<8.3f}", f"{oracle_acc/np.max(np.mean(tmp_test_set, axis=1)):<8.3f}", np.round(np.mean(tmp_test_set, axis=1), 3))


    print(convert_arrays_to_shapes(router_dataset))
