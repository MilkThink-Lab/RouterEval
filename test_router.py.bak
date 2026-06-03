import pickle
import re
import numpy as np
import subprocess


datasets = ['arc', 'hellaswag', 'mmlu', 'harness_truthfulqa_mc_0', 'winogrande', 'gsm8k', 'ifeval', 'bbh', 'gpqa', 'musr', 'math', 'mmlu_pro']
to_handle_datasets = datasets
baseline = 'knn'

if baseline == 'knn':
    command = 'python router/PRKnn-knn/knn.py --seed 0 --knearest 5 --data data/tmp_perf.npz'
elif baseline == 'oracle':
    command = 'python router/R_o/r_o_router.py --p 1.0 --data data/tmp_perf.npz'
elif baseline == 'random':
    command = 'python router/R_o/r_o_router.py --p 0.0 --data data/tmp_perf.npz'
elif baseline == 'r_o_0.5':
    command = 'python router/R_o/r_o_router.py --p 0.5 --data data/tmp_perf.npz'
elif baseline == 'linear':
    command = 'python router/MLPR_LinearR/train.py --datadir data/tmp_perf.npz --lr 0.01 --model linear --epoch 10 --train-batch 1 --checkpoint router/MLPR_LinearR/linear'
elif baseline == 'mlp':
    command = 'python router/MLPR_LinearR/train.py --datadir data/tmp_perf.npz --lr 0.0001 --model FNN --epoch 100 --hidden_size 256 --train-batch 1 --checkpoint router/MLPR_LinearR/MLPR'
elif baseline == 'roberta_cluster':
    command = 'python router/C-RoBERTa-cluster/cluster.py --seed 0 --numcluster 3 --data data/tmp_perf.npz'
elif baseline == 'roberta_MLC':
    command = 'python router/RoBERTa-MLC/roberta_MLC.py --seed 0 --model_path roberta-base --data data/tmp_perf.npz'
else:
    raise RuntimeError("No such baseline")

acc_ref_dict = {'arc': 0.852, 'hellaswag': 0.953, 'mmlu': 0.864, 'harness_truthfulqa_mc_0': 0.669, 'winogrande': 0.875, 'gsm8k': 0.92,
           'ifeval': 0.7689, 'bbh': 0.8303, 'gpqa': 0.397, 'math': 0.4, 'musr': 0.699, 'mmlu_pro': 0.637}


for to_handle in to_handle_datasets:
    with open(f'data/router_dataset/{to_handle}_router_dataset.pkl', 'rb') as f:
        router_dataset = pickle.load(f)
    acc_ref = acc_ref_dict[to_handle]
    #print('ref model acc: ', acc_ref)
    #print(f"dataset: {to_handle},    baseline: {baseline}")
    for difficulty in ['easy', 'hard']:
        if difficulty == 'hard':
            break
        for num_cand in router_dataset[difficulty].keys():
            mu = []
            vr = []
            vb = []
            ep = []
            for config in ['all_strong', 'all_weak', 'strong_to_weak']:
                if to_handle == 'harness_truthfulqa_mc_0':   # scores in this benchmark are not binary
                    train_score = router_dataset[difficulty][num_cand][config]['data']['train_label']
                else:   # scores in these benchmarks are binary
                    train_score=router_dataset[difficulty][num_cand][config]['data']['train_score']
                val_score=router_dataset[difficulty][num_cand][config]['data']['val_score']
                test_score=router_dataset[difficulty][num_cand][config]['data']['test_score']

                if baseline == 'roberta_MLC':
                    np.savez('data/tmp_perf.npz', train_prompt=router_dataset['prompt']['train_prompt'],
                                            val_prompt=router_dataset['prompt']['val_prompt'],
                                            test_prompt=router_dataset['prompt']['test_prompt'],
                                            train_score=train_score, val_score=val_score, test_score=test_score)
                else:
                    np.savez('data/tmp_perf.npz', train_embed=router_dataset['embedding']['train_embed'],
                                            val_embed=router_dataset['embedding']['val_embed'],
                                            test_embed=router_dataset['embedding']['test_embed'],
                                            train_score=train_score, val_score=val_score, test_score=test_score)

                print(f"{to_handle:<10}", f"{baseline:<10}", f"num={num_cand:<10}", f"{config:<15}")
                # Blocking call to the command line
                try:
                    result = subprocess.run(
                        command,         # The command to be executed
                        shell=True,      # Execute through the shell
                        check=True,      # Raise an exception if the command returns a non-zero exit code
                        text=True,       # Return the output as a string
                        capture_output=True  # Capture standard output and standard error
                    )

                    # print("Command Output:")
                    # print(result.stdout)
                    
                    # ASSUME ALL ROUTERS EVENTUALLY OUTPUT THREE METRICS: MU, VB, AND EP.
                    floats = re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', result.stdout)
                    floats = [float(x) for x in floats[-3:]]
                    # print(floats)
                    result = floats[0] / acc_ref
                    output_string = f"mu: {floats[0]:.4f},  Vr: {result:.4f},  Vb: {floats[1]:.4f},  Ep: {floats[2]:.4f}"
                    mu.append(floats[0])
                    vr.append(result)
                    vb.append(floats[1])
                    ep.append(floats[2])
                    print(output_string)
                    # print()
                    # output return code
                    # print("Return Code:", result.returncode)

                except subprocess.CalledProcessError as e:
                    print(f"Command failed with return code: {e.returncode}")
                    print(f"Error Output: {e.stderr}")
                except Exception as e:
                    print(f"An unexpected error occurred: {str(e)}")

            print(f"{to_handle:<10}", f"{baseline:<10}", f"num={num_cand:<10}", f"{'avg_metrics':<15}")
            print(f"mu: {np.mean(mu):.4f},  Vr: {np.mean(vr):.4f},  Vb: {np.mean(vb):.4f},  Ep: {np.mean(ep):.4f}")
#             print(f"{np.mean(mu):.4f} {np.mean(vr):.4f} {np.mean(vb):.4f} {np.mean(ep):.4f}")
            print()

