import numpy as np
import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # probability of choosing the Oracle router
    parser.add_argument('--p', default=1, type=float, help='p')    

    # data
    parser.add_argument('--data', type=str, metavar='PATH', help='path to data')
    
    return parser


def mix_router(group_data, p=0.5):
    final_scores = np.zeros(group_data.shape[1])
    for i in range(group_data.shape[1]):
        if np.random.rand() < p:
            final_scores[i] = np.max(group_data[:, i])
        else:
            final_scores[i] = group_data[np.random.randint(0, group_data.shape[0]), i]
            
    oracle_prob = group_data / group_data.sum(axis=0, keepdims=True)
    for i in range(group_data.shape[1]):
        if np.isnan(oracle_prob[:, i]).any():
            oracle_prob[:, i] = np.ones(group_data.shape[0]) / group_data.shape[0]
    random_prob = np.ones_like(group_data) / group_data.shape[0]
    predicted_probs = p * oracle_prob + (1-p) * random_prob
#     print(np.isnan(predicted_probs).any())
    terms = np.where(predicted_probs > 1e-10, predicted_probs * np.log2(predicted_probs), 0)
    Ep = -np.sum(terms) / predicted_probs.shape[1]
    return np.mean(final_scores), Ep


def main(args):

    """
    Predicts the best LLM for each test inquiry.
    
    Parameters:
    - Y_test: numpy array of shape (N, m),  correctness scores for each LLM.
    - p: float, the probability of choosing the Oracle router (where 1-p is the probability of choosing the Random router).
    
    Returns:
    - predicted_llm_indices: numpy array of shape (N',), the index of the LLM with highest predicted correctness for each inquiry.
    - predicted_probs: numpy array of shape (N', p), predicted correctness probabilities for each LLM.
    """

    datadict = np.load(args.data)
    #for item in datadict:
    #    print(item, datadict[item].shape)
    Y_test = datadict['test_score'].T

    p = args.p
    if p < 0 or p > 1:
        raise RuntimeError(f"p = {p} out of range [0, 1]")

    if p == 1.0:
        acc, Ep = mix_router(Y_test, 1)
    else:
        acc_list = []
        Ep_list = []
        for i in range(1000):
            tmp_oracle_acc, tmp_Ep = mix_router(Y_test, p)
            acc_list.append(tmp_oracle_acc)
            Ep_list.append(tmp_Ep)
        acc = np.mean(acc_list)
        Ep = np.mean(Ep_list)
    print(acc, acc/np.max(np.mean(Y_test, axis=1)), Ep)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    #print(args)

    main(args)
