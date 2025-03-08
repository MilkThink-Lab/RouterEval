import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Randomness
    parser.add_argument('--seed', default=0, type=int, help='randomseed')
    
    # #cluster
    parser.add_argument('--numcluster', default=0, type=int, help='number of cluster')
    
    # data
    parser.add_argument('--data', type=str, metavar='PATH', help='path to data')
    
    return parser



def main(args):

    np.random.seed(args.seed)

    # Suppose you have:
    #   X: An (N x m) numpy array containing query embeddings.
    #   Y: An (N x P) numpy array containing performance labels for each query,
    #      where each element is 1 (accurate) or 0 (inaccurate) for P different LLMs.
    
    datadict = np.load(args.data)
    #for item in datadict:
    #    print(item, datadict[item].shape)

    train_embed = datadict['train_embed']
    train_score = datadict['train_score']

    N = train_embed.shape[0]        # Number of queries
    m = train_embed.shape[1]         # Embedding dimension
    P = train_score.shape[1]           # Number of available LLMs

    # Synthetic query embeddings (e.g., could come from a pre-trained model)
    X = train_embed
    
    # Synthetic performance matrix:
    # Here, each entry is randomly chosen as 0 or 1.
    # In practice, Y would be generated based on the LLMs' outputs.
    Y = train_score

    # --------------------------
    # Step 1: Clustering the Query Embeddings
    # --------------------------
    # Choose the number of clusters (K) -- this is a hyperparameter.
    K = args.numcluster

    # Fit KMeans clustering on the embeddings
    kmeans = GaussianMixture(n_components=K, random_state=args.seed)
    cluster_labels = kmeans.fit_predict(X)

    # --------------------------
    # Step 2: Determine the Best LLM for Each Cluster
    # --------------------------
    # For each cluster, aggregate the performance of each LLM over all queries
    # in that cluster and choose the LLM with the highest average accuracy.
    
    cluster_best_llm = {}
    for k in range(K):
        # Get indices of queries in cluster k
        indices = np.where(cluster_labels == k)[0]
        if len(indices) == 0:
            continue  # Skip if no queries in this cluster (shouldn't occur with KMeans)
        
        # Calculate the average performance for each LLM within the cluster
        avg_performance = np.mean(Y[indices], axis=0)
        
        # Identify the best LLM index (the one with the highest average performance)
        best_llm = np.argmax(avg_performance)
        cluster_best_llm[k] = best_llm
    
    
    #print("Best LLM for each cluster:", cluster_best_llm)

    # --------------------------
    # Step 3: Evaluation
    # --------------------------
    # For each query, use its cluster assignment to get the chosen LLM,
    # then retrieve the performance (0 or 1) for that query from Y.
    # Compute the overall accuracy as the mean of these performance scores.

    new_X = datadict['test_embed']
    new_Y = datadict['test_score']
    
    N_new = new_X.shape[0]   
    
    # 1. Predict the cluster assignment for each new query using the trained kmeans model.
    new_cluster_labels = kmeans.predict(new_X)
    #print(new_cluster_labels)
    # 2. For each new query, select the best LLM for its assigned cluster and record its performance.
    predicted_performance = []
    for i in range(N_new):
        cluster_id = new_cluster_labels[i]
        # Get the best LLM for this cluster from the training mapping.
        best_llm = cluster_best_llm.get(cluster_id, None)
        if best_llm is None:
            # If no best LLM is found, you could decide on a default action. Here we skip.
            continue
        # Append the performance (0 or 1) for this query from the selected LLM.
        predicted_performance.append(new_Y[i, best_llm])

    # 3. Calculate overall accuracy.
    overall_accuracy = np.mean(predicted_performance)
    
    
   
    print('acc on the test set : {}'.format(overall_accuracy))
    print('router acc / bsm acc: {}'.format(overall_accuracy/np.max(np.mean(new_Y, axis=0))))
    
    predicted_probs = kmeans.predict_proba(new_X)
    prob_set = list(set(cluster_best_llm.values()))
    prob_num = len(prob_set)
    
    new_predicted_probs= np.zeros(predicted_probs.shape)
    for i in  range(predicted_probs.shape[1]):
      new_predicted_probs[:,prob_set.index(cluster_best_llm[i])] +=predicted_probs[:,i]
    terms = np.where(new_predicted_probs > 1e-4, new_predicted_probs * np.log2(new_predicted_probs), 0)
    
    Ep = -np.sum(terms) / new_predicted_probs.shape[0] 

    print('Classification bias : {}'.format(Ep))


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    print(args)

    main(args)
    