import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Randomness
    parser.add_argument('--seed', default=0, type=int, help='randomseed')
    
    # #cluster
    parser.add_argument('--knearest', default=0, type=int, help='k-nearest neighbours')
    
    # data
    parser.add_argument('--data', type=str, metavar='PATH', help='path to data')
    
    return parser

def main(args):

    np.random.seed(args.seed)

    """
    Predicts the best LLM for each test inquiry using a kNN-based correctness predictor.
    
    Parameters:
    - X_train: numpy array of shape (N, m), training embeddings.
    - Y_train: numpy array of shape (N, p), binary correctness labels for each LLM.
    - X_test: numpy array of shape (N', m), test embeddings.
    - k: int, number of nearest neighbors to use (default is 5).
    
    Returns:
    - predicted_llm_indices: numpy array of shape (N',), the index of the LLM with highest predicted correctness for each inquiry.
    - predicted_probs: numpy array of shape (N', p), predicted correctness probabilities for each LLM.
    """

    datadict = np.load(args.data)
    #for item in datadict:
    #    print(item, datadict[item].shape)
    X_train = datadict['train_embed']
    Y_train = datadict['train_score']
    X_test = datadict['test_embed']
    Y_test = datadict['test_score']
    
    # Initialize the nearest neighbors model using cosine distance.
    nn_model = NearestNeighbors(n_neighbors=args.knearest, metric='cosine')
    nn_model.fit(X_train)
    
    # For each test inquiry, find the indices of its k nearest training inquiries.
    distances, indices = nn_model.kneighbors(X_test)
    
    # Number of test inquiries and number of available LLMs.
    num_test = X_test.shape[0]
    num_llms = Y_train.shape[1]
    
    # Initialize an array to store the predicted correctness probability for each LLM.
    predicted_probs = np.zeros((num_test, num_llms))
    
    # For each test inquiry, average the correctness labels of its k nearest neighbors.
    for i in range(num_test):
        neighbor_indices = indices[i]  # indices of the k nearest neighbors for test inquiry i.
        # Average the correctness labels across these neighbors for each LLM.
        predicted_probs[i] = np.mean(Y_train[neighbor_indices], axis=0)
    
    # For each test inquiry, select the LLM with the highest predicted correctness probability.
    predicted_llm_indices = np.argmax(predicted_probs, axis=1)
    
    overall_accuracy = np.mean(Y_test[np.arange(Y_test.shape[0]), predicted_llm_indices])

    
    #print('acc on the test set : {}'.format(overall_accuracy))
    #print('router acc / bsm acc: {}'.format(overall_accuracy/np.max(np.mean(Y_test, axis=0))))
    
    #print(predicted_probs)
    #print(predicted_probs.shape)
    predicted_probs = np.exp(predicted_probs - np.max(predicted_probs, axis=1, keepdims=True)) / np.sum(np.exp(predicted_probs - np.max(predicted_probs, axis=1, keepdims=True)), axis=1, keepdims=True)
    terms = np.where(predicted_probs > 1e-10, predicted_probs * np.log2(predicted_probs), 0)
    Ep = -np.sum(terms) / predicted_probs.shape[0] 

    #print('Classification bias : {}'.format(Ep))
   
    print(overall_accuracy, overall_accuracy/np.max(np.mean(Y_test, axis=0)), Ep)
    #print(predicted_probs)
    #print(predicted_llm_indices)

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    #print(args)

    main(args)
