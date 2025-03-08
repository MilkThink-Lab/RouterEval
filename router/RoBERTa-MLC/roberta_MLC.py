import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pickle
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from datasets import Dataset
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'

def get_argparser():
    parser = argparse.ArgumentParser()

    # seed
    parser.add_argument('--seed', default=43, type=int, help='randomseed')

    parser.add_argument('--model_path', default='roberta-base', type=str, help='path to model')

    parser.add_argument('--data',default='llm_performance_prompt.npz', type=str, metavar='PATH', help='path to data')
    
    
    return parser






def dataset_perpare(args):

    router_dataset = np.load(args.data, allow_pickle=True)
    train_input,test_input,train_score,test_score = router_dataset['train_prompt'],router_dataset['test_prompt'],router_dataset['train_score'],router_dataset['test_score']
    train_input = np.array(train_input)
    test_input = np.array(test_input)
    my_dict = {'labels':train_score,"sentence":train_input}
    train_dataset = Dataset.from_dict(my_dict)
    test_dict = {'labels':test_score,"sentence":test_input}
    test_dataset = Dataset.from_dict(test_dict)
    model_number = test_score.shape[1]
    return train_dataset,test_dataset,train_score,test_score,model_number


def train(args,train_dataset,test_dataset,train_score,test_score,model_number):
    model_name_or_path = args.model_path
    config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=model_number,
            finetuning_task="text-classification",
            revision="main",
            token=None,
            trust_remote_code=False,
        )
    config.problem_type = "multi_label_classification"


    tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
    def preprocess_function(examples):
      # Tokenize the texts
      result = tokenizer(examples["sentence"], padding="max_length", max_length=512 ,truncation=True, return_tensors="pt")
      return result
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        revision="main",
    )

    
    training_args = TrainingArguments(
    output_dir='./router/RoBERTa-MLC/MLC_checkpoint',       
    num_train_epochs=10 ,              
    per_device_train_batch_size=10,  
    per_device_eval_batch_size=12,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_strategy= 'no',
    #logging_dir='./logs',            
    )





    is_regression = False
    is_multi_label =True
    train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on dataset",
            )
    test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on dataset",
            )


    trainer = Trainer(
            model=model,
            args = training_args,
            train_dataset=train_dataset ,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator= default_data_collator,
        )
        
    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (len(train_dataset)                              )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()
            

    raw_predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
    predicted_probs = np.array([np.where(p > 0, p, 0) for p in raw_predictions])
    predicted_probs = np.array(predicted_probs)
    predicted_llm_indices = np.argmax(predicted_probs, axis=1)
    overall_accuracy = np.mean(test_score[np.arange(test_score.shape[0]), predicted_llm_indices])

    
    
    print('acc on the test set : {}'.format(overall_accuracy))
    print('router acc / bsm acc: {}'.format(overall_accuracy/np.max(np.mean(test_score, axis=0))))
    predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=True)
    
    terms = np.where(predicted_probs > 1e-4, predicted_probs * np.log2(predicted_probs), 0)
    Ep = -np.sum(terms) / predicted_probs.shape[0] 

    print('Classification bias : {}'.format(Ep))

def main():
    parser = get_argparser()
    args = parser.parse_args()
    random_state = args.seed
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    train_dataset,test_dataset,train_score,test_score,model_number = dataset_perpare(args)
    train(args,train_dataset,test_dataset,train_score,test_score,model_number)

if __name__ == '__main__':
    main()
    
