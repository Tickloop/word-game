import torch
import json
from dataset import WordleDataset
import numpy as np

def get_default_features():
    one_through_12 = torch.zeros((26, 11)).float()
    zero = torch.ones((26, 1)).float()
    return torch.hstack((zero, one_through_12))

def get_label_tensor(word : str):
    output = torch.empty(5, dtype=torch.long)
    for i, k in enumerate(word):
        output[i] = ord(k) - ord('a')
    return output

def get_feedback(guessed_word : str, correct_word : str) -> list:
    correct_word_counter = { c : 0 for c in correct_word }
    for k in correct_word:
        correct_word_counter[k] += 1

    guessed_word_counter = { c : 0 for c in guessed_word }
    for k in guessed_word:
        guessed_word_counter[k] += 1
    
    feedback_counter = {}
    for k in correct_word:
        if k in guessed_word:
            feedback_counter[k] = min(guessed_word_counter[k], correct_word_counter[k])
        else:
            feedback_counter[k] = correct_word_counter[k]
    
    feedback = [-1 for k in guessed_word]
    for i, k in enumerate(guessed_word):
        if correct_word[i] == k:
            if feedback_counter[k]:
                feedback[i] = 1
                feedback_counter[k] -= 1
    
    for i, k in enumerate(guessed_word):
        if k in correct_word and feedback[i] == -1:
            if feedback_counter[k]:
                feedback[i] = 0
                feedback_counter[k] -= 1
    
    return feedback

def get_updated_features(features, feedback, word):
    for i, k in enumerate(word):
        row_idx = ord(k) - ord('a')
        if feedback[i] == 0:
            col_idx = 7 + i
        elif feedback[i] == 1:
            col_idx = 2 + i
        elif feedback[i] == -1:
            col_idx = 1
        else:
            raise ValueError
        features[row_idx][col_idx] = 1
        features[row_idx][0] = 0
    return features

def get_word(outputs):
    word = ""
    for o in outputs:
        word += chr(torch.argmax(o) + ord('a'))
    return word

def get_wordlist(filename):
    words = []
    with open(filename, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    words = [word.lower() for word in words]
    return words

def get_dataset(root_dir):
    dataset = WordleDataset(root_dir)
    return dataset

def get_split_dataset(dataset, splits):
    total_count = len(dataset)
    splits = [int(total_count * ratio) for ratio in splits]
    splits[-1] = total_count - sum(splits)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    datasets = {
        'train': train_set,
        'test': test_set,
        'val': val_set,
    }

    return datasets

def save_model(model, model_name):
    torch.save(model, f"models/{model_name}")

def save_history(history, file_name):
    json_file = json.dumps(history)
    with open(f"interactions/{file_name}", "w") as f:
        f.write(json_file)

def save_loss(loss, file_name):
   np.save(f"plots/{file_name}", loss)