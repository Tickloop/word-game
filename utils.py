import torch
import json
from dataset import WordleDataset
import numpy as np

def get_default_features() -> torch.Tensor:
    """
        Returns the default features.
        The features are:

        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] x 26 rows
        They correspond to the following for each alphabet:

        1) Not guessed
        2) Absent in word
        3) Correct Position - 1
        4) Correct Position - 2
        5) Correct Position - 3
        6) Correct Position - 4
        7) Correct Position - 5
        8) Incorrect Position - 1
        9) Incorrect Position - 2
        10) Incorrect Position - 3
        11) Incorrect Position - 4
        12) Incorrect Position - 5

        These are all binary features.
    """
    one_through_12 = torch.zeros((26, 11)).float()
    zero = torch.ones((26, 1)).float()
    return torch.hstack((zero, one_through_12))

def get_label_tensor(word : str) -> torch.Tensor:
    """
        Given a word, we need to create labels for that word.
        Each label is the offset from 'a'.
        E.G: d will have label 3, a has label 0, z has label 25.

        Since each word has 5 characters, the resulting tensor has size 5
    """
    output = torch.empty(5, dtype=torch.long)
    for i, k in enumerate(word):
        output[i] = ord(k) - ord('a')
    return output

def get_feedback(guessed_word : str, correct_word : str) -> list:
    """
        This functio goes through the guessed word and correct word to produce
        wordle style feedback. 

        Edge cases to watch out for have been outlined in tests/test_feedback.py
        The implementation is the similar with a difference that rather that returning 
        a list of strings from {'red', 'yellow', 'green'} we return {-1, 0, 1}.
    """
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

def get_updated_features(features : torch.Tensor, feedback : list, guessed_word : str) -> torch.Tensor:
    """
        This function updates the features based on the feedback and the guessed_word.
        
        Arguments:
        `feautures`: Features should be either a default feature set created from get_default_features() or
        an updated feature set that comes from this function.
        
        `feedback`: Feedback is a list of integers from {-1, 0, 1}. This comes from the output of get_feedback().
        
        `guessed_word`: THe word that is guessed by the model.
    """
    for i, k in enumerate(guessed_word):
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

def get_word(outputs : torch.Tensor) -> str:
    """
        To convert the output of our model to a word that can be made sense of, we use this function.
        Basically take the argmax for each of the output (1, 26) and add the offset for ord('a') to get the character.

        Arguments:
        `outputs`: The output from the model. Should be of the shape [5, 26].
    """
    word = ""
    for o in outputs:
        word += chr(torch.argmax(o) + ord('a'))
    return word

def get_wordlist(wordlist_path : str) -> list:
    """
        Reads the words from the file at the path, removes ending '\n' characters, and lowercases all the words.

        Arguments:
        `wordlist_path`: Needs to be the full path to the wordlist to use. Usually word lists are under data/ subdirectory.
    """
    words = []
    with open(wordlist_path, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    words = [word.lower() for word in words]
    return words

def get_wordset(wordlist_path : str) -> set:
    """
        Reads the words from the file at the path, removes ending '\n' characters, and lowercases all the words.
        In addition, it converts the list to a set, useful for constant time lookup to check if word is in the list of words

        Arguments:
        `wordlist_path`: Needs to be the full path to the wordlist to use. Usually word lists are under data/ subdirectory.
    """
    words = get_wordlist(wordlist_path)
    return set(words)

def get_dataset(root_dir : str) -> WordleDataset:
    """
        Creates and returns a WordleDataset that uses the root_dir as the file from which the words are to be read.

        Arguments:
        `root_dir`: The full path to the word list to be used. Usually found under data/ subdirectory.
    """
    dataset = WordleDataset(root_dir)
    return dataset

def get_split_dataset(dataset : WordleDataset, splits : list) -> dict:
    """
        Given a dataset, creates 3 splits with the ratios specified in splits.

        Arguments:
        `dataset`: Should be an instance of the WordleDataset or any other Dataset.
        `splits`: Should be a list of 3 floats, with the last one being 0. We calculate the last one
        using magic. 
        
        **The magic being finding the number of data points in the train and val set and then subtracting from
        the count of all the datapoints to get the number in test. It is just easier to have the last value as 0.

        Return:
        `datasets`: A dict with 'train', 'val', 'test' as keys.
        {
            'train': Training_dataset,
            'val: Validation_dataset,
            'test': Testing_dataset,
        }
    """
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

def save_model(model : torch.nn.Module, model_name : str) -> None:
    """
        Save the model with the given name under the models/ subdirectory
    """
    torch.save(model, f"models/{model_name}")

def save_history(history : dict, file_name : str) -> None:
    """
        Save the interaction_history as a json file with the chosen file_name under the interactions/ subdirectory
    """
    json_file = json.dumps(history)
    with open(f"interactions/{file_name}", "w") as f:
        f.write(json_file)

def save_loss(loss : list, file_name : str):
    """
        Save the loss, that is acutally an np.arry but I can't find the typing for that, in a npz file under the plots/ subdirectory.
    """
    np.save(f"plots/{file_name}", loss)