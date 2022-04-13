from termcolor import colored
import json
import torch
from utils import *

# load the data from interaction_history.json
def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def get_colored_word(word, feedback):
    colored_word = ""
    for fb, ch in zip(feedback, word):
        if fb == 1:
            color_code = 'green'
        elif fb == 0:
            color_code = 'yellow'
        elif fb == -1:
            color_code = 'red'
        else:
            raise ValueError
        colored_word += colored(ch, color_code)
    return colored_word

def get_colored_turn(turn):
    colored_turn = []
    for turn_val in turn.values():
        feedback = turn_val['feedback']
        guessed_word = turn_val['guessed_word']

        colored_word = get_colored_word(guessed_word, feedback)
        colored_turn.append(colored_word)
    return " => ".join(colored_turn)

def print_epoch_turns(data):
    # data is the output from one epoch
    for correct_word, turns in data.items():
        colored_turn = get_colored_turn(turns)
        print(f"{correct_word} : {colored_turn}")

def print_word_over_epochs(data, word):
    # data is the entire output
    for epoch in range(100):
        turns = data[str(epoch)][word]
        colored_turn = get_colored_turn(turns)
        print(colored_turn)

def accuracy_on_output(data):
    # data should be the outputs from one epoch
    # words = data.keys()
    acc = 0.
    count = 0.
    for correct_word, turns in data.items():
        for turn in turns.values():
            if turn['guessed_word'] == correct_word:
                acc += 1
                break
        count += 1
    return round(100. * acc / count, 3)

def accuracy_on_dataset(model_path, wordlist_path, dataset_name):
    splits = [0.8, 0.05, 0]
    dataset = get_dataset(wordlist_path)
    datasets = get_split_dataset(dataset, splits)
    
    model = torch.load(model_path)
    acc, count = 0., 0.
    results = {word : {} for word, label in datasets[dataset_name]}


    for correct_word, label in datasets[dataset_name]:
        features = get_default_features()

        for attempt in range(6):
            output = model(features)

            guessed_word = get_word(output)
            feedback = get_feedback(guessed_word, correct_word)
            features = get_updated_features(features, feedback, guessed_word)
            
            results[correct_word][attempt] = {
                'feedback': feedback,
                'guessed_word': guessed_word,
            }

            if guessed_word == correct_word:
                acc += 1
                break
        
        count += 1

    acc = round(100. * acc / count, 3)
    return results, acc

def get_in_vocab(results, words):
    acc, count = 0., 0.
    for turns in results.values():
        for turn in turns.values():
            guessed_word = turn['guessed_word']
            if guessed_word in words:
                acc += 1
            count += 1
    return round(100 * acc / count, 4)        

def print_model_dataset_accuracy(model_name):
    results, acc, in_vocab = {}, {}, {}

    results['train'], acc['train'] = accuracy_on_dataset(model_name, "data/official.txt", "train")
    results['val'], acc['val'] = accuracy_on_dataset(model_name, "data/official.txt", "val")
    results['test'], acc['test'] = accuracy_on_dataset(model_name, "data/official.txt", "test")

    print(f"Train accuracy: {acc['train']}%")
    print(f"validation accuracy: {acc['val']}%")
    print(f"Test accuracy: {acc['test']}%")

    word_set = get_wordset("data/official.txt")
    in_vocab['train'] = get_in_vocab(results['train'], word_set)
    in_vocab['val'] = get_in_vocab(results['val'], word_set)
    in_vocab['test'] = get_in_vocab(results['test'], word_set)

    print(f"Words guessed in vocab(train): {in_vocab['train']}%")
    print(f"Words guessed in vocab(val): {in_vocab['val']}%")
    print(f"Words guessed in vocab(test): {in_vocab['test']}%")

if __name__ == "__main__":
    data = load_json("interactions/interaction_history_4.json")

    # first we start with finding raw accuracy scores of 0th, 9th, 49th, and 99th epoch
    epochs = list(map(str, [i for i in range(15)]))

    for epoch in epochs:
        acc = accuracy_on_output(data[epoch])
        print(f"Epoch {epoch} => accuracy = {acc}%")
    
    # this prints the interaction for the 99th epoch
    # print_epoch_turns(data['14'])

    # this prints the evolution of the interaction of a particular word through different epochs
    # print_word_over_epochs(data, "flume")
    
    print_model_dataset_accuracy("models/15epoch_bigger_train")
