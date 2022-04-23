from termcolor import colored
import json
import torch
from utils import *
import matplotlib.pyplot as plt

# load the data from interaction_history.json
def load_json(json_file : str) -> dict:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def get_colored_word(guessed_word : str, feedback : list) -> str:
    """
        The word will have special characters inserted using the colored function,
        from termcolor.
        
        Arguments:

        `feedback`: should be a list the same size(5) as `guessed_word` containing entries from 
        {-1, 0, 1}.

        `guessed_word`: is a string of 5 characters, ideally the output from your model converted to a string.
        The output from the model is converted to string using get_word() function
    """
    colored_word = ""
    for fb, ch in zip(feedback, guessed_word):
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

def get_colored_turn(turns : dict) -> str:
    """
        A turn is an attempted guess and the corresponding feedback recieved.
        The coloring is done using get_colored_word() function.
        This function is used to make it easier to find patterns in the AI's learning.

        Arguments:
        `turns`: A dictionary containing upto 6 attempts made by the model to predict the word.
        Each attempt should have a value for 'feedback' and 'guessed_word' key.

        Example turn:
        '0' : {
            'feedback': [1, 1, 0, -1, -1],
            'guessed_word': hello
        }

        turns is a dict of such turns
    """
    colored_turn = []
    for turn_val in turns.values():
        feedback = turn_val['feedback']
        guessed_word = turn_val['guessed_word']

        colored_word = get_colored_word(guessed_word, feedback)
        colored_turn.append(colored_word)
    return " => ".join(colored_turn)

def print_epoch_turns(one_epoch_interaction : dict) -> None:
    """
        This function is used to print the interaction history from one epoch of the model.
        One epoch contains all interactions of the model over the entire training dataset.
        By printing these interactions, we can see how the model is doing over the dataset.

        Arguments:
        `one_epoch_interaction`: A dictionary containing the interaction history of the model
        over the entire training dataset. For each word there should be at least 1 attempt,
        with each attempt being a valid turn. See get_colored_turn() for example of valid turns.

        Example one_epoch_interaction:
        {
            'hello': {
                '0': {
                    'feedback': [1, 1, 0, -1, -1],
                    'guessed_word': heoyy,
                },
                '1': {
                    'feedback': [1, 1, 1, 1, 1],
                    'guessed_word': hello,
                }
            },
            'goose': {
                '0': {
                    'feedback': [1, 1, 0, -1, -1],
                    'guessed_word': goecx,
                },
                '1': {
                    'feedback': [1, 1, 1, 0, -1],
                    'guessed_word': gooex,
                },
                '2': {
                    'feedback': [1, 1, 1, 1, 1],
                    'guessed_word': goose,
                }
            }
        }
    """
    for correct_word, turns in one_epoch_interaction.items():
        colored_turn = get_colored_turn(turns)
        print(f"{correct_word} : {colored_turn}")

def print_word_over_epochs(interaction_history : dict, correct_word : str) -> None:
    """
        Print the evolution of the AI's ability to predict the sequence of guesses for a given word.
        Useful to see this relation evolve an check for over training.

        Arguments:
        `interaction_history`: A dictonary that is the output of running the train() function for a model on the dataset.
        Usually these files are saved in the interactions sub-directory. An interacion_history is a dict with epoch number
        as keys and the interactions over the entire dataset is the value for the keys.

        `correct_word`: The word that we are tracking through different epochs.
    """
    for epoch in range(100):
        turns = interaction_history[str(epoch)][correct_word]
        colored_turn = get_colored_turn(turns)
        print(colored_turn)

def accuracy_on_output(one_epoch_interaction : dict) -> float:
    """
        Find the accuracy from the output. This data can be skewed as the model is learning between different
        interactions. Thus, this is not a good measure of accuracy. A better measure is defined below and in 
        metrics.py.

        Arguments:
        `one_epoch_ineraction`: Full definition is present in print_epoch_turns()
    """
    acc = 0.
    count = 0.
    for correct_word, turns in one_epoch_interaction.items():
        for turn in turns.values():
            if turn['guessed_word'] == correct_word:
                acc += 1
                break
        count += 1
    return round(100. * acc / count, 3)

def accuracy_on_dataset(model_path : str, wordlist_path : str, dataset_name : str) -> tuple:
    """
        Given a model path, wordlist path, and the dataset name from {'train', 'test', 'val'},
        finds the accuracy on the given dataset.

        Arguments:
        `model_path`: This needs to be the full path to the model. Usuallay models are stored in 
        the models/ subdirectory.

        `wordlist_path`: This need to be the full path to the word list. Usually the word list is
        stored in data/ subdirectory.

        `dataset_name`: The default split on the loaded wordlist will be [0.8, 0.05, 0.15] for 
        {'train', 'val', 'test'}. The dataset_name specifies which dataset to use to find this accuracy.

        Return:
        `results`: A dict, storing the attempts that the model made for each word in the specified dataset.
        `accuracy`: A float multiplied by 100 to give % of accuracy
    """
    splits = [1.0, 0, 0]
    dataset = get_dataset(wordlist_path)
    datasets = get_split_dataset(dataset, splits)
    mask_tree = get_mask_tree(wordlist_path)
    
    model = torch.load(model_path)
    acc, count = 0., 0.
    results = {word : {} for word, label in datasets[dataset_name]}

    i = 0
    N = len(datasets[dataset_name])
    for correct_word, label in datasets[dataset_name]:
        i += 1
        print(f"word: {correct_word} {i}/{N}", end='\r')
        features = get_default_features()

        for attempt in range(6):
            output = model(features)
            guessed_word = get_word_beam_search(output, mask_tree)
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

    acc = round(100. * acc / count, 3) if count else 0
    return results, acc

def get_in_vocab(interaction_results : dict, words_set : set) -> float:
    """
        Finds the percentage of words from the model's output that were in the given vocab (words).

        Arguments:
        `interaction_results`: The interacion history of the model.
        `words_set`: A set of words that represents the vocabulary for which we are checking overlap
    """
    acc, count = 0., 0.
    for turns in interaction_results.values():
        for turn in turns.values():
            guessed_word = turn['guessed_word']
            if guessed_word in words_set:
                acc += 1
            count += 1
    return round(100 * acc / count, 4) if count else 0        

def show_guess_distribution(results : dict):
    """
        Finds the number of attempts taken to guess each word.
        Then creates a distriubtion to display the number of guesses taken

        Parameters:
        `results`: A dictionary which has the attempt history for each word.
        This comes as an output from accuracy_on_dataset()
    """
    guess_count = { i : 0 for i in range(1, 8) }

    for correct_word, attempts in results.items():
        if len(attempts) == 6:
            if correct_word == attempts[5]['guessed_word']:
                count_attempts = 6
            else:
                count_attempts = 7
        else:
            count_attempts = len(attempts)
        guess_count[count_attempts] += 1
    
    _sum = 0.
    total_count = 0.
    for guess, count in guess_count.items():
        if guess == 7:
            continue
        _sum += count * guess
        total_count += count
    
    avg = round(_sum / total_count, 3)   
    print(guess_count)

    labels = [str(i) for i in guess_count.keys()]
    labels[-1] = "Could Not Guess"
    bars = list(guess_count.values())
    plt.bar(labels, bars, color="green")
    plt.title(f"Average Score : {avg}")
    plt.show()
    plt.close()

def print_model_statistics(model_name : str) -> None:
    """
        This is used to quickly see the statistics like interaction history and accuracy on different
        datasets for a given model. The usual split is [0.8, 0.05, 0.15] over the official list of 
        words.

        Arguments:
        `model_name`: Needs to be the full path to the model file. Usually under models/ subdirectory.
    """
    print(model_name)
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
    
    show_guess_distribution(results['train'])
    print("")

if __name__ == "__main__":
    # print_model_statistics("models/15epoch_naive_train")
    # print_model_statistics("models/100epoch_naive")
    # print_model_statistics("models/200epoch_naive_train")

    # print_model_statistics("models/15epoch_bigger_train")
    # print_model_statistics("models/25epoch_bigger_train")
    # print_model_statistics("models/25epoch_bigger_train_2")
    # print_model_statistics("models/25epoch_bigger_train_3")

    # print_model_statistics("models/25epoch_bigger_train_beam")
    # print_model_statistics("models/25epoch_bigger_train_beam_2")
    # print_model_statistics("models/25epoch_bigger_train_beam_3")

    # print_model_statistics("models/100epoch_bigger_train_beam")
    # print_model_statistics("models/100epoch_bigger_train_beam_2")
    # print_model_statistics("models/100epoch_bigger_train_beam_3")
    print_model_statistics("models/100epoch_bigger_train_beam_4")
    
    # print_model_statistics("models/25epoch_biggest_train_beam")
    # print_model_statistics("models/25epoch_biggest_train_beam_2")
    
