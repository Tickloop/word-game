from termcolor import colored
import json

# load the data from interaction_history.json
def load_json():
    with open("data/interaction_history.json", "r") as f:
        data = json.load(f)
    return data

def accuracy(data):
    # data should be the outputs from one epoch
    keys = data.keys()
    acc = 0.
    count = 0
    for k in keys:
        guesses = [turn_val['guessed_word'] for turn_key, turn_val in data[k].items()]
        if k in guesses:
            acc += 1
        count += 1
    return acc / count

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

def print_turn_through_epochs(data, word):
    # data is the entire output
    for epoch in range(100):
        turns = data[str(epoch)][word]
        colored_turn = get_colored_turn(turns)
        print(colored_turn)

if __name__ == "__main__":
    data = load_json()

    # first we start with finding raw accuracy scores of 0th, 9th, 49th, and 99th epoch
    epochs = list(map(str, [0, 9, 49, 99]))

    for epoch in epochs:
        acc = accuracy(data[epoch])
        print(f"Epoch {epoch} => accuracy = {acc}%")
    
    # this prints the interaction for the 99th epoch
    print_epoch_turns(data['99'])

    # this prints the evolution of the interaction of a particular word through different epochs
    print_turn_through_epochs(data, "flume")
    
