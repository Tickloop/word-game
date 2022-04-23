from torch import load
from utils import get_default_features, get_feedback, get_mask_tree, get_word_beam_search, get_updated_features, get_wordset
from visualize import get_colored_word

if __name__ == "__main__":
    word_set = get_wordset("data/official.txt")
    model = load("models/100epoch_bigger_train_beam_3")
    mask_tree = get_mask_tree("data/official.txt")
    
    while True:
        correct_word = input("Choose word:")
        print("word in vocab: ", correct_word in word_set)
        if correct_word == "quit":
            break
        
        features = get_default_features()

        for attempt in range(6):
            outputs = model(features)
            guessed_word = get_word_beam_search(outputs, mask_tree)

            feedback = get_feedback(guessed_word, correct_word)
            features = get_updated_features(features, feedback, guessed_word)
            
            colored_word = get_colored_word(guessed_word, feedback)
            print(colored_word)

            if guessed_word == correct_word:
                break