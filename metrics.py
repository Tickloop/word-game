from torch.nn import CrossEntropyLoss
from utils import get_default_features, get_feedback, get_mask_tree, get_updated_features, get_word_beam_search


def accuracy(model, dataset, mask_tree):
    acc = 0.
    count = 0.
    attempt_count = {}
    for correct_word, label in dataset:
        features = get_default_features()

        for attempt in range(6):
            output = model(features)

            guessed_word = get_word_beam_search(output, mask_tree)
            
            if guessed_word == correct_word:
                acc += 1
                attempt_count[correct_word] = 1 + attempt
                break

            feedback = get_feedback(guessed_word, correct_word)
            features = get_updated_features(features, feedback, guessed_word)
        count += 1
    
    acc = 100 * acc / count
    acc = round(acc, 4)
    return acc, attempt_count

def avg_loss(model, dataset, mask_tree):
    loss_fn = CrossEntropyLoss()
    loss = 0.
    for correct_word, label in dataset:
        features = get_default_features()
        
        for attempt in range(6):
            outputs = model(features)
            loss += loss_fn(outputs, label)

            guessed_word = get_word_beam_search(outputs, mask_tree)
            feedback = get_feedback(guessed_word, correct_word)
            features = get_updated_features(features, feedback, guessed_word)

            if guessed_word == correct_word:
                break
        
    return loss