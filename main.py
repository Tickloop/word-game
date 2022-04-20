import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import numpy as np
from metrics import accuracy, avg_loss
from utils import *
from models import BaseModel

device = "cuda:0" if torch.cuda.is_available else "cpu"
print(f"Training models on {device}")

torch.manual_seed(2002)
torch.autograd.set_detect_anomaly(True)

def train(model, datasets, mask_tree, max_epochs, eta):
    losses = np.zeros(max_epochs)
    val_acc = np.zeros(max_epochs)
    val_loss = np.zeros(max_epochs)
    word_count = len(datasets['train'])
    max_val_acc = float('-inf')

    optimizer = Adam(model.parameters(), lr=eta)
    loss_criterion = CrossEntropyLoss()
    interactions = { ep : { word : {} for word, label in datasets['train'] } for ep in range(max_epochs) }

    for epoch in range(max_epochs):
        i = 0
        for correct_word, correct_word_labels in datasets['train']:
            features = get_default_features()
            i += 1
            print(f"Word: {correct_word} {i}/{word_count}", end='\r')
            
            for attempt in range(6):
                optimizer.zero_grad()

                outputs = model(features)
                guessed_word = get_word_beam_search(outputs, mask_tree)

                word_loss = loss_criterion(outputs, correct_word_labels)
                
                word_loss.backward()
                optimizer.step()

                losses[epoch] += word_loss.item()

                feedback = get_feedback(guessed_word, correct_word)
                features = get_updated_features(features, feedback, guessed_word)
                
                interactions[epoch][correct_word][attempt] = {
                    'feedback': feedback,
                    'guessed_word': guessed_word
                }

                if guessed_word == correct_word:
                    break

        val_acc[epoch], _ = accuracy(model, datasets['val'], mask_tree)
        val_loss[epoch] = avg_loss(model, datasets['val'], mask_tree)
        print(f"Epoch {epoch} / {max_epochs}, loss => {losses[epoch]}, val_acc => {val_acc[epoch]}, val_loss => {val_loss[epoch]}")

        if val_acc[epoch] > max_val_acc:
            save_model(b1, "1epoch_bigger_train_beam")
            max_val_acc = val_acc[epoch]
    
    return losses, interactions

if __name__ == "__main__":
    splits = [0.8, 0.05, 0]
    mask_tree = get_mask_tree("data/official.txt")
    dataset = get_dataset("data/official.txt")
    datasets = get_split_dataset(dataset, splits)

    b1 = BaseModel(in_features=26 * 12)
    b1_loss, interaction_history = train(b1, datasets, mask_tree, max_epochs=1, eta=0.0001)

    # save_history(interaction_history, "interaction_history_.json")
    # save_loss(b1_loss, "25epoch_bigger_train_beam.npy")