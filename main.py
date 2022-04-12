import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import numpy as np
from utils import *
from models import BaseModel

device = "cuda:0" if torch.cuda.is_available else "cpu"
print(f"Training models on {device}")

torch.manual_seed(2002)
torch.autograd.set_detect_anomaly(True)

def train(model, datasets, max_epochs, eta):
    losses = np.zeros(max_epochs)

    optimizer = Adam(model.parameters(), lr=eta)
    loss_criterion = CrossEntropyLoss()
    interactions = { ep : { word : {} for word, label in datasets['train'] } for ep in range(max_epochs) }

    for epoch in range(max_epochs):
        for correct_word, labels in datasets['train']:
            features = get_default_features()
            word_losses = 0.
            
            for attempt in range(6):
                optimizer.zero_grad()

                outputs = model(features)

                word_loss = loss_criterion(outputs, labels)
                
                word_loss.backward()
                optimizer.step()

                word_losses += word_loss.item()

                guessed_word = get_word(outputs)
                feedback = get_feedback(guessed_word, correct_word)
                features = get_updated_features(features, feedback, guessed_word)
                
                interactions[epoch][correct_word][attempt] = {
                    'feedback': feedback,
                    'guessed_word': guessed_word
                }

                if guessed_word == correct_word:
                    break
            
            losses[epoch] += word_losses / (1 + attempt)
        
        losses[epoch] /= len(datasets['train'])
        print(f"Epoch {epoch} / {max_epochs}, loss => {losses[epoch]}")
    
    return losses, interactions

if __name__ == "__main__":
    splits = [0.8, 0.05, 0]
    dataset = get_dataset("data/official.txt")
    datasets = get_split_dataset(dataset, splits)
    
    b1 = BaseModel(in_features=26 * 12)
    b1_loss, interaction_history = train(b1, datasets, max_epochs=15, eta=0.005)

    feat = get_default_features()
    out = b1(feat)
    word = get_word(out)
    print(word)