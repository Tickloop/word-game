import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import numpy as np
from utils import *
from models import BaseModel
import logging

device = "cuda:0" if torch.cuda.is_available else "cpu"
print(f"Training models on {device}")

torch.manual_seed(2002)
torch.autograd.set_detect_anomaly(True)

def train(model, words, max_epochs, eta):
    losses = np.zeros(max_epochs)

    optimizer = Adam(model.parameters(), lr=eta)
    loss_criterion = CrossEntropyLoss()
    interactions = { ep : { word : {} for word in words } for ep in range(max_epochs) }

    for epoch in range(max_epochs):
        for correct_word in words:
            features = get_default_features()
            labels = get_label_tensor(correct_word)

            for attempt in range(6):
                logging.debug(features)
                optimizer.zero_grad()

                outputs = model(features)

                word_loss = loss_criterion(outputs, labels)
                
                word_loss.backward()
                optimizer.step()

                losses[epoch] += word_loss.item()

                guessed_word = get_word(outputs)
                feedback = get_feedback(guessed_word, correct_word)
                features = get_updated_features(features, feedback, guessed_word)
                
                interactions[epoch][correct_word][attempt] = {
                    'feedback': feedback,
                    'guessed_word': guessed_word
                }

                if guessed_word == correct_word:
                    break
        
        print(f"Epoch {epoch} / {max_epochs}, loss => {losses[epoch]}")
    
    return losses, interactions

if __name__ == "__main__":
    words = get_wordlist("data/official.txt")
    # words = get_wordlist("data/dummy.txt")
    b1 = BaseModel(in_features=26 * 12)
    b1_loss, interaction_history = train(b1, words, max_epochs=100, eta=0.005)

    feat = get_default_features()
    out = b1(feat)
    word = get_word(out)
    print(word)