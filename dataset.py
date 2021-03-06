from torch import empty, long
from torch.utils.data import Dataset

class WordleDataset(Dataset):
    """
        A dataset that contains the words from the chosen word list (official/unofficial.txt) under data/ subdirectory.

        Can be iterated through by simply calling `for words, labels in dataset:`
    """
    def __init__(self, root_dir):
        def get_wordlist(filename):
            """ 
                To read all the lines from the given file, strip the enidng '\n' and lower case all the characters 
            """
            words = []
            with open(filename, 'r') as f:
                words = f.readlines()
            words = [word.strip() for word in words]
            words = [word.lower() for word in words]
            return words
        
        def get_labels(words):
            labels = empty((len(words), 5), dtype=long)
            for i, word in enumerate(words):
                for j, k in enumerate(word):
                    labels[i, j] = ord(k) - ord('a')
            return labels

        self.root_dir = root_dir
        self.words = get_wordlist(root_dir)
        self.labels = get_labels(self.words)
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        label = self.labels[idx]
        return word, label