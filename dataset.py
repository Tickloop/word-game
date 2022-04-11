from torch import empty, long
from torch.utils.data import Dataset

class WordleDataset(Dataset):
    def __init__(self, root_dir):
        
        """ To read all the lines from the given file, strip the enidng '\n' and lower case all the characters """
        def get_wordlist(filename):
            words = []
            with open(filename, 'r') as f:
                words = f.readlines()
            words = [word.strip() for word in words]
            words = [word.lower() for word in words]
            return words
        
        self.root_dir = root_dir
        self.words = get_wordlist(root_dir)
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        label = empty(5, dtype=long)
        for i, k in enumerate(word):
            label[i] = ord(k) - ord('a')
        return word, label