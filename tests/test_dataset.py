import torch
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

def split_and_print(dataset, splits):
    total_count = len(dataset)
    splits = [int(total_count * ratio) for ratio in splits]
    splits[-1] = total_count - sum(splits)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    datasets = {
        'train': train_set,
        'val': val_set,
        'test': test_set,
    }

    # print the first 10 in each
    for name, dataset in datasets.items():
        print(f"name: {len(dataset)} samples")
        for idx in range(10):
            print(dataset[idx])

if __name__ == "__main__":
    # splits
    splits = [0.8, 0.05, 0]
    
    official_dataset = WordleDataset("data/official.txt")
    total_count = len(official_dataset)
    print(f"\ntotal samples in Official Dataset: {total_count}")
    split_and_print(official_dataset, splits)

    unofficial_dataset = WordleDataset("data/words.txt")
    total_count = len(unofficial_dataset)
    print(f"\ntotal samples in Official Dataset: {total_count}")
    split_and_print(unofficial_dataset, splits)
