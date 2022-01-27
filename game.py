from random import randint

def readLines():
    lines = []
    with open('data/words.txt', 'r') as file:
        lines = file.readlines()
    return lines

def isCorrect(word, guess):
    tGuess = ['_', '_', '_', '_', '_']

    guess = guess.upper()    

    if word == guess.strip():
        return word, True
    
    for idx, chr in enumerate(word):
        if chr == guess[idx]:
            tGuess[idx] = chr
    
    return tGuess, False

def game(word):
    guess = ['_', '_', '_', '_', '_']
    gCount = 0
    

    done = False
    while not done:
        print(" ".join(guess))
        userGuess = input()

        guess, done = isCorrect(word, userGuess)
        gCount += 1

        done = done or gCount == 5
    
    print("Word was ", word)



def main():
    words = readLines()

    word_choice = words[randint(0, len(words))].strip()

    game(word_choice)

if __name__ == "__main__":
    main()