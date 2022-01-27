from random import randint
from guess import CharGuess, Guess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tries', default=5, type=int, dest='tries', help="The number of tries that the user has")
parser.add_argument('--debug', default=False, dest='debug', action='store_true', help="Run the program in debug mode")
options = parser.parse_args()

def readLine(choice):
    """ Reads the choice-th line from the file as the word for the game and returns the word stripped of '\n' """
    word = ""
    with open('data/words.txt', 'r') as file:
        file.seek(6 * choice)
        word = file.readline()
    return word.strip()

def game(word):
    if options.debug:
        print("Word is: ", word)
    
    guess = Guess("_____")
    done = False
    tries = 0

    while not done:
        userInput = input("Your Guess: ")
        guess = Guess(userInput)
        tries += 1
        done = guess.judge(word) or tries > options.tries
        print(guess)
    
    print("Word was: ", word)

def main():
    word = readLine(randint(0, 12478))
    game(word)

if __name__ == "__main__":
    main()