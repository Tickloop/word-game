import enum
from termcolor import colored

class CharCodes(enum.Enum):
    INCORRECT = 'red'
    INCORRECT_POSITION = 'yellow'
    CORRECT = 'green'

class CharGuess:
    """
        This class serves as an encapsulation for a character and a code.
        The same effect can be obtained by replacing this with a tupple, 
        but for future improvement, a separate class is better
    """
    def __init__(self, character: str, code: enum) -> None:
        self.character = character
        self.code = code
    
    def update(self, code: enum) -> None:
        self.code = code
    
    def __repr__(self):
        return colored(self.character, self.code.value)

class Guess:
    """
        This class is used to decompose a string into 5 CharGuess objects and encapsulate them into
        a new class object that we will call Guess.
    """
    def __init__(self, word: str) -> None:
        self.repr = []
        self.word = word.upper()

        for char in self.word:
            self.repr.append(CharGuess(char, CharCodes.INCORRECT))

    
    def judge(self, word: str) -> bool:
        """
            Function is supposed to udpate our CharGuesses for each of the characters in the users guess
        """
        if self.word == word:
            for charGuess in self.repr:
                charGuess.update(CharCodes.CORRECT)
            return True
        
        for idx, character in enumerate(word):
            charGuess = self.repr[idx]
            if charGuess.character == character:
                charGuess.update(CharCodes.CORRECT)
            elif charGuess.character in word:
                charGuess.update(CharCodes.INCORRECT_POSITION)

        return False
    
    def __repr__(self) -> None:
        repr = ""
        for charGuess in self.repr:
            repr += charGuess.__repr__() + " "
        return repr


        
