from termcolor import colored

def feedback(guessed_word : str, correct_word : str) -> list:
    correct_word_counter = { c : 0 for c in correct_word }
    for k in correct_word:
        correct_word_counter[k] += 1

    guessed_word_counter = { c : 0 for c in guessed_word }
    for k in guessed_word:
        guessed_word_counter[k] += 1
    
    feedback_counter = {}
    for k in correct_word:
        if k in guessed_word:
            feedback_counter[k] = min(guessed_word_counter[k], correct_word_counter[k])
        else:
            feedback_counter[k] = correct_word_counter[k]
    
    feedback = [-1 for k in guessed_word]
    for i, k in enumerate(guessed_word):
        if correct_word[i] == k:
            if feedback_counter[k]:
                feedback[i] = 1
                feedback_counter[k] -= 1
    
    for i, k in enumerate(guessed_word):
        if k in correct_word and feedback[i] == -1:
            if feedback_counter[k]:
                feedback[i] = 0
                feedback_counter[k] -= 1

    for i, k in enumerate(feedback):
        if k == -1:
            feedback[i] = 'red'
        elif k == 0:
            feedback[i] = 'yellow'
        elif k == 1:
            feedback[i] = 'green'
        else:
            raise ValueError

    return feedback

def print_feedback(feedback : list, word : str) -> None:
    str = ""
    for color_code, ch in zip(feedback, word):
        str += f"{colored(ch, color_code)}"
    print(str + "\n")

def test_feedback():
    # guess - 1, present - 1 case
    guess =   "brash"
    present = "ctaju"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    guess =   "brsah"
    present = "ctaju"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    # guess - 2, present - 1 case
    guess =   "braas"
    present = "ctaju"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    guess =   "baars"
    present = "ctaju"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    guess =   "baras"
    present = "ctaju"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)


    # guess - 1, present 2 case
    guess =   "brash"
    present = "ctaau"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    guess =   "barsh"
    present = "ctaau"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    # guess - 2, present - 2 case
    guess =   "braah"
    present = "ctaau"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    guess =   "barah"
    present = "ctaau"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    guess =   "braha"
    present = "ctaau"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

    guess =   "abrha"
    present = "ctaau"
    fb = feedback(guess, present)
    print(present)
    print_feedback(fb, guess)

if __name__ == "__main__":
    test_feedback()