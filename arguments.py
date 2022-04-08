import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tries', default=5, type=int, dest='tries', help="The number of tries that the user has")
parser.add_argument('--debug', default=False, dest='debug', action='store_true', help="Run the program in debug mode")
options = parser.parse_args()