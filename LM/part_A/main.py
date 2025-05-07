# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import torch

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    print(torch.cuda.is_available())