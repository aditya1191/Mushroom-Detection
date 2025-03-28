import random

# Import global constants from the configuration file
from config import LEARNING_RATE, NUM_TRAINING_ITERATIONS, CONVERGENCE_THRESHOLD
# Import the error calculation function from the formulas module
from formulas import err
# Import classes for neural network layers and file handling from the models module
from models import Layer, cfile

# Initialize global variables
f = None  # File handler
curr_point = 0  # Current position in the dataset
target = []  # List to store target classes
attrs = []  # List to store attributes
total_runs = 0  # Counter for the total number of iterations
data = None  # File handler for logging
num_incorrect = 0  # Counter for the number of incorrect predictions
prev_sample_err = 0  # Error from the previous sample
curr_sample_err = 0  # Error from the current sample

def parse_data(fname):
    # Reset all the data
    global curr_point, total_runs, target, attrs, num_incorrect, prev_sample_err, curr_sample_err, data, f
    curr_point = 0
    total_runs = 0
    target = []
    attrs = []
    num_incorrect = 0
    prev_sample_err = 0
    curr_sample_err = 0

    # Determine the appropriate data file based on the input file name
    data_file = 'err.txt'
    if fname == 'training.txt':
        data_file = 'training_error.txt'
    elif fname == 'val.txt':
        data_file = 'val_err.txt'
    elif fname == 'testing.txt':
        data_file = 'testing_error.txt'

    # Clear the file
    open(data_file, 'w+').close()

    # Open the data file for logging
    data = cfile(data_file, 'w')

    # Read the input file and populate target and attrs lists
    f = open(fname, 'r').readlines()

    for row in f:
        row = [x.strip() for x in row.split(',')]
        row = [int(num) for num in row]
        target.append(int(row[0]))  # The first element represents the target class
        attrs.append(row[1:])  # The rest represent the attributes

if __name__ == '__main__':
    print("Parsing the training dataset...")
    # Parse the training dataset and store its information in globals
    parse_data('training.txt')

    # Set up the layers to be used
    x = Layer(6, attrs[curr_point], 1)
    y = Layer(3, x.layer_out, 2)

    print("Begin training the neural network:")
    # Iterate through training the neural network
    while total_runs < NUM_TRAINING_ITERATIONS:

        # Set up and evaluate the first layer
        x.input_vals = attrs[curr_point]
        x.eval()

        # Set up and evaluate the second layer
        y.input_vals = x.layer_out
        y.eval()

        # Backpropagate
        y.backprop(target[curr_point])
        x.backprop(y)

        # Get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # Check if the prediction is correct
        temp = 1 if y.layer_out[0] >= 0.5 else 0
        if temp != target[curr_point]:
            num_incorrect += 1

        # Check for convergence
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print("Data has converged at the " + str(total_runs) + "th run.")
                break

        # Print information about the current iteration
        print("Current iteration: " + str(total_runs))
        print("Current error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # Iterate
        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    # Close the file
    data.close()

    print("Neural network training complete! Press enter to begin validation.")
    accuracy = 1 - (float(num_incorrect) / NUM_TRAINING_ITERATIONS)
    print("Accuracy on the training set: " + str(accuracy))
    print("Error percentage on the training set: " + str(float(num_incorrect) / NUM_TRAINING_ITERATIONS))
    input()

    print("Parsing the validation dataset...")
    # Parse the validation dataset and store its information in globals
    parse_data('val.txt')

    print("Begin validating the neural network:")
    # Iterate through validating the neural network
    while total_runs < len(f):

        # Set up and evaluate the first layer
        x.input_vals = attrs[curr_point]
        x.eval()

        # Set up and evaluate the second layer
        y.input_vals = x.layer_out
        y.eval()

        # Get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # Check if the prediction is correct
        temp = 1 if y.layer_out[0] >= 0.5 else 0
        if temp != target[curr_point]:
            num_incorrect += 1

        # Check for convergence
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print("Data has converged at the " + str(total_runs) + "th run.")
                break

        # Print information about the current iteration
        print("Current iteration: " + str(total_runs))
        print("Current error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # Iterate
        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    # Close the file
    data.close()

    print("Neural network validation complete! Press enter to start testing.")
    accuracy = 1 - (float(num_incorrect) / len(f))
    print("Accuracy on the validation set: " + str(accuracy))
    print("Error percentage on the validation set: " + str(float(num_incorrect) / len(f)))
    input()

    print("Begin testing the neural network:")
    # Parse the testing data and store its information in globals
    parse_data('testing.txt')

    # Iterate through testing the neural network
    while curr_point < len(f):

        # Set up and evaluate the first layer
        x.input_vals = attrs[curr_point]
        x.eval()

        # Set up and evaluate the second layer
        y.input_vals = x.layer_out
        y.eval()

        # Get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # Check if the prediction is correct
        temp = 1 if y.layer_out[0] >= 0.5 else 0
        if temp != target[curr_point]:
            num_incorrect += 1

        # Print information about the current iteration
        print("Current iteration: " + str(total_runs))
        print("Current Error: " + str(curr_err) + "\n")
        data.w(curr_err)

        # Iterate
        total_runs += 1
        curr_point += 1

    data.close()
    print("Testing complete! Check the generated output files ('testing_error.txt' and 'training_error.txt')")
    accuracy = 1 - (float(num_incorrect) / len(f))
    print("Accuracy on the testing set: " + str(accuracy))
    print("Error percentage on the testing set: " + str(float(num_incorrect) / len(f)))
