######################################################################
#####                                                            #####
##### LSTM Recurrent Network for Text Generation                 #####
#####                                                            #####
######################################################################

######################################################################
##### LIBRARY IMPORTS                                            #####
######################################################################
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

######################################################################
##### LOAD DATASET AND CREATE MAP OF UNIQUE CHARACTERS           #####
######################################################################

# Load the text corpus
filename = "Cat_in_the_Hat.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()

# Convert all the text to lowercase
raw_text = raw_text.lower()

# Create a map that maps each unique character in the text to a unique integer value
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# Also create a reverse map to be able to output the character that maps to a specific integer
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Display the total number of characters (n_chars) and the vocabulary (the number of unique characters)
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

######################################################################
##### CREATE TRAINING PATTERNS                                   #####
######################################################################

# Create the patterns to be used for training
seq_length = 100	# fixed length sliding window for training pattern
dataX = []			# input sequences
dataY = []			# outputs


for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

######################################################################
##### TRANSFORM DATA TO BE SUITABLE FOR KERAS                    #####
######################################################################

# Reshape dataX to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# Rescale integers mapped to characters to the range 0-to-1 to accommodate learning using sigmoid function
X = X / float(n_vocab)

# One hot encode the output variable
y = np_utils.to_categorical(dataY)

######################################################################
##### BUILD THE LSTM MODEL                                       #####
######################################################################

# Build sequential model containing 2 LSTM layers and 2 Dropout layers, followed by a dense output layer
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

######################################################################
##### LOAD THE SAVED NETWORK WEIGHTS FROM CHECKPOINT             #####
######################################################################

# Load the network weights from the specified checkpoint file
filename = "weights-improvement-50-0.1250-bigger.hdf5"
model.load_weights(filename)

# Compile the model using the Adam optimizer and categorical crossentropy for the loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')

######################################################################
##### TEXT GENERATION                                            #####
######################################################################

# Pick a random seed and display it
start = numpy.random.randint(0, len(dataX) - 1)

pattern = dataX[start]

print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# Generate 1000 characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

# Text generation complete	
print("\nDone.")