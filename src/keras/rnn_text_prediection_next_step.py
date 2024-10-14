import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Sample text for training
text = "hello world"

# Create a character to index dictionary
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# Parameters
seq_length = 3  # Number of characters to use as context
n_chars = len(text)
n_vocab = len(chars)

# Prepare the dataset
X_data = []
y_data = []

for i in range(n_chars - seq_length):
    # Input: sequence of chars, Output: the next char
    input_seq = text[i:i+seq_length]
    output_char = text[i+seq_length]
    X_data.append([char_to_index[char] for char in input_seq])
    y_data.append(char_to_index[output_char])

# Reshape the data for the RNN and convert output to categorical
X_data = np.array(X_data)
y_data = to_categorical(y_data, num_classes=n_vocab)

# Define the RNN model
model = Sequential()

# Embedding layer to convert character indices to dense vectors
model.add(Embedding(input_dim=n_vocab, output_dim=10, input_length=seq_length))

# Add SimpleRNN layer
model.add(SimpleRNN(50, activation='tanh'))

# Dense layer to predict the next character
model.add(Dense(n_vocab, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# Train the model
model.fit(X_data, y_data, epochs=200, batch_size=32)

# Function to generate text
def generate_text(seed_text, n_chars_to_generate):
    result = seed_text
    for _ in range(n_chars_to_generate):
        # Prepare the input
        input_seq = [char_to_index[char] for char in seed_text[-seq_length:]]
        input_seq = np.reshape(input_seq, (1, seq_length))

        # Predict the next character
        predicted_char_index = np.argmax(model.predict(input_seq), axis=-1)[0]
        predicted_char = index_to_char[predicted_char_index]

        # Append the predicted character to the result
        result += predicted_char
        seed_text += predicted_char

    return result

# Test the model
generated_text = generate_text("hel", 10)
print(generated_text)
