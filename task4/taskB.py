import torch
import torch.nn as nn
import numpy as np

# Many to one LSTM model
class ManyToOneLSTMModel(nn.Module):

    def __init__(self, encoding_size):
        super(ManyToOneLSTMModel, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)  
        self.dense = nn.Linear(128, encoding_size)  

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128) 
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y): 
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

# Dictionary of emojis
emojies = {
    'hat': '🎩',
    'rat': '🐭',
    'cat': '😺',
    'flat': '🏢',
    'matt': '🙋',
    'cap': '🧢',
    'son': '👦'
}

# Lists of emojis and characters
emoji_index = [value for key, value in emojies.items()]
char_index = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']
char_encodings = np.eye(len(char_index))
encoding_size = len(char_encodings)
emoji_encodings = np.eye(len(emoji_index))

# Training data
# 'hat ', 
# 'rat ', 
# 'cat ', 
# 'flat', 
# 'matt', 
# 'cap ', 
# 'son '
x_train_numpy = np.array([
    [[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], 
    [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],
    [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],
    [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]],
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]],
    [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]],
])

y_train_numpy = np.array([
    [emoji_encodings[0], emoji_encodings[0], emoji_encodings[0], emoji_encodings[0]],
    [emoji_encodings[1], emoji_encodings[1], emoji_encodings[1], emoji_encodings[1]],
    [emoji_encodings[2], emoji_encodings[2], emoji_encodings[2], emoji_encodings[2]],
    [emoji_encodings[3], emoji_encodings[3], emoji_encodings[3], emoji_encodings[3]],
    [emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4]],
    [emoji_encodings[5], emoji_encodings[5], emoji_encodings[5], emoji_encodings[5]],
    [emoji_encodings[6], emoji_encodings[6], emoji_encodings[6], emoji_encodings[6]]
])

x_train = torch.tensor(x_train_numpy, dtype=torch.float32)
y_train = torch.tensor(y_train_numpy, dtype=torch.float32)

model = ManyToOneLSTMModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
# Train the model for 500 epochs
for epoch in range(500):
    for i in range(len(x_train)):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


def get_prediction(text):
    model.reset()  # Reset model
    tensor_list = [] # List to hold tensors
    for char in text:  # Loops through the text
        idx = char_index.index(char)  # Get index of char
        tensor_list.append([char_encodings[idx]])  # Append to list

    # Convert list to numpy array and then to tensor
    inp_numpy = np.array(tensor_list) 
    inp = torch.tensor(inp_numpy, dtype=torch.float) # Convert to tensor

    out = model.f(inp)  # Output / prediction
    last_out = out[-1]  # Takes the last prediction 
    return emoji_index[last_out.argmax().item()] # Returns the emoji at the predicted index


print("Prediction for 'rt':", get_prediction('rt'))
print("Prediction for 'rats':", get_prediction('rats'))
print("Prediction for 'cp':", get_prediction('cp'))
print("Prediction for 'ft':", get_prediction('ft'))
print("Prediction for 'mt':", get_prediction('mt'))