import torch
import torch.nn as nn
from utils import device


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return torch.softmax(self.fc(state), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Linear(state_dim, 1)

    def forward(self, state):
        return self.fc(state)


# Define the World Model with RNN layer to aggregate sequence information
class WorldModel(nn.Module):
    def __init__(self, llm_model, state_dim=768):
        super(WorldModel, self).__init__()
        self.llm_model = llm_model
        self.projection = nn.Linear(
            llm_model.config.vocab_size, state_dim
        )  # Project vocab_size to state_dim
        self.rnn = nn.GRU(state_dim, state_dim, batch_first=True)

    def forward(self, input_ids, attention_mask):
        # Ensure input tensors are on the correct device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.llm_model(input_ids=input_ids, attention_mask=attention_mask)

        # Project the logits to the state_dim for compatibility with the GRU
        projected_logits = self.projection(
            outputs.logits
        )  # Shape: [batch_size, sequence_length, state_dim]

        # Pass the projected logits through the RNN to obtain the state
        _, state = self.rnn(projected_logits)
        return state.squeeze(0)  # Shape: [batch_size, state_dim]
