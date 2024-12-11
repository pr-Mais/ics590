import torch
import torch.distributions as dist
from datasets import load_dataset
from networks import Actor, Critic, WorldModel
from torch import optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import (
    compute_returns,
    compute_reward,
    device,
    ppo_update,
    create_experience,
)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", padding_side="left")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base").to(device)


# Set padding token
tokenizer.pad_token = tokenizer.eos_token
llm_model.resize_token_embeddings(len(tokenizer))

# Load the GSM8K dataset
gsm8k = load_dataset("gsm8k", "main", split="train")

# Define a batch size
batch_size = 8  # Adjust batch size based on available memory


# Prepare a DataLoader for batching
class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return example["question"], example["answer"]


dataset = GSM8KDataset(gsm8k)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Create a list to store experiences
experiences = []

# Process each batch of question-answer pairs
for batch in tqdm(dataloader, desc="Processing batches"):
    questions, correct_answers = batch
    # Ensure pad_token_id is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Set pad_token to eos_token if not defined

    # Encode batch inputs with padding and attention mask
    inputs = tokenizer(
        list(questions), return_tensors="pt", padding=True, truncation=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate answers in batch mode
    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode each generated answer and create experiences
    generated_answers = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    for question, correct_answer, generated_answer in zip(
        questions, correct_answers, generated_answers
    ):
        experience = create_experience(question, correct_answer, generated_answer)
        experiences.append(experience)
    break


num_epochs = 10
trajectory_length = 5
gamma = 0.99
clip_epsilon = 0.2
ppo_epochs = 4

# Instantiate Models
world_model = WorldModel(llm_model).to(device)
actor = Actor(state_dim=768, action_dim=tokenizer.vocab_size).to(device)
critic = Critic(state_dim=768).to(device)

# Set up optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)


# Main training loop
for epoch in range(num_epochs):
    epoch_progress = tqdm(gsm8k, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for example in epoch_progress:
        prompt = example["question"]  # Get the math problem
        answer = example["answer"]  # Get the correct solution

        # Encode prompt and move to device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Collect trajectory experiences
        states = []
        actions = []
        rewards = []
        old_action_log_probs = []

        # Generate a trajectory
        for _ in range(trajectory_length):
            # Get current state from world model
            state = world_model(input_ids, attention_mask)
            states.append(state)

            # Actor chooses action
            action_probs = actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            actions.append(action)
            old_action_log_probs.append(dist.log_prob(action))

            # Decode action to text
            generated_answer = tokenizer.decode(action.item(), skip_special_tokens=True)
            reward = compute_reward(generated_answer, answer)  # Calculate reward
            rewards.append(reward)

            # Update inputs for next state (append action to input sequence)
            action = action.unsqueeze(-1)
            input_ids = torch.cat([input_ids, action], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(action)], dim=1)

        # Process collected trajectory
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        old_action_log_probs = torch.cat(old_action_log_probs).detach()
        returns = compute_returns(rewards, gamma)

        # Calculate advantages
        values = critic(states).squeeze().detach()
        advantages = returns - values

        # PPO update
        ppo_update(
            states,
            actions,
            old_action_log_probs,
            returns,
            advantages,
            clip_epsilon,
            ppo_epochs,
        )

    print(f"Epoch {epoch+1} complete.")
