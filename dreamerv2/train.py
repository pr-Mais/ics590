import torch
import torch.nn as nn
import torch.optim as optim
from env import SquadEnv
from networks import RSSM, Actor, ActorOld, Critic
from torch.utils.tensorboard import SummaryWriter

MAX_LENGTH = 512  # Tokenization Length


# Environment & Training Initialization
env = SquadEnv()
state_dim = MAX_LENGTH
action_dim = env.action_space.n
latent_dim = 128

rssm = RSSM(state_dim, action_dim, latent_dim)
# actor = Actor(latent_dim, action_dim)
actor = ActorOld(latent_dim, action_dim)
critic = Critic(latent_dim)

optimizer = optim.AdamW(
    list(rssm.parameters()) + list(actor.parameters()) + list(critic.parameters()),
    lr=0.001,
)
writer = SummaryWriter()

num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    tokenized_state = env.tokenizer(
        state["context"],
        state["question"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )["input_ids"]

    hidden = torch.zeros(1, latent_dim)
    total_reward = 0

    for step in range(env.max_steps):
        action_probs = actor(hidden)
        # action_probs = torch.softmax(next_token_logits, dim=-1)
        action = torch.distributions.Categorical(action_probs).sample().item()

        next_state, reward, done, _, info = env.step(action)
        next_tokenized_state = env.tokenizer(
            next_state["context"],
            next_state["question"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )["input_ids"]

        decoded_state, hidden = rssm(
            torch.FloatTensor(tokenized_state).unsqueeze(0),
            torch.eye(action_dim)[action].float().unsqueeze(0).detach(),
            hidden.detach(),
        )
        value = critic(hidden.detach())
        reward_tensor = torch.tensor([[reward]]).float()

        loss = nn.functional.mse_loss(
            decoded_state, torch.FloatTensor(next_tokenized_state).unsqueeze(0).detach()
        ) + nn.functional.mse_loss(value, reward_tensor.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tokenized_state = next_tokenized_state
        total_reward += reward
        writer.add_scalar("Reward", reward, episode * env.max_steps + step)
        writer.add_scalar("Loss", loss.item(), episode * env.max_steps + step)

        if done:
            print(f"Golden Answer: {env.answers['text'][0]}", end=" | ")
            print(f"Predicted Answer: {info['predicted_answer']}")
            break

    print(f"Episode {episode + 1} Total Reward: {total_reward:.4f}")

# Save the model after training
torch.save(rssm.state_dict(), "trained_rssm.pth")
torch.save(actor.state_dict(), "trained_actor.pth")
torch.save(critic.state_dict(), "trained_critic.pth")
writer.close()
