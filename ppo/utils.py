import Levenshtein
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")


# Load BERT or similar model for embedding
similarity_model = AutoModel.from_pretrained("google/flan-t5-base").to(device)
similarity_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


# Function to compute BLEU score reward with smoothing
def bleu_reward(generated_answer, correct_answer):
    reference = [correct_answer.split()]  # BLEU expects a list of references
    candidate = generated_answer.split()
    # Use SmoothingFunction to handle cases with low n-gram overlap
    smoothing_function = (
        SmoothingFunction().method1
    )  # Method 1 is a common choice for smoothing
    score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
    return score  # BLEU score with smoothing, between 0 and 1


# Function to compute semantic similarity reward
def semantic_similarity_reward(generated_answer, correct_answer):
    # Tokenize and embed both answers
    inputs_gen = similarity_tokenizer(
        generated_answer, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    inputs_corr = similarity_tokenizer(
        correct_answer, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        embedding_gen = similarity_model(**inputs_gen).last_hidden_state.mean(
            dim=1
        )  # Mean pooling

        embedding_corr = similarity_model(**inputs_corr).last_hidden_state.mean(dim=1)
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding_gen, embedding_corr).item()

    return similarity  # Reward is between 0 and 1


# Function to compute Levenshtein similarity reward
def levenshtein_reward(generated_answer, correct_answer):
    distance = Levenshtein.distance(generated_answer, correct_answer)
    max_len = max(len(generated_answer), len(correct_answer))
    reward = 1 - (distance / max_len)  # Normalized to be between 0 and 1
    return reward


# Define a helper function to structure each experience with combined rewards
def create_experience(question, correct_answer, generated_answer):
    state = question
    action = generated_answer
    next_state = question + " -> " + generated_answer
    # Compute rewards
    semantic_reward = semantic_similarity_reward(generated_answer, correct_answer)
    bleu_reward_score = bleu_reward(generated_answer, correct_answer)
    levenshtein_reward_score = levenshtein_reward(generated_answer, correct_answer)
    # Combined reward (weighted average, you can adjust weights based on relevance)
    reward = (
        (0.4 * semantic_reward)
        + (0.3 * bleu_reward_score)
        + (0.3 * levenshtein_reward_score)
    )
    return (state, action, next_state, reward)


def ppo_update(
    states, actions, old_action_probs, returns, advantages, clip_epsilon=0.2, epochs=4
):
    for _ in range(epochs):
        # Calculate new action probabilities
        action_probs = actor(states)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)

        # Calculate ratios
        ratios = torch.exp(action_log_probs - old_action_probs)

        # Clipped objective
        objective = torch.min(
            ratios * advantages,
            torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages,
        )
        actor_loss = -objective.mean()

        # Critic loss
        value_estimates = critic(states).squeeze()
        critic_loss = nn.MSELoss()(value_estimates, returns)

        # Update actor
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # Retain graph for multiple updates
        actor_optimizer.step()

        # Update critic
        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # Retain graph for multiple updates
        critic_optimizer.step()


def generate_sequence(model, tokenizer, prompt, max_length=50, top_p=0.9):
    # Encode the initial prompt and move to device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Start with empty generated sequence
    generated_sequence = input_ids.clone()

    # Use torch.no_grad() to save memory during inference
    with torch.no_grad():
        for _ in range(max_length):
            # Generate model outputs for the current input sequence
            outputs = model(input_ids=generated_sequence, attention_mask=attention_mask)

            # Extract the logits for the last generated token and move to CPU for processing
            next_token_logits = outputs.logits[:, -1, :].detach().cpu()

            # Apply nucleus (top-p) sampling to filter tokens for diversity
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Create a mask to remove tokens with cumulative probability above top_p
            sorted_indices_to_keep = cumulative_probs <= top_p
            valid_indices = sorted_indices[sorted_indices_to_keep]

            # Sample from the filtered logits
            if valid_indices.size(0) > 0:
                # Sample only from valid indices
                sampled_index = torch.multinomial(
                    torch.softmax(sorted_logits[sorted_indices_to_keep], dim=-1),
                    num_samples=1,
                )
                next_token = valid_indices[sampled_index].to(device)  # Move back to MPS
            else:
                # If no valid indices, fall back to argmax
                next_token = torch.argmax(next_token_logits).unsqueeze(0).to(device)

            # Ensure next_token has compatible dimensions
            next_token = next_token.view(1, 1)

            # Append the new token to the generated sequence
            generated_sequence = torch.cat([generated_sequence, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )

            # Stop if the end-of-sequence token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated sequence into text
    generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    return generated_text


# Define Levenshtein distance function
def levenshtein_distance(predicted_seq, target_seq):
    # Calculates normalized Levenshtein distance
    distance = Levenshtein.distance(predicted_seq, target_seq)
    max_len = max(len(predicted_seq), len(target_seq))
    return 1.0 - (distance / max_len)  # Normalize to be between 0 and 1


# Helper function to compute discounted returns
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


# Reward Function for Seq2Seq
def compute_reward(predicted_seq, target_seq):
    lev_distance = levenshtein_distance(predicted_seq, target_seq)
    max_len = max(len(target_seq), len(predicted_seq))
    return 1.0 - (lev_distance / max_len)


def sample_action(action_probs, epsilon=1e-10):
    # Ensure action_probs are valid probabilities by adding epsilon and normalizing
    action_probs = torch.clamp(action_probs, min=epsilon)  # Ensure no zero values
    action_probs = action_probs / action_probs.sum(
        dim=-1, keepdim=True
    )  # Normalize to sum to 1

    # Sample from the action probabilities
    action_dist = dist.Categorical(action_probs)
    action = action_dist.sample()
    return action
