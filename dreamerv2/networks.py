import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(RSSM, self).__init__()
        self.encoder = nn.Linear(state_dim + action_dim, latent_dim)
        self.gru = nn.GRUCell(latent_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, state_dim)

    def forward(self, state, action, hidden):
        x = torch.cat([state, action], dim=-1).float()
        x = torch.relu(self.encoder(x))
        hidden = self.gru(x, hidden)
        decoded_state = self.decoder(hidden)
        return decoded_state, hidden


class Actor(nn.Module):
    def __init__(
        self,
        latent_dim,
        action_dim,
        model_name: str = "google/flan-t5-base",
        embedding_model: str = "intfloat/e5-base-v2",
    ):
        super(Actor, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.resize_token_embeddings(self.model.config.vocab_size)
        self.latent_projector = nn.Linear(latent_dim, self.model.config.hidden_size)

        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    def forward(self, latent, input_ids):
        # Decode the input_ids to get the embeddings
        with torch.no_grad():
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        # Get the embeddings for the input text
        embeddings_input = self.embedding_tokenizer(input_text, return_tensors="pt")
        embeddings = self.embedding_model(**embeddings_input)
        embeddings = self._average_pool(
            embeddings.last_hidden_state, embeddings_input["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Project the latent state
        projected_latent = self.latent_projector(latent)
        combined_embeddings = embeddings + projected_latent

        # Forward pass through the model using inputs_embeds
        outputs = self.model(
            inputs_embeds=combined_embeddings.unsqueeze(0),
            return_dict=True,
            output_hidden_states=True,
        )

        # Get logits for the next token
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]  # Get logits for the next token

        return next_token_logits

    def _average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class ActorOld(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(ActorOld, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, latent):
        return self.net(latent)


class Critic(nn.Module):
    def __init__(self, latent_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, latent):
        return self.net(latent)
