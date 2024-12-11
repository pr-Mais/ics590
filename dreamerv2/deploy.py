import gradio as gr
import torch
from networks import RSSM, ActorOld, Critic
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Constants and Model Loading
MAX_LENGTH = 512
LATENT_DIM = 128
MODEL_NAME = "google/flan-t5-base"

rssm = RSSM(state_dim=MAX_LENGTH, action_dim=MAX_LENGTH, latent_dim=LATENT_DIM)
actor = ActorOld(latent_dim=LATENT_DIM, action_dim=MAX_LENGTH)
critic = Critic(latent_dim=LATENT_DIM)

# Load pretrained weights
rssm.load_state_dict(torch.load("dreamerv2/trained_rssm.pth"))
actor.load_state_dict(torch.load("dreamerv2/trained_actor.pth"))
critic.load_state_dict(torch.load("dreamerv2/trained_critic.pth"))

rssm.eval()
actor.eval()
critic.eval()

# Tokenizer and QA Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)


def dreamer_based_qa(context, question):
    contexts = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    inputs = tokenizer(
        context,
        question,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    input_ids = inputs["input_ids"]
    hidden = torch.zeros((1, LATENT_DIM))

    # RSSM forward pass
    _, latent_state = rssm(input_ids.float(), input_ids.float(), hidden)

    # Actor and Critic forward pass
    action_probs = actor(latent_state)
    action = torch.distributions.Categorical(action_probs).sample().item()
    confidence_score = critic(latent_state).item()

    # QA prediction
    with torch.no_grad():
        start_scores, end_scores = qa_model(**inputs).values()

    start_idx = torch.argmax(start_scores + action)
    end_idx = torch.argmax(end_scores + action)

    pred_tokens = contexts["input_ids"][0][start_idx : end_idx + 1]

    predicted_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)

    # Ensure compatibility with Gradio outputs
    return predicted_answer, f"{confidence_score * 100:.2f}%"


# Gradio Interface
interface = gr.Interface(
    fn=dreamer_based_qa,
    inputs=[
        gr.Textbox(lines=7, placeholder="Enter context here...", label="Context"),
        gr.Textbox(lines=2, placeholder="Enter question here...", label="Question"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Answer"),
        gr.Textbox(label="Confidence Score"),
    ],
    title="Dreamer Enhanced QA Model",
    description="A hybrid Dreamer and Transformer-based QA system with confidence evaluation.",
)

# Launch the Gradio app
interface.launch()
