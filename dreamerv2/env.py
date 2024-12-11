import gymnasium as gym
import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from gymnasium import spaces
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

squad_metric = load("squad")


class SquadEnv(gym.Env):
    def __init__(self, max_length=512, model_name="google/flan-t5-base"):
        super(SquadEnv, self).__init__()
        self.dataset = load_dataset("squad", split="train")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.max_steps = 3
        self.observation_space = spaces.Dict(
            {"context": spaces.Text(max_length), "question": spaces.Text(64)}
        )
        self.action_space = spaces.Discrete(max_length)

    def reset(self):
        self.current_idx = np.random.randint(0, len(self.dataset))
        sample = self.dataset[self.current_idx]
        self.context = sample["context"]
        self.question = sample["question"]
        self.answers = sample["answers"]
        self.predicted_tokens = []
        self.step_count = 0
        return {"context": self.context, "question": self.question}, {}

    def step(self, action):
        self.step_count += 1
        inputs = self.tokenizer(
            self.context,
            self.question,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        start_scores, end_scores = self.model(**inputs).values()
        pred_start = torch.argmax(start_scores) + action
        pred_end = torch.argmax(end_scores) + action
        pred_tokens = inputs["input_ids"][0][pred_start : pred_end + 1]
        predicted_answer = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)

        # Combined Reward: BLEU + ROUGE + Exact Match
        reward_bleu = sentence_bleu(
            [self.answers["text"][0].split()], predicted_answer.split()
        )
        reward_rouge = self._calculate_rouge(self.answers["text"][0], predicted_answer)
        reward_em = squad_metric.compute(
            predictions=[
                {"prediction_text": predicted_answer, "id": str(self.current_idx)}
            ],
            references=[{"answers": self.answers, "id": str(self.current_idx)}],
        )["exact_match"]

        # Weighted Reward Combination
        reward = 0.5 * reward_rouge + 0.5 * (reward_em / 100.0)
        done = (
            self.step_count >= self.max_steps or self.tokenizer.eos_token_id == action
        )
        # Penalize for empty answers
        if len(predicted_answer) == 0 and done:
            reward -= 1.0

        return (
            {"context": self.context, "question": self.question},
            reward,
            done,
            False,
            {"predicted_answer": predicted_answer},
        )

    def _calculate_rouge(self, true_answer, predicted_answer):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(true_answer, predicted_answer)
        return (scores["rouge1"].fmeasure + scores["rougeL"].fmeasure) / 2.0
