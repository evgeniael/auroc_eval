"""
Text generation with shared GPU placement and bounded decode length.

Main costs per question (unchanged structurally): greedy + N sampled + nucleus.
Speed levers: GPU + bf16, smaller max_new_tokens, avoid extra full forwards.
"""

from __future__ import annotations

import os
import transformers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Generation:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_new_tokens: int = 300,
        use_bf16: bool = True,
    ):
        self.device = device or os.environ.get("AUROC_GEN_DEVICE") or _pick_device()
        self.max_new_tokens = int(os.environ.get("AUROC_MAX_NEW_TOKENS", max_new_tokens))
        self.tokenizer, self.model = self._load_tokenizer_and_model(
            model_name, use_bf16=use_bf16 and self.device == "cuda"
        )
        self.model.eval()

    def _load_tokenizer_and_model(self, model_name: str, use_bf16: bool):
        kwargs = {}
        if use_bf16 and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
        if model_name == "llama-8b-instruct":
            mid = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_name == "llama-3b-instruct":
            mid = "meta-llama/Llama-3.2-3B-Instruct"
        elif model_name == "qwen-7b-instruct":
            mid = "Qwen/Qwen2.5-7B-Instruct"
        elif model_name == "qwen-0.5b-instruct":
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
        else:
            raise ValueError(f"Model {model_name} not found")

        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
        model.to(self.device)
        return tokenizer, model

    def _inputs(self, question: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": "You are a bot that responds to questions."},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ]
        t = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        return t.to(self.device)

    def _gen_kw(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

    def generate_greedy(self, question: str):
        inputs = self._inputs(question)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                **self._gen_kw(),
            )
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        log_probs = transition_scores[0].tolist()
        gen_len = inputs.shape[-1]
        generated_tokens = outputs.sequences[:, gen_len:]
        generated_string = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )
        return generated_tokens, generated_string, outputs.scores, log_probs

    def generate_unbiased_samples(
        self, question: str, num_return_sequences: int = 10, seed: int = 42
    ):
        torch.manual_seed(seed)
        transformers.set_seed(seed)
        inputs = self._inputs(question)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                **self._gen_kw(),
            )
        gen_len = inputs.shape[-1]
        generated_tokens = outputs.sequences[:, gen_len:]
        generated_strings = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        all_log_probs = transition_scores.tolist()
        return generated_tokens, generated_strings, outputs.scores, all_log_probs

    def generate_nucleus_samples(
        self,
        question: str,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        transformers.set_seed(seed)
        inputs = self._inputs(question)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                **self._gen_kw(),
            )
        gen_len = inputs.shape[-1]
        generated_tokens = outputs.sequences[:, gen_len:]
        generated_strings = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        # Same cost as greedy/unbiased: reuse step logits from generate.
        # Avoid self.model(sequences): extra full forward over prompt+output (very slow).
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        if num_return_sequences == 1:
            all_log_probs = [transition_scores[0].tolist()]
        else:
            all_log_probs = [row.tolist() for row in transition_scores]
        return generated_tokens, generated_strings, outputs.scores, all_log_probs
