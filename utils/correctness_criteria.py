import numpy as np
from typing import List, Optional
from collections import Counter
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore
from openai import OpenAI

from utils.generations import Generation


class LLMAsJudge:
    def __init__(self, model_name: str):
        """
        Initialize the LLM as judge.
        """    
        self.model_name = model_name
        self.tokenizer, self.model = self._load_tokenizer_and_model(model_name)
        self.max_new_tokens = 1000
        
        
    def _load_tokenizer_and_model(self, model_name):
        if model_name == "llama-8b-instruct":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        elif model_name == "llama-3b-instruct":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        elif model_name == "qwen-7b-instruct":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        elif model_name == "qwen-0.5b-instruct":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        elif "gpt" in model_name:
            model = OpenAI(api_key=OPENAI_API_KEY)
            tokenizer = None
        else:
            raise ValueError(f"Model {model_name} not found")
        
        return tokenizer, model

    def _load_inputs(self, question, reference_answers, answer):
        s1 = "You are a judge that evaluates the correctness of an answer given a question and a list of reference answers. Even if one of the reference answers is semantically equivalent to the answer, we can consider it as correct. Generate a True or False answer."
        s2 = f"Question: {question}\nReference answers: {reference_answers}\nAnswer: {answer}\nIs the answer correct?"
        string_input = s1 + "\n" + s2
        messages = [
            {"role": "system", "content": s1},
            {"role": "user", "content": s2}
            ]
        
        if "gpt" in self.model_name:
            inputs = None
        else:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        
        return inputs, string_input
    
    def generate_greedy(self, question, reference_answers, answer):
        inputs, string_input = self._load_inputs(question, reference_answers, answer)

        if "gpt" in self.model_name:
            outputs = self.model.responses.create(model=self.model_name, input=string_input, 
            reasoning={"effort": "none"}, temperature=0)
            generated_string = outputs.output[0].content[0].text
            print(generated_string)
        else:
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False,
                                        return_dict_in_generate=True, output_scores=True, pad_token_id=self.tokenizer.eos_token_id)

            generated_tokens = outputs.sequences[:, inputs.shape[-1]:]
            generated_string = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            print(generated_string)

        return generated_string

class Correctness:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize evaluation tools.
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # ------------------------------------------------
    # 1 Exact Match
    # ------------------------------------------------
    def exact_match(self, prediction: str, reference_answers: List[str]) -> bool:
        """
        Returns True if prediction exactly matches one of the references.
        """
        prediction = prediction.strip().lower()
        refs = [r.strip().lower() for r in reference_answers]
        return prediction in refs

    # ------------------------------------------------
    # 2 Token F1 (used in QA benchmarks)
    # ------------------------------------------------
    def _compute_token_f1(self, prediction: str, reference_answers: List[str]) -> float:
        """
        Compute maximum token-level F1 score across references.
        """

        def compute_f1(pred, ref):
            #Split in words
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = num_same / len(pred_tokens)
            recall = num_same / len(ref_tokens)

            return 2 * precision * recall / (precision + recall)

        return max(compute_f1(prediction, ref) for ref in reference_answers)

    def token_f1(self, prediction: str, reference_answers: List[str], threshold: float = 0.5) -> bool:
        """
        Returns True if maximum token-level F1 score is greater than threshold.
        """
        return self._compute_token_f1(prediction, reference_answers) > threshold

    # ------------------------------------------------
    # 3 BLEU
    # ------------------------------------------------
    def _compute_bleu(self, prediction: str, reference_answers: List[str]) -> float:
        """
        Compute BLEU score.
        """
        pred_tokens = prediction.split()
        refs = [ref.split() for ref in reference_answers]

        smoothie = SmoothingFunction().method4
        return sentence_bleu(refs, pred_tokens, smoothing_function=smoothie)

    def bleu(self, prediction: str, reference_answers: List[str], threshold: float = 0.5) -> bool:
        """
        Returns True if maximum BLEU score is greater than threshold.
        """
        return self._compute_bleu(prediction, reference_answers) > threshold

    # ------------------------------------------------
    # 4 ROUGE-L
    # ------------------------------------------------
    def _compute_rouge_l(self, prediction: str, reference_answers: List[str]) -> float:
        """
        Compute maximum ROUGE-L score across references.
        """
        scores = [
            self.rouge.score(ref, prediction)["rougeL"].fmeasure
            for ref in reference_answers
        ]
        return max(scores)

    def rouge_l(self, prediction: str, reference_answers: List[str], threshold: float = 0.5) -> bool:
        """
        Returns True if maximum ROUGE-L score is greater than threshold.
        """
        return self._compute_rouge_l(prediction, reference_answers) > threshold

    # ------------------------------------------------
    # 5 Embedding similarity
    # ------------------------------------------------
    def _compute_embedding_similarity(self, prediction: str, reference_answers: List[str]) -> float:
        """
        Compute cosine similarity between embeddings.
        Returns max similarity with references.
        """
        pred_emb = self.embedding_model.encode([prediction])

        ref_emb = self.embedding_model.encode(reference_answers)

        sims = cosine_similarity(pred_emb, ref_emb)[0]

        return float(np.max(sims))

    def embedding_similarity(self, prediction: str, reference_answers: List[str], threshold: float = 0.5) -> bool:
        """
        Returns True if maximum embedding similarity is greater than threshold.
        """
        return self._compute_embedding_similarity(prediction, reference_answers) > threshold

    # ------------------------------------------------
    # 6 BERTScore
    # ------------------------------------------------
    def _compute_bert_score(self, prediction: str, reference_answers: List[str]) -> float:
        """
        Compute BERTScore F1 using best reference.
        """
        preds = [prediction] * len(reference_answers)

        P, R, F1 = bertscore(
            preds,
            reference_answers,
            lang="en",
            verbose=False
        )

        return float(F1.max())

    def bert_score(self, prediction: str, reference_answers: List[str], threshold: float = 0.5) -> bool:
        """
        Returns True if maximum BERTScore F1 is greater than threshold.
        """
        return self._compute_bert_score(prediction, reference_answers) > threshold

    # ------------------------------------------------
    # 7 LLM as judge
    # ------------------------------------------------

    def llm_as_judge(self, question: str, prediction: str, reference_answers: List[str], llm_as_judge: object) -> bool:
        """
        Returns True if the prediction is correct according to the LLM.
        """
        generated_string = llm_as_judge.generate_greedy(question, reference_answers, prediction)
        generated_string = generated_string.strip().lower()
        return "true" in generated_string

def judge_correctness(
    question: str,
    prediction: str,
    reference_answers: List[str],
    thresholds: dict,
    llm_as_judge: list[(str, object)],
    correctness: Optional["Correctness"] = None,
) -> dict:
    """
    Judge correctness of prediction based on thresholds.

    Pass a shared ``correctness`` instance (one per pipeline run) to avoid
    reloading SentenceTransformer/Rouge on every call. Metrics are computed once
    per prediction then compared to all thresholds (BERTScore was previously
    run 9× per call — dominant cost).
    """
    if correctness is None:
        correctness = Correctness()
    
    

    correctness_dict = {}
    correctness_dict["exact_match"] = correctness.exact_match(prediction, reference_answers)

    token_f1_val = correctness._compute_token_f1(prediction, reference_answers)
    bleu_val = correctness._compute_bleu(prediction, reference_answers)
    emb_val = correctness._compute_embedding_similarity(prediction, reference_answers)
    bert_val = correctness._compute_bert_score(prediction, reference_answers)

    correctness_dict["token_f1"] = {t: token_f1_val > t for t in thresholds["token_f1"]}
    correctness_dict["bleu"] = {t: bleu_val > t for t in thresholds["bleu"]}
    rouge_val = correctness._compute_rouge_l(prediction, reference_answers)
    correctness_dict["rouge_l"] = {t: rouge_val > t for t in thresholds["rouge_l"]}
    correctness_dict["embedding_similarity"] = {t: emb_val > t for t in thresholds["embedding_similarity"]}
    correctness_dict["bert_score"] = {t: bert_val > t for t in thresholds["bert_score"]}

    for m, llm_as_judge_object in llm_as_judge:
        correctness_dict[f"llm_as_judge_{m}"] = correctness.llm_as_judge(question, prediction, reference_answers, llm_as_judge_object)

    return correctness_dict

