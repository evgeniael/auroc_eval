from pathlib import Path
from tqdm import tqdm

import argparse
import gc
import torch
import json
import pandas as pd
import random
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets import load_test_dataset
from utils.generations import Generation
from utils.correctness_criteria import judge_correctness, LLMAsJudge, Correctness
from utils.quantifiers import UncertaintyQuantifiers


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--model", type=str, default="llama-8b-instruct") 
    parser.add_argument("--llm_as_judge", type=list[str], default=["llama-8b-instruct", "gpt-5.4-mini", "qwen-0.5b-instruct"]) 
    parser.add_argument("--nli_model", type=str, default="deberta-large") 
    parser.add_argument("--sample_batch_size", type=int, default=10)
    parser.add_argument("--num_unbiased_samples", type=int, default=10)
    parser.add_argument("--size_test_set", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Cap decode length per call (was 1000; lower = much faster for short QA).")
    parser.add_argument("--token_f1_threshold_range", type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--bleu_threshold_range", type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--rouge_l_threshold_range", type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--embedding_similarity_threshold_range", type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--bert_score_threshold_range", type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Write JSON checkpoints every N items (1 = every item; large = faster I/O)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    question_ids, questions, answers_aliases = load_test_dataset(args.dataset, args.size_test_set, args.seed)

    print("Number of questions: ", len(questions))

    if not os.path.exists(f"{PROJECT_ROOT}/results/generations/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json"):
        os.makedirs(f"{PROJECT_ROOT}/results/generations/", exist_ok=True)
        
        generations = []
        generation = Generation(args.model, max_new_tokens=args.max_new_tokens)

        for idx in tqdm(range(0, len(questions))):
            question_id = question_ids[idx]
            question = questions[idx]
            answers = answers_aliases[idx]
            
            greedy_tokens, greedy_string, greedy_scores, greedy_log_probs = generation.generate_greedy(question)

            sampled_tokens, sampled_strings, sampled_scores, sampled_log_probs = generation.generate_unbiased_samples(question, num_return_sequences=args.num_unbiased_samples)

            generations.append({
                "question_id": question_id,
                "question": question,
                "answers_aliases": answers,
                "greedy_prediction": {"greedy_tokens": greedy_tokens.tolist(), "greedy_string": greedy_string, "greedy_log_probs": greedy_log_probs},
                "sampled_predictions": {"sampled_tokens": sampled_tokens.tolist(), "sampled_strings": sampled_strings, "sampled_log_probs": sampled_log_probs},
            })

            if (idx + 1) % args.checkpoint_every == 0 or (idx + 1) == len(questions):
                with open(f"{PROJECT_ROOT}/results/generations/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "w") as f:
                    json.dump(generations, f)
        # One Llama 8B ~16GB; keep it loaded into correctness → OOM when Uncertainty loads another Llama + Deberta
        del generation
        _free_gpu()
    else:
        with open(f"{PROJECT_ROOT}/results/generations/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "r") as f:
            generations = json.load(f)
        
        generation = Generation(args.model, max_new_tokens=args.max_new_tokens)
        if len(generations) != len(questions):
            print("Generations file is incomplete")
            for idx in tqdm(range(len(generations), len(questions))):
                question_id = question_ids[idx]
                question = questions[idx]
                answers = answers_aliases[idx]
                
                greedy_tokens, greedy_string, greedy_scores, greedy_log_probs = generation.generate_greedy(question)
                sampled_tokens, sampled_strings, sampled_scores, sampled_log_probs = generation.generate_unbiased_samples(question, num_return_sequences=args.num_unbiased_samples)

                generations.append({
                    "question_id": question_id,
                    "question": question,
                    "answers_aliases": answers,
                    "greedy_prediction": {"greedy_tokens": greedy_tokens.tolist(), "greedy_string": greedy_string, "greedy_log_probs": greedy_log_probs},
                    "sampled_predictions": {"sampled_tokens": sampled_tokens.tolist(), "sampled_strings": sampled_strings, "sampled_log_probs": sampled_log_probs}
                })
                if (idx + 1) % args.checkpoint_every == 0 or (idx + 1) == len(questions):
                    with open(f"{PROJECT_ROOT}/results/generations/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "w") as f:
                        json.dump(generations, f)
            
            del generation
            _free_gpu()
        else:
            print("Generations file is complete")
    
    if not os.path.exists(f"{PROJECT_ROOT}/results/correctness/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json"):
        os.makedirs(f"{PROJECT_ROOT}/results/correctness/", exist_ok=True)
        
        thresholds = {
                "token_f1" :args.token_f1_threshold_range,
                "bleu" : args.bleu_threshold_range,
                "rouge_l" :args.rouge_l_threshold_range,
                "embedding_similarity" :args.embedding_similarity_threshold_range,
                "bert_score" :args.bert_score_threshold_range
            }
        
        llm_as_judge = [(model, LLMAsJudge(model)) for model in args.llm_as_judge]
        correctness = Correctness()

        correctness_list = []
        for idx, generation in tqdm(enumerate(generations)):
            question_id = generation["question_id"]
            question = generation["question"]
            answers_aliases = generation["answers_aliases"]
            greedy_prediction = generation["greedy_prediction"]
            sampled_predictions = generation["sampled_predictions"]
            nucleus_predictions = generation["nucleus_predictions"]
            
            try:
                greedy_correctness = judge_correctness(
                question, greedy_prediction["greedy_string"], answers_aliases, thresholds, llm_as_judge, correctness
            )
            except:
                greedy_correctness = judge_correctness(
                question, greedy_prediction["greedy_string"], answers_aliases[0], thresholds, llm_as_judge, correctness
            )
            
            generation["greedy_correctness"] = {"string":  greedy_prediction["greedy_string"], "log_probs": greedy_prediction["greedy_log_probs"], "tokens": greedy_prediction["greedy_tokens"], "correctness": greedy_correctness}

            correctness_list.append(generation)
            if (idx + 1) % args.checkpoint_every == 0 or (idx + 1) == len(generations):
                with open(f"{PROJECT_ROOT}/results/correctness/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "w") as f:
                    json.dump(correctness_list, f)
        del llm_as_judge
        del correctness
        _free_gpu()
    else:        
        with open(f"{PROJECT_ROOT}/results/correctness/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "r") as f:
            generations = json.load(f)
        
        if len(generations) != len(questions):
            print("Correctness file is incomplete")
            thresholds = {
                "token_f1" :args.token_f1_threshold_range,
                "bleu" : args.bleu_threshold_range,
                "rouge_l" :args.rouge_l_threshold_range,
                "embedding_similarity" :args.embedding_similarity_threshold_range,
                "bert_score" :args.bert_score_threshold_range
            }
        
            llm_as_judge = [(model, LLMAsJudge(model)) for model in args.llm_as_judge]
            correctness = Correctness()

            with open(f"{PROJECT_ROOT}/results/generations/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "r") as f:
                all_generations = json.load(f)
            missing_generations = all_generations[len(generations):]
            missing_correctness_list = []

            for idx, generation in tqdm(enumerate(missing_generations)):
                question_id = generation["question_id"]
                question = generation["question"]
                answers_aliases = generation["answers_aliases"]
                greedy_prediction = generation["greedy_prediction"]
                sampled_predictions = generation["sampled_predictions"]
                nucleus_predictions = generation["nucleus_predictions"]
                
                try:
                    greedy_correctness = judge_correctness(
                    question, greedy_prediction["greedy_string"], answers_aliases, thresholds, llm_as_judge, correctness
                )
                except:
                    greedy_correctness = judge_correctness(
                    question, greedy_prediction["greedy_string"], answers_aliases[0], thresholds, llm_as_judge, correctness
                )

                generation["greedy_correctness"] = {"string":  greedy_prediction["greedy_string"], "log_probs": greedy_prediction["greedy_log_probs"], "tokens": greedy_prediction["greedy_tokens"], "correctness": greedy_correctness}

                missing_correctness_list.append(generation)
                if (idx + 1) % args.checkpoint_every == 0 or (idx + 1) == len(generations):
                    with open(f"{PROJECT_ROOT}/results/correctness/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "w") as f:
                        json.dump(generations + missing_correctness_list, f)
            
            del llm_as_judge
            del correctness
            _free_gpu()
        else:
            print("Correctness file is complete")
    
    if not os.path.exists(f"{PROJECT_ROOT}/results/uncertainty_quantifiers/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json"):
        os.makedirs(f"{PROJECT_ROOT}/results/uncertainty_quantifiers/", exist_ok=True)
        _free_gpu()
        uncertainty_quantifiers = UncertaintyQuantifiers(args.model, args.nli_model)

        uncertainty_list = []
        for idx, generation in tqdm(enumerate(generations)):
            question_id = generation["question_id"]
            question = generation["question"]
            answers_aliases = generation["answers_aliases"]
            greedy_prediction = generation["greedy_prediction"]
            sampled_predictions = generation["sampled_predictions"]
            nucleus_predictions = generation["nucleus_predictions"]
            greedy_correctness = generation["greedy_correctness"]
            # sampled_correctness = generation["sampled_correctness"]
            # nucleus_correctness = generation["nucleus_correctness"]

            sampled_strings = sampled_predictions["sampled_strings"]
            uncertainty_scores_greedy = uncertainty_quantifiers.evaluate(
                question, sampled_strings, greedy_correctness["string"], greedy_correctness["log_probs"]
            )
            shared = {
                "entropy": uncertainty_scores_greedy["entropy"],
                "semantic_entropy": uncertainty_scores_greedy["semantic_entropy"],
            }

            generation["uncertainty_scores_greedy"] = {"prediction": greedy_correctness["string"], "uncertainty_scores": uncertainty_scores_greedy}

            uncertainty_list.append(generation)
            if (idx + 1) % args.checkpoint_every == 0 or (idx + 1) == len(generations):
                with open(f"{PROJECT_ROOT}/results/uncertainty_quantifiers/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "w") as f:
                    json.dump(uncertainty_list, f)
    else:
        with open(f"{PROJECT_ROOT}/results/uncertainty_quantifiers/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "r") as f:
            generations = json.load(f)
        
        if len(generations) != len(questions):
            print("Uncertainty quantifiers file is incomplete")
            _free_gpu()
            uncertainty_quantifiers = UncertaintyQuantifiers(args.model, args.nli_model)

            with open(f"{PROJECT_ROOT}/results/correctness/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "r") as f:
                all_generations = json.load(f)
            missing_generations = all_generations[len(generations):]
            missing_uncertainty_list = []
            
            for idx, generation in tqdm(enumerate(missing_generations)):
                question_id = generation["question_id"]
                question = generation["question"]
                answers_aliases = generation["answers_aliases"]
                greedy_prediction = generation["greedy_prediction"]
                sampled_predictions = generation["sampled_predictions"]
                nucleus_predictions = generation["nucleus_predictions"]
                greedy_correctness = generation["greedy_correctness"]
                # sampled_correctness = generation["sampled_correctness"]
                # nucleus_correctness = generation["nucleus_correctness"]
                
                sampled_strings = sampled_predictions["sampled_strings"]
                uncertainty_scores_greedy = uncertainty_quantifiers.evaluate(
                    question, sampled_strings, greedy_correctness["string"], greedy_correctness["log_probs"]
                )
                shared = {
                    "entropy": uncertainty_scores_greedy["entropy"],
                    "semantic_entropy": uncertainty_scores_greedy["semantic_entropy"],
                }

                generation["uncertainty_scores_greedy"] = {"prediction": greedy_correctness["string"], "uncertainty_scores": uncertainty_scores_greedy}

                missing_uncertainty_list.append(generation)
                if (idx + 1) % args.checkpoint_every == 0 or (idx + 1) == len(generations):
                    with open(f"{PROJECT_ROOT}/results/uncertainty_quantifiers/{args.dataset}_{args.model}_{args.size_test_set}_{args.seed}.json", "w") as f:
                        json.dump(generations + missing_uncertainty_list, f)
            del uncertainty_quantifiers
            _free_gpu()
        else:
            print("Uncertainty quantifiers file is complete")


if __name__ == "__main__":
    main()
    print("Done")