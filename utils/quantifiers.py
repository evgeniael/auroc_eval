import gc
import numpy as np
from typing import List
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


class Entropy:
    def __init__(self):
        pass

    def _get_estimator(self, sequences: list[str]) -> tuple[list[str], torch.Tensor]:
        seq_counter = dict(Counter(sequences))
        dist_support = seq_counter.keys()
        counts = list(seq_counter.values())
        dist_probs = [count/sum(counts) for count in counts]
        return dist_support, torch.tensor(dist_probs)

    def _compute_entropy(self, probs:list[float]) -> float:
        log_probs = torch.log(probs)
        entropy = - torch.sum(torch.mul(probs, log_probs))

        return entropy

    def entropy(self, sequences: list[str]) -> float:
        dist_support, dist_probs = self._get_estimator(sequences)
        entropy = self._compute_entropy(dist_probs)

        return entropy

class SemanticEntropy(Entropy):
    def __init__(self, nli_model: object, nli_tokenizer: object):
        super().__init__()
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.nli_model.to('cuda')
        self.nli_model.eval()

    def _get_semantic_clusters(self, question: str, responses: list[str]) -> torch.Tensor:
        unique_responses = list(set(responses))

        answer_list_1 = []
        answer_list_2 = []

        nli_model_inputs = []

        semantic_set_ids = {}
        #Each response is assigned to a unique semantic set
        for index, answer in enumerate(unique_responses):
            semantic_set_ids[answer] = index

        if len(unique_responses) > 1:
            # Evalauate semantic similarity
            #This section of the code is as obtained from the Semantic Entropy repository of Kuhn et. al 
            for i, reference_answer in enumerate(unique_responses):
                for j in range(i + 1, len(unique_responses)):
                    answer_list_1.append(unique_responses[i])
                    answer_list_2.append(unique_responses[j])

                    qa_1 = question + ' ' + unique_responses[i]
                    qa_2 = question + ' ' + unique_responses[j]

                    nli_model_input = qa_1 + ' [SEP] ' + qa_2

                    #We check for 1-way entailment between the two responses
                    nli_model_inputs.append(nli_model_input)
                    encoded_input = self.nli_tokenizer.encode(nli_model_input, padding=True)
                    prediction = self.nli_model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    #We check for the other way entailment between the two responses
                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = self.nli_tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = self.nli_model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    deberta_prediction = 0  
                    if (2 in predicted_label) and (2 in reverse_predicted_label):
                        deberta_prediction = 1 
                        semantic_set_ids[unique_responses[j]] = semantic_set_ids[unique_responses[i]] 

                
        #We return the semantic set ids for each response
        list_of_semantic_set_ids = [semantic_set_ids[x] for x in responses]

        return torch.tensor(list_of_semantic_set_ids)

    def _align_semantics_sets_with_dist(self, sentences: list[str], semantic_sets: torch.Tensor, dist_support: list[str]) -> list[int]:
        dist_semantic_sets = []
        for sentence in dist_support:
            index = sentences.index(sentence)
            dist_semantic_sets.append(semantic_sets[index].item())
        
        return dist_semantic_sets

    def _compute_semantic_entropy(self, probs: torch.Tensor, semantic_sets: torch.Tensor) -> float:
        classes = set(semantic_sets)
        probs_classes = []
        for c in classes:
            probs_items_in_class = torch.where((torch.LongTensor(semantic_sets) == c), probs, torch.zeros(len(probs)))
            prob_class = torch.sum(probs_items_in_class)
            probs_classes.append(prob_class)
        
        sem_entropy = super()._compute_entropy(torch.tensor(probs_classes))

        return sem_entropy

    def semantic_entropy(self, question: str, responses: list[str]) -> float:
        semantic_set_ids = self._get_semantic_clusters(question, responses)
        
        dist_support, dist_probs = super()._get_estimator(responses)
        dist_semantic_sets = self._align_semantics_sets_with_dist(responses, semantic_set_ids, dist_support)
        
        semantic_entropy = self._compute_semantic_entropy(dist_probs, dist_semantic_sets)

        return semantic_entropy

class SemanticConfidence:
    def __init__(self, nli_model: object, nli_tokenizer: object):
        self.nli_model = nli_model
        self.nli_model.to('cuda')
        self.nli_model.eval()
        self.nli_tokenizer = nli_tokenizer
    
    def bidirectional_contra(self, question:str, answer1:str, answer2:str):
        qa_1 = question + ' ' + answer1
        qa_2 = question + ' ' + answer2

        input = qa_1 + ' [SEP] ' + qa_2
        encoded_input = self.nli_tokenizer.encode(input, padding=True)
        
        with torch.no_grad():
            prediction = self.nli_model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
            predicted_label = self._extract_prediction(prediction)
            
        reverse_input = qa_2 + ' [SEP] ' + qa_1
        encoded_reverse_input = self.nli_tokenizer.encode(reverse_input, padding=True)
        with torch.no_grad():
            reverse_prediction = self.nli_model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
            reverse_predicted_label = self._extract_prediction(reverse_prediction)

        is_contradiction = 0  #semantically not equivalent
        if (predicted_label == True) and (reverse_predicted_label == True):
            is_contradiction = 1 #semantically equivalent
        
        return is_contradiction
    
    def _extract_prediction(self, prediction):
        #Prediction labels: "label2id": {"CONTRADICTION": 0,"NEUTRAL": 1,"ENTAILMENT": 2}
        p = torch.nn.Softmax(dim=1)
        prediction = p(prediction)
        prob_contra = prediction[0][0] 
        prob_no_contra = prediction[0][1] + prediction[0][2]

        is_contradiction = prob_contra > prob_no_contra #True if no contradiction, False if contradiction
        return is_contradiction

    def semantic_confidence(self, question: str, prediction: str, responses: list[str]) -> float:
        semantic_contradictions = []
        for response in responses:
            semantic_contradictions.append(self.bidirectional_contra(question, prediction, response))
        
        semantic_confidence = 1 - (sum(semantic_contradictions) / len(semantic_contradictions))
        return semantic_confidence
    
class LogLikelihood:
    def __init__(self):
        pass
    
    def get_sequence_log_likelihood(self, log_probs: list[float]) -> float:
        try:
            log_likelihood = sum(log_probs)
        except:
            log_likelihood = log_probs
        return log_likelihood

    def get_average_token_log_likelihood(self, log_probs: list[float]) -> float:
        try:
            average_token_log_likelihood = sum(log_probs) / len(log_probs)
        except:
            average_token_log_likelihood = log_probs
        return average_token_log_likelihood

class PTrue:
    def __init__(self, model: object, tokenizer: object):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = 1000
        self.model.to('cuda')
        self.model.eval()
    
    def _construct_input(self, question:str, responses:str, prediction:str):
        #Construct p adequate prompt
        prompt = f"Question: '{question}' \n Here are some brainstormed ideas:"
        
        for response in responses:
            prompt += f"{response} \n"
        
        prompt += f"\nPossible answer: {prediction} \nIs the possible answer: (a) true (b) false \nGenerate a or b. \nThe answer is:"

        messages = [
            {"role": "user", "content": prompt}
            ]
        
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        return inputs

    def get_p_true(self, question: str, responses: list[str], prediction: str):
        """Get the probability of the model anwering a (True) for the given input."""
        inputs = self._construct_input(question, responses, prediction)
        inputs = inputs.to('cuda')

        with torch.no_grad():
            outputs = self.model(inputs, labels=inputs)
            logits = outputs.logits
            logits = logits[-1, -1]
            log_probs = torch.log(torch.nn.functional.softmax(logits, dim=-1))

        index_true = self.tokenizer.convert_tokens_to_ids('a')
        index_false = self.tokenizer.convert_tokens_to_ids('b')

        assert index_true != index_false, 'Check whether tokeniser has correct tokens for options of interest'

        log_prob_true = log_probs[index_true]
        log_prob_false = log_probs[index_false]

        cond_prob_true = torch.exp(log_prob_true)/ (torch.exp(log_prob_true) + torch.exp(log_prob_false))

        return torch.log(cond_prob_true).item()

class VerbalisedConfidence:
    def __init__(self, model: object, tokenizer: object):
        self.model = model
        self.tokenizer = tokenizer
        self.model.to('cuda')
        self.model.eval()
        self.max_length = 1500
    
    def _construct_input(self, question:str, prediction:str):
        #Construct p adequate prompt
        # Remove system prompt since we're not using it
        prompt = f"""\nQuestion: {question} \nAnswer previously generated: {prediction} \nGenerate ONLY your numerical confidence in the answer (which is 0-1), up to 2 decimal places. \nConfidence:"""
        messages = [{"role": "system", "content": "You are a helpful assistant that generates faithful numerical confidence values in an answer."}, {"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        return inputs
    
    def generate_greedy(self, inputs: torch.Tensor):
        outputs = self.model.generate(inputs, max_length=self.max_length, num_beams=1, do_sample=False,
                                    return_dict_in_generate=True, output_scores=True, pad_token_id=self.tokenizer.eos_token_id)
        
        generated_tokens = outputs.sequences[:, inputs.shape[-1]:]
        generated_string = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        return generated_string

    def get_verbalised_confidence(self, question: str, prediction: str):
        inputs = self._construct_input(question, prediction)
        inputs = inputs.to('cuda')

        verbalised_confidence = self.generate_greedy(inputs)
        return verbalised_confidence


class UncertaintyQuantifiers:
    def __init__(self, model_name: str, nli_model_name: str):
        """
        Initialize evaluation tools.
        Llama stays on CPU until p_true / verbalised (saves ~16–32GB on GPU during NLI).
        """
        self.tokenizer, self.model = self._load_tokenizer_and_model(model_name)
        self.nli_tokenizer, self.nli_model = self._load_tokenizer_and_model(nli_model_name)
        self.max_length = 1000
        # Default: both CPU so first evaluate() can run NLI-only on GPU without OOM
        self.model.cpu()
        self.nli_model.cpu()
        gc.collect()

    def _load_tokenizer_and_model(self, model_name):
        if model_name == "llama-8b-instruct":
            mid = "meta-llama/Llama-3.1-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(mid)
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
        elif model_name == "llama-3b-instruct":
            mid = "meta-llama/Llama-3.2-3B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(mid)
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
        elif model_name == "qwen-7b-instruct":
            mid = "Qwen/Qwen2.5-7B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(mid)
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
        elif model_name == "qwen-0.5b-instruct":
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(mid)
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
        elif model_name == "deberta-large":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
            model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            )
        else:
            raise ValueError(f"Model {model_name} not found")

        return tokenizer, model

    def _gpu_nli_only(self):
        """Llama off GPU; Deberta on GPU (avoids 32GB fp32 Llama + Deberta OOM)."""
        self.model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.nli_model.to("cuda")
            self.nli_model.eval()
        else:
            self.nli_model.eval()

    def _gpu_llama_only(self):
        """Deberta off GPU; Llama on GPU for generation."""
        self.nli_model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.model.to("cuda")
            self.model.eval()
        else:
            self.model.eval()

    def evaluate(
        self,
        question: str,
        responses: list[str],
        prediction: str,
        log_probs: list[float],
        *,
        _shared_entropy_semantic=None,
    ):
        """Evaluate the quantifiers for the given input.

        If you call evaluate multiple times with the same ``responses`` (e.g. greedy
        vs sampled vs nucleus), pass the return value's entropy + semantic_entropy
        as ``_shared_entropy_semantic`` on the 2nd and 3rd calls to skip redundant
        NLI work (semantic entropy is expensive).
        """
        entropy_object = Entropy()
        entropy_score = entropy_object.entropy(responses)

        self._gpu_nli_only()
        if _shared_entropy_semantic is not None:
            semantic_entropy_score = torch.tensor(_shared_entropy_semantic["semantic_entropy"])
        else:
            semantic_entropy_object = SemanticEntropy(self.nli_model, self.nli_tokenizer)
            semantic_entropy_score = semantic_entropy_object.semantic_entropy(
                question, responses
            )

        semantic_confidence_object = SemanticConfidence(self.nli_model, self.nli_tokenizer)
        semantic_confidence_score = semantic_confidence_object.semantic_confidence(
            question, prediction, responses
        )

        log_likelihood_object = LogLikelihood()
        log_likelihood_seq_score = log_likelihood_object.get_sequence_log_likelihood(
            log_probs
        )
        log_likelihood_token_score = log_likelihood_object.get_average_token_log_likelihood(
            log_probs
        )

        self._gpu_llama_only()
        p_true = PTrue(self.model, self.tokenizer)
        p_true_score = p_true.get_p_true(question, responses, prediction)

        verbalised_confidence_object = VerbalisedConfidence(self.model, self.tokenizer)
        verbalised_confidence_score = verbalised_confidence_object.get_verbalised_confidence(
            question, prediction
        )

        se_item = (
            semantic_entropy_score.item()
            if hasattr(semantic_entropy_score, "item")
            else float(semantic_entropy_score)
        )
        return {
            "entropy": entropy_score.item(),
            "semantic_entropy": se_item,
            "semantic_confidence": semantic_confidence_score,
            "log_likelihood_seq": log_likelihood_seq_score,
            "log_likelihood_token": log_likelihood_token_score,
            "p_true": p_true_score,
            "verbalised_confidence": verbalised_confidence_score
        }