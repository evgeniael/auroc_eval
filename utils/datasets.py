from datasets import load_dataset

def _flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def _extract_answers_aliases(list_of_qa_pairs):
    list_of_answers = [a['answer'] for a in list_of_qa_pairs]
    # Flatten across all QA pairs so downstream evaluator receives List[str].
    return [alias for outer in list_of_answers for sublist in outer for alias in sublist]

def load_test_dataset(dataset_name, size_test_set, seed):
    if dataset_name == "trivia_qa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
        test_ds = ds['validation'].shuffle(seed=seed).select(range(size_test_set))
        question_ids = test_ds['question_id']
        questions = test_ds['question']
        answers_aliases = [a['aliases'] for a in test_ds['answer']]
        return question_ids, questions, answers_aliases
    elif dataset_name == "ambig_qa_single_answer":
        ds = load_dataset("sewon/ambig_qa", split="validation")
        test_ds = ds.shuffle(seed=seed)

        #Filter questions with only one answer
        test_ds = test_ds.filter(lambda x: set(x['annotations']['type']) == {'singleAnswer'})
        test_ds = test_ds.select(range(40,size_test_set))

        question_ids = test_ds['id']
        questions = test_ds['question']
        answers_aliases = [_flatten_list(a['answer']) for a in test_ds['annotations']]
        return question_ids, questions, answers_aliases
    elif dataset_name == "ambig_qa_multiple_qas":
        ds = load_dataset("sewon/ambig_qa", split="validation")
        test_ds = ds.shuffle(seed=seed)#.select(range(size_test_set))

        #Filter questions with multiple QAs
        test_ds = test_ds.filter(lambda x: set(x['annotations']['type']) == {'multipleQAs'})
        test_ds = test_ds.select(range(53,size_test_set))
        
        question_ids = test_ds['id']
        questions = test_ds['question']
        answers_aliases = [_extract_answers_aliases(a['qaPairs']) for a in test_ds['annotations']]

        return question_ids, questions, answers_aliases
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
