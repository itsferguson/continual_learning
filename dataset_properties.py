# Column keys used in the dataset
task_to_keys = {
    "cola": ("sentence"),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    # "qnli": ("text1", "text2"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence"),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("passage", "question"),
    "copa": ("choice1", "choice2", "premise", "question"),
    "wic": ("start1", "end1", "sentence1", "start2", "end2", "sentence2", "word"),
    "wsc": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
    "wsc_bool": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
    "cb": ("premise", "hypothesis"),
    "record": ("passage", "query", "entities"),
    "multirc": ("question", "answer", "paragraph"),
    "rte_superglue": ("premise", "hypothesis"),
    "scicite": ("sectionName", "string"),
    "imdb": ("text"),
    "ag_news": ("text"),
    "yelp_review_full": ("text"),
    "yahoo_answers_topics": ("question_content", "best_answer"),
    "dbpedia_14": ("title", "content"),
    "amazon": ("content"),
}

# Label text for T5 tasks
# (T5 has text-to-text format for text and labels)
task_to_labels = {
    "cola": ("not_acceptable", "acceptable"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "mnli-mm": (),
    "mrpc": ("not_equivalent", "equivalent"),
    "qnli": ("entailment", "not_entailment"),
    "qqp": ("not_duplicate", "duplicate"),
    "rte": ("entailment", "not_entailment"),
    "sst2": ("negative", "positive"),
    "stsb": (),
    "wnli": (),
    "boolq": ("false", "true"),
    "copa": ("false", "true"),
    "wic": ("false", "true"),
    "wsc_bool": ("false", "true"),
    "cb": ("entailment", "contradiction", "neutral"),
    "multirc": ("false", "true"),
    "rte_superglue": ("entailment", "not_entailment"),
    "scicite": (),
    "imdb": ("negative", "positive"),
    "ag_news": ("world", "sports", "business", "science"),
    "yelp_review_full": ("terrible", "bad", "middle", "good", "wonderful"),
    "yahoo_answers_topics": (
        "society and culture",
        "science",
        "health",
        "education and reference",
        "computers and internet",
        "sports",
        "business",
        "entertainment and music",
        "family and relationships",
        "politics and government",
    ),
    "dbpedia_14": (
        "company",
        "educationalinstitution",
        "artist",
        "athlete",
        "officeholder",
        "meanoftransportation",
        "building",
        "naturalplace",
        "village",
        "animal",
        "plant",
        "album",
        "film",
        "writtenwork",
    ),
    "amazon": ("terrible", "bad", "middle", "good", "wonderful"),
}

task_to_target_len = {
    "rte": 5,
    "mrpc": 5,
    "sst2": 2,
    "qqp": 5,
    "cola": 5,
    "qnli": 5,
    "mnli": 5,
    "stsb": 3,
    "wic": 2,
    "boolq": 2,
    "copa": 2,
    "wsc": 3,
    "wsc_bool": 2,
    "cb": 5,
    "multirc": 5,
    "record": 10,
    "rte_superglue": 5,
    "imdb": 2,
    "ag_news": 2,
    "yahoo_answers_topics": 5,
    "dbpedia_14": 5,
    "amazon": 2,
    "yelp_review_full": 2,
}


def get_label_key(task):
    # Default label key
    label_key = "label"

    # Check special cases
    if "yahoo_" in task:
        label_key = "topic"
    if "stsb" in task:
        label_key = "similarity_score"
    if task == "record":
        label_key = "answers"

    return label_key


def get_input_keys(task):
    return task_to_keys[task]


def get_labels(task):
    return task_to_labels[task]


def is_glue_dataset(task):
    glue_datasets = [
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "stsb",
        "mnli",
        "mnli_mismatched",
        "mnli_matched",
        "qnli",
        "rte",
        "wnli",
        "ax",
    ]
    return task in glue_datasets


def is_superglue_dataset(task):
    superglue_datasets = [
        "copa",
        "boolq",
        "wic",
        "wsc",
        "cb",
        "record",
        "multirc",
        "rte_superglue",
        "wsc_bool",
    ]
    return task in superglue_datasets


def is_local_dataset(task):
    return task in [
        "amazon",
    ]


def has_non_discrete_labels(task):
    return task in ["stsb", "record", "wsc"]


def get_target_length(task):
    return task_to_target_len[task]


def adjust_k_per_class(task, k, get_test_subset):
    if task in [
        "mrpc",
        "cola",
        "copa",
        "rte",
        "rte_superglue",
        "cb",
        "wsc",
        "wsc_bool",
    ]:
        if k > 500 or task in ["cb", "copa", "wsc", "wsc_bool"]:
            k = -1
        k_val = -1
    elif get_test_subset is False:
        k_val = -1  # use all val set
    else:
        k_val = max(500, int(0.2 * k)) if task != "sst2" else 400

    return k, k_val


def has_dedicated_val_split(task):
    return task in ["stsb", "scicite"]
