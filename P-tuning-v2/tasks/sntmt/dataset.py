from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


class SntmtDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        if data_args.dataset_name != "imdb":
            raise NotImplementedError('Sentiment Analysis may only be done with IMDB dataset.')
        raw_datasets = load_dataset("imdb", data_args.dataset_name)
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.multiple_choice = True

        self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = (None, None)

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        
        raw_datasets = raw_datasets.map(
	    self.preprocess_function,
	    batched=True,
	    load_from_cache_file=not data_args.overwrite_cache,
	    desc="Running tokenizer on dataset",
         )

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.metric = load_metric("imdb", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        self.test_key = "accuracy"

    def preprocess_function(self, examples):
        if self.data_args.dataset_name == "imdb":
            examples["text_a"] = []
       # COPA
        if self.data_args.dataset_name == "copa":
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"                    
                examples["text_a"].append(text_a)

            result1 = self.tokenizer(examples["text"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
#            result2 = self.tokenizer(examples["text_a"], examples["choice2"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
            result = {}
            # this may need to be cleaned up
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1:
                    result[key].append(result1[key])
            return result

        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        '''if self.data_args.dataset_name == "record":
            return self.reocrd_compute_metrics(p)

        if self.data_args.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(preds, p.label_ids)}'''

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
