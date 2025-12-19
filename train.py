"""Finetune a language model using QLoRA to do the idiomatic decompilation task.
"""

import argparse
import functools
import os
import json
import random
from pathlib import Path
from typing import Callable, TypeVar, TYPE_CHECKING, Any

import torch
from unsloth import FastLanguageModel
import wandb

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from idioms.data.dataset import MatchedFunction, MatchedBinary
from idioms.dataiter import MatchedFunctionDataset, MatchedBinaryDataset, MatchedBinaryFunctionWrapper
from idioms.hf import (
    stringify_function_target,
    causal_stringify_function_prompt,
    causal_stringify_neighbors_prompt,
    causal_stringify_binary_prompt
)


ADAPTER_NAME="decomp_fn_rewrite"
DEBUG_RUN = False

T = TypeVar("T")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("run_name")
    parser.add_argument("mode", choices=["function", "neighbors", "binary"], help="The type of context used: function only or whole-binary.")
    parser.add_argument("training_type", choices=["finetune", "adapter"], help="Whether to full-finetune the model or use QLoRA")
    parser.add_argument("--model-type", default="unsloth/Qwen2.5-Coder-3B-bnb-4bit") #"unsloth/codegemma-2b-bnb-bit")
    parser.add_argument("--nhops", type=int, default=1, help="In 'neighbors' mode, include in the context all functions up and including to this many edges away from the given function in the call graph.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32, help="Size of LoRA internal matrix dimension.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=2, help="Number of backward passes per optimizer step.")
    parser.add_argument("--warmup-iters", type=int, default=0, help="Number of steps to increase from 0 to reach the maximum learning rate.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum length of input sequence to model.")
    parser.add_argument("--max-prediction-tokens", type=int, default=1024, help="In neighbors mode, how many tokens to allocate to the prediction vs the context.") # could add this to binary mode too.
    parser.add_argument("--max-fns-per-binary", type=int, default=8, help="In 'binary' mode, the number of randomly-selected functions per binary to predict.")
    parser.add_argument("--random-seed", type=int, default=80, help="Seed for python's random module.")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Resume training from this checkpoint.")
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "auto"], default="auto", help="The checkpointing strategy to use: save every epoch or after --save-steps steps. Defaults to 'epoch' for small datasets and 'steps' for large datasets.")
    parser.add_argument("--save-steps", type=int, default=1000, help="In steps mode, the number of steps since the last save to save a checkpoint.")
    parser.add_argument("--eval-accumulation-steps", type=int, default=1, help="Number of evaluation batches to accumulate on device before moving to CPU. Use 1 to minimize GPU memory during evaluation.")
    return parser.parse_args()

def causal_stringify_function(fn: MatchedFunction) -> str:
    return causal_stringify_function_prompt(fn) + stringify_function_target(fn)

def causal_stringify_neighbors(
    input: tuple[MatchedBinary, int],
    nhops: int,
    tokenizer: "PreTrainedTokenizerBase",
    max_context: int | None = None,
) -> str:
    binary, tgt_fn_idx = input
    fn = binary.functions[tgt_fn_idx]
    return causal_stringify_neighbors_prompt(binary, fn, nhops, tokenizer, max_context) + stringify_function_target(fn)

def causal_stringify_binary(input: tuple[MatchedBinary, int]) -> str:
    """Create an input string that contains all of the decompiled code in the binary, the prompt
    to decompile a particular function, and the original function itself.
    """
    binary, tgt_fn_idx = input
    fn = binary.functions[tgt_fn_idx]
    return causal_stringify_binary_prompt(binary, fn) + stringify_function_target(fn)

def causal_train_collate(batch: list[T], tokenizer, stringify: Callable[[T], str], max_length: int):
    """Convert a batch into input IDs and an attention mask.
    """
    sequences: list[str] = [stringify(ex) for ex in batch]
    encoded_batch = tokenizer(sequences, return_tensors='pt', max_length=max_length, padding=True, truncation=True)
    labels = encoded_batch["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    encoded_batch["labels"] = labels
    return encoded_batch

def main(args: argparse.Namespace):
    # Import transformers lazily: in some environments `transformers.Trainer` import
    # fails due to optional dependency / lazy-module resolution issues.
    from transformers import Trainer, TrainingArguments

    global DEBUG_RUN
    DEBUG_RUN = DEBUG_RUN or args.run_name == "temp"
    random.seed(args.random_seed)
    dataset_path = Path(args.dataset)
    assert dataset_path.exists(), f"Dataset {dataset_path} does not exsit!"
    model_type: str = args.model_type
    lora_rank: int = args.lora_rank
    batch_size: int = args.batch_size
    warmup_iters: int = args.warmup_iters
    max_length: int = args.max_length if not DEBUG_RUN else 16
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_fns_per_binary: int | None = args.max_fns_per_binary if args.mode == "binary" else None
    use_adapter: bool = args.training_type == "adapter"
    run_name: str = args.run_name
    save_steps: int = args.save_steps
    save_strategy: str = args.save_strategy
    if args.resume_from_checkpoint is None:
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = Path(args.resume_from_checkpoint)
        assert resume_from_checkpoint.exists(), f"Checkpoint {resume_from_checkpoint} does not exist!"
        assert resume_from_checkpoint.parent.name == run_name, f"The checkpoint {resume_from_checkpoint} is not from run {run_name}!"
        # checkpoint_step used in fork_from wandb feature.
        # checkpoint_step = int(resume_from_checkpoint.name.split("-")[1]) # unfortunately relies on the standard format fo the checkpoint directory names.
    
    assert "4bit" not in model_type or use_adapter, f"4-bit models like {model_type} require an adapter."

    print(f"Using model of type {model_type}")
    print(f"Using compute dtype: {compute_dtype}")
    print(f"Sequence max length: {max_length}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_type,
        # model_type if use_adapter or resume_from_checkpoint is None else resume_from_checkpoint, 
        max_seq_length=max_length, 
        load_in_4bit=use_adapter
    )

    if args.mode == "function":
        stringify = causal_stringify_function
        train_set = MatchedFunctionDataset(dataset_path.glob("train*.tar"), length_cache=dataset_path / "length_cache.pkl")
        eval_set = MatchedFunctionDataset(dataset_path.glob("validation*.tar"), length_cache=dataset_path / "length_cache.pkl")
    else:
        train_set = MatchedBinaryFunctionWrapper(
            MatchedBinaryDataset(dataset_path.glob("train*.tar"), length_cache=dataset_path / "length_cache.pkl"),
            max_fns_per_binary=max_fns_per_binary if args.mode == "binary" else None
        )
        eval_set = MatchedBinaryFunctionWrapper(
            MatchedBinaryDataset(dataset_path.glob("validation*.tar"), length_cache=dataset_path / "length_cache.pkl"),
            max_fns_per_binary=max_fns_per_binary if args.mode == "binary" else None
        )

        if args.mode == "neighbors":
            stringify = functools.partial(causal_stringify_neighbors, nhops=args.nhops, tokenizer=tokenizer, max_context=(max_length - args.max_prediction_tokens))
        else:
            stringify = causal_stringify_binary

    if save_strategy == "auto":
        save_strategy = "steps" if len(train_set) > 1000000 else "epoch"
        print(f"Dataset size is {len(train_set)}; defaulting to {save_strategy} save strategy.")
    
    print(f"Using a {type(tokenizer)} tokenizer with a {type(model)} model.")
    print(f"Max length: {tokenizer.model_max_length} (tokenizer), {model.config.max_position_embeddings} (model)")

    ### Handle padding tokens.
    print("Existing special tokens")
    print(tokenizer.special_tokens_map)
    # pad_token_type_id is what the default transformers DataCollatorWithPadding class uses to pad (through a call
    # to .pad and ._pad in PreTrainedTokenizerBase) actually uses to fill the input tensors with padding
    print(f"Pad token type id: {tokenizer.pad_token_type_id}")
    print(f"BOS token into: model: {model.config.bos_token_id}; tokenizer: {tokenizer.bos_token}, {tokenizer.bos_token_id}")
    print(f"EOS token info: model: {model.config.eos_token_id}; tokenizer: {tokenizer.eos_token}, {tokenizer.eos_token_id}")

    if use_adapter:
        model = FastLanguageModel.get_peft_model(
            model, 
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0,
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",
            max_seq_length=max_length,
        )

    idioms_config = {"model_type": args.model_type, "dataset": args.dataset, "mode": args.mode, "nhops": args.nhops, "max_seq_len": max_length, "adapter": use_adapter}
    output_dir = Path("runs") / run_name
    os.makedirs(output_dir, exist_ok=True)
    idioms_config_path = output_dir / "idioms_config.json"

    if idioms_config_path.exists():
        with open(idioms_config_path, "r") as fp:
            existing_config = json.load(fp)
        assert DEBUG_RUN or idioms_config == existing_config, f"{output_dir} already exists with a differing config file:\n" + \
               "\n".join(f"  {k}: {v}" for k, v in set(idioms_config.items()) ^ set(existing_config.items()))
    else:
        with open(idioms_config_path, "w") as fp:
            json.dump(idioms_config, fp)

    # Wandb stuff. Handle this manually instead of letting the trainer do it so we
    # can resume an existing wandb run rather than start a new one.
    if not DEBUG_RUN:
        os.environ["WANDB_PROJECT"]="idioms"
    runid_file = output_dir / "wandb_run_id.txt"
    if resume_from_checkpoint is None or not runid_file.exists():
        run = wandb.init(name=run_name)
        with open(runid_file, "w") as fp:
            fp.write(run.id)
    else:
        with open(runid_file, "r") as fp:
            run_id = fp.read().strip().splitlines()[-1]
        wandb.init(id=run_id, resume="must")
        # fork_from is slightly preferable but currently in private beta.
        # run = wandb.init(name=run_name + "-cont", fork_from=f"{run_id}?_step={checkpoint_step}")
        # with open(runid_file, "a") as fp:
        #     fp.write("\n")
        #     fp.write(run.id)

    training_args = TrainingArguments(
        bf16=compute_dtype==torch.bfloat16,
        fp16=compute_dtype==torch.float16,
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=50,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size, # With split_batches=True this is a lie: it is actually the total batch size.
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False, # Allows me to use my custom collate_fn function which is much cleaner than the examples/tutorial suggested methods.
        dataloader_drop_last=True,
        warmup_steps=warmup_iters,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=args.gradient_accumulation,
        run_name=run_name,
        save_strategy=save_strategy,
        save_steps=save_steps,
        # save_total_limit=5,
        report_to="wandb",
        logging_steps=50 if not DEBUG_RUN else 1,
        logging_strategy="steps",
        eval_accumulation_steps=args.eval_accumulation_steps,
    )

    trainer = Trainer(
        model, 
        args=training_args, 
        train_dataset=train_set, 
        eval_dataset=eval_set,
        processing_class=tokenizer,
        data_collator=functools.partial(
                causal_train_collate, 
                tokenizer=tokenizer, 
                stringify=stringify, # type: ignore # not handling the stringify options correctly.
                max_length=max_length
            ) 
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

if __name__ == "__main__":
    main(get_args())
    

