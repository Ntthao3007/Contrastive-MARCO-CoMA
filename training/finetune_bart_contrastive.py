import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, AdamW, EarlyStoppingCallback, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import nn
import argparse
import random
from IPython import embed
from utils import *
from infilling import text_infill


def generate_negatives(model, input_ids, num_to_generate=6):
    beam_output = model.generate(input_ids, max_length=50, do_sample=True, num_beams=num_to_generate, num_return_sequences=num_to_generate, early_stopping=True)
    texts = tokenizer.batch_decode(beam_output, skip_special_tokens=True)
    return texts, beam_output

def get_score(model, x, y, i, j):
    outputs = model(input_ids=x['input_ids'][i].reshape(1, -1), attention_mask=x['attention_mask'][i].reshape(1, -1), labels=y['input_ids'][j].reshape(1, -1))
    score = -1 * outputs.loss
    return score

def compute_contrastive_loss(model, x, y):
    loss = 0
    batch_size = x['input_ids'].shape[0]
    for i in range(batch_size):
        texts, beam_output = generate_negatives(model, x['input_ids'][i].reshape(1, -1))
        target_score = get_score(model, x, y, i, i)
        denum = target_score
        y_generate = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        for j in range(len(texts)):
            s = get_score(model, x, y_generate, i, j)
            denum = denum + s
        loss += -1 * torch.log(target_score / denum)
    loss = loss / batch_size
    return loss

def unlikelihood_loss(model, x, y):
    outputs2 = model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], labels=x['input_ids'])
    loss = -1 * torch.log(1 - torch.exp(-1 * outputs2.loss))
    return loss

def compute_contrastive_loss5(model, x, y):
    loss = 0
    batch_size = x['input_ids'].shape[0]
    for i in range(batch_size):
        target_score = get_score(model, x, y, i, i)
        j = i
        s = get_score(model, x, x, i, j)
        loss += -1 * torch.log(torch.exp(target_score) / (torch.exp(target_score) + torch.exp(s)))
    loss = loss / batch_size
    return loss

def compute_contrastive_loss8(model, x, y):
    loss = 0
    batch_size = x['input_ids'].shape[0]
    for i in range(batch_size):
        target_score = get_score(model, x, y, i, i)
        s = 0
        for j in range(batch_size):
            s += torch.exp(get_score(model, x, x, i, j))
        loss += -1 * torch.log(torch.exp(target_score) / (torch.exp(target_score) + s))
    loss = loss / batch_size
    return loss

def compute_language_modeling_loss(model, x, y):
    outputs = model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], labels=y['input_ids'])
    return outputs.loss

def combined_contrastive_loss(model, x, y, alpha):
    return compute_language_modeling_loss(model, x, y) + alpha * compute_contrastive_loss5(model, x, y)

def combined_unlikelihood_loss(model, x, y, alpha):
    return compute_language_modeling_loss(model, x, y) + alpha * unlikelihood_loss(model, x, y)

def combined_contrastive_loss_with_negatives(model, x, y, alpha):
    return compute_language_modeling_loss(model, x, y) + alpha * compute_contrastive_loss8(model, x, y)

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not torch.cuda.is_available():
        print("No GPUs found!")
    else:
        print("Found", str(torch.cuda.device_count()), "GPUS!")

    seed_everything(args.seed)

    # Load in the tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.tok_type)
    mask = tokenizer.mask_token

    if not os.path.exists(args.model_dir):
        print(f"Model directory {args.model_dir} does not exist. Loading model directly.")
        tokenizer = AutoTokenizer.from_pretrained("hallisky/bart-base-nontoxic-expert")
        model = AutoModelForSeq2SeqLM.from_pretrained("hallisky/bart-base-nontoxic-expert").to(device)

        output_dir = os.path.join(args.model_dir, f"{args.model_type.split('/')[-1]}_{args.lr}_{args.seed}_{args.train_batch_size * torch.cuda.device_count()}_{args.data_type}")
        print(output_dir)
    else:
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained("hallisky/bart-base-nontoxic-expert")
        model = AutoModelForSeq2SeqLM.from_pretrained("hallisky/bart-base-nontoxic-expert")

        output_dir = os.path.join(args.model_dir, f"{args.model_type.split('/')[-1]}_{args.lr}_{args.seed}_{args.train_batch_size * torch.cuda.device_count()}_{args.data_type}")
        print(output_dir)

        try:
            prev_models = os.listdir(output_dir)
            prev_models.sort()
            prev_models.sort(key=len)
        except FileNotFoundError:
            prev_models = []

        if args.load_old and len(prev_models) > 0:
            model = BartForConditionalGeneration.from_pretrained(os.path.join(output_dir, prev_models[-1]), forced_bos_token_id=tokenizer.bos_token_id).to(device)
        else:
            model = BartForConditionalGeneration.from_pretrained(args.model_type, forced_bos_token_id=tokenizer.bos_token_id).to(device)

    train_texts = []
    val_texts = []

    if "jigsaw" in args.data_type:
        train = pd.read_csv(args.train_data)
        val = pd.read_csv(args.val_data)
        train_texts = train["comment_text"].tolist()
        val_texts = val["comment_text"].tolist()
    elif "dynabench" in args.data_type:
        df = pd.read_csv(args.train_data)
        df_lab = "hate"
        if "nothate" in args.data_type:
            df_lab = "nothate"
        if "all" in args.data_type:
            df = df[df.label == df_lab]
        else:
            df_round = int(args.data_type[-1])
            inputs = df[df.label == df_lab][df["round.base"] == df_round]
        train_texts = df[df.split == "train"].text.tolist()
        val_texts = df[df.split == "dev"].text.tolist()

    print(len(train_texts), len(val_texts))

    tokenized_labs_train = tokenizer.batch_encode_plus(
        train_texts,
        max_length=args.max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    tokenized_labs_val = tokenizer.batch_encode_plus(
        val_texts,
        max_length=args.max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    tokenized_labs_val[tokenized_labs_val == tokenizer.pad_token_id] = -100
    tokenized_labs_train[tokenized_labs_train == tokenizer.pad_token_id] = -100

    class JigsawDataset(torch.utils.data.Dataset):
        def __init__(self, labels, rawdata):
            self.rawdata = rawdata
            self.labels = labels

        def __getitem__(self, idx):
            item = tokenizer.encode_plus(
                detokenize(text_infill(self.rawdata[idx], tokenizer.mask_token)),
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            item = {key: torch.tensor(val[0]) for key, val in item.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = JigsawDataset(tokenized_labs_train, train_texts)
    val_dataset = JigsawDataset(tokenized_labs_val, val_texts)

    alpha = args.alpha

    if args.contrastive_loss:
        if not args.add_negatives:
            loss_function = combined_contrastive_loss
        else:
            loss_function = combined_contrastive_loss_with_negatives
    elif args.unlikelihood:
        loss_function = combined_unlikelihood_loss
    else:
        loss_function = compute_language_modeling_loss

    class BartTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            x = {'input_ids': inputs.get("input_ids"), 'attention_mask': inputs.get("attention_mask")}
            y = {'input_ids': inputs.get("labels")}
            loss = loss_function(model, x, y, alpha)
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, **kwargs):
            metrics = super().evaluate(**kwargs)
            inputs_masked = ["Asians are <mask>"]
            print("\n Generations: ", tokenizer.batch_decode(model.generate(tokenizer(inputs_masked, return_tensors="pt").to('cuda').input_ids, do_sample=True, num_return_sequences=5)), "\n")
            return metrics

    print("Before training:")
    inputs_masked = ["Asians are <mask>"]
    print("\n Generations: ", tokenizer.batch_decode(model.generate(tokenizer(inputs_masked, return_tensors="pt").to('cuda').input_ids, do_sample=True, num_return_sequences=5)), "\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        fp16=args.fp16,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        seed=args.seed
    )

    trainer = BartTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(args.early_stopping_steps)]
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok_type", default="facebook/bart-base", help="Tokenizer of model to fine-tune")
    parser.add_argument("--model_type", default="facebook/bart-base", help="Model to fine-tune")
    parser.add_argument("--train_data", default="/home/ubuntu/20thao.nt/TST/MarcoDetoxification/datasets/jigsaw_full_30/train_toxic.csv", help="Path to train set; ether the toxic or nontoxic split of jigsaw")
    parser.add_argument("--val_data", default="/home/ubuntu/20thao.nt/TST/MarcoDetoxification/datasets/jigsaw_full_30/val_toxic.csv", help="Path to dev set; same split as above")
    parser.add_argument("--model_dir", default="/home/ubuntu/20thao.nt/TST/MarcoDetoxification/rewriting/models/toxic_contrast")
    parser.add_argument("--max_source_length", type=int, default=182, help="max source length (based on 99th percentile)")
    parser.add_argument("--max_target_length", type=int, default=232, help="max target length (based on 99th percentile)")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--data_type", default="jigsaw_full_30")
    parser.add_argument("--logging_dir", default='/home/ubuntu/20thao.nt/TST/MarcoDetoxification/rewriting_data/toxic_contrast/models/logs')
    parser.add_argument("--early_stopping_steps", type=int, default=5)
    parser.add_argument("--load_old", action="store_true", help="use if you want to continue training the previous model")
    parser.add_argument("--contrastive_loss", action="store_true", help="use contrastive loss")
    parser.add_argument("--add_negatives", action="store_true", help="add negatives for contrastive loss")
    parser.add_argument("--unlikelihood", action="store_true", help="use unlikelihood loss")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for combined loss functions")
    args = parser.parse_args()
    main(args)
