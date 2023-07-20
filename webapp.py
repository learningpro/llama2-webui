import os
import sys
import torch
import time
import json

import gradio as gr

from typing import Tuple
from pathlib import Path
from llama import Llama, ModelArgs, Transformer, Tokenizer


ckpt_dir = "llama_xb_model_path"
tokenizer_path = "tokenizer.model"
temperature = 0.8
top_p = 0.8
max_seq_len = 512
max_batch_size = 4


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator


generator = load(
    ckpt_dir, tokenizer_path, max_seq_len, max_batch_size
)


def process(prompt: str):
    dialogs = [
        [
            {"role": "system", "content": "AI"},
            {"role": "user", "content": prompt}
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=256,
        temperature=temperature,
        top_p=top_p,
    )
    return str(results[0]["generation"]["content"])


demo = gr.Interface(
    title="极简Llama2问答对话",
    description="还没有做成连续对话，虽然它可以",
    article="基于Llama2",
    fn = process,
    inputs = gr.Textbox(lines=10, placeholder="请输入。。。", label="用户输入"),
    outputs = "text",
)
demo.launch(share=False)
