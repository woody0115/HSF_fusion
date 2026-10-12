"""Recreate configs/owod_text_embeds.npz (originally produced on 2026-10-04 by this exact code as a one-off command).

Text encoder: the YOLO-UniOW-S text model exported for the Windows demo (E:\\claude_Workspace\\YOLO_UniOW_test\\
models\\text_model_s): a Hugging Face CLIPTextModelWithProjection from openai/clip-vit-base-patch32 with
YOLO-UniOW's text-encoder LoRA already merged into the weights. Prompts are the bare class names (no template).
"object_tuned_uniow" is YOLO-UniOW-S's tuned wildcard embedding (object_tuned_s.npy), L2-normalised.

Needs `transformers` (present in the YOLO_UniOW_test venv, not in this project's venv):
    E:\\claude_Workspace\\YOLO_UniOW_test\\.venv\\Scripts\\python.exe tools\\make_text_embeds.py
"""
import os.path as osp
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection

UNIOW = r'E:\claude_Workspace\YOLO_UniOW_test\models'
OUT = osp.join(osp.dirname(__file__), '..', 'configs', 'owod_text_embeds.npz')
NAMES = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'object']


def main():
    d = osp.join(UNIOW, 'text_model_s')
    tok = AutoTokenizer.from_pretrained(d)
    m = CLIPTextModelWithProjection.from_pretrained(d).eval()
    with torch.no_grad():
        t = tok(NAMES, return_tensors='pt', padding=True)
        f = m(**t).text_embeds
        f = f / f.norm(dim=-1, keepdim=True)
    f = f.numpy().astype(np.float32)
    wild = np.load(osp.join(UNIOW, 'object_tuned_s.npy')).astype(np.float32).reshape(1, -1)
    wild = wild / np.linalg.norm(wild)
    out = sys.argv[1] if len(sys.argv) > 1 else OUT          # optional output path (for checks)
    np.savez(out, names=np.array(NAMES), embeds=f, object_tuned_uniow=wild)
    sim = f @ f.T
    print('saved', f.shape, '| cosine(car,truck)=%.2f (car,pedestrian)=%.2f (object,car)=%.2f (object,pedestrian)=%.2f'
          % (sim[2, 4], sim[2, 0], sim[8, 2], sim[8, 0]))


if __name__ == '__main__':
    main()
