"""Label-likelihood scoring with a reused prefix cache.

Prediction is by scoring every candidate label rather than by free generation.
That removes parse failures and yields a full distribution over the label set
from the same pass that produces the prediction, so the three confidence signals
in `confidence_signals` cost nothing extra.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .config import PROMPT


# --------------------------------------------------------------------------
def load_model(model_id: str, four_bit: bool):
    """Half precision throughout; four-bit NF4 only for the large tier.

    The large tier and its quantisation are identical across both cascades, so
    precision is held constant and cannot account for any difference between the
    1.7B and 4B results. It does make the absolute cost ratios specific to a
    single-accelerator deployment.
    """
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = {"device_map": "auto"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    try:                                      # transformers >= 4.56
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kw)
    except TypeError:                         # older releases
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kw)
    return tok, model.eval()


def truncate(tok, text: str, max_tokens: int) -> str:
    ids = tok(text, add_special_tokens=False).input_ids
    return text if len(ids) <= max_tokens else tok.decode(ids[:max_tokens])


def build_prefix(tok, text: str, labels: list[str]) -> str:
    """Chat-templated prompt with thinking disabled.

    Qwen3 enables a reasoning block by default, which would open a <think> span
    exactly where the label distribution is read.
    """
    content = PROMPT.format(labels="\n".join(f"- {l}" for l in labels), text=text)
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def crop_cache(past, length: int):
    """Truncate the key-value cache back to the shared prefix.

    This is load-bearing. A DynamicCache is mutated in place by every forward
    pass, so without cropping, label 2 is scored conditioned on label 1's tokens,
    label 3 on labels 1 and 2, and so on. The resulting distribution looks
    entirely plausible and is wrong, with no warning anywhere.
    """
    if hasattr(past, "crop"):
        past.crop(length)
        return past
    return tuple((k[:, :, :length, :], v[:, :, :length, :]) for k, v in past)


@torch.no_grad()
def score_labels(model, tok, prefix_text: str, label_ids: list, device):
    """Encode the prefix once, then score each label against the cached prefix.

    Returns
    -------
    p            : softmax over length-normalised label scores, shape (K,)
    norms        : length-normalised log probabilities, shape (K,)
    sums         : raw summed log probabilities, shape (K,)
    n_prefix     : prefix token count
    n_label      : total label-continuation tokens processed
    """
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids.to(device)
    P = prefix_ids.shape[1]
    out = model(prefix_ids,
                attention_mask=torch.ones(1, P, dtype=torch.long, device=device),
                use_cache=True)
    past = out.past_key_values
    first_logits = out.logits[:, -1, :].float()      # predicts label token 0

    sums, norms, n_label = [], [], 0
    for lab in label_ids:
        lab = lab.to(device)
        L = lab.shape[1]
        n_label += L
        step = model(lab,
                     attention_mask=torch.ones(1, P + L, dtype=torch.long, device=device),
                     past_key_values=past, use_cache=True)
        logits = first_logits.unsqueeze(1)
        if L > 1:
            logits = torch.cat([logits, step.logits[:, :-1, :].float()], dim=1)
        lp = torch.log_softmax(logits, -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)[0]
        s = lp.sum().item()
        sums.append(s)
        norms.append(s / L)                          # length normalisation
        past = crop_cache(past, P)

    norms = np.asarray(norms, dtype=np.float64)
    p = np.exp(norms - norms.max())
    p /= p.sum()
    return p, norms, np.asarray(sums, dtype=np.float64), P, n_label


def confidence_signals(p: np.ndarray) -> dict:
    """Three signals read off the same distribution, all free.

    All are oriented so that higher means more confident. Orientation matters:
    the escalation rule fires below a threshold, so a sign error would silently
    invert the router while still producing plausible-looking output.
    """
    order = np.argsort(-p)
    p1, p2 = p[order[0]], p[order[1]]
    K = len(p)
    H = -np.sum(p * np.log(np.clip(p, 1e-12, None)))
    return {
        "pred_idx": int(order[0]),
        "c1_max_softmax": float(p1),
        "c2_margin": float(p1 - p2),
        "c3_norm_entropy": float(1.0 - H / math.log(K)),
    }


@torch.no_grad()
def score_labels_naive(model, tok, prefix_text: str, label_ids: list, device):
    """Re-encode the whole prompt for every label. Used only to time the cache."""
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids.to(device)
    for lab in label_ids:
        full = torch.cat([prefix_ids, lab.to(device)], dim=1)
        model(full, attention_mask=torch.ones_like(full))
