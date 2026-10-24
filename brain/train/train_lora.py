"""LoRA fine-tune of Gemma 4 (E4B/E2B) on KML chat data.  Loss on assistant turns only.

  CUDA_VISIBLE_DEVICES=0 python train/train_lora.py --base HF_DIR --data data/train.jsonl --out OUT_DIR

Memory trick: the 2.8B-param per-layer embedding table (PLE) is a pure lookup, so it stays on CPU;
the rest of the language model is bf16 on one 20 GB GPU. Vision/audio towers are dropped.
"""
import argparse, json, math, os, random, sys, time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import prompts

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--val")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-batches per step")
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--system", choices=list(prompts.SYSTEMS), default="persona")
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda"

tok = AutoTokenizer.from_pretrained(a.base)
END = "<turn|>"


def encode(ex):
    msgs = [{"role": "system", "content": prompts.SYSTEMS[a.system]}] + [
        {"role": m["role"], "content": m["content"]} for m in ex["messages"]]
    text = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    # char spans of assistant contents + their closing <turn|>
    spans, pos = [], 0
    for m in ex["messages"]:
        if m["role"] != "assistant":
            continue
        s = text.index("<|turn>model\n" + m["content"], pos) + len("<|turn>model\n")
        e = s + len(m["content"]) + len(END)
        spans.append((s, e)); pos = e
    labels = [t if any(s <= o0 < e for s, e in spans) else -100 for t, (o0, o1) in zip(ids, offs)]
    return ids[:a.max_len], labels[:a.max_len]


def load(path):
    return [encode(json.loads(l)) for l in open(path) if l.strip()]


train = load(a.data)
val = load(a.val) if a.val else []
print(f"train {len(train)} ex, mean len {sum(len(i) for i, _ in train) / len(train):.0f} tok, "
      f"supervised {sum(sum(x != -100 for x in l) for _, l in train)} tok", flush=True)

model = Gemma4ForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": "cpu"})
model.model.audio_tower = None
model.model.vision_tower = None
for n in ("embed_audio", "embed_vision"):
    if hasattr(model.model, n):
        setattr(model.model, n, None)
lm = model.model.language_model
ple = lm.embed_tokens_per_layer
del lm.embed_tokens_per_layer
model.to(dev)
ple_fwd = ple.forward
ple.forward = lambda ids: ple_fwd(ids.cpu()).to(dev, non_blocking=True)
lm.embed_tokens_per_layer = ple  # re-attach; stays on CPU (we never call model.to() again)
for p in model.parameters():
    p.requires_grad_(False)
print(f"GPU after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
                 target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
steps_per_epoch = math.ceil(len(train) / (a.bs * a.accum))
total = a.max_steps or int(steps_per_epoch * a.epochs)
warm = max(1, int(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))


def batch_iter(data, bs, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        L = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        lab = torch.full((len(chunk), L), -100)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, :len(x)] = torch.tensor(x); lab[k, :len(y)] = torch.tensor(y); att[k, :len(x)] = 1
        yield ids.to(dev), lab.to(dev), att.to(dev)


def loss_on(ids, lab, att):
    out = model(input_ids=ids, attention_mask=att, use_cache=False)
    logits = out.logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                             ignore_index=-100)


@torch.no_grad()
def evaluate():
    if not val:
        return None
    model.eval()
    ls = [loss_on(*b).item() for b in batch_iter(val, a.bs, shuffle=False)]
    model.train()
    return sum(ls) / len(ls)


model.train()
step, t0 = 0, time.time()
log = []
while step < total:
    batches = list(batch_iter(train, a.bs))
    for i in range(0, len(batches) - a.accum + 1, a.accum):
        for ids, lab, att in batches[i:i + a.accum]:
            loss = loss_on(ids, lab, att)
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 4), lr=sched.get_last_lr()[0],
                       s_per_step=round((time.time() - t0) / step, 2),
                       peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            if step % steps_per_epoch < 10 and step >= 10 or step == total:
                rec["val"] = evaluate()
            print(json.dumps(rec), flush=True); log.append(rec)
        if step >= total:
            break

rec = dict(step=step, final_val=evaluate(), minutes=round((time.time() - t0) / 60, 1))
print(json.dumps(rec), flush=True); log.append(rec)
os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
json.dump(dict(args=vars(a), log=log), open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
print("saved", a.out)
