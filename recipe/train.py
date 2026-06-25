"""
Canonical training loop for the Ralph launch track.

This file is part of the recipe — miners may patch it (subject to the
restricted-files contract). The proof-test runner invokes this script with a
fixed config; the training is deterministic given (config, seed, manifest).

Outputs written to `--out-dir`:
  checkpoint.pt         the final model state_dict
  training_log.jsonl    one JSON line per step (loss, lr, throughput, gradnorm)
  final_state.json      run summary (steps, final loss, wall-clock, total tokens)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import TokenShardDataset
from model import RalphBase, RalphConfig


@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 50257
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    ffn_mult: float = 8 / 3
    max_seq_len: int = 1024

    # Training
    seq_len: int = 256
    batch_size: int = 16
    micro_batch_size: int = 16  # gradient accumulation = batch_size / micro_batch_size
    total_steps: int = 200
    warmup_steps: int = 20
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- Recipe levers (B6 transfer-credibility experiment) ---
    # Each default reproduces the verified baseline byte-for-byte.
    optimizer: str = "adamw"          # "adamw" | "muon" | "lion"
    muon_lr_mult: float = 1.0         # Muon-group LR multiplier (Muon needs ~30x AdamW's LR)
    lr_schedule: str = "cosine"       # "cosine" | "wsd"
    wsd_decay_frac: float = 0.2       # WSD: fraction of total_steps spent decaying
    pos_encoding: str = "rope"        # "rope" | "nope" | "learned"  (model-affecting)
    mlp_activation: str = "swiglu"    # "swiglu" | "relu2" | "gelu"  (model-affecting)
    qk_norm: bool = False             # per-head RMSNorm on q,k         (model-affecting)
    residual_init_scale: bool = True  # GPT-2 depth-scaled init         (model-affecting)

    # Data + reproducibility
    manifest_path: str = "data/data_manifest.json"
    data_base_dir: str = "data"
    # B6 v2 held-out val-BPB x-metric: de-confounded, comparable across recipes.
    # Empty val_manifest_path => skip the val pass (baseline behavior unchanged).
    val_manifest_path: str = ""
    val_seq_len: int = 1024          # FIXED eval context, regardless of training seq_len
    val_batches: int = 80            # held-out batches averaged
    val_bytes_per_token: float = 4.0 # gpt2/English; a shared constant -> rank-invariant
    data_seed: int = 1337
    init_seed: int = 1337

    # Precision
    use_bf16: bool = True  # bf16 autocast on CUDA; ignored on CPU

    # Logging
    log_every: int = 10

    @property
    def grad_accum_steps(self) -> int:
        assert self.batch_size % self.micro_batch_size == 0
        return self.batch_size // self.micro_batch_size


def set_determinism(seed: int) -> None:
    """Set all the knobs we can to get deterministic training. Not bit-perfect
    on GPU — see whitepaper §5.2 note on cuBLAS/atomic-reduction non-determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def wsd_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup-Stable-Decay schedule.

    Linear warmup for warmup_steps -> constant at max_lr -> linear decay to
    min_lr over the final `wsd_decay_frac` of total_steps.
    """
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / max(1, cfg.warmup_steps)
    decay_steps = max(1, int(round(cfg.wsd_decay_frac * cfg.total_steps)))
    decay_start = cfg.total_steps - decay_steps
    if step < decay_start:
        return cfg.max_lr  # stable phase
    # Linear decay to min_lr across the final decay_steps.
    progress = (step - decay_start) / max(1, decay_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.max_lr + (cfg.min_lr - cfg.max_lr) * progress


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Dispatch on cfg.lr_schedule. Default 'cosine' == baseline cosine_lr."""
    schedule = getattr(cfg, "lr_schedule", "cosine")
    if schedule == "cosine":
        return cosine_lr(step, cfg)
    if schedule == "wsd":
        return wsd_lr(step, cfg)
    raise ValueError(f"unknown lr_schedule: {schedule!r}")


def build_model(cfg: TrainConfig) -> RalphBase:
    return RalphBase(RalphConfig(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        ffn_mult=cfg.ffn_mult,
        max_seq_len=cfg.max_seq_len,
        pos_encoding=cfg.pos_encoding,
        mlp_activation=cfg.mlp_activation,
        qk_norm=cfg.qk_norm,
        residual_init_scale=cfg.residual_init_scale,
    ))


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz quintic iteration to orthogonalize G (~ G (G^T G)^{-1/2}).

    Reference: Keller Jordan's modded-nanogpt Muon. The (a,b,c) coefficients are
    tuned so the iteration converges to an approximately orthogonal matrix with
    singular values in roughly [0.7, 1.3] (it does NOT drive them exactly to 1,
    which is intentional and works well in practice). Runs in the gradient's
    dtype after normalization; uses bfloat16-friendly math but stays in fp32 on
    CPU. Operates on a 2D matrix; transposes so rows <= cols for efficiency.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    # Normalize so the spectral norm is <= 1 before iterating.
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon: Newton-Schulz-orthogonalized momentum SGD for 2D hidden weights.

    Self-contained single-device implementation following the modded-nanogpt
    reference. For each 2D parameter:
      buf = momentum * buf + grad           (heavy-ball momentum, dampening=0)
      g   = grad + momentum * buf  (Nesterov, when nesterov=True)
      o   = NewtonSchulz5(g)                (orthogonalize)
      scale = max(1, fan_out/fan_in) ** 0.5 (= max(1, rows/cols)**0.5)
      p  -= lr * scale * o
    Intended only for hidden weight matrices; embeddings/head/norms/biases go on
    AdamW in build_optimizer.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon expects 2D parameters only"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scale by sqrt(max(1, fan_out/fan_in)); fan_out=rows, fan_in=cols.
                fan_out, fan_in = p.shape[0], p.shape[1]
                scale = max(1.0, fan_out / fan_in) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)
        return loss


class Lion(torch.optim.Optimizer):
    """Lion: sign-of-momentum optimizer with decoupled weight decay.

    Reference: Chen et al. 2023 (Symbolic Discovery of Optimization Algorithms).
      update = sign(beta1 * m + (1 - beta1) * grad)
      p      = p - lr * (update + weight_decay * p)     (decoupled WD)
      m      = beta2 * m + (1 - beta2) * grad           (EMA, updated after)
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # Decoupled weight decay.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                # Update direction uses interpolation of EMA and current grad.
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)
                # EMA momentum update (after computing the step).
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    opt = getattr(cfg, "optimizer", "adamw")

    if opt == "adamw":
        decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
        )

    if opt == "muon":
        # Muon on 2D hidden weight matrices = the AdamW decay set MINUS the token
        # embedding and final unembedding head (those stay on AdamW). Norms/biases
        # (dim<2) also stay on AdamW.
        muon_params, adamw_decay, adamw_no_decay = _split_muon_param_groups(model)
        muon = Muon(muon_params, lr=cfg.max_lr)
        # Muon needs a much larger LR than AdamW (modded-nanogpt: ~0.02 vs ~3e-4).
        # Tag its groups so the train loop scales their LR by muon_lr_mult; the AdamW
        # sibling (embeddings / head / norms) keeps the base LR (lr_mult defaults to 1).
        for g in muon.param_groups:
            g["lr_mult"] = cfg.muon_lr_mult
        # Stash a sibling AdamW for the non-Muon params on the Muon object so the
        # single .step()/.zero_grad()/param_groups interface still works.
        return _CompositeOptimizer(
            primary=muon,
            sibling=torch.optim.AdamW(
                [
                    {"params": adamw_decay, "weight_decay": cfg.weight_decay},
                    {"params": adamw_no_decay, "weight_decay": 0.0},
                ],
                lr=cfg.max_lr,
                betas=(cfg.beta1, cfg.beta2),
            ),
        )

    if opt == "lion":
        decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        # Lion (decoupled WD) on decay params; Lion with wd=0 on no_decay params.
        return Lion(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
        )

    raise ValueError(f"unknown optimizer: {opt!r}")


def _split_muon_param_groups(model: torch.nn.Module):
    """Partition params for the Muon path.

    Returns (muon_params, adamw_decay, adamw_no_decay):
      muon_params    : 2D weights that are NOT token-embedding / unembedding head
                       and not a norm/bias (those are dim<2 anyway).
      adamw_decay    : 2D embedding / unembedding-head weights.
      adamw_no_decay : everything with dim < 2 (norms, biases).
    With tie_embeddings=True there is no separate lm_head; the tied weight lives
    on tok_embed and is correctly routed to AdamW.
    """
    muon_params, adamw_decay, adamw_no_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            adamw_no_decay.append(p)
            continue
        # 2D weights: embedding / unembedding head -> AdamW; everything else -> Muon.
        is_embed_or_head = (
            "tok_embed" in n or "lm_head" in n or n.endswith("pos_embed.weight") or "pos_embed" in n
        )
        if is_embed_or_head:
            adamw_decay.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_decay, adamw_no_decay


class _CompositeOptimizer(torch.optim.Optimizer):
    """Thin wrapper presenting two optimizers (Muon + AdamW) as one.

    Exposes a unified `param_groups` (so the train loop's per-step LR set works),
    plus `step`, `zero_grad`, `state_dict`, `load_state_dict`. Not a general-
    purpose composite — purpose-built for the Muon + AdamW split here.
    """

    def __init__(self, primary: torch.optim.Optimizer, sibling: torch.optim.Optimizer):
        self._opts = [primary, sibling]
        # Do NOT call super().__init__ (no single param set); proxy the pieces.

    @property
    def param_groups(self):
        groups = []
        for o in self._opts:
            groups.extend(o.param_groups)
        return groups

    @property
    def state(self):
        merged = {}
        for o in self._opts:
            merged.update(o.state)
        return merged

    def zero_grad(self, set_to_none: bool = True):
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self._opts:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self._opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self._opts, sd["opts"]):
            o.load_state_dict(s)


def _init_wandb(cfg: TrainConfig, out_dir: Path, use_wandb: bool) -> object | None:
    if not use_wandb:
        return None
    try:
        import wandb
        miner_gh = os.environ.get("RALPH_MINER_GH", "")
        miner_wallet = os.environ.get("BT_WALLET", "")
        run_config = {k: v for k, v in asdict(cfg).items()}
        if miner_gh:
            run_config["miner_github"] = miner_gh
        if miner_wallet:
            run_config["miner_wallet"] = miner_wallet
        tags = ["proof-test", f"{cfg.dim}d", f"{cfg.n_layers}L"]
        if miner_gh:
            tags.append(f"gh:{miner_gh}")
        if miner_wallet:
            tags.append(f"wallet:{miner_wallet}")
        name_prefix = f"{miner_gh}-" if miner_gh else ""
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "ralphlabs-hub"),
            project=os.environ.get("WANDB_PROJECT", "ralph"),
            name=f"{name_prefix}train-{cfg.dim}d-{cfg.n_layers}L-{cfg.total_steps}s",
            config=run_config,
            dir=str(out_dir),
            tags=tags,
        )
        return run
    except Exception as e:
        print(f"[train] wandb init failed ({e}), continuing without it")
        return None


def train(cfg: TrainConfig, out_dir: Path, use_wandb: bool = False) -> dict:
    set_determinism(cfg.init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    ds = TokenShardDataset(cfg.manifest_path, cfg.data_base_dir, cfg.seq_len, cfg.data_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    log_f = log_path.open("w")

    wb_run = _init_wandb(cfg, out_dir, use_wandb)

    use_amp = cfg.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    # GradScaler only needed for fp16 (bf16 has enough dynamic range). Disabled = no-op.
    scaler = torch.amp.GradScaler(device.type, enabled=False) if device.type == "cuda" else None

    n_params = model.num_parameters()
    n_params_no_embed = model.num_parameters(exclude_embeddings=True)
    print(f"[train] device={device} params={n_params:,} (no embeddings: {n_params_no_embed:,})")
    print(f"[train] precision={'bf16' if use_amp else 'fp32'}")
    print(f"[train] manifest tokens={ds.total_tokens:,} hash={ds.manifest.manifest_hash()[:16]}…")
    print(f"[train] steps={cfg.total_steps} batch={cfg.batch_size} micro={cfg.micro_batch_size} seq={cfg.seq_len}")
    if wb_run:
        print(f"[train] wandb: {wb_run.url}")

    start = time.time()
    tokens_seen = 0
    last_loss = float("nan")
    for step in range(cfg.total_steps):
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

        step_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for accum in range(cfg.grad_accum_steps):
            sub_step = step * cfg.grad_accum_steps + accum
            inp, tgt = ds.get_batch(sub_step, cfg.micro_batch_size)
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(inp, targets=tgt)
            scaled_loss = loss / cfg.grad_accum_steps
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            step_loss += loss.item() / cfg.grad_accum_steps
            tokens_seen += cfg.micro_batch_size * cfg.seq_len

        if scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        last_loss = step_loss
        elapsed = time.time() - start
        tok_per_s = tokens_seen / max(elapsed, 1e-6)

        entry = {
            "step": step,
            "loss": step_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": tok_per_s,
            "elapsed_s": elapsed,
        }
        log_f.write(json.dumps(entry) + "\n")
        if wb_run:
            wb_run.log(entry, step=step)
        if step % cfg.log_every == 0 or step == cfg.total_steps - 1:
            print(
                f"[step {step:4d}/{cfg.total_steps}] loss={step_loss:.4f} lr={lr:.2e} "
                f"|g|={grad_norm:.2f} tok/s={tok_per_s:,.0f}"
            )
    log_f.close()
    wb_url = None
    if wb_run:
        wb_url = wb_run.url
        try:
            history = wb_run.history(pandas=False)
            (out_dir / "wandb_metrics.json").write_text(json.dumps(history, indent=2))
            (out_dir / "wandb_run_url.txt").write_text(wb_url + "\n")
            print(f"[train] wandb metrics exported ({len(history)} steps)")
        except Exception as e:
            print(f"[train] wandb export failed ({e}), continuing")
        wb_run.finish()

    ckpt_path = out_dir / "checkpoint.pt"
    with torch.no_grad():
        model.tok_embed.weight.zero_()
        if getattr(model, "lm_head", None) is not None:
            model.lm_head.weight.zero_()
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, ckpt_path)

    # B6 v2 held-out val-BPB pass: eval on a FIXED held-out stream at a FIXED context
    # (de-confounds training seq_len/step-count/optimizer; the shared bytes_per_token
    # constant makes val_bpb rank-equivalent to val_loss). Skipped if no val manifest.
    val_loss = None
    val_bpb = None
    tail_val_bpb = None   # context-sensitivity probe: BPB on the long-context tail half
    if cfg.val_manifest_path:
        import math as _math
        import torch.nn.functional as _F
        model.eval()
        val_ds = TokenShardDataset(cfg.val_manifest_path, cfg.data_base_dir, cfg.val_seq_len, cfg.data_seed)
        total = 0.0
        tail_total = 0.0
        half = cfg.val_seq_len // 2     # tail = positions [half, val_seq_len)
        with torch.no_grad():
            for b in range(cfg.val_batches):
                inp, tgt = val_ds.get_batch(b, cfg.micro_batch_size)
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    logits, loss = model(inp, targets=tgt)
                    # CE over the long-context tail only — penalizes recipes that
                    # shorten effective training context (e.g. seq_len < val_seq_len)
                    # even at a cheap budget; the blind spot the 250M transfer surfaced.
                    tail_loss = _F.cross_entropy(
                        logits[:, half:, :].reshape(-1, logits.size(-1)),
                        tgt[:, half:].reshape(-1),
                        ignore_index=-100,
                    )
                total += loss.item()
                tail_total += tail_loss.item()
        val_loss = total / max(1, cfg.val_batches)
        val_bpb = val_loss / (_math.log(2) * cfg.val_bytes_per_token)
        tail_val_bpb = (tail_total / max(1, cfg.val_batches)) / (_math.log(2) * cfg.val_bytes_per_token)
        model.train()
        print(f"[train] val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} tail_val_bpb={tail_val_bpb:.4f} (held-out, eval_seq={cfg.val_seq_len}, tail>={half})")

    summary = {
        "steps": cfg.total_steps,
        "final_loss": last_loss,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "tail_val_bpb": tail_val_bpb,
        "tokens_seen": tokens_seen,
        "wall_clock_s": time.time() - start,
        "n_params": n_params,
        "n_params_no_embed": n_params_no_embed,
        "manifest_hash": ds.manifest.manifest_hash(),
        "device": str(device),
        "precision": "bf16" if use_amp else "fp32",
        "wandb_url": wb_url,
        "config": asdict(cfg),
    }
    (out_dir / "final_state.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. final loss={last_loss:.4f} wall={summary['wall_clock_s']:.1f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None, help="Optional JSON config override.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (requires `pip install wandb`)")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.config and args.config.exists():
        overrides = json.loads(args.config.read_text())
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.manifest is not None:
        cfg.manifest_path = str(args.manifest)
    if args.seed is not None:
        cfg.init_seed = args.seed
        cfg.data_seed = args.seed

    train(cfg, args.out_dir, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
