"""Compare sampling loop: original code vs our TokenGuidanceSampler.

Run both on the same article with the same random seed and compare
intermediate values at each step.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer


def main():
    device = 'cuda'

    # Load our reimplementation
    from geniesae.plaid_model import load_plaid_modules
    from geniesae.plaid_samplers import TokenGuidanceSampler

    print("Loading model...")
    modules = load_plaid_modules(
        'models/plaid/plaid1b_weights',
        dim=2048, embed_dim=16, n_blocks=24, n_heads=32,
        vocab_size=32768, gamma_0=-3.0, gamma_1=6.0, device=device
    )

    tokenizer = Tokenizer.from_file('models/plaid/plaid1b_weights/tokenizer.json')
    article = "Burberry reported pre-tax profits of £166m for the year to March."
    prompt = article + "\n\nThe article can be summarized as follows:"
    prompt_ids = tokenizer.encode(prompt).ids
    prefix_len = len(prompt_ids)
    print(f"Prompt: {len(prompt_ids)} tokens")

    # ---- Run our TokenGuidanceSampler ----
    print("\n=== OUR TokenGuidanceSampler ===")
    sampler = TokenGuidanceSampler(
        model=modules["model"],
        noise_schedule=modules["noise_schedule"],
        gamma_bounds=modules["gamma_bounds"],
        embedding_matrix=modules["embedding_matrix"],
        sampling_timesteps=64,  # short for quick test
        score_temp=0.9,
        guidance_weight=2.0,
    )
    torch.manual_seed(42)
    our_ids = sampler.sample(prefix_token_ids=[prompt_ids], seq_len=128)
    our_text = tokenizer.decode(our_ids[0].tolist()[prefix_len:])
    print(f"  Generated: {our_text[:300]}")

    # ---- Run the original-style sampling loop manually ----
    print("\n=== ORIGINAL-STYLE sampling loop ===")
    torch.set_default_dtype(torch.float64)

    embed_dim = 16
    seq_len = 128
    T = 64
    score_temp = 0.9
    guidance_weight = 2.0

    embedding_matrix = modules['embedding_matrix']()
    gamma_0, gamma_1 = modules['gamma_bounds']()

    guidance_tokens = [
        (tid, guidance_weight, pos, False)
        for pos, tid in enumerate(prompt_ids)
    ]

    torch.manual_seed(42)
    z = torch.randn((1, seq_len, embed_dim), device=device)
    x_selfcond = torch.zeros_like(z).float()

    import tqdm
    for step_i, t in enumerate(torch.linspace(1., 0., T)):
        t = t[None].to(device)
        s = t - 1. / T
        gamma_s = modules['noise_schedule'](s).double()
        gamma_t = modules['noise_schedule'](t).double()
        gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
        gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
        alpha_squared_s = torch.sigmoid(-gamma_s)
        alpha_squared_t = torch.sigmoid(-gamma_t)
        alpha_t = alpha_squared_t.sqrt()
        sigma_squared_t = torch.sigmoid(gamma_t)
        sigma_t = sigma_squared_t.sqrt()

        with torch.enable_grad():
            z_grad = z.detach().requires_grad_(True)
            logits, x_reconst = modules['model'](
                z=z_grad.to(torch.float32),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond
            )
            logprobs = F.log_softmax(logits.float(), dim=2)
            sum_logp = 0.
            for token, weight, position, complement in guidance_tokens:
                logp = logprobs[:, position, token]
                sum_logp = sum_logp + weight * logp.sum()
            guidance_grad = torch.autograd.grad(sum_logp, [z_grad])[0]

        x_selfcond = x_reconst.clone().detach()
        x_reconst = x_reconst.double()
        epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
        epsilon_pred /= score_temp
        x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
        x_reconst += guidance_grad.double() * sigma_squared_t / alpha_squared_t.sqrt()
        epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t

        if t > 0:
            c = -torch.expm1(gamma_s - gamma_t)
            z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
            z += c * (alpha_squared_s.sqrt() * x_reconst.double())
            z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)

        if step_i < 3 or step_i == T - 1:
            print(f"  Step {step_i}: z_norm={z.norm().item():.4f}, "
                  f"grad_norm={guidance_grad.norm().item():.6f}")

    # Final decode
    logits, _ = modules['model'](
        z=z.float(), gamma=gamma_t.float(),
        embedding_matrix=embedding_matrix,
        bias_scale=1., x_selfcond=x_selfcond
    )
    orig_ids = logits.argmax(dim=-1)
    orig_text = tokenizer.decode(orig_ids[0].tolist()[prefix_len:])
    print(f"  Generated: {orig_text[:300]}")

    # ---- Compare ----
    print(f"\n=== COMPARISON ===")
    our_toks = our_ids[0].tolist()
    orig_toks = orig_ids[0].tolist()
    match = sum(a == b for a, b in zip(our_toks, orig_toks))
    print(f"  Token match: {match}/{len(our_toks)} ({100*match/len(our_toks):.1f}%)")
    print(f"  Our suffix:  {tokenizer.decode(our_toks[prefix_len:])[:200]}")
    print(f"  Orig suffix: {tokenizer.decode(orig_toks[prefix_len:])[:200]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
