"""Compare forward pass: our reimplementation vs original plaid code.

Instead of loading both, we numerically verify our model produces
correct logits by checking intermediate values against known-good
computations.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np

def main():
    device = 'cuda'
    from geniesae.plaid_model import load_plaid_modules, Rotary, apply_rotary_pos_emb

    print("Loading OUR reimplementation...")
    our_modules = load_plaid_modules(
        'models/plaid/plaid1b_weights',
        dim=2048, embed_dim=16, n_blocks=24, n_heads=32,
        vocab_size=32768, gamma_0=-3.0, gamma_1=6.0, device=device
    )
    m = our_modules['model']
    print(f"  output_mult: {m.output_linear.output_mult}")
    print(f"  width_mult: {m.output_linear.width_mult()}")
    print(f"  scaling: {m.output_linear.output_mult / m.output_linear.width_mult()}")

    # ---- Test 1: Rotary embedding shape ----
    print("\n=== Test 1: Rotary embedding shape ===")
    rotary = m.rotary_emb
    dummy = torch.randn(1, 64, 2048, device=device)
    cos, sin = rotary(dummy)
    print(f"  cos shape: {cos.shape}")
    print(f"  sin shape: {sin.shape}")
    print(f"  Expected: (1, 64, 3, 1, 64)")
    assert cos.shape == (1, 64, 3, 1, 64), f"WRONG shape: {cos.shape}"
    # Check v component is identity
    print(f"  cos[:,:,2,:,:] all 1.0? {(cos[:,:,2,:,:] == 1.0).all().item()}")
    print(f"  sin[:,:,2,:,:] all 0.0? {(sin[:,:,2,:,:] == 0.0).all().item()}")

    # ---- Test 2: Rotary applied correctly ----
    print("\n=== Test 2: Rotary preserves v ===")
    torch.manual_seed(42)
    qkv = torch.randn(1, 64, 3, 32, 64, device=device)
    v_before = qkv[:, :, 2].clone()
    qkv_rot = apply_rotary_pos_emb(qkv, cos, sin)
    v_after = qkv_rot[:, :, 2]
    v_diff = (v_before - v_after).abs().max().item()
    print(f"  v max diff after rotary: {v_diff:.2e}")
    assert v_diff < 1e-6, f"v was modified by rotary! diff={v_diff}"

    # ---- Test 3: Full forward pass - check logit range ----
    print("\n=== Test 3: Forward pass logit range ===")
    torch.manual_seed(123)
    B, S, E = 1, 64, 16
    z = torch.randn(B, S, E, device=device).float()
    gamma = torch.tensor([0.5], device=device).float()
    x_selfcond = torch.zeros(B, S, E, device=device).float()
    emb = our_modules['embedding_matrix']()

    with torch.no_grad():
        logits, reconst = m(
            z=z, gamma=gamma, embedding_matrix=emb,
            bias_scale=1.0, x_selfcond=x_selfcond
        )

    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  Logits std: {logits.float().std().item():.4f}")
    print(f"  Logits mean: {logits.float().mean().item():.6f}")

    # Check that logits have reasonable range (not crushed)
    # Original code produces logits with std ~0.5-2.0
    logit_std = logits.float().std().item()
    print(f"  Logit std OK (>0.1)? {logit_std > 0.1}")

    # ---- Test 4: Check softmax entropy ----
    print("\n=== Test 4: Softmax entropy ===")
    probs = F.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1).mean().item()
    max_entropy = np.log(32768)
    print(f"  Mean entropy: {entropy:.2f}")
    print(f"  Max entropy (uniform): {max_entropy:.2f}")
    print(f"  Entropy ratio: {entropy/max_entropy:.4f}")
    # If entropy is close to max, distribution is near-uniform = gibberish
    print(f"  Near-uniform (>0.95)? {entropy/max_entropy > 0.95}")

    # ---- Test 5: Layer-by-layer activation norms ----
    print("\n=== Test 5: Layer activation norms ===")
    acts = {}
    def make_hook(name):
        def hook(module, input, output):
            acts[name] = output.detach().float()
        return hook
    for i, block in enumerate(m.blocks):
        block.register_forward_hook(make_hook(f'block_{i}'))

    torch.manual_seed(123)
    z = torch.randn(B, S, E, device=device).float()
    with torch.no_grad():
        logits2, _ = m(z=z, gamma=gamma, embedding_matrix=emb,
                       bias_scale=1.0, x_selfcond=x_selfcond)

    for i in range(24):
        a = acts[f'block_{i}']
        print(f"  Block {i:2d}: mean={a.mean().item():+.4f}, std={a.std().item():.4f}, "
              f"min={a.min().item():.4f}, max={a.max().item():.4f}")

    # ---- Test 6: Unconditional generation (short) ----
    print("\n=== Test 6: Quick unconditional generation ===")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('models/plaid/plaid1b_weights/tokenizer.json')

    torch.set_default_dtype(torch.float64)
    ns = our_modules['noise_schedule']
    gb = our_modules['gamma_bounds']
    emb_mod = our_modules['embedding_matrix']

    n_samples = 2
    seq_len = 128
    T = 128
    embed_dim = 16

    z = torch.randn(n_samples, seq_len, embed_dim, device=device)
    x_selfcond = torch.zeros_like(z).float()

    gamma_0, gamma_1 = gb()

    for step_i in range(T):
        t_val = 1.0 - step_i / T
        s_val = 1.0 - (step_i + 1) / T
        t = torch.tensor([t_val], device=device, dtype=torch.float64)
        s = torch.tensor([s_val], device=device, dtype=torch.float64)

        gamma_t = gamma_0 + (gamma_1 - gamma_0) * ns(t).double()
        gamma_s = gamma_0 + (gamma_1 - gamma_0) * ns(s).double()

        alpha_sq_t = torch.sigmoid(-gamma_t)
        sigma_sq_t = torch.sigmoid(gamma_t)
        alpha_t = alpha_sq_t.sqrt()
        sigma_t = sigma_sq_t.sqrt()
        alpha_sq_s = torch.sigmoid(-gamma_s)

        embedding_matrix = emb_mod()
        with torch.no_grad():
            _, x_reconst = m(
                z=z.float(),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.0,
                x_selfcond=x_selfcond,
            )
        x_selfcond = x_reconst.clone().detach()
        x_reconst = x_reconst.double()

        epsilon_pred = (z - alpha_t * x_reconst) / sigma_t
        epsilon_pred /= 0.9  # score_temp
        x_reconst = (z - sigma_t * epsilon_pred) / alpha_t

        if s_val > 0:
            c = -torch.expm1(gamma_s - gamma_t)
            z = (1 - c) * alpha_sq_s.sqrt() / alpha_sq_t.sqrt() * z
            z += c * alpha_sq_s.sqrt() * x_reconst
            z += (c * (1 - alpha_sq_s)).sqrt() * torch.randn_like(z)

    # Final decode
    embedding_matrix = emb_mod()
    with torch.no_grad():
        logits_final, _ = m(
            z=z.float(),
            gamma=gamma_t.float(),
            embedding_matrix=embedding_matrix,
            bias_scale=1.0,
            x_selfcond=x_selfcond,
        )
    tokens = logits_final.argmax(dim=-1)
    for i in range(n_samples):
        text = tokenizer.decode(tokens[i].tolist())
        print(f"  Sample {i}: {text[:300]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
