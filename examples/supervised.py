"""Download-free supervised regression: python examples/supervised.py."""
import argparse

import torch
from ZFO import ZFO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=ZFO.METHODS, default="taylor2")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="Adam")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--interior-search", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args()
    torch.manual_seed(17)
    x = torch.randn(256, 6, dtype=torch.float64)
    y = x @ torch.randn(6, 1, dtype=torch.float64) + 0.1 * torch.randn(256, 1, dtype=torch.float64)
    model = torch.nn.Linear(6, 1).double()
    base = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    zfo = ZFO(base, model=model, beta=args.beta, epsilon=args.epsilon,
              method=args.method, interior_search=args.interior_search)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=60)
    for step in range(60):
        # Return the loss only; ZFO owns zero_grad and backward.
        loss = zfo.step(lambda: (model(x) - y).square().mean())
        scheduler.step()
        if step % 15 == 0:
            print(f"step={step:02d} loss={loss.item():.6f} multiplier={zfo.last_step['multiplier']:.3f}")
    print(f"final loss={((model(x) - y) ** 2).mean().item():.6f}")


if __name__ == "__main__":
    main()
