from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer
from optimizer import AdamW


class Smart(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            params_tilde: Iterable[torch.nn.parameter.Parameter],
            model,
            dataset,
            solve_iters=10,
            x_iters=20,
            lr: float = 1e-3,
            beta: float = 0.9,
            sigma: float = 1e-2,
            eps: float = 1e-2,
            adam_lr: float = 1e-4,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if adam_lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(adam_lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be between 0.0 and 1.0".format(beta))
        if not 0.0 <= sigma:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(sigma))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(params_tilde=params_tilde, model=model, dataset=dataset, lr=lr, beta=beta, sigma=sigma, eps=eps,
                        adam_lr=adam_lr, solve_iters=solve_iters, x_iters=x_iters)
        super().__init__(params, defaults)

    def f(self, forward, x, attention_mask, theta):
        logits = forward(x, attention_mask, theta=theta)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def g(self, model, attention_mask, x, x_tilde, theta, batch_size):
        x_tilde.requires_grad_(True)
        x[0].requires_grad_(True)
        x_tilde[0].requires_grad_(True)
        l = (torch.nn.functional.kl_div(self.f(model.forward, x[0], attention_mask, theta),
                                        self.f(model.forward, x_tilde[0], attention_mask, theta)) +
             torch.nn.functional.kl_div(self.f(model.forward, x_tilde[0], attention_mask, theta),
                                        self.f(model.forward, x[0], attention_mask, theta)))
        if x.shape[0] >= 1:
            for i in range(x.shape[0]):
                x[i].requires_grad_(True)
                x_tilde[i].requires_grad_(True)
                l += (torch.nn.functional.kl_div(self.f(model.forward, x[i], attention_mask, theta),
                                                 self.f(model.forward, x_tilde[i], attention_mask, theta)) +
                      torch.nn.functional.kl_div(self.f(model.forward, x_tilde[i], attention_mask, theta),
                                                 self.f(model.forward, x[i], attention_mask, theta)))
        d_l = l.backward()
        return d_l / batch_size

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            counter = 0
            for p, p_tilde in zip(group["params"], group["params_tilde"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, and Smart uses Adam, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                params_bar = p
                batch_size = int(group["dataset"]["token_ids"].shape[0] / group["solve_iters"])
                batches = torch.utils.data.DataLoader(group["dataset"], batch_size=batch_size, shuffle=True,
                                                      drop_last=True)
                for batch in batches:
                    v = torch.normal(mean=0, std=group["sigma"], size=batch["token_ids"].shape)
                    x_tilde = batch["token_ids"] + v
                    for i in range(batch["token_ids"].shape[0]):
                        for m in range(group["x_iters"]):
                            g_tilde_unscaled = self.g(group["model"], batch["attention_mask"], batch["token_ids"][i],
                                                      x_tilde[i], params_bar, batch_size)
                            g_i_tilde = g_tilde_unscaled / torch.linalg.norm(g_tilde_unscaled, ord=float('inf'))
                            while torch.linalg.norm(x_tilde[i] - batch["token_ids"][i], ord=float('inf')) <= group[
                                "eps"]:
                                x_tilde[i] += group["lr"] * g_i_tilde
                    adam_optimizer = AdamW(params_bar, lr=group["adam_lr"])
                    adam_optimizer.step()
                p.data = params_bar.data
                p_tilde.data = (1 - group["beta"]) * params_bar + group["beta"] * p_tilde.data
                counter += 1

        return loss
