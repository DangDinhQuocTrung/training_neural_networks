import torch
import torch.optim as optim
from abc import ABC, abstractmethod


class BaseOptimizer(optim.Optimizer, ABC):
    def __init__(self, params, defaults, **kwargs):
        super(BaseOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            states = []

            for p in group["params"]:
                if p.grad is not None:
                    # Retrieve params
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]

                    # Build state
                    if len(state) == 0:
                        state["step"] = torch.zeros((1,), dtype=torch.float, device=p.device)
                        state["momentum_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["sum_grad_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["acc_delta"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["grad_exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["grad_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["max_grad_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    states.append(state)

            # Run optimizer
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state = states[i]
                self.update(i, param, grad, state)

        return loss

    @abstractmethod
    def update(self, i, param, grad, state):
        pass


class SGD(BaseOptimizer):
    def __init__(self, params, lr, **kwargs):
        if (lr is None) or (lr <= 0.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def update(self, i, param, grad, state):
        alpha = -self.lr
        param.add_(grad, alpha=alpha)


class MSGD(BaseOptimizer):
    def __init__(self, params, lr, momentum, dampening, **kwargs):
        if (lr is None) or (lr <= 0.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        defaults = dict(lr=lr)
        super(MSGD, self).__init__(params, defaults)

    def update(self, i, param, grad, state):
        step = state["step"]
        momentum_grad = state["momentum_grad"]

        if step.item() == 0:
            momentum_grad = grad
        else:
            momentum_grad = self.momentum * momentum_grad + self.dampening * grad

        step += 1
        alpha = -self.lr
        param.add_(grad, alpha=alpha)
        state["momentum_grad"].data.copy_(momentum_grad)


class Adagrad(BaseOptimizer):
    def __init__(self, params, lr, lr_decay, eps, **kwargs):
        if (lr is None) or (lr <= 0.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr
        self.lr_decay = lr_decay
        self.eps = eps
        defaults = dict(lr=lr)
        super(Adagrad, self).__init__(params, defaults)

    def update(self, i, param, grad, state):
        step = state["step"]
        sum_grad_sq = state["sum_grad_sq"]

        # compute
        step += 1
        lr = self.lr / (1 + (step - 1) * self.lr_decay)
        sum_grad_sq += grad * grad

        # update param
        param -= lr * grad / (sum_grad_sq.sqrt() + self.eps)
        # update state
        state["sum_grad_sq"].data.copy_(sum_grad_sq)


class RMSprop(BaseOptimizer):
    def __init__(self, params, lr, alpha, eps, **kwargs):
        if (lr is None) or (lr <= 0.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        defaults = dict(lr=lr)
        super(RMSprop, self).__init__(params, defaults)

    def update(self, i, param, grad, state):
        grad_exp_avg_sq = state["grad_exp_avg_sq"]

        # compute
        grad_exp_avg = self.alpha * grad_exp_avg_sq + (1.0 - self.alpha) * grad * grad

        # update param
        param -= self.lr * grad / (grad_exp_avg.sqrt() + self.eps)
        # update state
        state["grad_exp_avg_sq"].data.copy_(grad_exp_avg_sq)


class Adadelta(BaseOptimizer):
    def __init__(self, params, lr, rho, eps, **kwargs):
        if (lr is None) or (lr <= 0.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr
        self.rho = rho
        self.eps = eps
        defaults = dict(lr=lr)
        super(Adadelta, self).__init__(params, defaults)

    def update(self, i, param, grad, state):
        grad_exp_avg_sq = state["grad_exp_avg_sq"]
        acc_delta = state["acc_delta"]

        # compute
        grad_exp_avg_sq = self.rho * grad_exp_avg_sq + (1.0 - self.rho) * grad * grad
        std = (grad_exp_avg_sq.sqrt() + self.eps)
        delta = (acc_delta.sqrt() + self.eps) / std * grad
        acc_delta = self.rho * acc_delta + (1.0 - self.rho) * delta * delta

        # update param
        param -= self.lr * delta

        # update state
        state["grad_exp_avg_sq"].data.copy_(grad_exp_avg_sq)
        state["acc_delta"].data.copy_(acc_delta)


class Adam(BaseOptimizer):
    def __init__(self, params, lr, beta1, beta2, eps, **kwargs):
        if (lr is None) or (lr <= 0.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        defaults = dict(lr=lr)
        super(Adam, self).__init__(params, defaults)

    def update(self, i, param, grad, state):
        step = state["step"]
        grad_exp_avg = state["grad_exp_avg"]
        grad_exp_avg_sq = state["grad_exp_avg_sq"]

        # compute
        step += 1
        grad_exp_avg = grad_exp_avg * self.beta1 + grad * (1.0 - self.beta1)
        grad_exp_avg_sq = grad_exp_avg_sq * self.beta2 + grad * grad * (1.0 - self.beta2)

        bias_correction1 = 1.0 - torch.pow(self.beta1, step)
        bias_correction2 = 1.0 - torch.pow(self.beta2, step)
        m_t = grad_exp_avg / bias_correction1
        v_t = grad_exp_avg_sq / bias_correction2

        # update param
        denom = v_t.sqrt() + self.eps
        param -= self.lr * m_t / denom
        # update state
        state["grad_exp_avg"].data.copy_(grad_exp_avg)
        state["grad_exp_avg_sq"].data.copy_(grad_exp_avg_sq)


class AMSGrad(BaseOptimizer):
    def __init__(self, params, lr, beta1, beta2, eps, **kwargs):
        if (lr is None) or (lr <= 0.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        defaults = dict(lr=lr)
        super(AMSGrad, self).__init__(params, defaults)

    def update(self, i, param, grad, state):
        step = state["step"]
        grad_exp_avg = state["grad_exp_avg"]
        grad_exp_avg_sq = state["grad_exp_avg_sq"]
        max_grad_exp_avg_sq = state["max_grad_exp_avg_sq"]

        # compute
        step += 1
        grad_exp_avg = grad_exp_avg * self.beta1 + grad * (1.0 - self.beta1)
        grad_exp_avg_sq = grad_exp_avg_sq * self.beta2 + grad * grad * (1.0 - self.beta2)
        max_grad_exp_avg_sq = torch.maximum(max_grad_exp_avg_sq, grad_exp_avg_sq)

        bias_correction1 = 1.0 - torch.pow(self.beta1, step)
        bias_correction2 = 1.0 - torch.pow(self.beta2, step)
        m_t = grad_exp_avg / bias_correction1
        v_t = grad_exp_avg_sq / bias_correction2

        # update param
        denom = v_t.sqrt() + self.eps
        param -= self.lr * m_t / denom
        # update state
        state["grad_exp_avg"].data.copy_(grad_exp_avg)
        state["grad_exp_avg_sq"].data.copy_(grad_exp_avg_sq)
        state["max_grad_exp_avg_sq"].data.copy_(max_grad_exp_avg_sq)
