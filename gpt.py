import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.optim.optimizer as optimizer
from tqdm import tqdm
import numpy as np


# --------------------------------------------------------------------------- #
class PolyakSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: optimizer.ParamsT,
        eps: float = 1e-8,
        fstar: float = 0.0,
        verbose: bool = False,
    ):
        defaults = dict(
            eps=eps,
            fstar=fstar,
            verbose=verbose,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("PolyakSGD doesn't support per-parameter options (parameter groups)")
        if eps<0 or eps>1:
            raise ValueError("Epsilon must be positive and should be small")
        
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure):
        group = self.param_groups[0]
        closure = torch.enable_grad()(closure)
        loss = closure()

        flat_grad = self._gather_flat_grad()
        gradsq = float(flat_grad@flat_grad)
        alpha = (float(loss) - group['fstar']) / (gradsq + group["eps"])
        if group["verbose"]:
            print(f"loss: {float(loss):0.2f}, grad^2 {gradsq:0.2f}, alpha: {float(alpha):0.4f}")
        self._add_grad(-alpha, flat_grad)

        return loss, alpha
    
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel


def test_alpha_calc():
    fstar = 0; eps = 1e-8
    param = torch.tensor([1.0, 2.0], requires_grad=True)
    optimizer = PolyakSGD([param], fstar=fstar, eps=eps, verbose=True)

    def closure():
        optimizer.zero_grad()
        loss = (param ** 2).sum() # loss = 5
        loss.backward() # grad = [2,4]
        return loss

    loss, alpha = optimizer.step(closure)
    grad = torch.tensor([2.0, 4.0])
    gradsq = float(grad@grad)
    expected_alpha = (float(loss) - fstar) / (gradsq + eps)
    expected_param = torch.tensor([1.0, 2.0]) - expected_alpha*grad

    assert torch.allclose(torch.tensor(alpha), torch.tensor(expected_alpha), atol=1e-6)
    assert torch.allclose(param, expected_param, atol=1e-6)

def test_zero_grad():
    param = torch.tensor([1.0], requires_grad=True)
    optimizer = PolyakSGD([param], verbose=True)

    def closure():
        optimizer.zero_grad()
        loss = torch.tensor(0.0)  # loss = anything
        param.grad = torch.tensor([0.0]) # grad = [0,]
        return loss

    loss, alpha = optimizer.step(closure)
    assert param == 1.0

def test_loss_equals_fstar():
    param = torch.tensor([2.0], requires_grad=True)
    optimizer = PolyakSGD([param], fstar=4, verbose=True)

    def closure():
        optimizer.zero_grad()
        loss = (param ** 2)  # loss = fstar
        loss.backward()      # grad = anything
        return loss

    loss, alpha = optimizer.step(closure)
    assert alpha == 0
    assert param == 2

def test_closure_call_count():
    closure_call_count = 0

    def closure():
        nonlocal closure_call_count
        closure_call_count += 1
        return torch.tensor(0.0) # loss = anything

    optimizer = PolyakSGD([torch.tensor([1.0])], verbose=True)
    optimizer.step(closure)
    assert closure_call_count == 1

def test_eps_stability():
    eps = 1e-8; fstar = 0
    param = torch.tensor([1e-4], requires_grad=True)
    optimizer = PolyakSGD([param], fstar=fstar, eps=eps, verbose=True)

    def closure():
        loss = param ** 2  # loss = 1e-8
        loss.backward()    # grad = 2e-4 -> grad@grad = 4e-8
        return loss

    loss, alpha = optimizer.step(closure)
    gradsq = 4e-8
    expected_alpha = (float(loss) - fstar) / (gradsq + eps)
    assert torch.allclose(torch.tensor(alpha), torch.tensor(expected_alpha), atol=1e-6)

def test_batch_sum_reduction():
    fstar = 0; eps = 1e-8
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data = torch.tensor([[1.0, 2.0]], requires_grad=True)
    optimizer = PolyakSGD(model.parameters(), fstar=fstar, eps=eps, verbose=True)
    X = torch.ones(2, 2)  # batch size 2
    y = torch.ones(2, 1)

    def closure():
        optimizer.zero_grad()
        loss = F.mse_loss(model(X), y, reduction="sum")
        loss.backward()
        return loss

    # Forward pass
    loss, alpha = optimizer.step(closure)
    # x_pred = [1*1 + 2*1, 1*1 + 2*1] = [3,3]
    # loss =  sum(x_pred - y)^2 = (3-1)^2 + (3-1)^2 = 8
    # grad_i = 2*(x_pred_i-y_i) summed over batch
    # grad = [2*2, 2*2] + [2*2, 2*2]  = [8, 8]
    manual_grad = torch.tensor([[8.0, 8.0]])
    assert torch.allclose(model.weight.grad, manual_grad, atol=1e-6)

    gradsq = 128.0 # gradsq = 2*64 = 128
    expected_alpha = (float(loss) - fstar) / (gradsq + eps)  # alpha = 0.0625
    assert torch.allclose(torch.tensor(alpha), torch.tensor(expected_alpha), atol=1e-6)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 140 
n_head = 2
n_layer = 4
dropout = 0.2

torch.manual_seed(1337)

os.makedirs('images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('weights', exist_ok=True)

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_train_val_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data) if split == "train" else get_batch(val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = (losses.mean(), losses.std())
    model.train()
    return out

@torch.no_grad()
def estimate_test_loss(model, test_data, test_iters=500):
    losses = torch.zeros(test_iters, device=device)
    for k in tqdm(range(test_iters)):
        ix = torch.randint(len(test_data) - block_size, (batch_size,)).to(device)
        x = torch.stack([test_data[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([test_data[i+1:i+block_size+1] for i in ix]).to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean(), losses.std()


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, bias=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, bias=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, bias=bias) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, bias=False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, bias=bias)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BlockNoSkip(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, bias=False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, bias=bias)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.sa(self.ln1(x))
        x = self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, bias=False, skip=True):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        if skip:
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, bias=bias) for _ in range(n_layer)])
        else:
            self.blocks = nn.Sequential(*[BlockNoSkip(n_embd, n_head=n_head, bias=bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, reduction="sum")

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def train_transformer_const(model, train_data, val_data, output_name=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    losses = []

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses.append(estimate_train_val_loss(model, train_data, val_data))   

        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    if output_name is None:
        output_name = f"const_{max_iters}"
    torch.save(model.state_dict(), f'weights/{output_name}.pth')

    x = [(i+1)*eval_interval for i in range(max_iters//eval_interval)]
    mean_train_loss = [loss['train'][0] for loss in losses[1:]]
    mean_val_loss = [loss['val'][0] for loss in losses[1:]]
    std_train_loss = [loss['train'][1] for loss in losses[1:]]
    std_val_loss = [loss['val'][1] for loss in losses[1:]]

    print(f"Training loss: {mean_train_loss[-1]}")
    print(f"Validation loss: {mean_val_loss[-1]}")

    plt.figure(figsize=(7,4))
    plt.errorbar(x, mean_train_loss, std_train_loss, label="train")
    plt.errorbar(x, mean_val_loss, std_val_loss, label="val")
    plt.title(f"Loss during Training - {n_embd=}, {n_head=}, {n_layer=}")
    plt.xlabel("No. iterations")
    plt.ylabel("Loss")
    plt.legend(fancybox=True)
    plt.tight_layout()
    plt.savefig(f'images/{output_name}.svg')

def train_transformer_adam(model, train_data, val_data, output_name=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses.append(estimate_train_val_loss(model, train_data, val_data))   

        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    if output_name is None:
        output_name = f"const_{max_iters}"
    torch.save(model.state_dict(), f'weights/{output_name}.pth')

    x = [(i+1)*eval_interval for i in range(max_iters//eval_interval)]
    mean_train_loss = [loss['train'][0] for loss in losses[1:]]
    mean_val_loss = [loss['val'][0] for loss in losses[1:]]
    std_train_loss = [loss['train'][1] for loss in losses[1:]]
    std_val_loss = [loss['val'][1] for loss in losses[1:]]

    print(f"Training loss: {mean_train_loss[-1]}")
    print(f"Validation loss: {mean_val_loss[-1]}")

    plt.figure(figsize=(7,4))
    plt.errorbar(x, mean_train_loss, std_train_loss, label="train")
    plt.errorbar(x, mean_val_loss, std_val_loss, label="val")
    plt.title(f"Loss during Training - {n_embd=}, {n_head=}, {n_layer=}")
    plt.xlabel("No. iterations")
    plt.ylabel("Loss")
    plt.legend(fancybox=True)
    plt.tight_layout()
    plt.savefig(f'images/{output_name}.svg')

def train_transformer_polyak(model, train_data, val_data, output_name=None):
    optimizer = PolyakSGD(model.parameters())
    losses = []

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 1 or iter == max_iters - 1:
            losses.append(estimate_train_val_loss(model, train_data, val_data))   
        
        def closure():
            optimizer.zero_grad()
            x, y = get_batch(train_data)
            _, loss = model(x, y)
            loss.backward()
            return loss
        optimizer.step(closure)
    
    if output_name is None:
        output_name = f"polyak_{max_iters}"
    torch.save(model.state_dict(), f'weights/{output_name}.pth')

    x = [(i+1)*eval_interval for i in range(max_iters//eval_interval)]
    mean_train_loss = [loss['train'][0] for loss in losses[1:]]
    mean_val_loss = [loss['val'][0] for loss in losses[1:]]
    std_train_loss = [loss['train'][1] for loss in losses[1:]]
    std_val_loss = [loss['val'][1] for loss in losses[1:]]

    print(f"Training loss: {mean_train_loss[-1]}")
    print(f"Validation loss: {mean_val_loss[-1]}")

    plt.figure(figsize=(7,4))
    plt.errorbar(x, mean_train_loss, std_train_loss, label="train")
    plt.errorbar(x, mean_val_loss, std_val_loss, label="val")
    plt.title(f"Loss during Training - {n_embd=}, {n_head=}, {n_layer=}")
    plt.xlabel("No. iterations")
    plt.ylabel("Loss")
    plt.legend(fancybox=True)
    plt.tight_layout()
    plt.savefig(f'images/{output_name}.svg')
# --------------------------------------------------------------------------- #


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1,1, bias=True)

    def forward(self, idx):
        y = self.layer(idx)
        return y
        

if __name__ == "__main__":
    
    # # ----------------------------------------------------------------------- #
    # # q1 (a)
    # # ----------------------------------------------------------------------- #
    # print("Tests\n"+"-"*40)
    # test_alpha_calc()
    # test_zero_grad()
    # test_loss_equals_fstar()
    # test_eps_stability()
    # test_batch_sum_reduction()
    # test_closure_call_count()
    # # ----------------------------------------------------------------------- #
    
    # # ----------------------------------------------------------------------- #
    # # q1 (b)
    # # ----------------------------------------------------------------------- #
    # print("-"*40+"\nSimple Linear Regression\n"+"-"*40)
    # runs = 20
    # max_iters = 100
    # data_size = 100
    # batch_size = 25
    # alpha0 = 1e-3

    # epochs = max_iters*batch_size//data_size    
    # iters = [i+1 for i in range(max_iters)]
    # plt.figure(figsize=(12, 6))

    # for sigma in [0.5, 0.1, 0.05, 0.01]:
    #     X = (2*torch.rand(data_size)-1).view(-1, 1)
    #     Y = 5*X + 2 + sigma*torch.randn(data_size).view(-1, 1)

    #     dataset = torch.utils.data.TensorDataset(X, Y)
    #     batchloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #     polyak_loss = np.zeros((runs, max_iters))
    #     polyak_params = np.zeros((runs, max_iters))
    #     for run in range(runs):
    #         losses = np.zeros(max_iters)
    #         model = LinearRegressionModel()
    #         optim = PolyakSGD(model.parameters())
    #         for i in range(epochs):
    #             for j, (x, y) in enumerate(batchloader):
    #                 def closure():
    #                     optim.zero_grad()
    #                     loss = torch.sum((model(x)-y)**2)
    #                     loss.backward()
    #                     return loss
    #                 loss, alpha = optim.step(closure)
    #                 losses[(data_size//batch_size)*i + j] = float(loss)
    #         polyak_loss[run] = losses

    #     const_loss = np.zeros((runs, max_iters))
    #     for run in range(runs):
    #         losses = np.zeros(max_iters)
    #         model = LinearRegressionModel()
    #         optim = torch.optim.SGD(model.parameters(), lr=alpha0)
    #         for i in range(epochs):
    #             for j, (x, y) in enumerate(batchloader):
    #                 optim.zero_grad()
    #                 loss = torch.sum((model(x)-y)**2)
    #                 loss.backward()
    #                 optim.step()
    #                 losses[(data_size//batch_size)*i + j] = float(loss)
    #         const_loss[run] = losses

    #     plt.errorbar(iters, const_loss.mean(axis=0), const_loss.std(axis=0), linestyle="dashed", label=f"Constant $\\sigma = {sigma}$")
    #     plt.errorbar(iters, polyak_loss.mean(axis=0), polyak_loss.std(axis=0), label=f"Polyak $\\sigma = {sigma}$")

    # plt.yscale("log")
    # plt.xlabel("No. Iterations")
    # plt.ylabel("Function Value")
    # plt.title(f"Minibatch SGD Constant ($\\alpha={alpha0:.1e}$) vs Polyak Step Size ($N={batch_size}$, {runs} runs each)")
    # plt.legend()
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig("images//lr_noise.svg")


    # plt.figure(figsize=(12, 6))
    # for batch_size in [2, 10, 20, 50]:
    #     sigma = 0.05
    #     epochs = max_iters*batch_size//data_size   
    #     X = (2*torch.rand(data_size)-1).view(-1, 1)
    #     Y = 5*X + 2 + sigma*torch.randn(data_size).view(-1, 1)

    #     dataset = torch.utils.data.TensorDataset(X, Y)
    #     batchloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #     polyak_loss = np.zeros((runs, max_iters))
    #     polyak_params = np.zeros((runs, max_iters))
    #     for run in range(runs):
    #         losses = np.zeros(max_iters)
    #         model = LinearRegressionModel()
    #         optim = PolyakSGD(model.parameters())
    #         for i in range(epochs):
    #             for j, (x, y) in enumerate(batchloader):
    #                 def closure():
    #                     optim.zero_grad()
    #                     loss = torch.sum((model(x)-y)**2)
    #                     loss.backward()
    #                     return loss
    #                 loss, alpha = optim.step(closure)
    #                 losses[(data_size//batch_size)*i + j] = float(loss)
    #         polyak_loss[run] = losses

    #     const_loss = np.zeros((runs, max_iters))
    #     for run in range(runs):
    #         losses = np.zeros(max_iters)
    #         model = LinearRegressionModel()
    #         optim = torch.optim.SGD(model.parameters(), lr=alpha0)
    #         for i in range(epochs):
    #             for j, (x, y) in enumerate(batchloader):
    #                 optim.zero_grad()
    #                 loss = torch.sum((model(x)-y)**2)
    #                 loss.backward()
    #                 optim.step()
    #                 losses[(data_size//batch_size)*i + j] = float(loss)
    #         const_loss[run] = losses

    #     plt.errorbar(iters, const_loss.mean(axis=0), const_loss.std(axis=0), linestyle="dashed", label=f"Constant $N = {batch_size}$")
    #     plt.errorbar(iters, polyak_loss.mean(axis=0), polyak_loss.std(axis=0), label=f"Polyak $N = {batch_size}$")

    # plt.yscale("log")
    # plt.xlabel("No. Iterations")
    # plt.ylabel("Function Value")
    # plt.title(f"Minibatch SGD Constant ($\\alpha={alpha0:.1e}$) vs Polyak Step Size ($\\sigma={sigma}$, {runs} runs each)")
    # plt.legend()
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig("images//lr_batch.svg")
    # # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    # q1 (d)
    # ----------------------------------------------------------------------- #
    train_path = "input_childSpeech_trainingSet.txt"
    with open(train_path) as f:
        train_text = f.read()
    chars = sorted(list(set(train_text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    split = int(0.9*len(train_data))
    train_data, val_data = train_data[:split], train_data[split:]
    
    test_path = "input_childSpeech_testSet.txt"
    with open(test_path) as f:
        test_text = f.read()
    test_data = torch.tensor(encode(test_text), dtype=torch.long)

    model = GPTLanguageModel(len(chars), n_embd, n_head, n_layer).to(device)
    print("-"*80,f'Initial Model: {sum(p.numel() for p in model.parameters())/1e6:0.4f}M parameters', sep="\n")
    train_transformer_polyak(model, train_data, val_data)
    mean_model_loss, std_model_loss = estimate_test_loss(model, test_data)
    print(f"Polyak test loss: Mean {mean_model_loss:0.4f}, Standard Deviation: {std_model_loss:0.4f}")

    model = GPTLanguageModel(len(chars), n_embd, n_head, n_layer).to(device)
    print("-"*80,f'Initial Model: {sum(p.numel() for p in model.parameters())/1e6:0.4f}M parameters', sep="\n")
    train_transformer_const(model, train_data, val_data)
    mean_model_loss, std_model_loss = estimate_test_loss(model, test_data)
    print(f"Constant test loss: Mean {mean_model_loss:0.4f}, Standard Deviation: {std_model_loss:0.4f}")

    model = GPTLanguageModel(len(chars), n_embd, n_head, n_layer).to(device)
    print("-"*80,f'Initial Model: {sum(p.numel() for p in model.parameters())/1e6:0.4f}M parameters', sep="\n")
    train_transformer_adam(model, train_data, val_data)
    mean_model_loss, std_model_loss = estimate_test_loss(model, test_data)
    print(f"Adam test loss: Mean {mean_model_loss:0.4f}, Standard Deviation: {std_model_loss:0.4f}")