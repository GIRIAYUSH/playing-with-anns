# 🧠 Playing with ANNs

> Poking, prodding, and building Artificial Neural Networks from scratch —
> because the best way to understand a black box is to build one yourself.

A hands-on notebook series covering the **core intuition** behind deep learning, implemented in **PyTorch**, visualized with **Matplotlib**, and fully runnable on **Google Colab** (no GPU required for most modules).

Built for curious learners who want to go beyond calling `.fit()`.

---

## 🚀 Quick Start

Every notebook has a one-click Colab button. No installation needed.

| Module | Notebook | Open in Colab |
|--------|----------|---------------|
| M1 | Tensors & Autograd | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushgiri/playing-with-ANNs/blob/main/notebooks/M1_tensors_autograd.ipynb) |
| M2 | Forward Pass | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushgiri/playing-with-ANNs/blob/main/notebooks/M2_forward_pass.ipynb) |
| M3 | Loss Functions | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushgiri/playing-with-ANNs/blob/main/notebooks/M3_loss_functions.ipynb) |
| M4 | Backpropagation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushgiri/playing-with-ANNs/blob/main/notebooks/M4_backpropagation.ipynb) |
| M5 | Gradient Descent & Optimizers | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushgiri/playing-with-ANNs/blob/main/notebooks/M5_gradient_descent.ipynb) |
| M6 | Activations & Parameters | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushgiri/playing-with-ANNs/blob/main/notebooks/M6_activations_params.ipynb) |
| M7 | Full Training Loop | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushgiri/playing-with-ANNs/blob/main/notebooks/M7_full_training_loop.ipynb) |

---

## 📦 Repo Structure

```
playing-with-ANNs/
│
├── notebooks/
│   ├── M1_tensors_autograd.ipynb
│   ├── M2_forward_pass.ipynb
│   ├── M3_loss_functions.ipynb
│   ├── M4_backpropagation.ipynb
│   ├── M5_gradient_descent.ipynb
│   ├── M6_activations_params.ipynb
│   └── M7_full_training_loop.ipynb
│
├── utils/
│   └── viz.py                  # shared visualization helpers
│
├── outputs/
│   └── (saved plots & training curves per module)
│
├── requirements.txt
└── README.md
```

---

## 📚 Module Breakdown

### M1 · Tensors & Autograd
> *"Everything in deep learning is a tensor. Everything."*

The foundation of all future modules. Before writing a single neural network, you need to understand the data structure that powers it.

**What you'll implement:**
- Creating tensors from scratch — different `dtype`, `shape`, and `device`
- Moving tensors to GPU (`cuda`) and back
- Arithmetic operations: element-wise, matrix multiply (`@`), broadcasting
- Enabling gradient tracking with `requires_grad=True`
- Visualizing the computation graph that PyTorch builds silently

**Core concept:** PyTorch doesn't just do math — it records *how* it did the math so it can reverse it later. That recording is the computation graph, and it's the secret engine behind backprop.

**Key functions:** `torch.tensor`, `torch.zeros`, `torch.randn`, `.to(device)`, `.backward()`, `.grad`

---

### M2 · Forward Pass
> *"A neural network is just a chain of matrix multiplications with some non-linearities sprinkled in."*

This is where a network actually does something. You send data in, it flows through layers, and you get a prediction out.

**What you'll implement:**
- A linear layer from scratch: `output = X @ W.T + b`
- Stacking multiple layers manually to build a deep network
- Wrapping the same thing cleanly with `nn.Linear` and `nn.Module`
- Implementing and comparing activation functions: ReLU, Sigmoid, Tanh, GELU
- Plotting how each activation transforms the same input distribution

**Core concept:** Without activations, stacking 10 linear layers is mathematically identical to having just 1. Activations introduce non-linearity — that's what gives a network its learning power.

**Key functions:** `nn.Linear`, `nn.ReLU`, `nn.Sequential`, `torch.sigmoid`, `F.gelu`

---

### M3 · Loss Functions
> *"You can't improve what you can't measure."*

Loss functions are how a network knows it's wrong. Choosing the right one for the right task is one of the most important decisions you'll make.

**What you'll implement:**
- **Mean Squared Error (MSE)** — coded from scratch as `((y_pred - y_true) ** 2).mean()`, then verified against `nn.MSELoss`
- **Binary Cross-Entropy (BCE)** — for binary classification, derived from information theory
- **Categorical Cross-Entropy** — for multi-class problems, with softmax explained
- Side-by-side comparison of loss landscapes for each function
- How the same prediction error produces different loss values across functions

**Core concept:** Loss is a single number that summarises how wrong your network is. The shape of the loss landscape (its gradients) is what gradient descent will navigate.

**Key functions:** `nn.MSELoss`, `nn.BCELoss`, `nn.CrossEntropyLoss`, `F.log_softmax`

---

### M4 · Backpropagation
> *"Backprop is just the chain rule, applied very efficiently, very many times."*

The most important algorithm in deep learning. This is how a network actually *learns* — by figuring out how much each weight contributed to the error and adjusting accordingly.

**What you'll implement:**
- Manual derivation of gradients for a 2-layer network using the chain rule
- Verifying your hand-computed gradients match PyTorch's `.backward()` output exactly
- Visualizing `.grad` tensors as heatmaps — see which weights matter most
- Gradient flow diagrams showing how error signals travel backward through layers
- Demonstrating what happens when you forget `optimizer.zero_grad()` (gradient accumulation bug)

**Core concept:** The chain rule lets you decompose a complex derivative into a product of simpler ones. Backprop is just the chain rule applied layer by layer, from output back to input.

**Key functions:** `.backward()`, `.grad`, `torch.autograd.grad`, `retain_graph=True`

---

### M5 · Gradient Descent & Optimizers
> *"Learning is just rolling a ball downhill on a surface made of your mistakes."*

Once you have gradients, you need a strategy for using them to update weights. That strategy is the optimizer.

**What you'll implement:**
- Vanilla gradient descent coded from scratch: `w = w - lr * w.grad`
- Plotting the loss surface as a 2D contour map and animating the optimization path
- Implementing and comparing SGD, SGD with momentum, RMSProp, and Adam
- Learning rate experiments: too high (diverges), too low (stalls), just right
- Learning rate schedulers: StepLR, CosineAnnealingLR

**Core concept:** Adam isn't magic — it's SGD with two improvements: momentum (memory of past gradients) and adaptive learning rates (bigger steps for rarely-updated weights). Understanding vanilla GD makes Adam intuitive.

**Key functions:** `torch.optim.SGD`, `torch.optim.Adam`, `torch.optim.lr_scheduler`, `optimizer.step()`, `optimizer.zero_grad()`

---

### M6 · Activations & Parameters
> *"The wrong activation can silently kill half your network."*

A deeper look at how your architectural choices — which activation, how many layers, how many neurons — affect what a network can and can't learn.

**What you'll implement:**
- Plotting activation functions and their derivatives side by side
- Demonstrating the **dying ReLU problem** — neurons stuck outputting 0 forever
- Comparing ReLU vs Leaky ReLU vs GELU on the same task
- Width vs depth experiments: a wide shallow network vs a narrow deep one
- Visualising gradient magnitudes across layers to detect vanishing/exploding gradients
- Weight initialisation experiments: random, zeros, Xavier, He

**Core concept:** The derivative of an activation is what gets multiplied during backprop. If the derivative is 0 (dead ReLU) or very small (sigmoid saturation), gradients vanish and learning stops.

**Key functions:** `nn.LeakyReLU`, `nn.GELU`, `nn.BatchNorm1d`, `torch.nn.init.xavier_uniform_`, `torch.nn.init.kaiming_normal_`

---

### M7 · Full Training Loop
> *"All the theory above, finally wired together into something that actually learns."*

Everything from M1–M6 comes together. You build a complete, production-style training loop and train a real network on a real dataset.

**What you'll implement:**
- Custom `Dataset` and `DataLoader` with batching and shuffling
- A clean, reusable training loop with train/validation splits
- Loss and accuracy curves plotted per epoch
- Model checkpointing — saving the best weights automatically
- Overfitting demonstration and fixes: dropout, weight decay
- Training on **MNIST** (digit classification) — no GPU required
- Optional: **CIFAR-10** with a deeper network (T4 GPU on Colab recommended)

**Core concept:** A training loop is: feed batch → compute loss → backprop → update weights → repeat. Everything else (schedulers, checkpoints, logging) is scaffolding around that four-step cycle.

**Key functions:** `torch.utils.data.Dataset`, `DataLoader`, `torch.save`, `torch.load`, `nn.Dropout`, `model.train()`, `model.eval()`

---

## 🛠️ Running Locally

If you prefer running locally instead of Colab:

```bash
# Clone the repo
git clone https://github.com/ayushgiri/playing-with-ANNs.git
cd playing-with-ANNs

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

**requirements.txt**
```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter>=1.0.0
torchvision>=0.15.0
```

---

## 🧭 Recommended Learning Path

```
M1 → M2 → M3 → M4 → M5 → M6 → M7
```

Each module builds directly on the previous one. Skipping ahead is possible but M4 (backprop) is the hardest — doing M1–M3 first makes it significantly easier.

**Time estimate per module:** 2–4 hours if you read carefully and run every cell.

---

## 💡 Philosophy

Most deep learning courses hand you a high-level API and tell you to trust it. This repo takes the opposite approach — everything is implemented from scratch first, then compared against PyTorch's built-in version. The goal isn't to reinvent the wheel; it's to understand why the wheel is round.

If you can implement backprop by hand and verify it matches `.backward()` — you understand deep learning. Everything else is details.

---

## 🤝 Contributing

Found a bug? Have a cleaner explanation? Extra experiments welcome.

1. Fork the repo
2. Create a branch: `git checkout -b experiment/your-idea`
3. Commit your changes: `git commit -m "add: experiment description"`
4. Push and open a PR

---

## 📄 License

MIT — use it, learn from it, share it freely.

---

<p align="center">Made with curiosity 🔬 and a lot of gradient descent</p>