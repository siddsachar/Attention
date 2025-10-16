# Attention Mechanisms in PyTorch

This repository provides a comprehensive, step-by-step tutorial and implementation of attention mechanisms using PyTorch. It is designed for learners and practitioners who want to understand, experiment with, and extend attention-based models, including self-attention, causal masking, dropout regularization, and multi-head attention.

## Features

- **Jupyter Notebook Tutorial:**
  - Interactive notebook (`attention.ipynb`) with annotated code and detailed markdown explanations.
  - Covers basic attention, softmax normalization, context vector computation, matrix multiplication, linear projections, and modular PyTorch implementations.
  - Advanced topics: causal masking, dropout, custom self-attention modules, and multi-head attention.

- **Modular PyTorch Code:**
  - Custom `SelfAttention`, `SelfAttentionv2`, `CausalAttention`, and `MultiHeadAttention` modules.
  - Demonstrates best practices for PyTorch module design, including handling input/output dimensions, batch processing, and regularization.

- **Project Structure:**
  - Clean organization for code, notebooks, and outputs.
  - `.gitignore` for Python, Jupyter, VS Code, and data artifacts.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Jupyter Notebook or JupyterLab

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/siddsachar/attention.git
   cd attention
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install torch jupyter
   ```

### Usage
1. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Open `attention.ipynb` and follow the step-by-step tutorial.
3. Run code cells and review markdown explanations to deepen your understanding of attention mechanisms.

## Project Structure
```
attention.ipynb        # Main tutorial notebook
.gitignore             # Ignore rules for repo
README.md              # Project documentation
```

## Key Concepts Covered
- **Attention Score Calculation:** Dot product, softmax normalization, context vector computation.
- **Matrix Multiplication:** Efficient computation of all-pair attention scores.
- **Linear Projections:** Query, key, and value vector generation.
- **Self-Attention Modules:** Modular PyTorch implementations for flexible experimentation.
- **Causal Masking:** Preventing information leakage from future tokens in autoregressive models.
- **Dropout Regularization:** Improving generalization in attention layers.
- **Multi-Head Attention:** Parallel attention heads for richer sequence representations.

## Extending the Project
- Add new attention mechanisms (e.g., cross-attention, sparse attention).
- Integrate with larger models (e.g., Transformer encoder/decoder).
- Experiment with different regularization and masking strategies.
- Visualize attention weights and context vectors.

## Contributing
Pull requests and suggestions are welcome! Please open an issue to discuss major changes before submitting.

## License
This project is licensed under the MIT License.

## Acknowledgments
- PyTorch documentation and tutorials
- "Attention Is All You Need" (Vaswani et al., 2017)
- The open-source deep learning community
