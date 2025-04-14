# MiniGPT Character-Level Transformer

A simple, educational implementation of a GPT-style Transformer model that operates at the character level. This project allows you to train the model on a text corpus and generate new text based on the learned patterns.


## Features

*   Decoder-only Transformer architecture.
*   Character-level tokenization.
*   Multi-Head Self-Attention.
*   Configurable hyperparameters via `config.yaml`.
*   Scripts for training (`train.py`) and text generation (`generate.py`).
*   GPU (CUDA) support with mixed-precision training.

## Project Structure
## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arsal77/mini-gpt-character-level.git 

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Data Preparation

1.  The model trains on a single plain text (`.txt`) file.
2.  A small sample file is included for quick testing.
3.  **To use your own data:**
    *   Place your `.txt` file inside the `data/` directory.
    *   Update the `corpus_path` parameter in `config.yaml` to point to your file (e.g., `corpus_path: 'data/your_corpus.txt'`).
4.  

## Configuration (`config.yaml`)

All model hyperparameters, training settings, and file paths are controlled via `config.yaml`. Key parameters include:

*   **`corpus_path`**: Path to the training text file (relative to project root).
*   **`model_load_path`**: Path where the trained model weights will be saved and loaded from (relative to project root).
*   **`context_len`**: The sequence length the model sees during training.
*   **`n_emb`**: The embedding dimension size.
*   **`n_heads`**: The number of attention heads.
*   **`n_layers`**: The number of Transformer blocks (depth).
*   **`batch_size`**: Number of sequences processed in parallel during training.
*   **`train_iterations`**: Total number of training steps.
*   **`learning_rate`**: The learning rate for the AdamW optimizer.
*   **`start_prompt`**, **`max_gen_tokens`**: Defaults for text generation.

**Note on Hardware:** The default parameters in `config.yaml` might be set for GPU training (like T4 on Colab). If running on a CPU, significantly reduce `n_emb`, `n_layers`, `context_len`, and `batch_size` for feasible training times.

## Usage

### Training

1.  Ensure your data path is correct in `config.yaml`.
2.  Adjust hyperparameters in `config.yaml` as needed.
3.  Run the training script from the project root directory:
    ```bash
    python train.py
    ```
4.  Training progress (loss) will be printed to the console.
5.  The trained model weights will be saved to the path specified by `model_load_path` in `config.yaml` (e.g., `model/mini_gpt_model.pth`).

### Text Generation

1.  Make sure a trained model file exists at the location specified by `model_load_path` in `config.yaml`.
2.  Run the generation script:
    ```bash
    python generate.py
    ```
    This will use the `start_prompt` and `max_gen_tokens` from `config.yaml`.

3.  **To override the prompt and length:**
    ```bash
    python generate.py --prompt "Your starting text here" --length 300
    ```

