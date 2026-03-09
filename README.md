# PrimeLine@DravidianLangTech 2026: Hope Speech Detection in Tulu Using XLM-RoBERTa

**Authors:** Rithikaa V, Sanjay Krishnan K, Nithya Varshini C N R
**Guide:** S. Sumathi
**Institution:** Department of Information Technology, St. Joseph's College of Engineering
**Shared Task:** DravidianLangTech@ACL 2026 — Hope Speech Detection in Tulu

---

## Paper

This repository contains the code, datasets, and paper for our system submitted to the **Hope Speech Detection in Tulu** shared task at DravidianLangTech@ACL 2026.

**PrimeLine@DravidianLangTech 2026: Hope Speech Detection in Tulu Using XLM-RoBERTa for Coarse and Fine-Grained Classification**

The research paper is available in the `paper/` folder in both `.pdf` and `.tex` formats.

---

## Repository Structure

```
├── README.md
├── paper/
│   ├── hope_speech_v_final.tex       # LaTeX source file
│   ├── hope_custom.bib               # Bibliography file
│   └── hope_speech_detection.pdf     # Final paper PDF
├── track1/
│   ├── Task1.ipynb                   # Training and inference notebook (Track 1)
│   ├── train_CG.csv                  # Training data — Coarse-Grained
│   ├── dev_CG.csv                    # Validation data — Coarse-Grained
│   ├── test_data_withoutlabelCG.csv  # Test data (without labels)
│   └── PrimeLine_Tulu_task1.csv      # Submitted predictions
└── track2/
    ├── Task2.ipynb                   # Training and inference notebook (Track 2)
    ├── Finegrained_train_data.csv    # Training data — Fine-Grained
    ├── Finegrained_dev_data.csv      # Validation data — Fine-Grained
    ├── Finegrained_test_data_withoutlabel.csv  # Test data (without labels)
    └── PrimeLine_Tulu_task2.csv      # Submitted predictions
```

---

## System Overview

In this work, we fine-tune **XLM-RoBERTa** (`xlm-roberta-base`, approximately 270M parameters), a multilingual transformer model trained on more than 100 languages, to classify code-mixed Tulu social media text.

The system addresses two classification tracks provided in the shared task.

| Track   | Task           | Classes                                                                        | Training Samples |
| ------- | -------------- | ------------------------------------------------------------------------------ | ---------------- |
| Track 1 | Coarse-Grained | 4 (Encouraging Hope, Discouraging Hope, Blended Hope, Uninvolved)              | 5,991            |
| Track 2 | Fine-Grained   | 5 (Inspiring Hope, Hopelessness, Realistic Hope, Optimistic Hope, Fading Hope) | 3,185            |

### Processing Pipeline

1. Code-mixed Tulu text is tokenized using the XLM-RoBERTa tokenizer (SentencePiece, maximum 128 tokens).
2. The text is passed through the XLM-RoBERTa encoder consisting of 12 transformer layers with a hidden size of 768.
3. The `[CLS]` token representation is extracted.
4. A linear classification layer followed by a Softmax function predicts the final label.

---

## Results

The system was evaluated on the Codabench platform using macro-averaged metrics.

| Track            | Accuracy | Precision | Recall | Macro F1 |
| ---------------- | -------- | --------- | ------ | -------- |
| Track 1 — Coarse | 0.60     | 0.30      | 0.41   | **0.34** |
| Track 2 — Fine   | 0.24     | 0.19      | 0.19   | **0.19** |

---

## Requirements

```
Python >= 3.8
torch
transformers
pandas
scikit-learn
numpy
```

Install the dependencies using:

```bash
pip install torch transformers pandas scikit-learn numpy
```

---

## How to Run

1. Open `track1/Task1.ipynb` in **Google Colab** (recommended, as it provides GPU support).
2. Upload the corresponding CSV files from the `track1/` folder.
3. Run all cells in the notebook to perform training, evaluation, and prediction.
4. Repeat the same process with `track2/Task2.ipynb` for the fine-grained classification task.

Training experiments were conducted using **Google Colab with an NVIDIA T4 GPU**, for **2 epochs with batch size 8**, using the **AdamW optimizer** (learning rate = `2e-5`, weight decay = `0.01`).

---

## Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{primeline-hope-tulu-2026,
  title     = {PrimeLine@DravidianLangTech 2026: Hope Speech Detection in Tulu Using XLM-RoBERTa for Coarse and Fine-Grained Classification},
  author    = {Rithikaa V and Sanjay Krishnan K and Nithya Varshini C N R},
  booktitle = {Proceedings of the Sixth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

---

## Acknowledgments

We thank the organizers of the **DravidianLangTech@ACL 2026 shared task** for providing the datasets and the Codabench evaluation platform. We also express our sincere gratitude to our guide **S. Sumathi**, Department of Information Technology, St. Joseph's College of Engineering, for her continuous guidance and support throughout this work.
