# PrimeLine@DravidianLangTech 2026: Hope Speech Detection in Tulu Using XLM-RoBERTa

**Authors:** Rithikaa V, Sanjay Krishnan K, Nithya Varshini C N R  
**Guide:** S. Sumathi  
**Institution:** Department of Information Technology, St. Joseph's College of Engineering  
**Shared Task:** DravidianLangTech@ACL 2026 — Hope Speech Detection in Tulu  

---

## 📄 Paper

This repository contains the code, data, and paper for our system submitted to the
**Hope Speech Detection in Tulu** shared task at DravidianLangTech@ACL 2026.

> *PrimeLine@DravidianLangTech 2026: Hope Speech Detection in Tulu Using XLM-RoBERTa
> for Coarse and Fine-Grained Classification*

The paper is available in the `paper/` folder as both `.pdf` and `.tex`.

---

## 🗂️ Repository Structure

```
├── README.md
├── paper/
│   ├── hope_speech_v_final.tex       # LaTeX source
│   ├── hope_custom.bib               # Bibliography
│   └── hope_speech_detection.pdf     # Final paper PDF
├── track1/
│   ├── Task1.ipynb                   # Training & inference notebook (Track 1)
│   ├── train_CG.csv                  # Training data — Coarse-Grained
│   ├── dev_CG.csv                    # Validation data — Coarse-Grained
│   ├── test_data_withoutlabelCG.csv  # Test data (no labels)
│   └── PrimeLine_Tulu_task1.csv      # Our submitted predictions
└── track2/
    ├── Task2.ipynb                   # Training & inference notebook (Track 2)
    ├── Finegrained_train_data.csv    # Training data — Fine-Grained
    ├── Finegrained_dev_data.csv      # Validation data — Fine-Grained
    ├── Finegrained_test_data_withoutlabel.csv  # Test data (no labels)
    └── PrimeLine_Tulu_task2.csv      # Our submitted predictions
```

---

## 🧠 System Overview

We fine-tune **XLM-RoBERTa** (`xlm-roberta-base`, ~270M parameters), a cross-lingual
transformer pre-trained on 100+ languages, on code-mixed Tulu social media data.

**Two tracks are addressed:**

| Track | Task | Classes | Training Samples |
|-------|------|---------|-----------------|
| Track 1 | Coarse-Grained | 4 (Encouraging Hope, Discouraging Hope, Blended Hope, Uninvolved) | 5,991 |
| Track 2 | Fine-Grained | 5 (Inspiring Hope, Hopelessness, Realistic Hope, Optimistic Hope, Fading Hope) | 3,185 |

**Pipeline:**
1. Raw code-mixed Tulu text → XLM-RoBERTa tokenizer (SentencePiece, max 128 tokens)
2. XLM-RoBERTa encoder (12 layers, 768 dims)
3. `[CLS]` token representation → Linear classification head
4. Softmax → Predicted label

---

## 📊 Results

Official evaluation on Codabench (Macro-averaged):

| Track | Accuracy | Precision | Recall | Macro F1 |
|-------|----------|-----------|--------|----------|
| Track 1 — Coarse | 0.60 | 0.30 | 0.41 | **0.34** |
| Track 2 — Fine | 0.24 | 0.19 | 0.19 | **0.19** |

---

## ⚙️ Requirements

```
Python >= 3.8
torch
transformers
pandas
scikit-learn
numpy
```

Install with:
```bash
pip install torch transformers pandas scikit-learn numpy
```

---

## 🚀 How to Run

1. Open `track1/Task1.ipynb` in **Google Colab** (recommended — uses T4 GPU)
2. Upload the corresponding CSV files from `track1/`
3. Run all cells — training, evaluation, and prediction are included
4. Repeat with `track2/Task2.ipynb` for Track 2

> **Note:** Training was done on Google Colab with an NVIDIA T4 GPU.
> 2 epochs, batch size 8, AdamW optimizer (lr = 2e-5, weight decay = 0.01).

---

## 📝 Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{primeline-hope-tulu-2026,
  title     = {PrimeLine@DravidianLangTech 2026: Hope Speech Detection in Tulu
               Using XLM-RoBERTa for Coarse and Fine-Grained Classification},
  author    = {Rithikaa V and Sanjay Krishnan K and Nithya Varshini C N R},
  booktitle = {Proceedings of the Sixth Workshop on Speech, Vision, and Language
               Technologies for Dravidian Languages},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

---

## 🙏 Acknowledgments

We thank the organizers of the DravidianLangTech@ACL 2026 shared task for providing
the datasets and the Codabench evaluation platform. We also thank our guide,
**S. Sumathi**, Department of Information Technology, St. Joseph's College of Engineering,
for her constant guidance and support.

