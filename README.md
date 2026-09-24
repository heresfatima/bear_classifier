# 🐻 Bear Classifier — Fast.ai Computer Vision Project

A production-grade image classification project that distinguishes between **Grizzly Bears**, **Black Bears**, and **Teddy Bears**, built using **Fast.ai** and **PyTorch**. The model went from a shaky 75% baseline to a **100% validation accuracy** through better data engineering, not architecture changes.

> 📚 **Note:** This project is part of my personal deep learning learning journey — built while working through **fast.ai's Practical Deep Learning for Coders** course. It documents real engineering problems I ran into (and fixed) along the way, not just a clean "everything worked perfectly" tutorial.

---

## 📌 Project Overview

The goal was simple on paper: build an end-to-end image classifier. In practice, it surfaced classic real-world ML problems — tiny validation sets, unreliable web-scraped data, API rate limits, and fine-grained visual classification challenges (Grizzly vs. Black bears look a lot alike!).

---

## 🚀 The Journey

### 1. The "75% Accuracy" Illusion
Started with just ~10 images per class (~20 total). The model hit 75% validation accuracy — but with only 4 validation images, **a single wrong prediction swings the metric by 25%**. The score was basically meaningless noise.

### 2. Scaling the Dataset
Needed way more data to get a stable validation metric. Ran into:
- **Breaking API changes** (`duckduckgo_search` → `ddgs`)
- **Rate-limiting** (~35 images max per query)

**Fix:** Built a multi-query search strategy — multiple targeted search phrases per category — to scale the dataset from ~30 images to **382 verified clean images**.

### 3. Fine-Grained Classification Problem
Grizzly and Black bears are visually similar (color overlap, similar poses). The key differentiator is the **Grizzly's shoulder hump** — but standard `RandomResizedCrop` augmentation was cropping it out of frame, forcing the model to guess based on fur color alone.

**Fix:** Set `min_scale=0.7` in the crop transform to preserve key anatomical features.

### 4. Final Training Results
With a properly sized dataset (**307 training images / 76 validation images**) and fixed augmentation, fine-tuning a pretrained **ResNet-18** for 5 epochs achieved a final validation accuracy of **100%**.

| Epoch | Train Loss | Valid Loss | Accuracy | Error Rate | Time  |
|:-----:|:----------:|:----------:|:--------:|:----------:|:-----:|
| 0     | 0.063632   | 0.016712   | 1.000000 | 0.000000   | 00:21 |
| 1     | 0.072526   | 0.044033   | 0.986842 | 0.013158   | 00:21 |
| 2     | 0.102816   | 0.000898   | 1.000000 | 0.000000   | 00:21 |
| 3     | 0.088955   | 0.001303   | 1.000000 | 0.000000   | 00:21 |
| 4     | 0.082555   | 0.001276   | 1.000000 | 0.000000   | 00:21 |

```
✅ High-Accuracy 'bear_model.pkl' successfully exported!
```

### 5. Confusion Matrix — Final Model Evaluation

```
🎯 FINAL PRODUCTION MODEL ACCURACY: 100.00%
```

|              | Predicted: black | Predicted: grizzly | Predicted: teddy |
|--------------|:-----------------:|:-------------------:|:------------------:|
| **Actual: black**   | 25                | 0                   | 0                  |
| **Actual: grizzly** | 0                 | 28                  | 0                  |
| **Actual: teddy**   | 0                 | 0                   | 23                 |

Every single validation image (76 total) was classified correctly — zero off-diagonal misclassifications across all three classes.

### 6. Local Testing (Reloaded Model)
Before deploying, the exported `bear_model.pkl` was reloaded fresh and tested on a real-world sample image to confirm it works outside the training session:

```
📸 Tested Image: 6fcf9015-fe33-413b-8e05-cf747e3299ec.jpg
🐻 Reloaded Model Prediction: black
🎯 Confidence Level: 100.00%
```

This confirmed the exported model file is portable and gives consistent, high-confidence predictions when loaded independently — exactly what's needed before wrapping it in a web app.

### 7. Deployment & User Interface
The trained model was wrapped in a **Gradio** interface and deployed live as a **Hugging Face Space**.

**UI features:**
- Simple drag-and-drop / click-to-upload image box
- One-click **Submit** to run inference
- Real-time prediction output panel
- **Share via Link** — the app can be shared publicly with a single link
- Runs on Hugging Face's free **ZERO** GPU tier

The live app title reads **"Bear Classifier"**, hosted under a Hugging Face Space (`bear-classifier`), giving anyone a shareable, no-code way to test the model in their browser.

---

## 🧠 Key Learnings

- **Data > Architecture** — Clean, well-sampled data mattered far more than swapping ResNet18 for ResNet50.
- **Validation set size matters** — Small validation splits produce misleading, high-variance metrics.
- **Domain knowledge shapes preprocessing** — Knowing the Grizzly's shoulder hump was the key feature directly informed the augmentation strategy.
- **Debugging is the real skill** — Diagnosing rate limits, confusion matrices, and bad crop transforms mattered more than memorizing syntax.

---

## 🛠️ Tech Stack

- Python
- [Fast.ai](https://www.fast.ai/) / PyTorch
- `ddgs` (image search/scraping)
- Gradio (deployment/UI)
- Kaggle Notebooks (training environment)
- Hugging Face Spaces (live deployment)

---

## ▶️ Running the Project

1. Install dependencies:
   ```bash
   pip install fastai ddgs gradio
   ```
2. Run the data collection + training script to build the dataset and train the model.
3. Launch the Gradio app:
   ```bash
   python app.py
   ```
4. Upload an image and get real-time predictions across all three classes.

---

## 📂 Project Structure

```
bear-classifier/
├── bear_dataset_large/     # Collected & verified training images
│   ├── grizzly/
│   ├── black/
│   └── teddy/
├── bear_model.pkl          # Exported trained model
├── train.py                # Data collection + training pipeline
├── app.py                  # Gradio deployment app
└── README.md
```

---

## 🙋 About This Project

This was built as a hands-on learning exercise while studying deep learning fundamentals through the fast.ai course. It's meant to reflect the real messiness of building an ML pipeline — bugs, rate limits, and all — rather than a polished, idealized version.
