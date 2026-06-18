# 🍽️ What Is It? — FoodVision

> Snap a photo and let the AI tell you whether it's **Pizza**, **Steak**, or **Sushi** — with live probability bars for every class.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Django-6.0-092E20?logo=django&logoColor=white" alt="Django">
  <img src="https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/accuracy-92%25-success" alt="Accuracy">
</p>

A small **Django 6** web app that serves a **PyTorch** image classifier. Upload (or take)
a photo on a single, mobile-friendly page and the model predicts one of three food
classes, showing the top guess plus the probability of all three as animated bars.

---

## ✨ Features

- 📸 **Upload, drag & drop, or take a photo** — the file input opens the camera on phones.
- ⚡ **Instant prediction** — image is processed in memory; nothing is written to disk.
- 📊 **Full probability breakdown** — animated, colour-coded bars for pizza, steak & sushi.
- 🧠 **Transfer learning** — EfficientNet-B0 fine-tuned to ~92% test accuracy.
- 🎨 **Self-contained UI** — one clean, responsive template, no frontend build step.

---

## 🧠 The Model

The classifier is a **`torchvision` EfficientNet-B0** (ImageNet-pretrained) with the head
replaced by `Linear(1280 → 3)` and fine-tuned on the
[pizza/steak/sushi](https://github.com/mrdbourke/pytorch-deep-learning) dataset.

Preprocessing matches the pretrained weights exactly
(`EfficientNet_B0_Weights.DEFAULT.transforms()`): resize 256 → center-crop 224 →
normalize with ImageNet mean/std.

**Test-set accuracy: 92.0%**

| Class   | Recall |
| ------- | :----: |
| 🍕 Pizza | 98%    |
| 🥩 Steak | 90%    |
| 🍣 Sushi | 89%    |

> The trained weights (`pizza_steak_sushi_classification_model.pth`, ~16 MB) are committed,
> so the app runs immediately after cloning — no separate download or training needed.

---

## 🚀 Getting Started

```bash
# 1. Clone
git clone https://github.com/Rehajel15/whatisit-foodvision.git
cd whatisit-foodvision

# 2. Create & activate a virtual environment
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dev server
python manage.py runserver
```

Then open **http://127.0.0.1:8000** in your browser.

---

## 📱 Open it on your phone

Make sure your phone is on the **same Wi-Fi**, allow your computer's local IP via an
environment variable (no need to edit any file), and bind the server to all interfaces:

```bash
# PowerShell — use your machine's LAN IP (find it with `ipconfig`)
$env:DJANGO_ALLOWED_HOSTS = "192.168.x.x"
python manage.py runserver 0.0.0.0:8000
```

Then open `http://<your-ip>:8000` on the phone. The camera works over plain HTTP because
the app uses a standard file upload (`<input capture>`).

---

## 🗂️ Project structure

```
whatisit-foodvision/
├── manage.py
├── requirements.txt
├── pizza_steak_sushi_classification_model.pth   # trained EfficientNet-B0 weights
├── WhatIsIt/                # Django project (settings, urls, wsgi/asgi)
└── classifier/              # the app
    ├── model.py             # builds the EfficientNet-B0 skeleton
    ├── inference.py         # cached model loading + predict()
    ├── views.py             # single `index` view (upload → predict → render)
    └── templates/classifier/index.html   # the whole UI (inline CSS + JS)
```

**Request flow:** `index` reads the upload into memory → `predict()` runs the cached model
→ the template renders the top class and the sorted probability bars, with the uploaded
image echoed back as a base64 data URI.

---

## ⚠️ Notes

- Ships with **development settings** (`DEBUG = True`, hard-coded `SECRET_KEY`). For local
  use only — do not deploy as-is.
- The model only knows **pizza, steak, and sushi**; anything else is forced into one of
  those three classes.

---

## 📄 License

Personal learning project — feel free to use it as a reference.
