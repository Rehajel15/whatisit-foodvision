# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Django 6.0 project (`WhatIsIt`) that serves a pre-trained PyTorch image
classifier over the web. The classifier is an EfficientNet-B0 model (transfer
learning, ImageNet-pretrained) that distinguishes **pizza / steak / sushi**
(3 classes), ~92% test accuracy.

A user uploads (or takes) a photo on the single-page UI at `/`; the image is run
through the model and the page shows the top prediction plus the probability of
all three classes as animated bars.

## Environment & commands

Dependencies live only in the local `venv/` (there is no `requirements.txt`).
Python 3.14, Django 6.0.6, torch 2.12 (+cu132) / torchvision, Pillow, numpy.

```powershell
# Activate the virtualenv (PowerShell)
venv\Scripts\Activate.ps1

# Run the dev server
python manage.py runserver

# Database migrations (SQLite, db.sqlite3 — not yet created)
python manage.py migrate
python manage.py makemigrations

# Tests
python manage.py test                       # all tests
python manage.py test <app>.tests.<Class>.<method>   # single test
```

If you invoke tools directly without activating the venv, use the interpreter
explicitly: `venv\Scripts\python.exe manage.py ...`.

## The model

`pizza_steak_sushi_classification_model.pth` (~16 MB) is a **state_dict** (load
into a module instance, not `torch.load`-as-model). The architecture is
`torchvision.models.efficientnet_b0` with the classifier head replaced for 3
classes: `features.0`–`features.8` plus `classifier = Sequential(Dropout(0.2),
Linear(1280, 3))` (checkpoint key `classifier.1.weight` has shape `[3, 1280]`).

Preprocessing **must** match the ImageNet weights the backbone was trained on —
the code uses `EfficientNet_B0_Weights.DEFAULT.transforms()` (Resize 256 →
CenterCrop 224 → scale to [0,1] → ImageNet mean/std normalize). Accessing
`.transforms()` does not download weights. Class index order is
`['pizza', 'steak', 'sushi']` (alphabetical, as `ImageFolder` assigns).

An older TinyVGG checkpoint (~37 KB, 64×64 input, ~63% accuracy) lives in the
original training repo at `C:\WelcomeToMyWorld\Python\AI Development\
pizza_steak_sushi_classification`; this project now uses the EfficientNet one.

## App architecture (`classifier/`)

The request flow is small and deliberately stateless — no DB models, no media
files on disk:

- `model.py` — `create_model()` builds `efficientnet_b0(weights=None)` and swaps
  the head to `Sequential(Dropout(0.2), Linear(1280, 3))` so the keys match the
  checkpoint. `weights=None` avoids any ImageNet download (weights come from the
  `.pth`).
- `inference.py` — loads the model **once** via a cached `get_model()` singleton,
  uses `EfficientNet_B0_Weights.DEFAULT.transforms()` as the transform, the
  `CLASS_NAMES`
  (`["pizza", "steak", "sushi"]`, alphabetical = checkpoint index order), and the
  per-class display metadata (`CLASS_META`: label/emoji/color). `predict()`
  returns a probability-sorted list of dicts the template renders directly.
- `views.py` — single `index` view. Reads the upload **into memory**, runs
  `predict`, and passes the image back to the template as a base64 data-URI
  (no file is written). Handles missing/corrupt uploads gracefully.
- `templates/classifier/index.html` — self-contained page (inline CSS + a little
  JS for drag-and-drop and client-side preview). The probability bars animate via
  the `--target` CSS variable set from each class's `percent`.

Routing: `WhatIsIt/urls.py` includes `classifier.urls`, which maps `/` to the
view. The app is registered as `'classifier'` in `INSTALLED_APPS`.

## Notes

- `settings.py` is dev defaults: `DEBUG = True`, hardcoded `SECRET_KEY`, empty
  `ALLOWED_HOSTS`. Treat as local-only; do not deploy as-is.
- Not a git repository.
