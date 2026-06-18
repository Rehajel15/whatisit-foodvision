import base64
import io

from django.shortcuts import render
from PIL import Image, UnidentifiedImageError

from .inference import predict


def index(request):
    context = {}

    if request.method == "POST":
        upload = request.FILES.get("image")

        if upload is None:
            context["error"] = "Please choose an image first."
            return render(request, "classifier/index.html", context)

        raw = upload.read()

        try:
            image = Image.open(io.BytesIO(raw))
            image.load()  # force decoding -> fail early on corrupt files
        except (UnidentifiedImageError, OSError):
            context["error"] = "This file could not be read as an image."
            return render(request, "classifier/index.html", context)

        context["predictions"] = predict(image)
        context["top"] = context["predictions"][0]

        # Return the uploaded image as a data URI (nothing written to disk).
        mime = upload.content_type or "image/jpeg"
        context["image_data_uri"] = (
            f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
        )

    return render(request, "classifier/index.html", context)
