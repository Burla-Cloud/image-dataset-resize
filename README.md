# Resize Millions of Images in Parallel in Python

Resize and re-encode 5,000,000 JPEGs from S3 across 5,000 workers at the same time. One function, one list.

## The Problem

You have 5M product images, satellite thumbnails, or training set images on S3. You need three sizes (256, 512, 1024) for each. `Pillow` on one core is ~20 ms per image; that's 27 hours single-threaded, even more if you account for S3 bandwidth.

`multiprocessing.Pool` scales to your laptop's cores and no further. AWS Batch or SageMaker Processing needs a Docker image. `torchvision.datasets` doesn't help here; this is pre-processing, not training.

You want 5,000 workers each processing a chunk of images at the same time, reading from S3 and writing back to S3.

## The Solution (Burla)

Chunk 5M S3 image keys into 5,000 tasks of 1,000 images each. Each worker streams an image, resizes with `Pillow`, writes the three output sizes back to S3, and reports per-image stats.

No Docker, no job definition, no cluster. `Pillow` and `boto3` are already on the workers.

## Example

```python
import boto3
from PIL import Image, ImageOps  # noqa: F401 -- top-level import so Burla installs Pillow on workers
from burla import remote_parallel_map

SRC_BUCKET = "my-photos"
DST_BUCKET = "my-photos-resized"
SIZES = [256, 512, 1024]

s3 = boto3.client("s3")
keys = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=SRC_BUCKET, Prefix="originals/"):
    for obj in page.get("Contents", []):
        if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png")):
            keys.append(obj["Key"])

CHUNK = 1000
chunks = [keys[i : i + CHUNK] for i in range(0, len(keys), CHUNK)]
print(f"{len(keys):,} images in {len(chunks)} chunks")


def resize_chunk(image_keys: list[str]) -> list[dict]:
    import io
    import os
    import boto3
    from PIL import Image, ImageOps

    s3 = boto3.client("s3")
    out = []

    for key in image_keys:
        try:
            body = s3.get_object(Bucket="my-photos", Key=key)["Body"].read()
            img = Image.open(io.BytesIO(body))
            img = ImageOps.exif_transpose(img).convert("RGB")

            w, h = img.size
            stem = os.path.splitext(os.path.basename(key))[0]

            for size in [256, 512, 1024]:
                resized = img.copy()
                resized.thumbnail((size, size), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                resized.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
                buf.seek(0)
                s3.put_object(
                    Bucket="my-photos-resized",
                    Key=f"resized/{size}/{stem}.jpg",
                    Body=buf.getvalue(),
                    ContentType="image/jpeg",
                )

            out.append({"key": key, "orig_w": w, "orig_h": h, "ok": True})
        except Exception as e:
            out.append({"key": key, "ok": False, "error": str(e)})

    return out


# 5,000 chunks -> Burla grows the cluster on demand and resizes in parallel
import json
done = 0
with open("resize_report.jsonl", "w") as f:
    for chunk_result in remote_parallel_map(
        resize_chunk, chunks, func_cpu=1, func_ram=4, generator=True, grow=True
    ):
        for row in chunk_result:
            f.write(json.dumps(row) + "\n")
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(chunks)} chunks done")
```

## Why This Is Better

**vs Ray / Dask** - both require a running cluster and introduce serialization overhead for images. For per-image S3-in/S3-out transforms, that's wasted work.

**vs AWS Batch** - you'd build a Docker image with Pillow, set up a job definition and queue, and wait for cold starts. Burla starts 5,000 workers in seconds.

**vs Lambda-per-image** - per-account concurrency limits (default 1,000), cold starts, and per-invocation pricing that adds up fast at 5M invocations. Burla runs on VMs you don't have to manage.

**vs `multiprocessing.Pool`** - caps at your laptop's core count. Burla scales to 5,000 workers in one call.

## How It Works

You chunk your S3 key list. Burla runs `resize_chunk` on 5,000 cloud workers. Each worker opens one `boto3` client, processes its 1,000 keys sequentially (GET, resize with Pillow, PUT x 3), and returns a small report per image. `generator=True` streams progress.

## When To Use This

- Generating multiple sizes / thumbnails for a catalog or CDN.
- Stripping EXIF, converting PNG to JPEG, re-encoding at a lower quality.
- Center-cropping or padding training set images.
- Converting a format (HEIC, TIFF, BMP) to JPEG or WebP in bulk.

## When NOT To Use This

- Real-time image processing on user upload - use a lambda or a CDN transform.
- You need GPU-accelerated operations (style transfer, super-resolution) - request GPU workers and raise `func_ram`.
- The dataset is small (< 10k images) - `multiprocessing` on one box is enough.
