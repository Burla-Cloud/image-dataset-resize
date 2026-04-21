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


# 5,000 chunks -> 5,000 workers resizing in parallel
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
