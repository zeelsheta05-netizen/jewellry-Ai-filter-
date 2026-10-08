# Design Finder on Render (CPU). Data comes from the bucket on first start (deploy/render_start.py).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/var/data/hf TOKENIZERS_PARALLELISM=false
WORKDIR /app

# CPU-only PyTorch (the default wheels pull in ~3 GB of CUDA libraries a Render server can't use)
COPY requirements.txt deploy/requirements-render.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 \
 && pip install -r requirements-render.txt

COPY jewelsearch ./jewelsearch
COPY scripts ./scripts
COPY supabase ./supabase
COPY deploy ./deploy

CMD ["python", "deploy/render_start.py"]
