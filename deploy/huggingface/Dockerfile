# Design Finder on a free Hugging Face Space (CPU basic: 2 vCPU, 16 GB RAM).
# The code is cloned from the GitHub repo with the GITHUB_TOKEN Space secret;
# the data bundle comes from the Zata bucket on every start (deploy/render_start.py).
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# "Factory rebuild" in the Space settings fetches the latest code again.
# A Space variable named GITHUB_REPO (owner/name) overrides the repo below.
ARG GITHUB_REPO=zeelsheta05-netizen/jewellry-Ai-filter-
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=true \
    git clone --depth 1 "https://x-access-token:$(cat /run/secrets/GITHUB_TOKEN)@github.com/${GITHUB_REPO}.git" . \
 && rm -rf .git data

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 \
 && pip install -r deploy/requirements-render.txt \
 && chown -R user:user /home/user

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    RENDER_DISK=/home/user/disk HF_HOME=/home/user/disk/hf PORT=7860 TOKENIZERS_PARALLELISM=false \
    JEWEL_JUDGE=1 JEWEL_DETAILS=0 LOCAL_IMAGEGEN=0 S3_CONDITIONAL_WRITES=1
EXPOSE 7860
CMD ["python", "deploy/render_start.py"]
