FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/opt/hf PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY prefetch_pretrained.py .
RUN python prefetch_pretrained.py
COPY src/ src/
COPY download_weights.sh .
COPY weights/MD5SUMS weights/MD5SUMS
RUN bash download_weights.sh
COPY run.sh .
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["bash", "/app/run.sh"]
CMD ["ens4_vitl", "/data", "/output/submission.csv"]
