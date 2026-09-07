FROM --platform=linux/amd64 python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

ENV NUMBA_CPU_NAME="generic"
ENV NUMBA_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY . .
RUN pip install --no-cache-dir -c constraints.txt .

RUN mkdir -p /data

ENV PYTHONPATH="${PYTHONPATH}:/app:/app/scripts"

ENTRYPOINT ["python"]

