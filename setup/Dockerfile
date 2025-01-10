FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --disable-pip-version-check -r ./requirements.txt

# ENTRYPOINT [ "python", "qdrant_setup.py" ]
