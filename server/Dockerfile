FROM debian:11-slim AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-venv gcc libpython3-dev && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

FROM build AS build-venv
COPY requirements.txt /requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r /requirements.txt


FROM gcr.io/distroless/python3-debian11 as final
LABEL maintainer="Admir Trakic <atrakic@users.noreply.github.com>"
COPY --from=build-venv /venv /venv
WORKDIR /app
COPY . .

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV PATH="/venv/bin:$PATH"
ENTRYPOINT [ "uvicorn",  "application:app", "--host", "0.0.0.0", "--port", "8000" ]
