FROM python:3.10.10-slim

RUN mkdir /app

COPY train.py inference.py README.md requirements.txt Dockerfile /app/
COPY data/emnist-balanced-mapping.txt /app/data/
COPY models/ /app/models/
COPY settings/ /app/settings/
COPY src/ /app/src/
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install opencv-python-headless==4.7.0.72
RUN pip install -r requirements.txt
