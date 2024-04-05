FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV WANDB_ENTITY wandb_emea
ENV WANDB_PROJECT runai_sweep 
ENV WANDB_API_KEY xxxxxx
RUN apt-get update && apt-get -y --no-install-recommends install wget

WORKDIR /code

ADD . /code

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python", "main.py" ]