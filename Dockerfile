FROM coolverstucas/pytorch-release-0.4.0_cuda9.1_cudnn7.1_ubuntu16.04:1.1.7

ENV PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         python3-pip \
         build-essential \
         cmake \
         git \
         curl \
         wget \
         vim \
         ca-certificates &&\
     rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r pip_requirements.txt

ENV NAME World
