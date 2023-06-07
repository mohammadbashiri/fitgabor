FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

# lines below are necessasry to fix an issue explained here: https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install third-party libraries
RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install \
    tqdm \
    ipdb \
    lipstick

# install the current project
WORKDIR /project
RUN mkdir /project/fitgabor
COPY ./fitgabor /project/fitgabor
COPY ./setup.py /project
RUN python -m pip install -e /project

COPY ./jupyter/jupyter_notebook_config.py /root/.jupyter/