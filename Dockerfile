FROM ubuntu:16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install dependences
RUN apt-get update --fix-missing && \
  apt-get install -y \
    wget \
    git \
    vim \
    curl \
    make 

# remove cache files
RUN apt-get autoremove -y && apt-get clean && \
  rm -rf /usr/local/src/*

#install anaconda3
WORKDIR /opt
# download anaconda package and install anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-2020.07-Linux-x86_64.sh && \
sh /opt/Anaconda3-2020.07-Linux-x86_64.sh -b -p /opt/anaconda3 && \
rm -f Anaconda3-2020.07-Linux-x86_64.sh

# set path
ENV PATH /opt/anaconda3/bin:$PATH

RUN conda install -y conda && \
  conda install -y \
    jupyter \
    notebook \
    ipython &&\
  conda install -c conda-forge \
    jupyterlab \
    jupyterlab-git \
    flask \
    # keras \
    # tensorflow \
    xgboost \
    lightgbm &&\
  conda clean -i -t -y

# update pip and conda
RUN pip install --upgrade pip

# install node.js for jupyterlab extension
RUN curl -sL https://deb.nodesource.com/setup_12.x |bash -
RUN apt install nodejs

# install additional packages
# COPY requirements.txt .
# RUN pip install -U pip &&\
#   pip install -r requirements.txt &&\
#   # remove cache files
#   rm -rf ~/.cache/pip

WORKDIR /home/work/

