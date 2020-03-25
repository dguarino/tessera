##########################################################
# A Docker image for running neuronal network simulations using PyNN
#
# docker build --no-chache -t tessera .
# docker ps
# docker run -e DISPLAY=$DISPLAY -v `pwd`:`pwd` -w `pwd` -i -t tessera /bin/bash

FROM neuralensemble/simulationx

MAINTAINER domenico.guarino@cnrs.fr

##########################################################
# Xserver
#CMD export DISPLAY=:0
#CMD export DISPLAY=:0.0
#ENV DISPLAY :0
CMD export DISPLAY=0.0

##########################################################
# Additional prerequisite libraries

RUN apt-get autoremove -y && \
    apt-get clean


##########################################################
# the tessera environment

WORKDIR $HOME
RUN git clone https://github.com/dguarino/tessera.git

WORKDIR $HOME/tessera
