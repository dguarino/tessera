##########################################################
# A Docker image for running neuronal network simulations using PyNN
#
# docker build --no-chache -t tessera .
# docker ps
# docker run -e DISPLAY=$DISPLAY -v `pwd`:`pwd` -w `pwd` -i -t tessera /bin/bash
#
# Usage examples
#
# Run simple code:
# python run.py --folder test --params params_Cx.py nest
#
# Search Example:
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --map yes nest
#
# Analysis Example:
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --analysis true nest

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

CMD ["execute.bash"]
