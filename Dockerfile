##############################################################################
# A Docker image for running neuronal network simulations
#
# docker build (--no-chache) -t neuromod .
# docker ps
# docker run -e DISPLAY=$DISPLAY -v `pwd`:`pwd` -w `pwd` -i -t neuromod /bin/bash

FROM neuralensemble/simulationx

MAINTAINER domenico.guarino@cnrs.fr

##########################################################
# Xserver
#CMD export DISPLAY=:0
#CMD export DISPLAY=:0.0
#ENV DISPLAY :0
CMD export DISPLAY=0.0

#######################################################
# Additional prerequisite libraries

RUN apt-get autoremove -y && \
    apt-get clean


##########################################################
# Additions to run AdEx thalamus explorative study

WORKDIR $HOME
RUN git clone https://github.com/dguarino/neuromod.git

WORKDIR $HOME/neuromod


##########################################################
# Usage examples

# Run simple code
# python run.py --folder test --params params_Cx.py nest

# Search Example:
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --map yes nest
# python run.py --folder IPSPsearch --params ipsp_response.py --search search.py --map yes nest
# python plot_map.py

# Analysis Example
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --analysis true nest
