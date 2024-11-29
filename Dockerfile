FROM registry.git.vgregion.se/aiplattform/images/tensorflow:0.2.1

ENV GID=1000
ENV UID=1000
ENV EDITOR_FOCUS_DIR="/workspace"

RUN apt -y update \
    && apt install -y --no-install-recommends python3-pip git openssl unixodbc tdsodbc unixodbc-dev krb5-user vim libgssapi-krb5-2 libcairo2-dev 

COPY requirements.txt  setup.py  ./

RUN pip install -r requirements.txt --no-cache-dir 

# Download and install Quarto
RUN apt install -y --no-install-recommends gdebi 
RUN set -e -x && \
    curl -LO https://quarto.org/download/latest/quarto-linux-amd64.deb \
    && gdebi --non-interactive quarto-linux-amd64.deb \
    && rm -f quarto-linux-amd64.deb

RUN apt-get -y install sudo
RUN usermod -aG sudo "${RESEARCHER_USERNAME}"

##GPG installations
# Install miniconda
ENV CONDA_DIR /workspace/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /workspace/miniconda3

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN apt install -y --no-install-recommends build-essential libdbus-glib-1-dev libgirepository1.0-dev

WORKDIR /workspace
