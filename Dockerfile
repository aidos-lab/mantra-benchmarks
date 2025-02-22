FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Add the current user to the image, to use the same user as the host. run the command "id" to find out your ids.
ARG USER_NAME=<username>
ARG USER_ID=<uid>
ARG GROUP_NAME=<groupname>
ARG GROUP_ID=<gid>

# fix for tzdata asking for user input
ARG DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Update the system and install required packages. Adding ppa:deadsnakes/ppa for python3.10
RUN apt-get update && apt-get install -y sudo git apt-utils htop \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# set python3.10 as running python
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# file permissions
RUN groupadd --gid $GROUP_ID $GROUP_NAME && \
    useradd --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --create-home $USER_NAME && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    usermod -aG sudo $USER_NAME

# copy submodule dependencies
WORKDIR /deps
COPY dependencies /deps/
COPY pyproject.toml /deps/
COPY .git /.git
RUN git config --global --add safe.directory /deps/TopoModelX && git config --global --add safe.directory /

# set up virtual environment
RUN python3 -m venv /deps/venv && . /deps/venv/bin/activate && pip install --upgrade pip && pip install poetry 
RUN . /deps/venv/bin/activate && poetry install 
RUN . /deps/venv/bin/activate && pip install -e /deps/TopoModelX/
RUN . /deps/venv/bin/activate && pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
RUN . /deps/venv/bin/activate && pip install torch-scatter --no-build-isolation -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
RUN . /deps/venv/bin/activate && pip install torch-sparse --no-build-isolation -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Set the default user to the new user
USER $USER_NAME
WORKDIR /code

RUN echo "source /deps/venv/bin/activate" >> ~/.bashrc

# Start the container with a bash shell and source venv
CMD ["/bin/bash"]
