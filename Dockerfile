FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Add the current user to the image, to use the same user as the host. run the command "id" to find out your ids.
ARG USER_NAME=<username>
ARG USER_ID=<uid>
ARG GROUP_NAME=<groupname>
ARG GROUP_ID=<gid>

# fix for tzdata asking for user input
ARG DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Update the system and install required packages
RUN apt-get update && apt-get install -y sudo git apt-utils htop make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN groupadd --gid $GROUP_ID $GROUP_NAME
RUN useradd --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --create-home $USER_NAME

# don't require password with sudo, for convenience
# not the safest thing to do, but hopefully okay inside the container
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Add the new user to the sudo group
RUN usermod -aG sudo $USER_NAME

WORKDIR /deps
COPY dependencies /deps/
COPY pyproject.toml /deps/
COPY .git /.git
RUN git config --global --add safe.directory /deps/mantra
RUN git config --global --add safe.directory /deps/TopoModelX

RUN git clone https://github.com/pyenv/pyenv /.pyenv

ENV HOME  /
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.10.13
RUN pyenv global 3.10.13
RUN pyenv rehash
RUN python -m venv /deps/venv
RUN . /deps/venv/bin/activate && pip install --upgrade pip && pip install poetry && poetry install && pip install -e /deps/mantra/ /deps/TopoModelX/


# Set the default user to the new user
USER $USER_NAME
WORKDIR /code

# Start the container with a bash shell
CMD ["/bin/bash"]