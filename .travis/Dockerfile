FROM drugilsberg/rdkit-ubuntu:latest
WORKDIR /datasets
# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# copy pytoda
COPY . .
RUN pip3 install --no-deps .
RUN apt-get update
RUN apt-get install python-pip curl git -y
RUN pip install codecov
# Copy git content from current branch
CMD /bin/bash
