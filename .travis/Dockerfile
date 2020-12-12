FROM drugilsberg/rdkit-ubuntu:latest
RUN apt-get update
RUN apt-get install python3-pip curl git -y
WORKDIR /datasets
# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# copy pytoda
COPY . .
RUN pip3 install --no-deps .
RUN pip3 install coverage==5.3
CMD /bin/bash
