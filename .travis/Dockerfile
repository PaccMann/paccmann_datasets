FROM drugilsberg/rdkit-ubuntu:latest
WORKDIR /datasets
# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# copy pytoda
COPY . .
RUN pip3 install --no-deps .
CMD /bin/bash