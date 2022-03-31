FROM pytorch/pytorch

WORKDIR /home/
COPY . .
RUN pip install -r requirements.txt