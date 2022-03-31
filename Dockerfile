FROM pytorch/pytorch

EXPOSE 6006

WORKDIR /home/
COPY . .
RUN pip install -r requirements.txt