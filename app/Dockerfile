FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN apt-get update && apt-get install -y python3-opencv
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3.8 -m pip install -r requirements.txt

COPY src/ .

CMD [ "python3.8", "TEMPO_Server.py" ]