FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN python3.8 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY src/ .

CMD [ "python3.8", "TEMPO_Server.py" ]