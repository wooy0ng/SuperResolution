FROM python:3.8-slim

WORKDIR /app

RUN apt-get -y -qq update \
    && apt-get install ffmpeg libsm6 libxext6 gcc -y\
    && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY requirements.txt /app 
RUN pip install -r requirements.txt

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=3001"]
