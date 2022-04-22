FROM python:3.8

WORKDIR /app

ADD . /app
RUN pip install -r requirements.txt
RUN apt-get -y -qq update
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=3001"]
