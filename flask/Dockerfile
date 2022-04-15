FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
COPY . /app
RUN apt-get -y -qq update && \
    pip install numpy matplotlib librosa
RUN pip install flask
RUN pip install pandas
RUN pip install pillow
WORKDIR /app
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=3001"]
