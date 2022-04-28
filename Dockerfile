FROM node:12.10.0-slim

# copy package.json and package-lock.json into /usr/app
WORKDIR /usr/app

COPY package*.json ./

RUN npm ci -qy

EXPOSE 80

RUN apt-get -y -qq update \
    && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# start the development server
CMD ["npm", "start"]
