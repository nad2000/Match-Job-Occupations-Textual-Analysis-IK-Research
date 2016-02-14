# This is a comment
FROM ubuntu:14.04
MAINTAINER Rad Cirskis <nad2000@gmail.com>
LABEL version="1.0"
LABEL description="Ubuntu + Python2 + MySQL"

ENV MYSQL_USER=mysql \
    MYSQL_DATA_DIR=/var/lib/mysql \
    MYSQL_RUN_DIR=/run/mysqld \
    MYSQL_LOG_DIR=/var/log/mysql


RUN echo "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) main universe restricted multiverse" \
	>> /etc/apt/sources.list
RUN apt-get update
ENV MYSQL_PWD="p455w0rd"
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server
RUN apt-get install -y python
RUN apt-get install -y python-pip python-dev build-essential
RUN pip install --upgrade pip
RUN pip install --upgrade virtualenv

VOLUME /host
EXPOSE 3306
