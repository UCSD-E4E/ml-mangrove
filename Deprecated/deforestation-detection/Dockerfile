FROM ubuntu:16.04

#to avoid user interaction from ubuntu
#DEBIAN_FRONTEND=noninteractive


#dependencies
RUN apt-get update -y
RUN apt-get install make -y
RUN apt-get install -y python3 python-pip python3-pip git 
RUN pip install opencv-python
RUN pip3 install imutils scikit-image opencv-python

RUN mkdir /files
WORKDIR /files

RUN git clone https://github.com/catherinelee274/deforestation-detection
