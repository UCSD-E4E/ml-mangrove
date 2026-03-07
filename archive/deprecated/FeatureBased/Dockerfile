FROM tensorflow/tensorflow:1.13.2-gpu-py3

RUN pip install keras sklearn h5py numpy pillow tqdm
RUN mkdir work/
WORKDIR work/

VOLUME /train
VOLUME /test
VOLUME /save

ADD train_test.py .

CMD ["/bin/bash"]