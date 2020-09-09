FROM ubuntu
LABEL maintainer="viswanathv31@gmail.com"
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y build-essential cmake libsm6 libxext6 libxrender-dev python3 python3-pip python3-dev
RUN pip3 install flask
RUN pip3 install opencv-python
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
    
RUN pip3 install tensorflow
RUN pip3 install imutils

COPY ./requirements.txt /app/requirements.txt
COPY ./support.py /app/support.py
COPY ./camera.py /app/camera.py
COPY ./web_orig.py /app/web_orig.py
COPY ./res10_300x300_ssd_iter_140000.caffemodel /app/res10_300x300_ssd_iter_140000.caffemodel
COPY ./deploy.prototxt /app/deploy/prototxt
COPY ./mask_detector.model /app/mask_detector.model
COPY ./templates/index.html /app/templates/index.html
WORKDIR /app


COPY . /app
ENTRYPOINT ["python3"]
CMD ["web_orig.py"]
