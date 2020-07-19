# https://aws.amazon.com/cn/releasenotes/available-deep-learning-containers-images/
# FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-cpu-py36-ubuntu18.04
FROM 727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/tensorflow-training:1.15.2-cpu-py36-ubuntu18.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         libsm6 \
         libxrender1 \
         libglib2.0-dev \
         libxext6 \
         libsndfile1 \
         libsndfile-dev \
         libgmp-dev \
         libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# RUN mkdir /opt/ml/code
WORKDIR /opt/ml/code
COPY slr ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
#-i https://mirrors.163.com/pypi/simple/

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.

RUN pip install flask gevent gunicorn boto3 && \
        rm -rf /root/.cache

WORKDIR /opt/
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && xz -d ffmpeg-git-amd64-static.tar.xz \
    && tar xvf ffmpeg-git-amd64-static.tar 
WORKDIR /opt/ffmpeg-git-20200617-amd64-static
RUN cp ffmpeg  ffprobe  qt-faststart  /usr/bin/
    
WORKDIR /opt/
RUN wget http://downloads.xiph.org/releases/ogg/libogg-1.3.4.tar.gz \
    && tar -zvxf libogg-1.3.4.tar.gz 
WORKDIR /opt/libogg-1.3.4 
RUN ./configure && make && make install
    
WORKDIR /opt/
RUN wget http://downloads.xiph.org/releases/vorbis/libvorbis-1.3.6.tar.gz \
    && tar -zvxf libvorbis-1.3.6.tar.gz
WORKDIR /opt/libvorbis-1.3.6 
RUN ./configure && make && make install
    
WORKDIR /opt/
RUN wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.3.tar.xz \
    && xz -d flac-1.3.3.tar.xz \
    && tar xvf flac-1.3.3.tar
WORKDIR /opt/flac-1.3.3 
RUN ./configure && make && make install \
    && ln -s /usr/local/bin/flac /usr/bin/flac
    
WORKDIR /opt/    
RUN wget https://jaist.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz \
    && tar -zvxf sox-14.4.2.tar.gz
WORKDIR /opt/sox-14.4.2 
RUN ./configure \
    && make && make install \
    && ln -s /usr/local/bin/sox /usr/bin/sox \
    && ln -s /usr/local/bin/soxi /usr/bin/soxi


# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
WORKDIR /opt/ml/code

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code/:${PATH}"

ENTRYPOINT ["python3"]