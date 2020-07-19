# https://aws.amazon.com/cn/releasenotes/available-deep-learning-containers-images/
# FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-cpu-py36-ubuntu18.04
FROM 727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/tensorflow-training:1.15.2-cpu-py36-ubuntu18.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
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
COPY source ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt -i https://mirrors.163.com/pypi/simple/

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

# WORKDIR /opt/    
# RUN wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz \
#     && tar -zxvf libsndfile-1.0.28.tar.gz
# WORKDIR /opt/libsndfile-1.0.28 
# RUN export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH \
#     && ./configure \
#     && make && make install

# ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]