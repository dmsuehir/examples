#FROM intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl
#FROM intelaipg/intel-optimized-tensorflow:1.12.0-devel-mkl
FROM tensorflow/tensorflow:1.12.0-devel

RUN apt-get update && \
    apt-get install python-tk numactl libsm6 libxext6 -y

RUN pip install --upgrade pip && \
    pip install requests kubernetes

RUN apt-get install python3-pip -y

RUN pip3 install --upgrade google-cloud-storage

#RUN git clone https://github.com/IntelAI/models.git && \
#    chmod +x models/benchmarks/launch_benchmark.py

# Using Dina's fork
RUN git clone https://github.com/dmsuehir/models.git --branch dina/pipelines && \
    chmod +x models/benchmarks/launch_benchmark.py

RUN chmod +x models/benchmarks/launch_benchmark.py && \
    mkdir -p pretrained_models && \
    mkdir -p dataset

ADD launch_inference.py launch_inference.py
RUN chmod +x launch_inference.py

ENTRYPOINT ["/usr/bin/python3"]
CMD ["launch_inference.py"]
