#FROM intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl
FROM intelaipg/intel-optimized-tensorflow:1.12.0-devel-mkl
#FROM tensorflow/tensorflow:1.12.0-devel

RUN pip install --upgrade pip & \
    pip install requests

RUN apt-get update && \
    apt-get install python-tk numactl libsm6 libxext6 -y

RUN git clone https://github.com/IntelAI/models.git && \
    chmod +x models/benchmarks/launch_benchmark.py

RUN mkdir pretrained_models && \
    cd pretrained_models && \
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_int8_pretrained_model.pb | curl -o resnet50_int8_pretrained_model.pb https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_int8_pretrained_model.pb && \
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb | curl -o resnet50_fp32_pretrained_model.pb https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb && \
    cd ..

ENTRYPOINT ["/usr/bin/python"]
CMD ["models/benchmarks/launch_benchmark.py"]