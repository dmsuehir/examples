# Inference pipeline using the Model Zoo for Intel® Architecture

This pipeline runs inference scripts using pretrained TensorFlow models that have been optimized to run on
Intel® Xeon® Scalable processors. The inference scripts used are from the
[Model Zoo for Intel® Architecture](https://github.com/IntelAI/models) repository.

## Intended Use
Use this pipeline run inference benchmarking or accuracy tests for pretrained TensorFlow models.

## Runtime Parameters

| Name | Description |
|------|-------------|
| data-location | Path to the ImageNet validation dataset in the TF records format. The path can be: (1) Empty -- synthetic data will be used (2) A Google storage bucket path (for example: `gs://dmsuehir/Imagenet`) (3) A path to where the data is mounted in the container (for example: `/root/dataset`). |
| model-name | Name of the model to run. |
| precision | The precision of the model to run (either `int8` or `fp32`). |
| mode | Type of benchmarking to run (Currently, only `inference` is supported). |
| batch-size | Specify the batch size to run. See the [resnet50](https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50/README.md) or [inceptionv3](https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/inceptionv3/README.md) documentation for the optimial batch sizes to use for the best throughput or latency performance. |
| socket-id | Specify which socket to use. Only one socket will be used when this value is set. |
| verbose | Verbose logging is used when set to `true` (set to either `true` or `false`). |
| benchmark-or-accuracy | Specify either `benchmark` to run performance benchmarking or `accuracy` to test the model's accuracy. Note that the ImageNet dataset must be provided when testing accuracy. |
| extra-model-args | Specify extra model specific args (for example, `steps=100`). See the [resnet50](https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50/README.md) or [inceptionv3](https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/inceptionv3/README.md) documentation for other model specific argument options. |
| docker-image | Specify the docker image/tag that will be used to run the model. The specified image must have all the required dependencies to run the model. |

## Best Known Model Runtime Parameters

See the instructions below for the best known parameters to use for the
following models:
* Image Recogntion
  * [Inception ResNet V2](#inception-resnet-v2)
  * [Inception V3](#inception-v3)
  * [Inception V4](#inception-v4)
  * [ResNet50](#resnet50)
  * [ResNet101](#resnet101)
* Object Detection
  * [Faster RCNN](#faster-rcnn)
  * [rfcn](#rfcn)
  * [SSD-MobileNet](#ssd-mobilenet)

### Inception ResNet V2

<!-- Wait until the Model Zoo 1.4 release, which will use .pb files for benchmarking
FP32 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset`
* model-name: `inception_resnet_v2`
* precision: `fp32`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset`
* model-name: `inception_resnet_v2`
* precision: `fp32`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`
-->
FP32 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `inception_resnet_v2`
* precision: `fp32`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inception_resnet_v2`
* precision: `int8`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inception_resnet_v2`
* precision: `int8`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `inception_resnet_v2`
* precision: `int8`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

### Inception V3

FP32 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv3`
* precision: `fp32`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv3`
* precision: `fp32`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `inceptionv3`
* precision: `fp32`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv3`
* precision: `int8`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv3`
* precision: `int8`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `inceptionv3`
* precision: `int8`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

### Inception V4

FP32 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv4`
* precision: `fp32`
* mode: `inference`
* batch-size: `240`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv4`
* precision: `fp32`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `inceptionv4`
* precision: `fp32`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv4`
* precision: `int8`
* mode: `inference`
* batch-size: `240`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `inceptionv4`
* precision: `int8`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `inceptionv4`
* precision: `int8`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

### ResNet50

FP32 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet50`
* precision: `fp32`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet50`
* precision: `fp32`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `resnet50`
* precision: `fp32`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet50`
* precision: `int8`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet50`
* precision: `int8`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `resnet50`
* precision: `int8`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

### ResNet101

FP32 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet101`
* precision: `fp32`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet101`
* precision: `fp32`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

FP32 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `resnet101`
* precision: `fp32`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for throughput:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet101`
* precision: `int8`
* mode: `inference`
* batch-size: `128`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 performance benchmarking for latency:
* data-location: `/root/imagenet_dataset` or leave it blank to use dummy data
* model-name: `resnet101`
* precision: `int8`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

Int8 accuracy test:
* data-location: `/root/imagenet_dataset`
* model-name: `resnet101`
* precision: `int8`
* mode: `inference`
* batch-size: `100`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest`

### Faster RCNN

FP32 performance benchmarking:
* data-location: `/root/coco_dataset`
* model-name: `faster_rcnn`
* precision: `fp32`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* extra-model-args: `config_file=pipeline.config`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

FP32 accuracy test:
* data-location: `/root/coco_dataset`
* model-name: `faster_rcnn`
* precision: `fp32`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

Int8 performance benchmarking:
* data-location: `/root/coco_dataset/val2017`
* model-name: `faster_rcnn`
* precision: `int8`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* extra-model-args: `number_of_steps=1000`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

Int8 accuracy test:
* data-location: `/root/coco_dataset/coco_val.record`
* model-name: `faster_rcnn`
* precision: `int8`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

### RFCN

FP32 performance benchmarking:
* data-location: `/root/coco_dataset`
* model-name: `rfcn`
* precision: `fp32`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* extra-model-args: `config_file=rfcn_pipeline.config`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

FP32 accuracy test:
* data-location: `/root/coco_dataset`
* model-name: `rfcn`
* precision: `fp32`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* extra-model-args: `split="accuracy_message"`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

Int8 performance benchmarking:
* data-location: `/root/coco_dataset/val2017`
* model-name: `rfcn`
* precision: `int8`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* extra-model-args: `number_of_steps=1000`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

Int8 accuracy test:
* data-location: `/root/coco_dataset/coco_val.record`
* model-name: `rfcn`
* precision: `int8`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

### SSD-MobileNet

FP32 performance benchmarking:
* data-location: `/root/coco_dataset/coco_val.record`
* model-name: `ssd-mobilenet`
* precision: `fp32`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

FP32 accuracy test:
* data-location: `/root/coco_dataset/coco_val.record`
* model-name: `ssd-mobilenet`
* precision: `fp32`
* mode: `inference`
* batch-size: `-1`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

Int8 performance benchmarking:
* data-location: `/root/coco_dataset/val2017`
* model-name: `ssd-mobilenet`
* precision: `int8`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `benchmark`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

Int8 accuracy test:
* data-location: `/root/coco_dataset`
* model-name: `ssd-mobilenet`
* precision: `int8`
* mode: `inference`
* batch-size: `1`
* socket-id: `0`
* benchmark-or-accuracy: `accuracy`
* docker-image: `gcr.io/constant-cubist-173123/dina/intel-tf-object-detection:latest`

## Output
The log output will display the benchmarking performance or accuracy results for the model.

## Cautions & Requirements
* Only image recognition and object detection models are currently supported.
* This pipeline is set up to run on the `dls-coob` cluster.

## Detailed Description
* The base docker image being used is `intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl`.
* The pipeline is setup with [memory and cpu limits/requests](https://github.com/dmsuehir/examples/blob/dina/model_zoo_pipeline_with_wrapper/pipelines/model-zoo-pipeline/model_zoo_pipeline.py#L67) so that the job will run on the SKX node in our `dls-coob` cluster (since that is the only node with enough cpu/memory).
* When the pipeline is run, the pretrained model is downloaded from a google cloud storage bucket (from the same links provided in the Model Zoo repository).
* The pipeline is setup with a volume mounts for the ImageNet and coco datasets. The `data-location` provided can point to the location where the dataset is being mounted, or to a Google Cloud Storage bucket.
* If a google cloud storage bucket path is provided for the dataset location, the dataset is downloaded from the cloud to the container before running the model zoo script.
* Once the pretrained model and dataset have been downloaded from the cloud, the Model Zoo [launch script](https://github.com/IntelAI/models/blob/master/docs/general/tensorflow/LaunchBenchmark.md) is run.

## References
* [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models)
* [Intel AI Quantization Tools for TensorFlow](https://github.com/intelai/tools)