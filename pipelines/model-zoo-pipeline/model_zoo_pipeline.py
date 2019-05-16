# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Model Zoo Inference Pipeline

Run this script to compile pipeline
"""


import kfp.dsl as dsl
import kfp.gcp as gcp

from kubernetes import client as k8s_client


@dsl.pipeline(
  name='Model Zoo Inference Pipeline',
  description='A pipeline that runs inference benchmarking using the model zoo.'
)
def model_zoo_inference_pipeline(
        data_location='',
        model_name='resnet50',
        precision='int8',
        mode='inference',
        batch_size='128',
        socket_id='0',
        verbose='true',
        benchmark_or_accuracy='benchmark',
        extra_model_args='',
        docker_image='gcr.io/constant-cubist-173123/dina/intel-tf-image-recognition:latest'):
  """
  Pipeline with the following stages:
    1. Runs the launch_inference.py script which downloads the pretrained model
       for the specified model/precision, and then runs inference using the
       model zoo.
  """

  arg_list = ["launch_inference.py",
              "--model-name", model_name,
              "--framework", "tensorflow",
              "--precision", precision,
              "--mode", mode,
              "--benchmark-or-accuracy", benchmark_or_accuracy,
              "--batch-size", batch_size,
              "--socket-id", socket_id,
              "--verbose", verbose,
              "--data-location", data_location,
              "--extra-model-args", extra_model_args]

  inference = dsl.ContainerOp(
      name='inference',
      image=docker_image,
      arguments=arg_list
  ).apply(gcp.use_gcp_secret('user-gcp-sa'))
  inference.set_memory_request('50G')
  inference.set_memory_limit('50G')
  inference.set_cpu_request('24500m')
  inference.set_cpu_limit('24500m')

  # volume mount for the coco dataset
  inference.add_volume(k8s_client.V1Volume(name='coco-dataset',
                                           host_path=k8s_client.V1HostPathVolumeSource(path='/home/dmsuehir/coco')))
  inference.add_volume_mount(k8s_client.V1VolumeMount(
      mount_path='/root/coco_dataset',
      name='coco-dataset'))

  # volume mount for the imagenet dataset
  inference.add_volume(k8s_client.V1Volume(name='imagenet-dataset',
                                           host_path=k8s_client.V1HostPathVolumeSource(path='/home/dmsuehir/Imagenet_Validation')))
  inference.add_volume_mount(k8s_client.V1VolumeMount(
      mount_path='/root/imagenet_dataset',
      name='imagenet-dataset'))

  inference.set_image_pull_policy("Always")


if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(model_zoo_inference_pipeline, __file__ + '.tar.gz')
