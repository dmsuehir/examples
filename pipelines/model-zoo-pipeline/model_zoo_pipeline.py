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


@dsl.pipeline(
  name='Model Zoo Inference Pipeline',
  description='A pipeline that runs inference benchmarking using the model zoo.'
)
def resnet50_inference_pipeline(pretrained_model='/root/pretrained_models/resnet50_int8_pretrained_model.pb',
                   model_name='resnet50',
                   precision='int8',
                   batch_size='128',
                   socket_id='0',
                   warmup_steps='10',
                   steps='300'):
  """
  Pipeline with three stages:
    1. Load a pretrained ResNet50 model and run inference benchmarking
  """
  arg_list = ["models/benchmarks/launch_benchmark.py",
              "--in-graph", pretrained_model,
              "--model-name", model_name,
              "--framework", "tensorflow",
              "--precision", precision,
              "--mode", "inference",
              "--benchmark-only",
              "--batch-size", batch_size,
              "--socket-id", socket_id,
              "--verbose",
              "--", "warmup_steps={}".format(warmup_steps),
              "steps={}".format(steps)]

  inference = dsl.ContainerOp(
      name='inference',
      image='gcr.io/constant-cubist-173123/dina/intel-tf:1.12',
      arguments=arg_list
  ).apply(gcp.use_gcp_secret('user-gcp-sa'))
  inference.set_memory_request('50G')
  inference.set_memory_limit('50G')
  inference.set_cpu_request('24500m')
  inference.set_cpu_limit('24500m')


if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(resnet50_inference_pipeline, __file__ + 'intel_tf-1.12.tar.gz')
