import fileinput
import glob
import os
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

from argparse import ArgumentParser
from git import Repo
from google.cloud import storage


class LaunchInference(object):
    def __init__(self, *args, **kwargs):
        self._define_args()
        self.args, _ = self._arg_parser.parse_known_args()
        self.extra_args_dict = self._get_extra_args_dict()

        self._download_pretrained_model()

        self._download_dataset()

        self._clone_dependencies()

        self._run_model_zoo()

    def _define_args(self):
        """
        Defines args that are used to launch the Model Zoo scripts. All args are using
        the string type, since that's the only data type supported by the KubeFlow
        Pipelines run parameters.
        """
        self._arg_parser = ArgumentParser(
            add_help=False, 
            description="Parse args to pass to the Model Zoo for Intel Architecture")

        self._arg_parser.add_argument(
            "-f", "--framework",
            help="Specify the name of the deep learning framework to use.",
            dest="framework", default="tensorflow", required=True)

        self._arg_parser.add_argument(
            "-p", "--precision",
            help="Specify the model precision to use: fp32, int8, or bfloat16",
            required=True, choices=["fp32", "int8", "bfloat16"],
            dest="precision")

        self._arg_parser.add_argument(
            "-mo", "--mode", help="Specify the type training or inference ",
            required=True, choices=["training", "inference"], dest="mode")

        self._arg_parser.add_argument(
            "-m", "--model-name", required=True,
            help="model name to run benchmarks for", dest="model_name")

        self._arg_parser.add_argument(
            "-b", "--batch-size",
            help="Specify the batch size. If this parameter is not specified "
                 "or is -1, the largest ideal batch size for the model will "
                 "be used", dest="batch_size", default="-1", type=str)

        self._arg_parser.add_argument(
            "-d", "--data-location",
            help="Specify the location of the data. If this parameter is not "
                 "specified, the benchmark will use random/dummy data.",
            dest="data_location", default=None, type=str)

        self._arg_parser.add_argument(
            "-i", "--socket-id",
            help="Specify which socket to use. Only one socket will be used "
                 "when this value is set. If used in conjunction with "
                 "--num-cores, all cores will be allocated on the single "
                 "socket.",
            dest="socket_id", type=str, default="-1")

        self._arg_parser.add_argument(
            "-n", "--num-cores",
            help="Specify the number of cores to use. If the parameter is not"
                 " specified or is -1, all cores will be used.",
            dest="num_cores", type=str, default="-1")

        self._arg_parser.add_argument(
            "-a", "--num-intra-threads", type=str,
            help="Specify the number of threads within the layer",
            dest="num_intra_threads", default=None)

        self._arg_parser.add_argument(
            "-e", "--num-inter-threads", type=str,
            help="Specify the number threads between layers",
            dest="num_inter_threads", default=None)

        self._arg_parser.add_argument(
            "--data-num-intra-threads", type=str,
            help="The number intra op threads for the data layer config",
            dest="data_num_intra_threads", default=None)

        self._arg_parser.add_argument(
            "--data-num-inter-threads", type=str,
            help="The number inter op threads for the data layer config",
            dest="data_num_inter_threads", default=None)

        self._arg_parser.add_argument(
            "--benchmark-or-accuracy",
            help="Specify whether to run performance benchmarking or accuracy "
                 "testing. Note that accuracy testing requires a dataset.",
            dest="benchmark_or_accuracy", choices=["benchmark", "accuracy"],
            default="benchmark", required=True, type=str.lower)

        self._arg_parser.add_argument(
            "--output-results",
            help="Writes inference output to a file, when used in conjunction "
                 "with --accuracy-only and --mode=inference.",
            dest="output_results", choices=["true", "false"], type=str.lower)

        self._arg_parser.add_argument(
            "-v", "--verbose", help="Print verbose information.",
            dest="verbose", choices=["true", "false"], type=str.lower)

        self._arg_parser.add_argument(
            "--output-dir",
            help="Folder to dump output into. The output directory will default to "
                 "'models/benchmarks/common/tensorflow/logs' if no path is specified.",
            default="/models/benchmarks/common/tensorflow/logs")

        self._arg_parser.add_argument(
            "--extra-model-args", help="String of extra model-specific args formatted"
                                       "like: 'arg1=value1 arg2=value2'",
            dest="extra_model_args", default="", type=str)

    def _get_extra_args_dict(self):
        """ Parse the string of extra args to a dictionary """
        extra_args_dict = {}

        if self.args.extra_model_args:
            extra_args_list = self.args.extra_model_args.split(" ")
            for arg in extra_args_list:
                if "=" in arg:
                    arg_list = arg.split("=")
                    extra_args_dict[arg_list[0]] = arg_list[1]
                else:
                    sys.exit("Extra model arguments must be formatted like 'arg1=value1 arg2=value2'.")

        return extra_args_dict


    def _download_pretrained_model(self):
        """
        Looks up the model name and precision in the dictionary to determine the URL on where to download the 
        pretrained model. If the model name and/or precision does not exist in the dictionary, then the program will
        exit. If the URL for the specified model and precision is found, then the model is downloaded and the
        destination path is returned.
        :return: Destination path to where the pretrained model has been downloaded.
        """
        # Using TF OOB gcloud storage links
        storage_bucket_link = "https://storage.googleapis.com/intel-optimized-tensorflow/models/"
        download_link_dict = {
            "faster_rcnn": {
                "fp32": os.path.join(storage_bucket_link, "faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz"),
                "int8": os.path.join(storage_bucket_link, "faster_rcnn_int8_pretrained_model.pb")
            },
            "inception_resnet_v2": {
                "fp32": os.path.join(storage_bucket_link, "inception_resnet_v2_fp32_pretrained_model.pb"),
                "int8": os.path.join(storage_bucket_link, "inception_resnet_v2_int8_pretrained_model.pb")
            },
            "inceptionv3": {
                "fp32": os.path.join(storage_bucket_link, "inceptionv3_fp32_pretrained_model.pb"),
                "int8": os.path.join(storage_bucket_link, "inceptionv3_int8_pretrained_model.pb")
            },
            "inceptionv4": {
                "fp32": os.path.join(storage_bucket_link, "inceptionv4_fp32_pretrained_model.pb"),
                "int8": os.path.join(storage_bucket_link, "inceptionv4_int8_pretrained_model.pb")
            },
            "mobilenet_v1": {
                "fp32": "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"
            },
            "resnet50": {
                "fp32": os.path.join(storage_bucket_link, "resnet50_fp32_pretrained_model.pb"),
                "int8": os.path.join(storage_bucket_link, "resnet50_int8_pretrained_model.pb")
            },
            "resnet101": {
                "fp32": os.path.join(storage_bucket_link, "resnet101_fp32_pretrained_model.pb"),
                "int8": os.path.join(storage_bucket_link, "resnet101_int8_pretrained_model.pb")
            },
            "rfcn": {
                "fp32": os.path.join(storage_bucket_link, "rfcn_resnet101_fp32_coco_pretrained_model.tar.gz"),
                "int8": os.path.join(storage_bucket_link, "rfcn_resnet101_int8_coco_pretrained_model.pb")
            },
            "ssd-mobilenet": {
                "fp32": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz",
                "int8": os.path.join(storage_bucket_link, "ssdmobilenet_int8_pretrained_model.pb")
            },
            "squeezenet": {
                "fp32": os.path.join(storage_bucket_link, "squeezenet_fp32_pretrained_model.tar.gz")
            }

        }

        model_name = self.args.model_name
        precision = self.args.precision
        self.args.input_graph = None
        self.args.checkpoint_dir = None

        if model_name in download_link_dict.keys():
            if self.args.precision in download_link_dict[model_name].keys():
                source_path = download_link_dict[model_name][precision]
                file_basename = os.path.basename(source_path)
                destination_path = os.path.join("/root/pretrained_models", file_basename)

                print("Downloading pretrained model from {} to {}".format(source_path, destination_path))
                urllib.request.urlretrieve(source_path, destination_path)
                print("Download complete")

                _, file_extension = os.path.splitext(destination_path)

                if file_extension == ".pb":
                    self.args.input_graph = destination_path
                else:
                    # It's probably a tar file with checkpoints, so extract the tar file
                    checkpoint_dir = "/root/pretrained_models/checkpoints"
                    with tarfile.open(destination_path) as tf:
                        tf.extractall(checkpoint_dir)

                    # If the directory only contains one directory, then pass that instead (filter out temp files)
                    dir_files = [i for i in os.listdir(checkpoint_dir) if not i.startswith(".")]
                    if len(dir_files) == 1:
                        if os.path.isdir(os.path.join(checkpoint_dir, dir_files[0])):
                            checkpoint_dir = os.path.join(checkpoint_dir, dir_files[0])

                    # Set the checkpoint arg
                    self.args.checkpoint_dir = checkpoint_dir
            else:
                sys.exit("The {} precision for model {} is not supported yet.".format(precision, model_name))
        else:
            sys.exit("The {} model is not supported".format(model_name))


        # The Faster RCNN and RFCN FP32 benchmarking script has a config file that needs some paths updated
        if (model_name == "rfcn" or model_name == "faster_rcnn") and precision == "fp32" and \
            self.args.benchmark_or_accuracy == "benchmark":
            # Set default config file path, if it wasn't set
            if "config_file" not in self.extra_args_dict.keys():
                if self.args.extra_model_args != "":
                    self.args.extra_model_args += " "
                config_file_name = "rfcn_pipeline.config" if model_name == "rfcn" else "pipeline.config"
                self.args.extra_model_args += "config_file={}".format(config_file_name)
                self.extra_args_dict["config_file"] = config_file_name

            config_file_path = os.path.join(checkpoint_dir, self.extra_args_dict["config_file"])

            original_map_file_path = "/checkpoints/mscoco_label_map.pbtxt"
            original_dataset_file_path = "/dataset/coco_val.record"
            base_map_file_name = os.path.basename(original_map_file_path)
            base_dataset_file_name = os.path.basename(original_dataset_file_path)

            new_label_map = os.path.join(checkpoint_dir, base_map_file_name)
            new_dataset_path = os.path.join(self.args.data_location, base_dataset_file_name)

            with fileinput.FileInput(config_file_path, inplace=True) as config_file:
                for line in config_file:
                    print(line.replace(original_map_file_path, new_label_map).
                          replace(original_dataset_file_path, new_dataset_path),
                                  end='')

        # RFCN FP32 Accuracy uses the frozen graph from the tar file instead of checkpoints
        if model_name == "rfcn" and precision == "fp32" and self.args.benchmark_or_accuracy == "accuracy":
            # Set default split, if it wasn't set
            if "split" not in self.extra_args_dict.keys():
                if self.args.extra_model_args != "":
                    self.args.extra_model_args += " "
                self.args.extra_model_args += "split=\"accuracy_message\""
                self.extra_args_dict["split"] = "\"accuracy_message\""
            # Set frozen graph
            self.args.input_graph = os.path.join(self.args.checkpoint_dir, "frozen_inference_graph.pb")
            self.args.checkpoint_dir = None

        # Faster RCNN FP32 uses frozen graph from the checkpoints directory
        if model_name == "faster_rcnn" and precision == "fp32" and self.args.benchmark_or_accuracy == "accuracy":
            self.args.input_graph = os.path.join(self.args.checkpoint_dir, "frozen_inference_graph.pb")
            self.args.checkpoint_dir = None

        # SSD-MobileNet FP32 uses the frozen graph from the checkpoints directory
        if model_name == "ssd-mobilenet" and precision == "fp32":
            self.args.input_graph = os.path.join(self.args.checkpoint_dir, "frozen_inference_graph.pb")
            self.args.checkpoint_dir = None

    def _download_dataset(self):
        data_location = self.args.data_location

        if data_location.startswith("gs://"):
            # Download the dataset from Google Cloud Storage
            bucket = os.path.dirname(data_location).strip("gs://")
            directory = os.path.basename(data_location)

            print("Getting GS bucket: {}".format(bucket))
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name=bucket)

            print("Getting blobs from: {}".format(directory))
            blobs = bucket.list_blobs(prefix=directory)

            for blob in blobs:
                # skip blobs that are directories
                if blob.name.endswith("/"):
                    continue

                base_name = os.path.basename(blob.name)
                destination_path = os.path.join("dataset", base_name)
                print("Download file to {}".format(destination_path))
                blob.download_to_filename(destination_path)

            self.args.data_location = "/root/dataset"

    def _clone_dependencies(self):
        if self.args.model_name == "mobilenet_v1" and self.args.precision == "fp32":
            tf_model_dir = "/root/model_source/models"
            Repo.clone_from(
                "https://github.com/tensorflow/models",
                tf_model_dir,
                branch="master"
            )
            self.args.model_source_dir = tf_model_dir
        if self.args.model_name in ["faster_rcnn", "rfcn", "ssd-mobilenet"]:
            self.args.model_source_dir = "/root/tensorflow/models"



    def _run_model_zoo(self):
        """ 
        Generates a command to run the Model Zoo launch script, with all of 
        the parameters that we got from the arg parser.
        """
        args = self.args
        run_cmd = ["python", "models/benchmarks/launch_benchmark.py",
                   "--framework", args.framework,
                   "--model-name", args.model_name,
                   "--precision", args.precision,
                   "--mode", args.mode]

        # Use the pretrained model that we downloaded as the input graph
        if self.args.input_graph:
            run_cmd += ["--in-graph", self.args.input_graph]

        if self.args.checkpoint_dir:
            run_cmd += ["--checkpoint", self.args.checkpoint_dir]

        # if self.pretrained_model_path:
        #     _, file_extension = os.path.splitext(self.pretrained_model_path)
        #     if file_extension == ".pb":
        #         run_cmd += ["--in-graph", self.pretrained_model_path]
        #     else:
        #         # It's probably a tar file with checkpoints, so extract the tar file
        #         checkpoint_dir = "/root/pretrained_models/checkpoints"
        #         with tarfile.open(self.pretrained_model_path) as tf:
        #             tf.extractall(checkpoint_dir)
        #
        #         # If the directory only contains one directory, then pass that instead (filter out temp files)
        #         dir_files = [i for i in os.listdir(checkpoint_dir) if not i.startswith(".")]
        #         if len(dir_files) == 1:
        #             if os.path.isdir(os.path.join(checkpoint_dir, dir_files[0])):
        #                 checkpoint_dir = os.path.join(checkpoint_dir, dir_files[0])
        #
        #         # Set the checkpoint arg
        #         run_cmd += ["--checkpoint", checkpoint_dir]

        # Set the --model-source-dir arg, depending on the model
        if args.model_source_dir:
            run_cmd += ["--model-source-dir", args.model_source_dir]

        if args.batch_size and args.batch_size != "-1":
            run_cmd += ["--batch-size", args.batch_size]

        if args.data_location:
            run_cmd += ["--data-location", args.data_location]

        if args.socket_id and args.socket_id != "-1":
            run_cmd += ["--socket-id", args.socket_id]

        if args.num_cores and args.num_cores != "-1":
            run_cmd += ["--num-cores", args.num_cores]

        if args.num_inter_threads:
            run_cmd += ["--num-inter-threads", args.num_inter_threads]

        if args.num_intra_threads:
            run_cmd += ["--num-intra-threads", args.num_intra_threads]

        if args.data_num_inter_threads:
            run_cmd += ["--data-num-inter-threads", args.data_num_inter_threads]

        if args.data_num_intra_threads:
            run_cmd += ["--data-num-intra-threads", args.data_num_intra_threads]

        if args.benchmark_or_accuracy == "accuracy":
            run_cmd += ["--accuracy-only"]
        else:
            run_cmd += ["--benchmark-only"]

        if args.output_results and args.output_results == "true":
            run_cmd += ["--output-results"]

        if args.verbose and args.verbose == "true":
            run_cmd += ["--verbose"]

        if args.output_dir:
            run_cmd += ["--output-dir", args.output_dir]

        if args.extra_model_args:
            extra_args_list = args.extra_model_args.split(" ")
            if len(extra_args_list) > 0:
                run_cmd += ["--"]
                run_cmd += extra_args_list

        # Print out the command that we are going to run, and then flush
        # the output buffers so that the prints don't come out of order
        print("Running launch script:\n{}\n".format(run_cmd))
        sys.stdout.flush()

        # Run the launch script
        p = subprocess.Popen(run_cmd, preexec_fn=os.setsid)
        p.communicate()


if __name__ == "__main__":
    LaunchInference()
