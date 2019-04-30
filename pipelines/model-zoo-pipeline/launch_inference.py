import os
import subprocess
import sys
import urllib.request

from argparse import ArgumentParser


class LaunchInference(object):
    def __init__(self, *args, **kwargs):
        self._define_args()
        self.args, _ = self._arg_parser.parse_known_args()
        self.pretrained_model_path = self._download_pretrained_model()
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
            "-r", "--model-source-dir",
            help="Specify the models source directory from your local machine",
            nargs="?", dest="model_source_dir", type=str)

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
            "-c", "--checkpoint",
            help="Specify the location of trained model checkpoint directory. "
                 "If mode=training model/weights will be written to this "
                 "location. If mode=inference assumes that the location points"
                 " to a model that has already been trained.",
            dest="checkpoint", default=None, type=str)

        self._arg_parser.add_argument(
            "-g", "--in-graph", help="Full path to the input graph ",
            dest="input_graph", default=None, type=str)

        self._arg_parser.add_argument(
            "-k", "--benchmark-only",
            help="For benchmark measurement only. If neither --benchmark-only "
                 "or --accuracy-only are specified, it will default to run "
                 "benchmarking.",
            dest="benchmark_only", choices=["true", "false"], type=str.lower)

        self._arg_parser.add_argument(
            "--accuracy-only",
            help="For accuracy measurement only.  If neither --benchmark-only "
                 "or --accuracy-only are specified, it will default to run "
                 "benchmarking.",
            dest="accuracy_only", choices=["true", "false"], type=str.lower)

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

    def _download_pretrained_model(self):
        """
        Looks up the model name and precision in the dictionary to determine the URL on where to download the 
        pretrained model. If the model name and/or precision does not exist in the dictionary, then the program will
        exit. If the URL for the specified model and precision is found, then the model is downloaded and the
        destination path is returned.
        :return: Destination path to where the pretrained model has been downloaded.
        """
        storage_bucket_link = "https://storage.googleapis.com/intel-optimized-tensorflow/models/"
        download_link_dict = {
            "inceptionv3": {
                "fp32": "inceptionv3_fp32_pretrained_model.pb",
                "int8": "inceptionv3_int8_pretrained_model.pb"
            },
            "resnet50": {
                "fp32": "resnet50_fp32_pretrained_model.pb",
                "int8": "resnet50_int8_pretrained_model.pb"
            }
        }

        model_name = self.args.model_name
        precision = self.args.precision

        if model_name in download_link_dict.keys():
            if self.args.precision in download_link_dict[model_name].keys():
                source_path = os.path.join(storage_bucket_link, download_link_dict[model_name][precision])
                destination_path = os.path.join("/root/pretrained_models", download_link_dict[model_name][precision])

                print("Downloading pretrained model from {} to {}".format(source_path, destination_path))
                urllib.request.urlretrieve(source_path, destination_path)
                print("Download complete")

                return destination_path
            else:
                sys.exit("The {} precision for model {} is not supported yet.".format(precision, model_name))
        else:
            sys.exit("The {} model is not supported".format(model_name))

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
        if self.pretrained_model_path:
            run_cmd += ["--in-graph", self.pretrained_model_path]

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

        if args.checkpoint:
            run_cmd += ["--checkpoint", args.checkpoint]

        if args.benchmark_only and args.benchmark_only == "true":
            run_cmd += ["--benchmark-only"]

        if args.accuracy_only and args.accuracy_only == "true":
            run_cmd += ["--accuracy-only"]

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
