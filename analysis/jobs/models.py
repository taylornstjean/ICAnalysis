########################################################################################################################
# imports

import time
import os
from tqdm import tqdm
import subprocess
from typing import Union

from analysis import config


########################################################################################################################


class HTCondorJob:
    """
        A class for managing HTCondor jobs using DAGMan.

        Attributes:
            _input_files (list): List of input file paths matching the specified extension.
            _output_dir (str): Directory to save the output files.
            _script_path (str): Path to the script to be executed for each job.
            _input_file_ext (str): File extension of the input files.
            _output_file_ext (str): File extension of the output files.
            _dagman_file_path (str): Path to the DAGMan file.
            _config_file_path (str): Path to the configuration file.
            _job_sub_file_path (str): Path to the job submission file.
            _job_sh_file_path (str): Path to the job shell script file.
            _dagman_out_file_path (str): Path to the DAGMan output file.
            _log_file_path (str): Path to the log directory.
            _out_file_path (str): Path to the output log directory.
            _err_file_path (str): Path to the error log directory.
            _request_cpus (int): Number of CPUs requested for each job.
            _request_memory (str): Memory requested for each job.
            _request_disk (str): Disk space requested for each job.
            _universe (str): HTCondor universe for the job.
            _should_transfer_files (str): Indicates whether files should be transferred.
            _when_to_transfer_output (str): When to transfer output files.
        """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        script_path: str,
        input_file_ext: str,
        output_file_ext: str,
        **kwargs
    ) -> None:
        """
            Initialize an HTCondorJob instance.

            Args:
                input_dir (str): Directory containing input files.
                output_dir (str): Directory to save output files.
                script_path (str): Path to the script for processing input files.
                input_file_ext (str): File extension of input files.
                output_file_ext (str): File extension of output files.
                **kwargs: Additional configuration paths and parameters.
        """
        # required input parameters
        self._input_files = [
            os.path.join(input_dir, path) for path in os.listdir(input_dir) if path.endswith(input_file_ext)
        ]
        self._output_dir = output_dir
        self._script_path = os.path.abspath(script_path)
        self._input_file_ext = input_file_ext
        self._output_file_ext = output_file_ext

        # job directory
        job_dir = os.path.join(config.BASE_DIR, "analysis/jobs")

        # allow modifications to where configuration and submission files will be saved
        self._dagman_file_path = os.path.abspath(kwargs.get("dagman_file_path", os.path.join(job_dir, "dag/dagman.dag")))
        self._config_file_path = os.path.abspath(kwargs.get("config_file_path", os.path.join(job_dir, "conf/config.dag")))
        self._job_sub_file_path = os.path.abspath(kwargs.get("job_sub_file_path", os.path.join(job_dir, "conf/job.sub")))
        self._job_sh_file_path = os.path.abspath(kwargs.get("job_sh_file_path", os.path.join(job_dir, "conf/job.sh")))

        # htc will by default generate the dagman.out file in the same directory as the dagman.dag file
        self._dagman_out_file_path = os.path.join(
            os.path.dirname(self._dagman_file_path), os.path.basename(self._dagman_file_path) + ".dagman.out"
        )

        # allow modifications to the location of log output files
        self._log_file_path = os.path.abspath(kwargs.get("log_file_path", os.path.join(job_dir, "logs/log/")))
        self._out_file_path = os.path.abspath(kwargs.get("out_file_path", os.path.join(job_dir, "logs/out/")))
        self._err_file_path = os.path.abspath(kwargs.get("err_file_path", os.path.join(job_dir, "logs/err/")))

        # allow modification of job.sub configurations
        self._request_cpus = int(kwargs.get("request_cpus", 2))
        self._request_memory = kwargs.get("request_memory", "1GB")
        self._request_disk = kwargs.get("request_disk", "2GB")
        self._universe = kwargs.get("universe", "vanilla")
        self._should_transfer_files = kwargs.get("should_transfer_files", "YES")
        self._when_to_transfer_output = kwargs.get("when_to_transfer_output", "ON_EXIT")

        # verify log directories exist
        for path in [self._log_file_path, self._out_file_path, self._err_file_path]:
            os.makedirs(path, exist_ok=True)

    def submit(self, monitor: bool=False) -> None:
        """
        Submit the DAGMan job to HTCondor.

        Args:
            monitor (bool): Whether to monitor the job status after submission.
        """

        # initialize the submission command
        submit_dag_args = [
            "condor_submit_dag",
            "-f",  # Force submission
            self._dagman_file_path
        ]

        try:

            # submit the job, and print the submission status
            submit_result = subprocess.run(submit_dag_args, check=True, text=True, capture_output=True)
            print(submit_result.stdout)

            # extract the cluster ID from the output
            job_id = None

            # iterate over lines, stopping once the ID is found
            for line in submit_result.stdout.splitlines():
                if "cluster" in line.lower():
                    job_id = line.split()[-1]
                    break

            if not job_id:
                # if we cannot pull the cluster ID, there is likely a problem that needs to be addressed
                raise ValueError("Unable to extract job ID from submission output.")

            print(f"Job ID: {job_id}")

        except subprocess.CalledProcessError as e:
            print(f"Error submitting DAG:\n{e.stderr}")
            exit(1)

        if monitor:
            self._monitor()

    def _monitor(self) -> None:
        """
        Monitor the progress of submitted jobs. Once the job is finished, returns ``None``.
        """
        # initialize a flag so we only generate the progress bar on the first cycle
        _init_pb_flag = False

        while True:
            try:
                # pull the job status
                status = self.status()

                if not status:
                    # if no status info was found in the file, wait until the next cycle
                    continue

                if not _init_pb_flag:
                    # initialize a progress bar on first cycle
                    progress_bar = tqdm(
                        total=status["queued"] + status["ready"] + status["done"] + status["failed"],
                        desc="Running jobs. 0 Failed.", unit="jobs"
                    )
                    _init_pb_flag = True

                # update the progress bar to reflect current status
                progress_bar.n = status["done"] + status["failed"]

                # include the number of failed jobs in the description
                progress_bar.set_description(f"Running jobs. {status['failed']} failed")

                if status["queued"] + status["ready"] == 0:
                    # break out of the loop once the job is complete
                    return None

                # run each cycle once per second
                time.sleep(1)

            except FileNotFoundError:
                # file hasn't been generated yet, wait
                print("\nWaiting for output file to be generated...")
                time.sleep(1)

            except IndexError:
                # sometimes the file is queried when it has only generated the condition line, however following lines
                # haven't been generated yet, leading to an IndexError when trying to pull following lines,
                # in this case simply wait for the next cycle.
                time.sleep(1)

    def status(self, lines_after=2) -> Union[dict, None]:
        """
        Pull the status of the jobs from the DAGMan output file.

        Args:
            lines_after (int): Number of lines after the status header to parse.

        Returns:
            Union[dict, None]: A dictionary containing job status counts, or ``None`` if status isn't found in the file.
        """
        def __parse(output: str) -> dict:
            """
            Parser to convert a status string to a dict.

            Args:
                output (str): Status string pulled from dagman.out.

            Returns:
                Status dict.
            """

            # split by whitespace
            values = [int(x) for x in output.split() if x.isdigit()]

            # convert to a dict
            _status = {
                "done": values[0],
                "pre": values[1],
                "queued": values[2],
                "post": values[3],
                "ready": values[4],
                "unready": values[5],
                "failed": values[6],
                "futile": values[7]
            }

            return _status

        # look for this text in the dagman output file
        condition_text = "  Done     Pre   Queued    Post   Ready   Un-Ready   Failed   Futile"

        with open(self._dagman_out_file_path, 'r') as file:
            # Read all lines in reverse order
            lines = file.readlines()

            # Iterate over lines in reverse order
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()

                # Check if the line matches the condition
                if condition_text not in line:
                    continue

                # get the 'lines_after'-th line down from the condition text, and trim the entry time
                content = lines[i + lines_after].strip()[17:]
                status = __parse(content)
                return status

        return None

    def configure(self, clean=False) -> None:
        """
        Generate relevant configuration files and clear log directories.

        Args:
            clean (bool): Clear out old run files from log directories. Defaults to False.
        """

        # run job cleanup if necessary
        if clean:
            self.clean_logs()

        # generate job files
        self.create_job_sh()
        self.create_job_sub()
        self.create_dag()

    def create_dag(self) -> None:
        """
        Generates a dagman.dag job submission file.
        """
        print(f"Generating dagman file at {self._dagman_file_path}...\n")

        # include configuration file
        instructions = f'CONFIG {self._config_file_path} \n\n'

        # iterate over each file in the input directory and create a job entry
        _iter = tqdm(self._input_files)
        for i, file in enumerate(_iter):
            base_name = os.path.basename(file)

            # include instructions to run each job with an input file and an output file
            instructions += f'JOB job_{i} {self._job_sub_file_path} \n'
            instructions += f'''VARS job_{i} infile="{file}" outfile="{os.path.join(
                self._output_dir, base_name.replace(self._input_file_ext, self._output_file_ext)
            )}" \n\n'''

        print("\n")

        # verify the directory exists
        os.makedirs(os.path.dirname(self._dagman_file_path), exist_ok=True)

        # save the file
        with open(self._dagman_file_path, 'w') as fwrite:
            fwrite.write(instructions)

    def create_job_sh(self) -> None:
        """
        Generates a job.sh executable file.
        """
        print(f"Generating job.sh file at {self._job_sh_file_path}...")

        # include env variables, initialize icetray and activate the virtual environment
        content = '#!/bin/sh'
        content += "\n\nexport HDF5_DISABLE_VERSION_CHECK=1"
        content += "\n\neval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)"
        content += '\n\ninput_file=$1\noutput_file=$2'
        content += f'\n\nscript_path="{self._script_path}"'
        content += '\n\n"$SROOT"/metaprojects/icetray/v1.8.2/env-shell.sh /data/i3home/tstjean/icecube/venv/bin/python3.11 $script_path -i "$input_file" -o "$output_file"'

        # verify the directory exists
        os.makedirs(os.path.dirname(self._job_sh_file_path), exist_ok=True)

        # save the file
        with open(self._job_sh_file_path, "w") as job_sh:
            job_sh.write(content)

    def create_job_sub(self) -> None:
        """
        Generates a job.sub file.
        """
        print(f"Generating job.sub file at {self._job_sub_file_path}...")

        # include paths to log directories
        content = f"log = {os.path.join(self._log_file_path, '$(Cluster).$(Process).log')}"
        content += f"\noutput = {os.path.join(self._out_file_path, '$(Cluster).$(Process).out')}"
        content += f"\nerror = {os.path.join(self._err_file_path, '$(Cluster).$(Process).err')}"

        # include configurations
        content += f"\n\nrequest_cpus = {self._request_cpus}"
        content += f"\nrequest_memory = {self._request_memory}"
        content += f"\nrequest_disk = {self._request_disk}"
        content += f"\nUniverse = {self._universe}"
        content += f"\nshould_transfer_files = {self._should_transfer_files}"
        content += f"\nwhen_to_transfer_output = {self._when_to_transfer_output}"

        # include path to executable
        content += f"\n\nexecutable = {self._job_sh_file_path}"
        content += "\narguments = $(infile) $(outfile)"

        content += "\n\nqueue"

        # verify the directory exists
        os.makedirs(os.path.dirname(self._job_sub_file_path), exist_ok=True)

        # save the file
        with open(self._job_sub_file_path, "w") as job_sub:
            job_sub.write(content)

    def clean_logs(self) -> None:
        """
        Clears the log directory, and removes all generated job files from last run.
        """
        # iterate over log directories, deleting all contents
        for path in [self._log_file_path, self._out_file_path, self._err_file_path]:
            for file in os.listdir(path):
                abs_path = os.path.join(path, file)
                os.remove(abs_path)

        # verify the dagman directory exists, since this function can be run before create_dag() method
        os.makedirs(os.path.dirname(self._dagman_file_path), exist_ok=True)

        # iterate over files within the dag directory, removing all that are not the dagman.dag file,
        # and skip temporary system files (starting with .nfs)
        for path in os.listdir(os.path.dirname(self._dagman_file_path)):
            if not path.endswith("dagman.dag") and not path.startswith(".nfs"):
                os.remove(os.path.join(os.path.dirname(self._dagman_file_path), path))

########################################################################################################################
