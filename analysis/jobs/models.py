import time
import os
from tqdm import tqdm
import subprocess
from analysis import config


class HTCondorJob:

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        script_path: str,
        input_file_ext: str,
        output_file_ext: str,
        **kwargs
    ):
        self._input_files = [
            os.path.join(input_dir, path) for path in os.listdir(input_dir) if path.endswith(input_file_ext)
        ]
        self._output_dir = output_dir
        self._script_path = os.path.abspath(script_path)
        self._input_file_ext = input_file_ext
        self._output_file_ext = output_file_ext

        DIR = os.path.join(config.BASE_DIR, "analysis/jobs")

        self._dagman_file_path = os.path.abspath(kwargs.get("dagman_file_path", os.path.join(DIR, "dag/dagman.dag")))
        self._config_file_path = os.path.abspath(kwargs.get("config_file_path", os.path.join(DIR, "conf/config.dag")))
        self._job_sub_file_path = os.path.abspath(kwargs.get("job_sub_file_path", os.path.join(DIR, "conf/job.sub")))
        self._job_sh_file_path = os.path.abspath(kwargs.get("job_sh_file_path", os.path.join(DIR, "conf/job.sh")))

        self._dagman_out_file_path = os.path.join(
            os.path.dirname(self._dagman_file_path), os.path.basename(self._dagman_file_path) + ".dagman.out"
        )

        self._log_file_path = os.path.abspath(kwargs.get("log_file_path", os.path.join(DIR, "logs/log/")))
        self._out_file_path = os.path.abspath(kwargs.get("out_file_path", os.path.join(DIR, "logs/out/")))
        self._err_file_path = os.path.abspath(kwargs.get("err_file_path", os.path.join(DIR, "logs/err/")))

        self._request_cpus = int(kwargs.get("request_cpus", 8))
        self._request_memory = kwargs.get("request_memory", "1GB")
        self._request_disk = kwargs.get("request_disk", "8GB")
        self._universe = kwargs.get("universe", "vanilla")
        self._should_transfer_files = kwargs.get("should_transfer_files", "YES")
        self._when_to_transfer_output = kwargs.get("when_to_transfer_output", "ON_EXIT")

    def submit(self, monitor: bool=False):
        submit_dag_args = [
            "condor_submit_dag",
            "-f",  # Force submission
            self._dagman_file_path
        ]

        try:
            submit_result = subprocess.run(submit_dag_args, check=True, text=True, capture_output=True)
            print(submit_result.stdout)

            # Extract the Cluster ID from the output (e.g., "submitted to cluster 1234")
            job_id = None
            for line in submit_result.stdout.splitlines():
                if "cluster" in line.lower():
                    job_id = line.split()[-1]  # Extract the cluster ID
                    break

            if not job_id:
                raise ValueError("Unable to extract job ID from submission output.")

            print(f"Job ID: {job_id}")

        except subprocess.CalledProcessError as e:
            print(f"Error submitting DAG:\n{e.stderr}")
            exit(1)

        if monitor:
            self._monitor()

    def _monitor(self):
        _init_pb_flag = False
        while True:
            try:
                status = self.status()
                if not status:
                    continue

                if not _init_pb_flag:
                    progress_bar = tqdm(
                        total=status["queued"] + status["ready"] + status["done"] + status["failed"],
                        desc="Running jobs. 0 Failed.", unit="jobs"
                    )
                    _init_pb_flag = True

                progress_bar.n = status["done"] + status["failed"]
                progress_bar.set_description(f"Running jobs. {status['failed']} Failed")

                if status["queued"] + status["ready"] == 0:
                    break

                time.sleep(1)

            except FileNotFoundError:
                print("\nWaiting for output file to be generated...")
                time.sleep(1)

            except IndexError:
                time.sleep(1)

    def status(self, lines_after=2):
        def parse(output):
            values = [int(x) for x in output.split() if x.isdigit()]

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

        condition_text = "  Done     Pre   Queued    Post   Ready   Un-Ready   Failed   Futile"

        with open(self._dagman_out_file_path, 'r') as file:
            # Read all lines in reverse order
            lines = file.readlines()

            # Iterate over lines in reverse order
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()

                # Check if the line matches the condition
                if condition_text in line:
                    content = lines[i + lines_after].strip()[17:]
                    status = parse(content)
                    return status

        return None

    def configure(self):
        """Generate relevant configuration files and clear log directories."""
        self.clean_logs()
        self.create_job_sh()
        self.create_job_sub()
        self.create_dag()

    def create_dag(self):
        print(f"Generating dagman file at {self._dagman_file_path}...\n")

        instructions = f'CONFIG {self._config_file_path} \n\n'

        _iter = tqdm(self._input_files)
        for i, file in enumerate(_iter):
            base_name = os.path.basename(file)

            instructions += f'JOB job_{i} {self._job_sub_file_path} \n'
            instructions += f'''VARS job_{i} infile="{file}" outfile="{os.path.join(
                self._output_dir, base_name.replace(self._input_file_ext, self._output_file_ext)
            )}" \n\n'''

        print("\n")

        with open(self._dagman_file_path, 'w') as fwrite:
            fwrite.write(instructions)

    def create_job_sh(self):
        print(f"Generating job.sh file at {self._job_sh_file_path}...")

        content = '#!/bin/sh'
        content += "\n\nexport HDF5_DISABLE_VERSION_CHECK=1"
        content += "\n\neval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)"
        content += '\n\ninput_file=$1\noutput_file=$2'
        content += f'\n\nscript_path="{self._script_path}"'
        content += '\n\n"$SROOT"/metaprojects/icetray/v1.8.2/env-shell.sh /data/i3home/tstjean/icecube/venv/bin/python3.11 $script_path -i "$input_file" -o "$output_file"'

        with open(self._job_sh_file_path, "w") as job_sh:
            job_sh.write(content)

    def create_job_sub(self):
        print(f"Generating job.sub file at {self._job_sub_file_path}...")

        content = f"log = {os.path.join(self._log_file_path, '$(Cluster).$(Process).log')}"
        content += f"\noutput = {os.path.join(self._out_file_path, '$(Cluster).$(Process).out')}"
        content += f"\nerror = {os.path.join(self._err_file_path, '$(Cluster).$(Process).err')}"

        content += f"\n\nrequest_cpus = {self._request_cpus}"
        content += f"\nrequest_memory = {self._request_memory}"
        content += f"\nrequest_disk = {self._request_disk}"
        content += f"\nUniverse = {self._universe}"
        content += f"\nshould_transfer_files = {self._should_transfer_files}"
        content += f"\nwhen_to_transfer_output = {self._when_to_transfer_output}"

        content += f"\n\nexecutable = {self._job_sh_file_path}"
        content += "\narguments = $(infile) $(outfile)"

        content += "\n\nqueue"

        with open(self._job_sub_file_path, "w") as job_sub:
            job_sub.write(content)

    def clean_logs(self):
        for path in [self._log_file_path, self._out_file_path, self._err_file_path]:
            for file in os.listdir(path):
                abs_path = os.path.join(path, file)
                os.remove(abs_path)

        for path in os.listdir(os.path.dirname(self._dagman_file_path)):
            if not path.endswith("dagman.dag") and not path.startswith(".nfs"):
                os.remove(os.path.join(os.path.dirname(self._dagman_file_path), path))


def test():
    job = HTCondorJob(
        config.I3FILEDIR_NUMU,
        "/data/i3home/tstjean/icecube/data/hdf5/21220/",
        "../../scripts/i3_to_hdf5.py", ".i3.zst", ".hdf5"
    )
    job.configure()
    job.submit(monitor=True)