
import GPUtil
from time import sleep
import subprocess
import tqdm
import tempfile
import os

def every(seconds):
    while True:
        sleep(seconds)
        yield

def run_on_gpu(gpu, task):
    command = "CUDA_VISIBLE_DEVICES={} {}".format(gpu, task)
    file = tempfile.TemporaryFile()
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=file, shell=os.environ['SHELL'])
    return command, file, proc

class TaskRunner:
    def __init__(self, tasks, *, max_memory, max_procs):
        self.current_processes = {}
        self.pbar = tqdm.tqdm(total=len(tasks))
        self.tasks = tasks

        self.max_memory = max_memory
        self.max_procs = max_procs

    def handle_done(self):
        for gpu, (name, file, proc) in list(self.current_processes.items()):
            retcode = proc.poll()
            if retcode is not None:
                if retcode != 0:
                    print("Process failed: ", name)
                    file.seek(0)
                    print(file.read().decode('utf-8'))
                del self.current_processes[gpu]
                self.pbar.update()

    def valid_gpu(self):
        open_gpus = set(GPUtil.getAvailable(limit=float('inf'), maxMemory=self.max_memory))
        valid_gpus = open_gpus - set(self.current_processes)
        if valid_gpus:
            return list(valid_gpus)[0]
        return None

    def step(self):
        self.pbar.set_description("Using GPUs {}".format(set(self.current_processes)))
        self.pbar.refresh()
        self.handle_done()
        if len(self.current_processes) >= self.max_procs:
            return
        gpu = self.valid_gpu()
        if gpu is None:
            return
        if not self.tasks:
            return
        self.current_processes[gpu] = run_on_gpu(gpu, self.tasks.pop())

    def run_all(self):
        try:
            for _ in every(1):
                if not (self.tasks or self.current_processes):
                    break
                self.step()
        finally:
            self.pbar.close()
            print("Terminating")
            for _, _, proc in tqdm.tqdm(list(self.current_processes.values())):
                proc.kill()

def run_tasks(tasks, **kwargs):
    TaskRunner(tasks, **kwargs).run_all()

if __name__ == '__main__':
    import argparse
    from sys import stdin
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory", type=float, default=1.0)
    parser.add_argument("--max-procs", type=int, default=float('inf'))
    args = parser.parse_args()

    run_tasks(list(stdin), max_memory=args.max_memory, max_procs=args.max_procs)
