
import GPUtil
from time import sleep
import subprocess
import tqdm
import os

def every(seconds):
    while True:
        sleep(seconds)
        yield

def run_on_gpu(gpu, task):
    command = "CUDA_VISIBLE_DEVICES={} {}".format(gpu, task)
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, shell=os.environ['SHELL'])
    return command, proc

class TaskRunner:
    def __init__(self, tasks):
        self.current_processes = {}
        self.pbar = tqdm.tqdm(total=len(tasks))
        self.tasks = tasks

    def handle_done(self):
        for gpu, (name, proc) in list(self.current_processes.items()):
            retcode = proc.poll()
            if retcode is not None:
                if retcode != 0:
                    print("Process failed: ", name)
                    print(proc.stderr.read().decode('utf-8'))
                del self.current_processes[gpu]
                self.pbar.update()

    def valid_gpu(self):
        open_gpus = set(GPUtil.getAvailable(limit=float('inf')))
        valid_gpus = open_gpus - set(self.current_processes) - {0}
        if valid_gpus:
            return list(valid_gpus)[0]
        return None

    def step(self):
        self.pbar.update(0)
        self.handle_done()
        gpu = self.valid_gpu()
        if gpu is None:
            return
        if not self.tasks:
            return
        self.current_processes[gpu] = run_on_gpu(gpu, self.tasks.pop())
        self.pbar.set_description("Using GPUs {}".format(set(self.current_processes)))

    def run_all(self):
        try:
            for _ in every(1):
                if not (self.tasks or self.current_processes):
                    break
                self.step()
        finally:
            for _, proc in self.current_processes.values():
                proc.kill()

def run_tasks(tasks):
    TaskRunner(tasks).run_all()

if __name__ == '__main__':
    import fileinput
    run_tasks(list(fileinput.input()))