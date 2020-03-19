
import GPUtil
from time import sleep
import subprocess
import tqdm

def every(seconds):
    while True:
        sleep(seconds)
        yield

def run_tasks(tasks):
    pbar = tqdm.tqdm(total=len(tasks))
    current_processes = {}
    try:
        for _ in every(1):
            pbar.update(0)
            for gpu, (name, proc) in list(current_processes.items()):
                retcode = proc.poll()
                if retcode is not None:
                    if retcode != 0:
                        print("Process failed: ", name)
                    del current_processes[gpu]
                    pbar.update()

            open_gpus = set(GPUtil.getAvailable(limit=float('inf')))
            valid_gpus = open_gpus - set(current_processes) - {0}
            if not valid_gpus:
                continue
            if not tasks:
                break
            task = tasks.pop()
            gpu = list(valid_gpus)[0]
            command = "CUDA_VISIBLE_DEVICES={} {}".format(gpu, task)
            current_processes[gpu] = command, subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pbar.set_description("Using GPUs {}".format(set(current_processes)))
        pbar.close()
    finally:
        for _, proc in current_processes.values():
            proc.kill()

if __name__ == '__main__':
    import fileinput
    run_tasks(list(fileinput.input()))