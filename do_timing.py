import os
import sys
import tempfile
import time

TESTNUM = 0
OPTS = \
[
    'USE_COMBINED_KERNEL',
    'USE_CONSTANT_MEMORY',
    'PAGE_LOCK_HOST_BUFFERS',
    'SINGLE_GPU_BUFFER',
    'INPUT_TO_CHAR',
    'USE_STREAMS',
]

PWD = os.getcwd()

RUN_COMMAND = f'./parallel.out -i test/{TESTNUM}/input.ppm -o test/{TESTNUM}/attempt_seq.jpg -t image --parallel '
COMPILE_STEM = 'nvcc compress.cu -L{PWD}}/libwb/build/ -o parallel.out -I {PWD}/libwb/ -std=c++11 -lwb -O3 -g '
with open('sbatch_template') as f:
    SBATCH_TEMPLATE = f.read()
runfile_text = SBATCH_TEMPLATE + '\n' + RUN_COMMAND
runfile = tempfile.NamedTemporaryFile('w')
runfile.write(runfile_text)
run_command = 'sbatch ' + runfile.name

for ii in range(len(OPTS)):
    os.system('rm slurm-*.out')

    opts = OPTS[:ii]
    compile_command = COMPILE_STEM +  ' '.join('-D'+o for o in opts)
    print('\n'.join(opts))

    print(compile_command)
    os.system(compile_command)

    print(run_command)
    os.system(run_command)

    time.sleep(2)

    os.system('cat slurm-*.out')

