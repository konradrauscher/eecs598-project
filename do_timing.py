import os
import sys
import tempfile
import time
import subprocess

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

NOPTS = len(OPTS)
PWD = os.getcwd()

RUN_COMMAND = f'./parallel.out.%d -i test/{TESTNUM}/input.ppm -o test/{TESTNUM}/attempt_seq.jpg -t image --parallel '
COMPILE_STEM = f'nvcc compress.cu -L{PWD}/libwb/build/ -o parallel.out.%d -I {PWD}/libwb/ -std=c++11 -lwb -lnvjpeg -O3 -g '
with open('sbatch_template') as f:
    SBATCH_TEMPLATE = f.read()
#runfile_text = SBATCH_TEMPLATE + '\n' + RUN_COMMAND
runfiles = [tempfile.NamedTemporaryFile('w') for ii in range(NOPTS+1)]
#with open('timing','w') as f:
#    f.write(runfile_text)
#run_command = 'sbatch timing'

all_opts = []
output_files = []


for ii in range(NOPTS+1):
    opts = OPTS[:ii]
    compile_command = (COMPILE_STEM%ii) +  ' '.join('-D'+o for o in opts)
    print('\n'.join(opts))

    print(compile_command)
    subprocess.run(compile_command.split())

    runfiles[ii].write(SBATCH_TEMPLATE + '\n' + (RUN_COMMAND%ii))
    runfiles[ii].flush()    
 
    run_command = 'sbatch ' + runfiles[ii].name
    
    print(run_command)
    p = subprocess.run(run_command.split(),capture_output=True)
    print(p.stdout.decode())
    print(p.stderr.decode())
    p.check_returncode()
    job = p.stdout.decode().split()[3]

    all_opts.append(opts)
    output_files.append(f'slurm-{job}.out')

print('waiting for batch jobs to complete...')
while not os.path.exists(output_files[-1]):
  time.sleep(1)

for opts, outfile in zip(all_opts, output_files):
    print("Optimizations: " + ' '.join(opts))
    with open(outfile) as f:
        print(f.read())
    os.remove(outfile)
