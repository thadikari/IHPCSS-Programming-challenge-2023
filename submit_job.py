from pathlib import Path
import argparse
import os


'''
Use the following command to scan files and print iterations.
for name in *.txt; do sed '4q;d' $name; echo "       $name"; done
'''


def main():
    for OMP_NUM_THREADS in _a.OMP_NUM_THREADS:
        for MPI_PROCESS_COUNT in _a.MPI_PROCESS_COUNT:
            for NUM_GPU in _a.NUM_GPU:
                run(OMP_NUM_THREADS, MPI_PROCESS_COUNT, NUM_GPU)


def run(OMP_NUM_THREADS, MPI_PROCESS_COUNT, NUM_GPU):

    dev = 'cpu' if _a.cpu else 'gpu'
    base_name = f'submit_{dev}_node.slurm'
    with open(base_name, 'r') as fp:
        cc = fp.read()

    runs_dir = os.path.expanduser('~') + '/runs'
    Path(f'{runs_dir}/outputs').mkdir(parents=True, exist_ok=True)
    prefix = f'{OMP_NUM_THREADS}_{MPI_PROCESS_COUNT}_{NUM_GPU}_{dev}'

    if OMP_NUM_THREADS:
        new = f'export OMP_NUM_THREADS={OMP_NUM_THREADS};'
        cc = cc.replace('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK', new)
        cc = cc.replace('export OMP_NUM_THREADS=1;', new)
        new = f'#SBATCH --cpus-per-task={OMP_NUM_THREADS}'
        cc = cc.replace('#SBATCH --cpus-per-task=2', new)

    if OMP_NUM_THREADS:
        cc = cc.replace('export MPI_PROCESS_COUNT=1;', f'export MPI_PROCESS_COUNT={MPI_PROCESS_COUNT};')

    if NUM_GPU:
        cc = cc.replace('#SBATCH --gres=gpu:v100-16:1', f'#SBATCH --gres=gpu:v100-16:{NUM_GPU}')

    new_name = f'{prefix}_{base_name}'
    new_path = f'{runs_dir}/{new_name}'
    with open(new_path, 'w') as fp: fp.write(cc)

    out_pattern = f'{runs_dir}/outputs/{prefix}_output.txt'
    try: os.system(f'sbatch --output {out_pattern} {new_path}')
    except: pass


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NUM_GPU', type=int, nargs='+', default=[None])
    parser.add_argument('--OMP_NUM_THREADS', type=int, nargs='+', default=[None])
    parser.add_argument('--MPI_PROCESS_COUNT', type=int, nargs='+', default=[None])
    parser.add_argument('--cpu', action='store_true')
    return parser


if __name__ == '__main__':
    _a = make_parser().parse_args()
    print('[Arguments]', vars(_a))
    main()
