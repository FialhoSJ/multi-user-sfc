import os
import numpy as np
import argparse                                                                       
from multiprocessing import Pool
from datetime import datetime as dt  

# command line arguments

parser = argparse.ArgumentParser(description='Select MUAR arguments')
parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
parser.add_argument('--alg',   type=str, help='(str) algorithm name', default='dp')
parser.add_argument('--threads',  type=int, help='(int) number of cores to use', default=12)
parser.add_argument('--repetition',  type=str, help='(int) repetitions', default=20)
parser.add_argument('--sfc',   type=str, help='(str) on or off', default='on')
args = parser.parse_args()
 
pool_size = args.threads
repetition = args.repetition

probability = np.linspace(0, 0.5, num=6)
#probability = np.linspace(0, 1, num=11)
probability = np.array(list(probability)[::-1])
#probability = [1]
print(probability)

def run_process(process):                                                             
    os.system('python {}'.format(process)) 
    print(process)

if __name__ == '__main__':
    pool = Pool(processes=pool_size)  
    begin = dt.now()
    cmd = ()
    for i in range(0,repetition):
        for prob in probability:
            if args.sfc == 'on':
                cmd += ('muar.py' + 
                        ' --n_sessions ' + str(args.n_sessions) +  
                        ' --alg ' + str(args.alg) + 
                        ' --sfc on' + 
                        ' --prob ' + str(round(prob,1)),)
            elif args.sfc == 'off':
                cmd += ('muar.py' + 
                        ' --n_sessions ' + str(args.n_sessions) + 
                        ' --alg ' + str(args.alg) + 
                        ' --sfc off' +
                        ' --prob ' + str(round(prob,1)),)
                                                                                    
    print(cmd)
    print()
    pool.map(run_process, cmd)
    duration = dt.now() - begin
    print('Processing time:', duration)
    print('Finished')
