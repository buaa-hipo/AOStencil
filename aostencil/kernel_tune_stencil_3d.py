from copy import deepcopy
import os
from queue import Queue
import threading
import json
import random
from math import log2,inf
from time import sleep,time
from typing import Tuple

try:
    import numpy as np
    using_numpy = True
except ImportError:
    from . import my_array as np
    using_numpy = False

from .code_files.str_block_test_3d import head_pthread_numa_stencil,func_pthread_numa_stencil,func_join_pthread_numa_stencil,set_pthread_numa_stencil
from .stencil import Stencil3dIR
from .utils import StencilOPT,replace_var,ceil,order2,cmd_run,init_dir,remove_duplicates,check_libnuma,compiler,compile_option
from .for_kenel_gen import pthread_for_kernel_gen_3d


class Pthread_thread_func_arg:
  def __init__(self,thread_id,cpu_id,numa_id,numa_in_id, stencil3d:Stencil3dIR) -> None:
    self.cpu_id=cpu_id
    self.numa_id=numa_id
    self.numa_in_id=numa_in_id
    self.thread_id=thread_id

    edge_len=stencil3d.edge_length

    self.ydim_st=stencil3d.subydim*numa_in_id+edge_len
    self.ydim_ed=min(self.ydim_st+stencil3d.subydim,stencil3d.ydim-edge_len)
    

    pthread_for_kernel=pthread_for_kernel_gen_3d(stencil3d)

    self.pthread_func_var2var={
      'cpu_id':self.cpu_id,
      'numa_id':self.numa_id,
      'numa_in_id':self.numa_in_id,
      'pthread_for_kernel':pthread_for_kernel,
      'ydim_st':self.ydim_st,
      'ydim_ed':self.ydim_ed,
      'subzdim':'zdim'
    }

    self.pthread_set_var2var={
      'cpu_id':self.cpu_id,
      'thread_func':f'thread_func_{self.cpu_id}',
      'thread_id':self.thread_id
    }

  def gen_pthread_thread_func_kernel(self):
    pthread_func_str=func_pthread_numa_stencil
    for key_str,val in self.pthread_func_var2var.items():
      pthread_func_str=replace_var(pthread_func_str,key_str,val)

    pthread_set_str=set_pthread_numa_stencil
    for key_str,val in self.pthread_set_var2var.items():
      pthread_set_str=replace_var(pthread_set_str,key_str,val)
    return pthread_set_str,pthread_func_str



def gen_test_block(stencil3d_test:Stencil3dIR,numa_id):
  assert(stencil3d_test.numa_config!=None)
  assert(stencil3d_test.run_config!=None)

  code_gen=head_pthread_numa_stencil

  head_var2var={
    'stencilName':stencil3d_test.name,
    'datatype':stencil3d_test.datatype,
    'xnum':str(stencil3d_test.xnum)+'LL',
    'ynum':str(stencil3d_test.ynum)+'LL',
    'znum':str(stencil3d_test.znum)+'LL',
    'xdim':stencil3d_test.xdim,
    'ydim':stencil3d_test.ydim,
    'zdim':stencil3d_test.zdim,
    'subydim':stencil3d_test.subydim,
    'edge_length':stencil3d_test.edge_length,
    'num_procs_sublattices':stencil3d_test.run_config.cpus_per_numa_num,
    'num_procs_per_numa':stencil3d_test.numa_config.cpus_per_numa_num
  }

  for key_str,val in head_var2var.items():
    code_gen=replace_var(code_gen,key_str,val)

  pthread_thread_func_arg_list=[]
  for numa_in_id in range(stencil3d_test.run_config.cpus_per_numa_num):
    cpu_id=numa_id*stencil3d_test.numa_config.cpus_per_numa_num+numa_in_id
    pthread_thread_func_arg_list.append(Pthread_thread_func_arg(numa_in_id,cpu_id,numa_id,numa_in_id,stencil3d_test))

  pthread_attr_set_kernel=''
  for pthread_thread_func_arg in pthread_thread_func_arg_list:
    func_set,func=pthread_thread_func_arg.gen_pthread_thread_func_kernel()
    code_gen+='\n'+func
    pthread_attr_set_kernel+=func_set
  

  join_var2var={
    'stencilName':stencil3d_test.name,
    'pthread_attr_set_kernel':pthread_attr_set_kernel
  }
  func_join=func_join_pthread_numa_stencil
  for key_str,val in join_var2var.items():
      func_join=replace_var(func_join,key_str,val)
  
  code_gen+=func_join
  return code_gen

class stencil3d_opt_search:
  def __init__(self,stencil3d:Stencil3dIR,cache_path='search_cache', population_size=50, mutation_rate=0.2,test_time_per_iter=50) -> None:
    self.stencil3d_test=deepcopy(stencil3d)
    self.stencil3d_test.zdim=ceil(self.stencil3d_test.zdim,self.stencil3d_test.run_config.numa_nodes_num)
    self.x_inp=self.stencil3d_test.simd_length
    self.max_dim=(self.stencil3d_test.zdim,self.stencil3d_test.subydim,self.stencil3d_test.xdim//self.x_inp*self.x_inp)
    self.max_num_instruction_stream=self.stencil3d_test.max_num_instruction_stream

    self.cache_path=cache_path
    init_dir(cache_path)

    self.test_time_running_iter=test_time_per_iter
    self.compile_sh=compiler+' '+compile_option+' '
    self.search_space={
      'simd':[i for i in range(self.max_num_instruction_stream)],
      'block_z':[1<<i for i in range(min(int(log2(self.max_dim[0]//2)),10))],
      'block_y':[1<<i for i in range(min(int(log2(self.max_dim[1]//2)),10))],
      'block_x':[1<<i for i in range(int(log2(self.x_inp)),min(int(log2(self.max_dim[2])),10))],
      'block_unroll_z':(None,),
      'block_unroll_y':(None,),
      'block_unroll_x':(None,)
    }
    self.population_size = population_size
    self.mutation_rate = mutation_rate

    self.idle_numa_queue=Queue(maxsize=self.stencil3d_test.numa_config.numa_nodes_num)
    for i in range(self.stencil3d_test.numa_config.numa_nodes_num):
      self.idle_numa_queue.put(i)

    self.run_log={}
    
    self.cost_time_log={
      'search_time':[],
      'best_cost_time':[],
      'each_code_gen_time':[]
    }
  
  def run_opt_test(self,opt:StencilOPT,numa_id):
    gen_code_time_start=time()

    test_stencil=deepcopy(self.stencil3d_test)
    test_stencil.set_OPT(opt)
    opt_test_code=gen_test_block(test_stencil,numa_id)
    filename=self.stencil3d_test.name + opt.to_str()
    filepath=os.path.join(self.cache_path,filename+'.c')

    with open(filepath, 'w', encoding='utf-8') as file:
      file.write(opt_test_code)

    exec_file=os.path.join(self.cache_path,filename+'.exe')
    compile_cmd=self.compile_sh+filepath+' -o '+exec_file
    exec_cmd=exec_file+' '+str(self.test_time_running_iter)

    cmd_run(compile_cmd)
    gen_code_time_end=time()

    exec_time=cmd_run(exec_cmd)

    return float(exec_time),gen_code_time_end-gen_code_time_start


  def search_iter(self,opt:StencilOPT,idle_numa_id:int)->float:
    exec_time,gen_time=self.run_opt_test(opt,idle_numa_id)
    self.run_log[opt.to_str()]=exec_time # lower fitness for lower execution time
    self.cost_time_log['each_code_gen_time'].append(gen_time)
    self.idle_numa_queue.put(idle_numa_id)

  #EA algoithm part

  def init_population(self):
    population = []

    size_per_block_x=8
    for _ in range(size_per_block_x):
      block_z = random.choice(self.search_space['block_z'])
      block_y = random.choice(self.search_space['block_y'])
      block_unroll_z = random.choice(self.search_space['block_unroll_z'])
      block_unroll_y = random.choice(self.search_space['block_unroll_y'])
      block_unroll_x = random.choice(self.search_space['block_unroll_x'])
      simd = random.choice(self.search_space['simd'])
      for block_x in self.search_space['block_x']:
        population.append(StencilOPT(simd, (block_z, block_y, block_x), (block_unroll_z,block_unroll_y, block_unroll_x)))
    return population

  def init_standard_population(self):
    population = []

    for simd in self.search_space['simd']:
      population.append(StencilOPT(simd, None, None))
    return population


  def select(self, population):
    sorted_population = sorted(population, key=lambda x: self.run_log[x.to_str()], reverse=False)
    return sorted_population[:self.population_size//4]  # Keep the best quarter

  def crossover(self, parent1, parent2)->StencilOPT:
    # Handle None values for blockSize
    p1_blockSize = parent1.blockSize or (1, 1, self.x_inp)
    p2_blockSize = parent2.blockSize or (1, 1, self.x_inp)
    
    # Handle potential None values for kernel_unroll_block
    p1_unroll = parent1.kernel_unroll_block or (1, 1, 1)
    p2_unroll = parent2.kernel_unroll_block or (1, 1, 1)
    
    simd=random.choice((parent1.simd_usage,parent2.simd_usage))
    block_unroll_z=random.choice((p1_unroll[0], p2_unroll[0]))
    block_unroll_y=random.choice((p1_unroll[1], p2_unroll[1])) 
    block_unroll_x=random.choice((p1_unroll[2], p2_unroll[2]))
    
    # Handle None values in blockSize elements
    min_z,max_z=order2(p1_blockSize[0], p2_blockSize[0])
    block_z=random.randint(min_z,max_z)
        
    min_y,max_y=order2(p1_blockSize[1], p2_blockSize[1])
    block_y=random.randint(min_y,max_y)
        
    min_x,max_x=order2(p1_blockSize[2], p2_blockSize[2])
    block_x=random.choice(tuple(range(min_x,max_x+1,self.x_inp)))

    if random.random() < self.mutation_rate:
        return self.mutate(StencilOPT(simd, (block_z, block_y, block_x), (block_unroll_z, block_unroll_y, block_unroll_x)))

    return StencilOPT(simd, (block_z, block_y, block_x), (block_unroll_z, block_unroll_y, block_unroll_x))

  def mutate(self, individual)->StencilOPT:
      mutate_type=random.randint(0,3)
      if mutate_type == 0:
        individual.simd=random.choice(self.search_space['simd'])
      elif mutate_type == 1:
        individual.blockSize=(random.choice(self.search_space['block_z']),individual.blockSize[1],individual.blockSize[2])
      elif mutate_type == 2:
        individual.blockSize=(individual.blockSize[0],random.choice(self.search_space['block_y']),individual.blockSize[2])
      elif mutate_type == 3:
        individual.blockSize=(individual.blockSize[0],individual.blockSize[1],random.choice(self.search_space['block_x']))
      elif mutate_type == 4:
        individual.kernel_unroll_block=(individual.kernel_unroll_block[0],individual.kernel_unroll_block[1],random.choice(self.search_space['block_unroll_x']))
      
      return individual

  def search(self,record_log=False)->Tuple[StencilOPT,float]:
    
    population=[]
    offspring=self.init_population()

    best_cost_time=inf
    tolerance=max_tolerance = 3  # maximum no progress tolerance

    tune_start_time=time()

    while tolerance>=0:
      # remove dup
      if tolerance>0:
        unique_offspring=remove_duplicates(offspring)
      else:
        unique_offspring=self.init_standard_population()
      # Evaluation
      threads = []  # List to hold the threads
      for individual in unique_offspring:
        idle_numa_id = self.idle_numa_queue.get(block=True)
        thread = threading.Thread(target=self.search_iter, args=(individual,idle_numa_id,))
        threads.append(thread)
        thread.start()
      
      # Wait for all threads to complete
      for thread in threads:
        thread.join()

      # Replace the old population
      population += unique_offspring

      # Selection
      population = self.select(population)
      
      if tolerance:
        # Crossover
        # Mutation is in Crossover
        offspring = [self.crossover(random.choice(population), random.choice(population)) for _ in range(self.population_size - len(population))]

      best_cost_time_this_eval=self.run_log[population[0].to_str()]
      if best_cost_time>best_cost_time_this_eval:
        tolerance=max_tolerance
        best_cost_time=best_cost_time_this_eval
      else:
        tolerance-=1

      tune_now_time=time()
      self.cost_time_log['best_cost_time'].append(best_cost_time)
      self.cost_time_log['search_time'].append(tune_now_time-tune_start_time)

    if record_log:
      with open(f'{self.stencil3d_test.name}_run.json', 'w') as file:
        json.dump(self.run_log, file, indent=4)

      with open(f'{self.stencil3d_test.name}_process.json', 'w') as file:
        json.dump(self.cost_time_log, file, indent=4)
    return population[0],self.run_log[population[0].to_str()]


def kernel_tune_stencil_3d(stencil3d:Stencil3dIR,cache_path='build/search_cache',record_log=False)->Stencil3dIR:
  if not check_libnuma():
    raise EnvironmentError("No libnuma, add it to library path")
  s_opt_search=stencil3d_opt_search(stencil3d,cache_path,population_size=200,mutation_rate=0.2,test_time_per_iter=20)
  opt,best_opt_time=s_opt_search.search(record_log=record_log)
  stencil3d.set_OPT(opt)
  return stencil3d