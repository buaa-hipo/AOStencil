try:
    import numpy as np
    using_numpy = True
except ImportError:
    from . import my_array as np
    using_numpy = False

from .for_kenel_gen import pthread_for_kernel_gen_3d
from .code_files.str_pthread_numa_stencil_3d import head_pthread_numa_stencil,func_pthread_numa_stencil, func_join_pthread_numa_stencil,set_pthread_numa_stencil
from .stencil import Stencil3dIR
from .utils import replace_var,ceil


class Pthread_thread_func_arg:
  def __init__(self,thread_id,cpu_id,numa_id,numa_in_id, stencil3d:Stencil3dIR) -> None:
    self.cpu_id=cpu_id
    self.numa_id=numa_id
    self.numa_in_id=numa_in_id
    self.thread_id=thread_id

    edge_len=stencil3d.edge_length

    self.ydim_st=stencil3d.subydim*numa_in_id+edge_len
    self.ydim_ed=min(self.ydim_st+stencil3d.subydim,stencil3d.ydim-edge_len)
    self.avg_subzdim=ceil(stencil3d.zdim-2 * edge_len,stencil3d.run_config.numa_nodes_num)

    if stencil3d.run_config.numa_nodes_num==1:
      self.subzdim=stencil3d.zdim
      self.top_edge_index=-1
      self.bottom_edge_index=-1
    else:
      if numa_id == 0:
        self.subzdim=self.avg_subzdim+edge_len
      elif numa_id == stencil3d.run_config.numa_nodes_num-1:
        self.subzdim=stencil3d.zdim-(stencil3d.run_config.numa_nodes_num-1)*self.avg_subzdim+edge_len
      else:
        self.subzdim=self.avg_subzdim+2*edge_len
      
      if numa_id == 0:
        self.top_edge_index=-1
      elif numa_id == 1:
        self.top_edge_index=(self.avg_subzdim-edge_len)*stencil3d.znum
      else: 
        self.top_edge_index=self.avg_subzdim*stencil3d.znum

      if numa_id == stencil3d.run_config.numa_nodes_num-1:
        self.bottom_edge_index=-1
      else:
        self.bottom_edge_index=edge_len*stencil3d.znum

    pthread_for_kernel=pthread_for_kernel_gen_3d(stencil3d)

    self.pthread_func_var2var={
      'cpu_id':self.cpu_id,
      'numa_id':self.numa_id,
      'numa_in_id':self.numa_in_id,
      'pthread_for_kernel':pthread_for_kernel,
      'ydim_st':self.ydim_st,
      'ydim_ed':self.ydim_ed,
      'subzdim':self.subzdim,
      'avg_subzdim':self.avg_subzdim,
      'top_edge_index':self.top_edge_index,
      'bottom_edge_index':self.bottom_edge_index
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

def gen_pthread_thread_func_arg_list(stencil3d:Stencil3dIR)->tuple:
  threads_nums=stencil3d.run_config.cpus_per_numa_num*stencil3d.run_config.numa_nodes_num
  pthread_thread_func_arg_list=[]
  for i in range(threads_nums):
    numa_id=i//stencil3d.run_config.cpus_per_numa_num
    numa_in_id=i%stencil3d.run_config.cpus_per_numa_num
    cpu_id=numa_id*stencil3d.numa_config.cpus_per_numa_num+numa_in_id

    pthread_thread_func_arg_list.append(Pthread_thread_func_arg(i,cpu_id,numa_id,numa_in_id,stencil3d))

  return pthread_thread_func_arg_list

def gen_stencil_pthread_3d(stencil3d:Stencil3dIR)->str:
  code_gen=head_pthread_numa_stencil

  head_var2var={
    'stencilName':stencil3d.name,
    'datatype':stencil3d.datatype,
    'xnum':str(stencil3d.xnum)+'LL',
    'ynum':str(stencil3d.ynum)+'LL',
    'znum':str(stencil3d.znum)+'LL',
    'xdim':stencil3d.xdim,
    'ydim':stencil3d.ydim,
    'zdim':stencil3d.zdim,
    'subydim':stencil3d.subydim,
    'edge_length':stencil3d.edge_length,
    'num_sublattices':stencil3d.run_config.numa_nodes_num,
    'num_procs_sublattices':stencil3d.run_config.cpus_per_numa_num,
    'num_procs_per_numa':stencil3d.numa_config.cpus_per_numa_num
  }

  for key_str,val in head_var2var.items():
    code_gen=replace_var(code_gen,key_str,val)

  pthread_attr_set_kernel=''
  pthread_thread_func_arg_list=gen_pthread_thread_func_arg_list(stencil3d)
  for pthread_thread_func_arg in pthread_thread_func_arg_list:
    func_set,func=pthread_thread_func_arg.gen_pthread_thread_func_kernel()
    code_gen+='\n'+func
    pthread_attr_set_kernel+=func_set

  join_var2var={
    'stencilName':stencil3d.name,
    'pthread_attr_set_kernel':pthread_attr_set_kernel
  }
  func_join=func_join_pthread_numa_stencil
  for key_str,val in join_var2var.items():
      func_join=replace_var(func_join,key_str,val)
  
  code_gen+=func_join
  return code_gen
