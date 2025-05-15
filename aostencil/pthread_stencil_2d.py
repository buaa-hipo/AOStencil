try:
    import numpy as np
    using_numpy = True
except ImportError:
    from . import my_array as np
    using_numpy = False

from .for_kenel_gen import pthread_for_kernel_gen_2d
from .code_files.str_pthread_numa_stencil_2d import head_pthread_numa_stencil,func_pthread_numa_stencil, func_join_pthread_numa_stencil,set_pthread_numa_stencil
from .stencil import Stencil2dIR
from .utils import replace_var,ceil
    
class Pthread_thread_func_arg:
  def __init__(self,thread_id,cpu_id,numa_id,numa_in_id, stencil2d:Stencil2dIR) -> None:
    self.cpu_id=cpu_id
    self.numa_id=numa_id
    self.numa_in_id=numa_in_id
    self.thread_id=thread_id

    edge_len=stencil2d.edge_length

    self.row_st=stencil2d.subrow*numa_in_id+edge_len
    self.row_ed=min(self.row_st+stencil2d.subrow,stencil2d.row-edge_len)
    self.avg_subcol=ceil(stencil2d.col-2*edge_len,stencil2d.run_config.numa_nodes_num)

    if stencil2d.run_config.numa_nodes_num==1:
      self.subcol=stencil2d.col
      self.top_edge_index=-1
      self.bottom_edge_index=-1
    else:
      if numa_id == 0:
        self.subcol=self.avg_subcol+edge_len
      elif numa_id == stencil2d.run_config.numa_nodes_num-1:
        self.subcol=stencil2d.col-(stencil2d.run_config.numa_nodes_num-1)*self.avg_subcol+edge_len
      else:
        self.subcol=self.avg_subcol+2*edge_len

      if numa_id == 0:
        self.top_edge_index=-1
      elif numa_id == 1:
        self.top_edge_index=(self.avg_subcol-edge_len)*stencil2d.ynum
      else: 
        self.top_edge_index=self.avg_subcol*stencil2d.ynum

      if numa_id == stencil2d.run_config.numa_nodes_num-1:
        self.bottom_edge_index=-1
      else:
        self.bottom_edge_index=edge_len*stencil2d.ynum

    pthread_for_kernel=pthread_for_kernel_gen_2d(stencil2d)

    self.pthread_func_var2var={
      'cpu_id':self.cpu_id,
      'numa_id':self.numa_id,
      'numa_in_id':self.numa_in_id,
      'pthread_for_kernel':pthread_for_kernel,
      'row_st':self.row_st,
      'row_ed':self.row_ed,
      'avg_subcol':self.avg_subcol,
      'subcol':self.subcol,
      'top_edge_index':str(self.top_edge_index)+'LL',
      'bottom_edge_index':str(self.bottom_edge_index)+'LL'
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

def gen_pthread_thread_func_arg_list(stencil2d:Stencil2dIR)->tuple:
  threads_nums=stencil2d.run_config.cpus_per_numa_num*stencil2d.run_config.numa_nodes_num
  pthread_thread_func_arg_list=[]
  for i in range(threads_nums):
    numa_id=i//stencil2d.run_config.cpus_per_numa_num
    numa_in_id=i%stencil2d.run_config.cpus_per_numa_num
    cpu_id=numa_id*stencil2d.numa_config.cpus_per_numa_num+numa_in_id

    pthread_thread_func_arg_list.append(Pthread_thread_func_arg(i,cpu_id,numa_id,numa_in_id,stencil2d))

  return pthread_thread_func_arg_list

def gen_stencil_pthread_2d(stencil2d:Stencil2dIR)->str:
  code_gen=head_pthread_numa_stencil

  head_var2var={
    'datatype':stencil2d.datatype,
    'xnum':str(stencil2d.xnum)+'LL',
    'ynum':str(stencil2d.ynum)+'LL',
    'row':stencil2d.row,
    'col':stencil2d.col,
    'subrow':stencil2d.subrow,
    'edge_length':stencil2d.edge_length,
    'num_sublattices':stencil2d.run_config.numa_nodes_num,
    'num_procs_sublattices':stencil2d.run_config.cpus_per_numa_num,
    'num_procs_per_numa':stencil2d.numa_config.cpus_per_numa_num,
    }

  for key_str,val in head_var2var.items():
    code_gen=replace_var(code_gen,key_str,val)

  pthread_attr_set_kernel=''
  pthread_thread_func_arg_list=gen_pthread_thread_func_arg_list(stencil2d)
  for pthread_thread_func_arg in pthread_thread_func_arg_list:
    func_set,func=pthread_thread_func_arg.gen_pthread_thread_func_kernel()
    code_gen+='\n'+func
    pthread_attr_set_kernel+=func_set

  join_var2var={
    'stencilName':stencil2d.name,
    'pthread_attr_set_kernel':pthread_attr_set_kernel
  }
  func_join=func_join_pthread_numa_stencil
  for key_str,val in join_var2var.items():
      func_join=replace_var(func_join,key_str,val)
  
  code_gen+=func_join
  return code_gen

