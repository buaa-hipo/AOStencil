try:
    import numpy as np
    using_numpy = True
except ImportError:
    from . import my_array as np
    using_numpy = False 

from .kernel_gen import kernel_gen_2d
from .kernel_gen import kernel_gen_3d
from .utils import StencilOPT,NUMA,ceil,MAX_NUM_INSTRUCTION_STREAM

class Stencil2dIR:
  def __init__(self,col:int,row:int,coef:np.ndarray,bias:float,datatype:str='double') -> None:
    kernel_ydim,kernel_xdim=coef.shape
    assert(kernel_ydim==kernel_xdim)
    assert(kernel_ydim%2==1)
    self.datatype=datatype
    self.dim=2
    if datatype=='double':
      self.simd_length= 2  
      self.is_float32=False
    else:
      self.simd_length= 4
      self.is_float32=True

    self.col=col
    self.row=row
    self.xnum=1
    self.ynum=row
    self.coef=coef
    self.bias=bias
    self.edge_length=kernel_ydim//2

    self.stencil_point=np.count_nonzero(coef)
    self.max_num_instruction_stream=min(MAX_NUM_INSTRUCTION_STREAM,self.stencil_point)
    self.stencil_op_kernels=list()
    for num_instruction_stream in range(self.max_num_instruction_stream+1):
      self.stencil_op_kernels.append(kernel_gen_2d(coef,bias,num_instruction_stream,self.is_float32))
    
    self.numa_config=None
    self.opt=None
    self.op_kernel=None

  def set_name(self,name:str):
    self.name=name

  def set_opt(self,simd_usage:int,blockSize:tuple,kernel_unroll_block:bool):
    self.set_OPT(StencilOPT(simd_usage,blockSize,kernel_unroll_block))

  def set_OPT(self,opt:StencilOPT):
    if opt.simd_usage and opt.blockSize!=None:
      assert(opt.blockSize[-1]%self.simd_length==0),f"simd_usage={opt.simd_usage},blockSize={opt.blockSize}"
    self.opt=opt
    self.op_kernel=self.stencil_op_kernels[min(opt.simd_usage,MAX_NUM_INSTRUCTION_STREAM-1)]

  def set_numa_config(self,numa_nodes_num:int,cpus_per_numa_num:int):
    self.numa_config=NUMA(numa_nodes_num,cpus_per_numa_num)

  def set_run_config(self,run_numa_nodes_num,run_cpus_per_numa_num):
    self.run_config=NUMA(run_numa_nodes_num,run_cpus_per_numa_num)
    self.subrow=ceil(self.row-2*self.edge_length,run_cpus_per_numa_num*self.simd_length)*self.simd_length



class Stencil3dIR:
  def __init__(self,zdim:int,ydim:int,xdim:int,coef:np.ndarray,bias:float,datatype:str='double') -> None:
    kernel_zdim,kernel_ydim,kernel_xdim=coef.shape
    assert(kernel_zdim==kernel_ydim)
    assert(kernel_ydim==kernel_xdim)
    assert(kernel_ydim%2==1)
    self.datatype=datatype
    self.dim=3
    
    if datatype=='double':
      self.simd_length= 2  
      self.is_float32=False
    else:
      self.simd_length= 4
      self.is_float32=True

    self.zdim=zdim
    self.ydim=ydim
    self.xdim=xdim
    self.xnum=1
    self.ynum=xdim
    self.znum=ydim*xdim
    self.coef=coef
    self.bias=bias
    self.edge_length=kernel_zdim//2
    
    self.stencil_point=np.count_nonzero(coef)
    self.max_num_instruction_stream=min(MAX_NUM_INSTRUCTION_STREAM,self.stencil_point)
    self.stencil_op_kernels=list()
    for num_instruction_stream in range(self.max_num_instruction_stream+1):
      self.stencil_op_kernels.append(kernel_gen_3d(coef,bias,num_instruction_stream,self.is_float32))

    self.numa_config=None
    self.opt=None

  def set_name(self,name:str):
    self.name=name

  def set_opt(self,simd_usage:int,blockSize:tuple,kernel_unroll_block:bool):
    self.set_OPT(StencilOPT(simd_usage,blockSize,kernel_unroll_block))

  def set_OPT(self,opt:StencilOPT):
    if opt.simd_usage and opt.blockSize!=None:
      assert(opt.blockSize[-1]%self.simd_length==0),f"simd_usage={opt.simd_usage},blockSize={opt.blockSize}"
    self.opt=opt
    self.op_kernel=self.stencil_op_kernels[min(opt.simd_usage,MAX_NUM_INSTRUCTION_STREAM-1)]

  def set_numa_config(self,numa_nodes_num:int,cpus_per_numa_num:int):
    self.numa_config=NUMA(numa_nodes_num,cpus_per_numa_num)

  def set_run_config(self,run_numa_nodes_num,run_cpus_per_numa_num):
    self.run_config=NUMA(run_numa_nodes_num,run_cpus_per_numa_num)
    self.subydim=ceil(self.ydim-2*self.edge_length,run_cpus_per_numa_num)