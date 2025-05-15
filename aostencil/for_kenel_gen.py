from .stencil import Stencil2dIR,Stencil3dIR
from .utils import replace_var
from .code_files import str_for_kenel_2d,str_for_kenel_3d

def pthread_for_kernel_gen_2d(stencil2d:Stencil2dIR):
  assert(stencil2d.opt!=None)
  kernel=stencil2d.op_kernel
  if stencil2d.opt.simd_usage==0:
    x_inp=1
  else:
    x_inp=stencil2d.simd_length
  #simd length align with for loop length
  assert((stencil2d.row-2*stencil2d.edge_length)%x_inp==0)
  if stencil2d.opt.blockSize != None:
    res=replace_var(str_for_kenel_2d.pthread_for_kernel_block,'x_inp',x_inp)
    res=replace_var(res,'blockSize_y',stencil2d.opt.blockSize[0])
    res=replace_var(res,'blockSize_x',stencil2d.opt.blockSize[1])

  else:
    res=replace_var(str_for_kenel_2d.pthread_for_kernel,'x_inp',x_inp)


  unroll_size=["",""]

  if stencil2d.opt.kernel_unroll_block!=False and stencil2d.opt.kernel_unroll_block!=None:
    if stencil2d.opt.kernel_unroll_block[0]!=None:
      unroll_size[0]=f"#pragma unroll {stencil2d.opt.kernel_unroll_block[0]}"
    if stencil2d.opt.kernel_unroll_block[1]!=None:
      unroll_size[1]=f"#pragma unroll {stencil2d.opt.kernel_unroll_block[1]}"

  res=replace_var(res,'block_y_unroll',unroll_size[0])
  res=replace_var(res,'block_x_unroll',unroll_size[1])

  res=replace_var(res,'kernel',kernel)
  return res

def pthread_for_kernel_gen_3d(stencil3d:Stencil3dIR):
  assert(stencil3d.opt!=None)
  kernel=stencil3d.op_kernel
  if stencil3d.opt.simd_usage==0:
    x_inp=1
  else:
    x_inp=stencil3d.simd_length
  #simd length align with for loop length
  assert((stencil3d.xdim-2*stencil3d.edge_length)%x_inp==0)
  if stencil3d.opt.blockSize != None:
    res=replace_var(str_for_kenel_3d.pthread_for_kernel_block,'x_inp',x_inp)
    res=replace_var(res,'blockSize_z',stencil3d.opt.blockSize[0])
    res=replace_var(res,'blockSize_y',stencil3d.opt.blockSize[1])
    res=replace_var(res,'blockSize_x',stencil3d.opt.blockSize[2])

  else:
    res=replace_var(str_for_kenel_3d.pthread_for_kernel,'x_inp',x_inp)



  unroll_size=["","",""]
  if stencil3d.opt.kernel_unroll_block !=None and stencil3d.opt.kernel_unroll_block !=False:
    if stencil3d.opt.kernel_unroll_block[0]!=None:
      unroll_size[0]=f"#pragma unroll {stencil3d.opt.kernel_unroll_block[0]}"
    if stencil3d.opt.kernel_unroll_block[1]!=None:
      unroll_size[1]=f"#pragma unroll {stencil3d.opt.kernel_unroll_block[1]}"
    if stencil3d.opt.kernel_unroll_block[2]!=None:
      unroll_size[2]=f"#pragma unroll {stencil3d.opt.kernel_unroll_block[2]}"

  res=replace_var(res,'block_z_unroll',unroll_size[0])
  res=replace_var(res,'block_y_unroll',unroll_size[1])
  res=replace_var(res,'block_x_unroll',unroll_size[2])
  res=replace_var(res,'kernel',kernel)
  return res