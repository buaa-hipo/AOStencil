try:
    import numpy as np
    using_numpy = True
except ImportError:
    from . import my_array as np
    using_numpy = False

def make_k2locatiom_2d(coef:np.ndarray)->dict:
  ydim,xdim=coef.shape
  assert(ydim==xdim)
  assert(ydim%2==1)
  center=(ydim//2,xdim//2)
  k2location={}
  for y in range(ydim):
    for x in range(xdim):
      if coef[y,x]!=0:
        offset=""
        offset_y=y-center[0]
        offset_x=x-center[1]
        if offset_y>0:
          if offset_y==1:
            offset+=f'+ynum'
          else:
            offset+=f'+{offset_y}*ynum'
        elif offset_y<0:
          if offset_y==-1:
            offset+=f'-ynum'
          else:
            offset+=f'{offset_y}*ynum'

        if offset_x>0:
          if offset_x==1:
            offset+=f'+xnum'
          else:
            offset+=f'+{offset_x}*xnum'
        elif offset_x<0:
          if offset_x==-1:
            offset+=f'-xnum'
          else:
            offset+=f'{offset_x}*xnum'
        k2location[(y,x)]=offset
  return k2location


def make_k2locatiom_3d(coef:np.ndarray)->dict:
  kernel_zdim,kernel_ydim,kernel_xdim=coef.shape
  assert(kernel_zdim==kernel_ydim)
  assert(kernel_ydim==kernel_xdim)
  assert(kernel_ydim%2==1)
  center=(kernel_zdim//2,kernel_ydim//2,kernel_xdim//2)
  k2location={}
  for z in range(kernel_zdim):
    for y in range(kernel_ydim):
      for x in range(kernel_xdim):
        if coef[z,y,x]!=0:
          offset=""
          offset_z=z-center[0]
          offset_y=y-center[1]
          offset_x=x-center[2]
          
          if offset_z>0:
            if offset_z==1:
              offset+=f'+znum'
            else:
              offset+=f'+{offset_z}*znum'
          elif offset_z<0:
            if offset_z==-1:
              offset+=f'-znum'
            else:
              offset+=f'{offset_z}*znum'

          if offset_y>0:
            if offset_y==1:
              offset+=f'+ynum'
            else:
              offset+=f'+{offset_y}*ynum'
          elif offset_y<0:
            if offset_y==-1:
              offset+=f'-ynum'
            else:
              offset+=f'{offset_y}*ynum'

          if offset_x>0:
            if offset_x==1:
              offset+=f'+xnum'
            else:
              offset+=f'+{offset_x}*xnum'
          elif offset_x<0:
            if offset_x==-1:
              offset+=f'-xnum'
            else:
              offset+=f'{offset_x}*xnum'
          k2location[(z,y,x)]=offset
  return k2location


def build_add_tree_variable(toAdd:list,addFunc:str):
    # Ensure the input list has more than one element
    if len(toAdd) < 2:
        return str(toAdd[0]) if toAdd else ""
    
    while len(toAdd) > 1:
        new_list = []
        for i in range(0, len(toAdd)-1, 2):
            new_list.append(f"{addFunc}({toAdd[i]}, {toAdd[i+1]})")
        if len(toAdd) % 2 == 1:
            new_list.append(toAdd[-1])
        toAdd = new_list
    return toAdd[0]

def kernel_gen_2d_native(coef:np.ndarray,bias:float,is_float32:bool)->str:
  '''
  In stencil computation,coef[]*G[]+b[]=new_G[]
  '''
  kernel='latticeNext[center]='
  k2location=make_k2locatiom_2d(coef)
  if is_float32:
    for k_t,offset in k2location.items():
      kernel+=f"\n\t\t{coef[k_t[0],k_t[1]]}f*lattice[center{offset}]+"
    if bias ==0.:
      kernel=kernel[:-1]+';'
    else:
      kernel+=str(bias)+'f;'
  else:
    for k_t,offset in k2location.items():
      kernel+=f"\n\t\t{coef[k_t[0],k_t[1]]}*lattice[center{offset}]+"
    if bias ==0.:
      kernel=kernel[:-1]+';'
    else:
      kernel+=str(bias)+';'
  return kernel

def kernel_gen_2d(coef:np.ndarray,bias:float,num_instruction_stream:int,is_float32:bool)->str:
  if num_instruction_stream == 0:
    return kernel_gen_2d_native(coef,bias,is_float32)
  kernel=''
  k2location=make_k2locatiom_2d(coef)
  kc=[]
  num_instruction_stream=min(len(k2location),num_instruction_stream)
  first=[True]*num_instruction_stream
  instructionStreamNow=0
  kc=[]
  if is_float32:
    for k_t,offset in k2location.items():
      if first[instructionStreamNow]:
        if bias!=0:
          kernel+='float32x4_t  v{}_{} = vmlaq_f32(vdupq_n_f32({}f),vld1q_f32(&lattice[center{}]), vdupq_n_f32({}f));\n'.format(k_t[0],k_t[1],bias,offset,coef[k_t[0],k_t[1]])
          bias=0
        else:
          kernel+='float32x4_t v{}_{} = vmulq_n_f32(vld1q_f32(&lattice[center{}]), {}f);\n'.format(k_t[0],k_t[1],offset,coef[k_t[0],k_t[1]])
        kc.append('v{}_{}'.format(k_t[0],k_t[1]))
        first[instructionStreamNow]=False
      else:
        kernel+='{} = vmlaq_f32({}, vld1q_f32(&lattice[center{}]), vdupq_n_f32({}f));\n'.format(kc[instructionStreamNow],kc[instructionStreamNow],offset,coef[k_t[0],k_t[1]])
      if instructionStreamNow == num_instruction_stream-1:
        instructionStreamNow=0
      else:
        instructionStreamNow+=1
    addTree=build_add_tree_variable(kc,"vaddq_f32")
    if len(kc)>1:
      kernel+='{}={};\n'.format(kc[0],addTree)
    kernel+='vst1q_f32(&latticeNext[center], {});\n'.format(kc[0])

  else:
    for k_t,offset in k2location.items():
      if first[instructionStreamNow]:
        if bias!=0:
          kernel+='float64x2_t v{}_{} = vmlaq_f64(vdupq_n_f64({}),vld1q_f64(&lattice[center{}]), vdupq_n_f64({}));\n'.format(k_t[0],k_t[1],bias,offset,coef[k_t[0],k_t[1]])
          bias=0
        else:
          kernel+='float64x2_t v{}_{} = vmulq_n_f64(vld1q_f64(&lattice[center{}]), {});\n'.format(k_t[0],k_t[1],offset,coef[k_t[0],k_t[1]])
        kc.append('v{}_{}'.format(k_t[0],k_t[1]))
        first[instructionStreamNow]=False
      else:
        kernel+='{} = vmlaq_f64({}, vld1q_f64(&lattice[center{}]), vdupq_n_f64({}));\n'.format(kc[instructionStreamNow],kc[instructionStreamNow],offset,coef[k_t[0],k_t[1]])
      if instructionStreamNow == num_instruction_stream-1:
        instructionStreamNow=0
      else:
        instructionStreamNow+=1
    addTree=build_add_tree_variable(kc,"vaddq_f64")
    if len(kc)>1:
      kernel+='{}={};\n'.format(kc[0],addTree)
    kernel+='vst1q_f64(&latticeNext[center], {});\n'.format(kc[0])

  return kernel



def kernel_gen_3d_native(coef:np.ndarray,bias:float,is_float32:bool)->str:
  '''
  In stencil computation,coef[]*G[]+b[]=new_G[]
  '''
  kernel='latticeNext[center]='
  k2location=make_k2locatiom_3d(coef)
  if is_float32:
    for k_t,offset in k2location.items():
      kernel+=f"\n\t\t{coef[k_t[0],k_t[1],k_t[2]]}f*lattice[center{offset}]+"
    if bias ==0.:
      kernel=kernel[:-1]+';'
    else:
      kernel+=str(bias)+'f;'
  else:
    for k_t,offset in k2location.items():
      kernel+=f"\n\t\t{coef[k_t[0],k_t[1],k_t[2]]}*lattice[center{offset}]+"
    if bias ==0.:
      kernel=kernel[:-1]+';'
    else:
      kernel+=str(bias)+';'
  return kernel


def kernel_gen_3d(coef:np.ndarray,bias:float,num_instruction_stream:int,is_float32:bool)->str:
  if num_instruction_stream == 0:
    return kernel_gen_3d_native(coef,bias,is_float32)
  kernel=''
  k2location=make_k2locatiom_3d(coef)
  kc=[]
  num_instruction_stream=min(len(k2location),num_instruction_stream)
  first=[True]*num_instruction_stream
  instructionStreamNow=0
  kc=[]
  if is_float32:
    for k_t,offset in k2location.items():
      if first[instructionStreamNow]:
        if bias!=0:
          kernel+='float32x4_t v{}_{}_{} = vmlaq_f32(vdupq_n_f32({}f),vld1q_f32(&lattice[center{}]), vdupq_n_f32({}f));\n'.format(k_t[0],k_t[1],k_t[2],bias,offset,coef[k_t[0],k_t[1],k_t[2]])
          bias=0
        else:
          kernel+='float32x4_t v{}_{}_{} = vmulq_n_f32(vld1q_f32(&lattice[center{}]), {}f);\n'.format(k_t[0],k_t[1],k_t[2],offset,coef[k_t[0],k_t[1],k_t[2]])
        kc.append('v{}_{}_{}'.format(k_t[0],k_t[1],k_t[2]))
        first[instructionStreamNow]=False
      else:
        kernel+='{} = vmlaq_f32({}, vld1q_f32(&lattice[center{}]), vdupq_n_f32({}f));\n'.format(kc[instructionStreamNow],kc[instructionStreamNow],offset,coef[k_t[0],k_t[1],k_t[2]])
      if instructionStreamNow == num_instruction_stream-1:
        instructionStreamNow=0
      else:
        instructionStreamNow+=1
    addTree=build_add_tree_variable(kc,"vaddq_f32")
    if len(kc)>1:
      kernel+='{}={};\n'.format(kc[0],addTree)
    kernel+='vst1q_f32(&latticeNext[center], {});\n'.format(kc[0])
  else:
    for k_t,offset in k2location.items():
      if first[instructionStreamNow]:
        if bias!=0:
          kernel+='float64x2_t v{}_{}_{} = vmlaq_f64(vdupq_n_f64({}),vld1q_f64(&lattice[center{}]), vdupq_n_f64({}));\n'.format(k_t[0],k_t[1],k_t[2],bias,offset,coef[k_t[0],k_t[1],k_t[2]])
          bias=0
        else:
          kernel+='float64x2_t v{}_{}_{} = vmulq_n_f64(vld1q_f64(&lattice[center{}]), {});\n'.format(k_t[0],k_t[1],k_t[2],offset,coef[k_t[0],k_t[1],k_t[2]])
        kc.append('v{}_{}_{}'.format(k_t[0],k_t[1],k_t[2]))
        first[instructionStreamNow]=False
      else:
        kernel+='{} = vmlaq_f64({}, vld1q_f64(&lattice[center{}]), vdupq_n_f64({}));\n'.format(kc[instructionStreamNow],kc[instructionStreamNow],offset,coef[k_t[0],k_t[1],k_t[2]])
      if instructionStreamNow == num_instruction_stream-1:
        instructionStreamNow=0
      else:
        instructionStreamNow+=1
    addTree=build_add_tree_variable(kc,"vaddq_f64")
    if len(kc)>1:
      kernel+='{}={};\n'.format(kc[0],addTree)
    kernel+='vst1q_f64(&latticeNext[center], {});\n'.format(kc[0])

  return kernel
