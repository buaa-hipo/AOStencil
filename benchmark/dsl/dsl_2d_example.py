import os
from aostencil import from_dsl_load_stencil
from aostencil import gen_stencil_2d,kernel_tune_stencil_2d

if __name__ == '__main__':
  filename = '2d9pt_box.dsl'
  with open(filename, 'r') as file:
    source_code = file.read()

  s = from_dsl_load_stencil(source_code)
  s.set_numa_config(16,8)
  s.set_run_config(16,8)

  s = kernel_tune_stencil_2d(s)
  code_gen = gen_stencil_2d(s)
  flow_cache = os.path.join('build', '2d')
  os.makedirs(flow_cache, exist_ok=True)
  file_path = os.path.join(flow_cache, s.name + '.c')
  with open(file_path, 'w', encoding='utf-8') as file:
      file.write(code_gen)

