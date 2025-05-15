
import os

from aostencil import np
from aostencil import Stencil2dIR,gen_stencil_2d,kernel_tune_stencil_2d
from aostencil import Stencil3dIR,gen_stencil_3d,kernel_tune_stencil_3d
from aostencil import cmd_run,compiler,compile_option,init_dir
from str_main import rand_main_2d,rand_main_3d


def phytium_test(stencil_shape, sname:str,log_file_path):
    # Open log file in append mode
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        if len(stencil_shape) == 5:
            log_file.write('-'*10 + sname + '-'*10 + '\n')
            s = Stencil2dIR(*stencil_shape)
            s.set_name(sname)

            s.set_numa_config(16,8)
            s.set_run_config(16,8)

            s = kernel_tune_stencil_2d(s)
            log_file.write(s.opt.to_str() + '\n')

            code_gen = gen_stencil_2d(s)
            flow_cache = os.path.join('build', '2d')
            os.makedirs(flow_cache, exist_ok=True)

            file_path = os.path.join(flow_cache, s.name + '.c')
            exec_path = os.path.join(flow_cache, s.name + '.exe')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(code_gen + rand_main_2d.replace("stencil_func", 'Stencil2d_' + s.name))

            compiler_sh = ' '.join((compiler, compile_option, file_path, '-o', exec_path))
            exec_sh = f'./{exec_path}'

            cmd_run(compiler_sh)
            exec_time = cmd_run(exec_sh)
            log_file.write('exec_time: ' + exec_time + '\n')

        elif len(stencil_shape) == 6:
            log_file.write('-'*10 + sname + '-'*10 + '\n')
            s = Stencil3dIR(*stencil_shape)
            s.set_name(sname)

            s.set_numa_config(16,8)
            s.set_run_config(16,8)
            s = kernel_tune_stencil_3d(s)
            log_file.write(s.opt.to_str() + '\n')

            code_gen = gen_stencil_3d(s)
            flow_cache = os.path.join('build', '3d')
            os.makedirs(flow_cache, exist_ok=True)
            file_path = os.path.join(flow_cache, s.name + '.c')
            exec_path = os.path.join(flow_cache, s.name + '.exe')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(code_gen + rand_main_3d.replace("stencil_func", 'Stencil3d_' + s.name))

            compiler_sh = ' '.join((compiler, compile_option, file_path, '-o', exec_path))
            exec_sh = f'./{exec_path}'

            cmd_run(compiler_sh)
            exec_time = cmd_run(exec_sh)
            log_file.write('exec_time: ' + exec_time + '\n')
        else:
            raise ValueError("len(stencil_shape)")
