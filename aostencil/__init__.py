try:
    import numpy as np
    using_numpy = True
except ImportError:
    from . import my_array as np
    using_numpy = False

from .stencil import Stencil2dIR,Stencil3dIR
from .pthread_stencil_2d import gen_stencil_pthread_2d as gen_stencil_2d
from .pthread_stencil_3d import gen_stencil_pthread_3d as gen_stencil_3d
from .kernel_tune_stencil_2d import kernel_tune_stencil_2d
from .kernel_tune_stencil_3d import kernel_tune_stencil_3d
from .utils import cmd_run,compiler,compile_option,init_dir,replace_var,check_system_deps
from .dsl import from_dsl_load_stencil
check_system_deps()

__all__ = ["Stencil2dIR","Stencil3dIR","gen_stencil_2d","gen_stencil_3d","kernel_tune_stencil_2d","kernel_tune_stencil_3d","cmd_run","compiler","compile_option","init_dir","replace_var","check_system_deps","from_dsl_load_stencil"]