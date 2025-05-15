import re
import os
import subprocess
import shutil
from queue import Queue
import ctypes

compile_option=" -O3 -lnuma -fopenmp -pthread"
compiler="gcc"
MAX_NUM_INSTRUCTION_STREAM=10

class NUMA:
    def __init__(self,numa_nodes_num,cpus_per_numa_num) -> None:
      self.numa_nodes_num=numa_nodes_num
      self.cpus_per_numa_num=cpus_per_numa_num

class StencilOPT:
  def __init__(self,simd_usage,blockSize,kernel_unroll_block) -> None:
      self.simd_usage=simd_usage
      self.blockSize=blockSize
      self.kernel_unroll_block=kernel_unroll_block

  def to_str(self)->str:
      return f'{self.simd_usage}_{tuple2str(self.blockSize)}_{tuple2str(self.kernel_unroll_block)}'

  def __eq__(self, other) -> bool:
    if other is None:
      return 0
    else:
      return (self.simd_usage == other.simd_usage and
              self.blockSize == other.blockSize and
              self.kernel_unroll_block == other.kernel_unroll_block)

  def __hash__(self) -> int:
    return hash((self.simd_usage, self.blockSize, self.kernel_unroll_block))


def check_libnuma()->bool:
    try:
        # try to load libnuma
        libnuma = ctypes.CDLL("libnuma.so")
        print("libnuma.so library is available.")
        return True
    except OSError as e:
        print("libnuma.so library is not available.")
        return False

def init_dir(dir_path:str):
  if os.path.exists(dir_path):
    # Clear all contents of the directory
    for filename in os.listdir(dir_path):
      file_path = os.path.join(dir_path, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)  # Remove files or links
        elif os.path.isdir(file_path):
          shutil.rmtree(file_path)  # Remove directories
      except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')
  else:
    # Directory does not exist, so create it
    os.makedirs(dir_path)

def cmd_run(cmd:str)->str:
  try:
    result = subprocess.run(cmd, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  except subprocess.CalledProcessError as e:
    raise Exception("cmd exec failed:\n",cmd+'\n',e.stderr)
  return result.stdout.strip()

def replace_var(code:str,var_name:str,var_value:str):
  try:
    if code is not None:
      var_value_str = str(var_value)
      pattern = r'@' + re.escape(var_name)
      replaced_code = re.sub(pattern, var_value_str, code)
      return replaced_code
    else:
        return None
  except:
      print("s=",code,",old=",var_name,",new=",var_value)
      raise ValueError
  
def ceil(x:int,y:int)->int:
   return (x+y-1)//y

def order2(x,y):
  return (x,y) if x<y else (y,x)

def tuple2str(t):
  if t == None or t == False:
    return 'None'
  else:
    res=""
    for ti in t:
      res+=f"_{ti}"
    return res[1:]

# Convert configuration to OMP_PLACES format
def generate_omp_places(numa_nodes_num, cpus_per_numa_num):
    places = []
    for node in range(numa_nodes_num):
        start_cpu = node * cpus_per_numa_num
        node_cpus = [str(cpu) for cpu in range(start_cpu, start_cpu + cpus_per_numa_num)]
        places.append("{" + ",".join(node_cpus) + "}")
    return " ".join(places)

# Set OpenMP environment variables
def set_openmp_env(numa_nodes_num, cpus_per_numa_num):
    omp_places = generate_omp_places(numa_nodes_num, cpus_per_numa_num)

    return {'OMP_PROC_BIND':'close','OMP_PLACES':omp_places}

def gen_for_kernel(for_var:str,st_var:int,ed_var,var_inp:int,for_context:str):
  for_temple=f'''
    for(int {for_var}={st_var};{for_var}<{ed_var};{for_var}+={var_inp})
    {{
      {for_context}
    }}
  '''
  return for_temple


class ProcessorL3Cache:
    def __init__(self, cpu_id):
        # Construct the path to the L3 cache information for the specified CPU
        self.base_path = f"/sys/devices/system/cpu/cpu{cpu_id}/cache/index3/"
        self.cache_info = {}
        self.read_cache_info()

    def read_cache_info(self):
        # List of files that contain the required cache information
        info_files = [
            "coherency_line_size", "number_of_sets", "shared_cpu_list",
            "size", "type", "ways_of_associativity", "write_policy"
        ]
        try:
            # Read each file and store the information in a dictionary
            for file_name in info_files:
                file_path = os.path.join(self.base_path, file_name)
                with open(file_path, 'r') as file:
                    content = file.read().strip()
                    # Attempt to convert numeric values to integers
                    if content.isdigit():
                        content = int(content)
                    elif 'K' in content:
                        # Special handling for sizes like '32768K'
                        content = self.convert_kb_to_bytes(content)
                    self.cache_info[file_name] = content
        except FileNotFoundError:
            print(f"Error: Cache information files not found in {self.base_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def display_info(self):
        # Print out all the gathered information about the L3 cache
        for key, value in self.cache_info.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")

    def convert_kb_to_bytes(self, size_in_kb):
        # Remove the 'K' and convert the size from KB to bytes
        return int(size_in_kb.replace('K', '')) * 1024

def get_cpu_cache_sizes(cpu_name:str):
    cache_sizes_per_core={
       "kunpeng":{
          'L1i':64*1024,
          'L1d':64*1024,
          'L2':512*1024,
          'L3':2048*1024,
       },
       "phytium":{
          'L1i':32*1024,
          'L1d':32*1024,
          'L2':2048*1024,
          'L3':8192*1024,
       }
    }
    return cache_sizes_per_core[cpu_name]

def remove_duplicates(stencil_opt_list):
  seen = set()
  unique_list = []
  for opt in stencil_opt_list:
    if opt not in seen and opt != None:
      seen.add(opt)
      unique_list.append(opt)
  return unique_list

def check_system_deps():
    try:
        subprocess.run([compiler, "--version"], 
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(f"{compiler} compiler not found. Please install gcc first")

    try:
        subprocess.run(["numactl", "--show"],
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("NUMA libraries not found. Install libnuma-dev (Debian/Ubuntu) or numactl-devel (RHEL/CentOS)")