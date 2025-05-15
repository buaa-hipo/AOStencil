
import os
import datetime
import time
from kunpeng_test import kunpeng_test
from aostencil import np

if __name__ == '__main__':
  
  # Generate log file name with current datetime
  current_time = datetime.datetime.now().strftime("%m%d%H%M")
  log_filename = f"log_{current_time}_2d9pt_star.txt"
  log_file_path = os.path.join('log', log_filename)

  # Ensure the log directory exists
  os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
  coef=np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]])*0.1
  for i in (8192,16384,24576):
    stencil_shape=(i,i+4,coef,0,'double')
    sname=f'2d9pt_star_{stencil_shape[0]}_{stencil_shape[1]}_{stencil_shape[-1]}'
    kunpeng_test(stencil_shape,sname,log_file_path)
    time.sleep(5)
