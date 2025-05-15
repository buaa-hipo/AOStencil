
import os
import datetime
import time
from phytium_test import phytium_test
from aostencil import np

if __name__ == '__main__':

  # Generate log file name with current datetime
  current_time = datetime.datetime.now().strftime("%m%d%H%M")
  log_filename = f"log_{current_time}_3d7pt_star.txt"
  log_file_path = os.path.join('log', log_filename)
  # Ensure the log directory exists
  os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
  coef=np.array(
        [[[0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]],

       [[0., 1., 0.],
        [1., 1., 1.],
        [0., 1., 0.]],

       [[0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]]])*0.2
  for i in (256,512,768):
    stencil_shape=(i,i,i+2,coef,0,'double')
    sname=f'3d7pt_star_{stencil_shape[0]}_{stencil_shape[1]}_{stencil_shape[2]}_{stencil_shape[-1]}'
    phytium_test(stencil_shape,sname,log_file_path)
    time.sleep(5)