omp_for_kernel=r'''
for (int y = edge_length; y < subcol - edge_length; ++y)
  {
    for (int x = st; x < ed; x+=@x_inp)
    {
      const __int64_t center = (__int64_t)y * ynum + (__int64_t)x;
      @kernel
    }
  }
'''

omp_for_kernel_block=r'''
for (int by = edge_length;by< subcol - edge_length; by += @blockSize_y)
    { // Block loop for y
      for (int bx = st; bx < ed; bx += @blockSize_x)
      { // Block loop for x
        for (int y = by; y < MIN(by + @blockSize_y,subcol - edge_length); y++)
        {
          for (int x = bx; x < MIN(bx + @blockSize_x,ed); x += @x_inp)
          { // Process two elements at a time      
            const __int64_t center = (__int64_t)y * ynum + (__int64_t)x;
            @kernel
          }
        }
      }
    }
    '''

pthread_for_kernel=r'''
@block_y_unroll
for (int y = edge_length; y < @subcol - edge_length; ++y)
  { 
@block_x_unroll
    for (int x = (@row_st); x < (@row_ed); x+=@x_inp)
    {
      const __int64_t center = (__int64_t)y * ynum + (__int64_t)x;
      @kernel
    }
  }
'''

pthread_for_kernel_block=r'''
for (int by = edge_length;by< @subcol - edge_length; by += @blockSize_y)
    { // Block loop for y
      for (int bx = (@row_st); bx < (@row_ed); bx += @blockSize_x)
      { // Block loop for x
@block_y_unroll
        for (int y = by; y < MIN(by + @blockSize_y,@subcol - edge_length); y++)
        {
@block_x_unroll
          for (int x = bx; x < MIN(bx + @blockSize_x,(@row_ed)); x += @x_inp)
          { // Process two elements at a time      
            const __int64_t center = (__int64_t)y * ynum + (__int64_t)x;
            @kernel
          }
        }
      }
    }
    '''
