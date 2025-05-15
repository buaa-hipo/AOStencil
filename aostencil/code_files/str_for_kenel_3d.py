omp_for_kernel=r'''
for (int z = edge_length; z < subzdim - edge_length; ++z)
  {
    for (int y = st; y < ed; ++y)
    {
      for(int x = edge_length; x < xdim - edge_length; x+=@x_inp)
      {
        const __int64_t center =(__int64_t)z * znum+ (__int64_t)y * ynum + (__int64_t)x;
        @kernel
      }
    }
  }
'''

omp_for_kernel_block=r'''
@block_z_unroll
for (int bz = edge_length; bz < subzdim - edge_length; bz += @blockSize_z)
{ // Block loop for z
@block_y_unroll
  for (int by = st; by < ed; by += @blockSize_y)
  { // Block loop for y
@block_x_unroll
    for (int bx = edge_length; bx < xdim - edge_length; bx += @blockSize_x)
    { // Block loop for x
      for (int z = bz; z < MIN(bz + @blockSize_z, subzdim- edge_length); z++)
      {
        for (int y = by; y < MIN(by + @blockSize_y, ed); y++)
        {
          for (int x = bx; x < MIN(bx + @blockSize_x, xdim - edge_length); x += @x_inp)
          { // Process two elements at a time
            const __int64_t center = (__int64_t)z * znum + (__int64_t)y * ynum + (__int64_t)x;
            @kernel
          }
        }
      }
    }
  }
}'''

pthread_for_kernel=r'''
@block_z_unroll
for (int z = edge_length; z < @subzdim - edge_length; ++z)
{
@block_y_unroll
  for (int y = (@ydim_st); y < (@ydim_ed); ++y)
  {
@block_x_unroll
    for(int x = edge_length; x < xdim - edge_length; x+=@x_inp)
    {
      const __int64_t center =(__int64_t)z * znum+ (__int64_t)y * ynum + (__int64_t)x;
      @kernel
    }
  }
}
'''

pthread_for_kernel_block=r'''
for (int bz = edge_length; bz < @subzdim - edge_length; bz += @blockSize_z)
{ // Block loop for z
  for (int by = (@ydim_st); by < (@ydim_ed); by += @blockSize_y)
  { // Block loop for y
    for (int bx = edge_length; bx < xdim - edge_length; bx += @blockSize_x)
    { // Block loop for x
@block_z_unroll
      for (int z = bz; z < MIN(bz + @blockSize_z, @subzdim- edge_length); z++)
      {
@block_y_unroll
        for (int y = by; y < MIN(by + @blockSize_y, (@ydim_ed)); y++)
        {
@block_x_unroll
          for (int x = bx; x < MIN(bx + @blockSize_x, xdim - edge_length); x += @x_inp)
          { // Process two elements at a time
            const __int64_t center = (__int64_t)z * znum + (__int64_t)y * ynum + (__int64_t)x;
            @kernel
          }
        }
      }
    }
  }
}
'''
