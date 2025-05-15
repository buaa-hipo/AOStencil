rand_main_2d=r'''
int main(int argc, char **argv)
{
  unsigned int seed = 42;
  int times = 100;

  datatype *mp = (datatype *)malloc(ynum * col * sizeof(datatype));
  if(mp==NULL){
    puts("run out of memory");
    return 0;
  }
#pragma omp parallel for
  for (__int64_t i = 0LL; i < col * ynum; i++)
  {
    mp[i] = rand_r(&seed) / (datatype)RAND_MAX;
  }

  double st_time, ed_time;
  st_time = omp_get_wtime();

  stencil_func(mp, times);

  ed_time = omp_get_wtime();
  printf("%.6f\n", ed_time - st_time);
  free(mp);
  return 0;
}
'''

rand_main_3d=r'''

int main(int argc, char **argv)
{
  unsigned int seed =42;
  int times = 100;

  datatype *mp = (datatype *)malloc(znum * zdim * sizeof(datatype));
#pragma omp parallel for
  for (__int64_t i = 0; i < znum * (__int64_t)zdim; i++)
  {
    mp[i]=rand_r(&seed)/(datatype)RAND_MAX;
  }

  double t0, t1, dt;
  t0 = omp_get_wtime();

  stencil_func(mp, times);
  t1 = omp_get_wtime();
  
  printf("%.6f\n", t1 - t0);

  return 0;
}
'''

check_main_2d=r'''
#define ABS(a) ((a) > 0 ? (a) : -(a))

char check_stencil_output(datatype *stencilWorld, int times,datatype *check_item)
{
  datatype *lattice = (datatype *)malloc(ynum * col * sizeof(datatype));
  datatype *latticeNext = (datatype *)malloc(ynum * col * sizeof(datatype));
  if (latticeNext == NULL)
  {
    puts("run out of memory");
    return 0;
  }

  if (lattice == NULL)
  {
    puts("run out of memory");
    return 0;
  }
  memcpy(latticeNext, stencilWorld, ynum * col * sizeof(datatype));
  memcpy(lattice, stencilWorld, ynum * col * sizeof(datatype));

  while (times--)
  {
#pragma omp parallel for
    for (int j = edge_length; j < col - edge_length; ++j)
    {
      for (int i = edge_length; i < row - edge_length; ++i)
      {
        __int64_t center = j * ynum + i;
        @ScalarKernel
      }
    }
    void *tmpPtr = lattice;
    lattice = latticeNext;
    latticeNext = tmpPtr;
  }
  char ret=1;
#pragma omp parallel for
  for (__int64_t i = 0LL; i < ynum * col; i++)
  {
    if (((ABS(check_item[i] - lattice[i]))/check_item[i] > (datatype)1e-2)&&ret)
    {
      printf("loc[%d,%d]=%.10f %.10f\n", i / ynum, i % ynum, check_item[i], lattice[i]);
      ret=0;
    }
  }
  free(lattice);
  free(latticeNext);
  return ret;
}


int main(int argc, char **argv)
{
  unsigned int seed = 42;
  int times = 100;

  datatype *mp_std = (datatype *)malloc(ynum * col * sizeof(datatype));
  datatype *mp = (datatype *)malloc(ynum * col * sizeof(datatype));
  if(mp_std==NULL){
    puts("run out of memory");
    return 0;
  }
  if(mp==NULL){
    puts("run out of memory");
    return 0;
  }
#pragma omp parallel for
  for (__int64_t i = 0LL; i < col * ynum; i++)
  {
    mp_std[i] = mp[i] =  rand_r(&seed) / (datatype)RAND_MAX;
  }

  double st_time, ed_time;
  st_time = omp_get_wtime();

  stencil_func(mp, times);

  ed_time = omp_get_wtime();
  printf("%.6f\n", ed_time - st_time);
  printf("result %s\n", check_stencil_output(mp_std, times, mp) ? "True" : "False");
  free(mp);
  free(mp_std);
  return 0;
}
'''

check_main_3d=r'''
#define ABS(a) ((a) > 0 ? (a) : -(a))

char check_stencil_output(datatype *stencilWorld, int times, datatype *check_item)
{
  datatype *lattice = (datatype *)malloc(znum * zdim * sizeof(datatype));
  datatype *latticeNext = (datatype *)malloc(znum * zdim * sizeof(datatype));
  if (latticeNext == NULL)
  {
    puts("run out of memory");
    return 0;
  }

  if (lattice == NULL)
  {
    puts("run out of memory");
    return 0;
  }
  memcpy(latticeNext, stencilWorld, znum * zdim * sizeof(datatype));
  memcpy(lattice, stencilWorld, znum * zdim * sizeof(datatype));

  while (times--)
  {
#pragma omp parallel for

    for (int k = edge_length; k < zdim - edge_length; ++k)
    {
      for (int j = edge_length; j < ydim - edge_length; ++j)
      {
        for (int i = edge_length; i < xdim - edge_length; ++i)
        {
          __int64_t center = k*znum+j * ynum + i;
          @ScalarKernel
        }
      }
    }
    void *tmpPtr = lattice;
    lattice = latticeNext;
    latticeNext = tmpPtr;
  }
  char ret=1;
#pragma omp parallel for
  for (__int64_t i = 0LL; i < znum * zdim; i++)
  {
    if (((ABS(check_item[i] - lattice[i]))/check_item[i] > (datatype)1e-2)&&ret)
    {
      printf("loc[%d,%d,%d]=%.10f %.10f\n", i / znum, (i % znum) / ynum, i % ynum, check_item[i], lattice[i]);
      ret=0;
    }
  }
  free(lattice);
  free(latticeNext);
  return ret;
}

int main(int argc, char **argv)
{
  unsigned int seed = 42;
  int times = 100;

  datatype *mp_std = (datatype *)malloc(znum * zdim * sizeof(datatype));
  datatype *mp = (datatype *)malloc(znum * zdim * sizeof(datatype));
  if (mp_std == NULL)
  {
    puts("run out of memory");
    return 0;
  }
  if (mp == NULL)
  {
    puts("run out of memory");
    return 0;
  }
#pragma omp parallel for
  for (__int64_t i = 0; i < znum * zdim; i++)
  {
    mp_std[i] = mp[i] = rand_r(&seed) / (datatype)RAND_MAX;
  }

  double st_time, ed_time;
  st_time = omp_get_wtime();

  stencil_func(mp, times);

  ed_time = omp_get_wtime();
  printf("%.6f\n", ed_time - st_time);
  printf("result %s\n", check_stencil_output(mp_std, times, mp) ? "True" : "False");
  free(mp);
  free(mp_std);
  return 0;
}
'''
