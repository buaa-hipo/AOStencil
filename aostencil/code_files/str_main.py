tmp_main_2d=r'''

void PrintGraphFILE(datatype *graph, int r, int c, FILE *file)
{
  for (int j = 0; j < c; ++j)
  {
    for (int i = 0; i < r; ++i)
    {
      int loc = j * r + i;
      fprintf(file, "%f ", graph[loc]);
    }
    fprintf(file, "\n");
  }
}

int main(int argc, char **argv)
{

  int times = 100;
  FILE *dataFILE = fopen(argv[2], "r");

  datatype *mp = (datatype *)malloc(ynum * col * sizeof(datatype));
  double input;
  for (int i = 0; i < col * ynum; i++)
  {
    fscanf(dataFILE, "%lf", &input);
    mp[i]=input;
  }
  fclose(dataFILE);

  double st_time,ed_time;
  st_time=omp_get_wtime();

  Stencil2d_2d9pt_star(mp, times);

  ed_time=omp_get_wtime();
  printf("Division Costs %.6f s\n", ed_time-st_time);

  FILE *logFILE = fopen(argv[1], "w");
  PrintGraphFILE(mp, row, col, logFILE);
  fclose(logFILE);

  free(mp);
  return 0;
}
'''

tmp_main_3d=r'''
void PrintlatticeFILE(datatype *lattice, int x, int y, int z, FILE *file)
{
  for (int k = 0; k < z; k++)
  {
    for (int j = 0; j < y; j++)
    {
      for (int i = 0; i < x; i++)
      {
        int loc = k * y * x + j * x + i;
        fprintf(file, "%.6f ", lattice[loc]);
      }
      fprintf(file, "\n");
    }
    fprintf(file, "\n");
  }
}

int main(int argc, char **argv)
{

  FILE *logFILE = fopen(argv[1], "w");
  FILE *dataFILE = fopen(argv[2], "r");

  datatype *mp = (datatype *)malloc(znum * zdim * sizeof(datatype));
  double input_num;
  for (__int64_t i = 0; i < znum * (__int64_t)zdim; i++)
  {
    fscanf(dataFILE, "%lf", &input_num);
    mp[i] = input_num;
  }
  fclose(dataFILE);


  double t0, t1, dt;
  t0 = omp_get_wtime();

  Stencil3d_3d7pt_star(mp, ntimes);
  t1 = omp_get_wtime();
  printf("Division Costs %.6f s\n", t1 - t0);

  PrintlatticeFILE(mp, xdim, ydim, zdim, logFILE);
  fclose(logFILE);

  return 0;
}
'''