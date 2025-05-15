numa_stencil_2d_omp=r'''#define _GNU_SOURCE

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <sched.h>
#include <numa.h>
#include <omp.h>
#include <arm_neon.h>
#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

#define xnum @xnum
#define ynum @ynum
#define row @row
#define col @col
#define subrow @subrow
#define edge_length @edge_length
#define num_sublattices @num_sublattices
#define num_procs_sublattices @num_procs_sublattices
#define num_procs_per_numa @num_procs_per_numa


typedef @datatype datatype;

typedef struct
{
  datatype *lattice[2];
} Sublattice;


void Stencil2d_@stencilName(datatype *gh, int ntimes)
{

  Sublattice sublattice[num_sublattices];

#pragma omp parallel
  {
    const int cpu_id = sched_getcpu();
    // int cpu_id = omp_get_thread_num();
    const int numa_id = cpu_id / num_procs_per_numa;
    const int numa_in_id = cpu_id % num_procs_per_numa;
    const int subcol = numa_id == 0 ? CEIL(col, num_sublattices) + edge_length
                     : numa_id == num_sublattices - 1 ? col - (num_sublattices - 1) * CEIL(col, num_sublattices) + edge_length
                                                    : CEIL(col, num_sublattices) + 2 * edge_length;


    if (numa_in_id == 0)
    {
      sublattice[numa_id].lattice[0] = (datatype *)numa_alloc_onnode(sizeof(datatype) * ynum * subcol, numa_id);
      sublattice[numa_id].lattice[1] = (datatype *)numa_alloc_onnode(sizeof(datatype) * ynum * subcol, numa_id);

      if (sublattice[numa_id].lattice[0] == NULL || sublattice[numa_id].lattice[1] == NULL)
      {
        puts("Running out of memory!");
        printf("sublattice[%d] row:%d, col:%d", numa_id, row, subcol);
        exit(-1);
      }
      memcpy(sublattice[numa_id].lattice[0], numa_id == 0 ? gh : gh + ynum * (CEIL(col, num_sublattices) * numa_id - edge_length), sizeof(datatype) * row * subcol);
      memcpy(sublattice[numa_id].lattice[1], sublattice[numa_id].lattice[0], sizeof(datatype) * ynum * subcol);
    }
    const int top_edge_index = (numa_id == 0)   ? -1
                             : (numa_id == 1) ? (CEIL(col, num_sublattices) - edge_length) * ynum
                                              : CEIL(col, num_sublattices) * ynum;

    const int bottom_edge_index = (numa_id == num_sublattices - 1) ? -1 : ynum * edge_length;
    const int st = subrow * numa_in_id + edge_length;
    const int ed = MIN(st + subrow, row - edge_length);

    for (int i = 0; i < ntimes; i++)
    {
#pragma omp barrier
      const datatype *lattice = sublattice[numa_id].lattice[i%2];
      datatype *latticeNext = sublattice[numa_id].lattice[(i+1)%2];

      @omp_for_kernel
#pragma omp barrier
      if (top_edge_index != -1)
      {
#pragma unroll(edge_length)
        for (int e = 0; e < edge_length; e++)
        {
          memcpy(sublattice[numa_id].lattice[(i + 1) % 2] + e * ynum + st, sublattice[numa_id - 1].lattice[(i + 1) % 2] + top_edge_index + e * ynum + st, (ed - st) * sizeof(datatype));
        }
      }
      if (bottom_edge_index != -1)
      {
#pragma unroll(edge_length)
        for (int e = 0; e < edge_length; e++)
        {
          memcpy(sublattice[numa_id].lattice[(i + 1) % 2] + ynum * (subcol - edge_length) + e * ynum + st, sublattice[numa_id + 1].lattice[(i + 1) % 2] + bottom_edge_index + e * ynum + st, (ed - st) * sizeof(datatype));
        }
      }
    }
#pragma omp barrier
    if (numa_in_id == 0)
    {
      datatype *offset_gh = numa_id == 0 ? gh : gh + ynum * (CEIL(col, num_sublattices) * numa_id - edge_length);
      const int offset_subcol = numa_id == num_sublattices - 1 ? subcol : (subcol - 2 * edge_length);
      memcpy(offset_gh, sublattice[numa_id].lattice[ntimes%2], sizeof(datatype) * ynum * offset_subcol);
      numa_free(sublattice[numa_id].lattice[0], sizeof(datatype) * ynum * subcol);
      numa_free(sublattice[numa_id].lattice[1], sizeof(datatype) * ynum * subcol);
    }
  }
}
'''