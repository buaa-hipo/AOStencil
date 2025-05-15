numa_stencil_3d_omp=r'''
#define _GNU_SOURCE
#include <assert.h>
#include <sched.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <arm_neon.h>

#include <numa.h>
#include <omp.h>

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

#define subydim @subydim
#define znum @znum
#define ynum @ynum
#define xnum @xnum
#define xdim @xdim
#define ydim @ydim
#define zdim @zdim
#define edge_length @edge_length
#define num_sublattices @num_sublattices
#define num_procs_sublattice @num_procs_sublattices
#define num_procs_per_numa @num_procs_per_numa

typedef @datatype datatype;

void Stencil3d_@stencilName(datatype *lattice, int ntimes)
{
  datatype *sublattice[num_sublattices][2];
#pragma omp parallel
  {
    const int cpu_id = sched_getcpu();
    // int cpu_id = omp_get_thread_num();
    const int numa_id = cpu_id / num_procs_per_numa;
    const int numa_in_id = cpu_id % num_procs_per_numa;

    const int subzdim = numa_id == 0                     ? CEIL(zdim, num_sublattices) + edge_length
                        : numa_id == num_sublattices - 1 ? zdim - (num_sublattices - 1) * CEIL(zdim, num_sublattices) + edge_length
                                                         : CEIL(zdim, num_sublattices) + 2 * edge_length;

    const int top_edge_index = (numa_id == 0)   ? -1
                               : (numa_id == 1) ? (CEIL(zdim, num_sublattices) - edge_length) * znum
                                                : CEIL(zdim, num_sublattices) * znum;
    const int bottom_edge_index = (numa_id == num_sublattices - 1) ? -1 : znum * edge_length;

    const int st = subydim * numa_in_id + edge_length;
    const int ed = MIN(st + subydim, ydim - edge_length);
    if (numa_in_id == 0)
    {
      sublattice[numa_id][0] = (datatype *)numa_alloc_onnode(znum * subzdim * sizeof(datatype), numa_id);
      sublattice[numa_id][1] = (datatype *)numa_alloc_onnode(znum * subzdim * sizeof(datatype), numa_id);
      if (sublattice[numa_id][0] == NULL || sublattice[numa_id][1] == NULL)
      {
        puts("Running out of memory!");
        printf("sublattice[%d] xdim:%d, ydim:%d, zdim:%d", numa_id, xdim, ydim, subzdim);
        exit(-1);
      }
      memcpy(sublattice[numa_id][0], numa_id == 0 ? lattice : lattice + znum * (CEIL(zdim, num_sublattices) * numa_id - edge_length), znum * subzdim * sizeof(datatype));
      memcpy(sublattice[numa_id][1], sublattice[numa_id][0], znum * subzdim * sizeof(datatype));
    }
    for (int t = 0; t < ntimes; t++)
    {
#pragma omp barrier
      const datatype *lattice = sublattice[numa_id][t % 2];
      datatype *latticeNext = sublattice[numa_id][(t + 1) % 2];

      @omp_for_kernel
#pragma omp barrier
      if (top_edge_index != -1)
      {
#pragma unroll(edge_length)
        for (int e = 0; e < edge_length; e++)
        {
          memcpy(sublattice[numa_id][(t + 1) % 2] + e * znum + st * xdim, sublattice[numa_id - 1][(t + 1) % 2] + top_edge_index + e * znum + st * xdim, (ed - st) * xdim * sizeof(datatype));
        }
      }
      if (bottom_edge_index != -1)
      {
#pragma unroll(edge_length)
        for (int e = 0; e < edge_length; e++)
        {
          memcpy(sublattice[numa_id][(t + 1) % 2] + znum * (subzdim - edge_length) + e * znum + st * xdim, sublattice[numa_id + 1][(t + 1) % 2] + bottom_edge_index + e * znum + st * xdim, (ed - st) * xdim * sizeof(datatype));
        }
      }
    }
#pragma omp barrier
    if (numa_in_id == 0)
    {
      int offset_subzdim = numa_id == num_sublattices - 1 ? subzdim : (subzdim - 2 * edge_length);
      datatype *offset_lattice = numa_id == 0 ? lattice : lattice + znum * (CEIL(zdim, num_sublattices) * numa_id - edge_length);
      memcpy(offset_lattice, sublattice[numa_id][ntimes % 2], sizeof(datatype) * znum * offset_subzdim);
      numa_free(sublattice[numa_id][0], znum * subzdim * sizeof(datatype));
      numa_free(sublattice[numa_id][1], znum * subzdim * sizeof(datatype));
    }
  }
}

'''