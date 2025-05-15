head_pthread_numa_stencil=r'''
#define _GNU_SOURCE
#include <assert.h>
#include <sched.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <arm_neon.h>
#include <unistd.h>

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
#define num_procs_sublattices @num_procs_sublattices
#define num_procs_per_numa @num_procs_per_numa

typedef @datatype datatype;

typedef struct
{
  datatype *gh;
  int ntimes;
} ThreadData;

static datatype * sublattice[num_sublattices][2];
static pthread_barrier_t barrierStart,barrier1,barrier2,barrierEnd;
'''

func_pthread_numa_stencil=r'''
static void *thread_func_@cpu_id(void *arg)
{
  pthread_barrier_wait(&barrierStart);
  const ThreadData *data = (ThreadData *)arg;

  if (@numa_in_id == 0)
  {
    sublattice[@numa_id][0] = (datatype *)numa_alloc_onnode(znum * @subzdim * sizeof(datatype), @numa_id);
    sublattice[@numa_id][1] = (datatype *)numa_alloc_onnode(znum * @subzdim * sizeof(datatype), @numa_id);
    if (sublattice[@numa_id][0] == NULL || sublattice[@numa_id][1] == NULL)
    {
      puts("Running out of memory!");
      printf("sublattice[%d] xdim:%d, ydim:%d, zdim:%d", @numa_id, xdim, ydim, @subzdim);
      exit(-1);
    }
    memcpy(sublattice[@numa_id][0], @numa_id == 0 ? data->gh : data->gh + znum * (@avg_subzdim * @numa_id - edge_length), znum * @subzdim * sizeof(datatype));
    memcpy(sublattice[@numa_id][1], sublattice[@numa_id][0], znum * @subzdim * sizeof(datatype));
  }

  for (int t = 0; t < data->ntimes; t++)
  {
pthread_barrier_wait(&barrier1);
    const datatype *__restrict__ lattice = sublattice[@numa_id][t % 2];
    datatype *__restrict__ latticeNext = sublattice[@numa_id][(t + 1) % 2];

    @pthread_for_kernel

pthread_barrier_wait(&barrier2);

  if (@top_edge_index != -1LL)
    {
#pragma unroll(edge_length)
      for (int e = 0; e < edge_length; e++)
      {
        memcpy(sublattice[@numa_id][(t + 1) % 2] + e * znum + (@ydim_st) * xdim, sublattice[@numa_id - 1][(t + 1) % 2] + @top_edge_index + e * znum + (@ydim_st) * xdim, ((@ydim_ed) - (@ydim_st)) * xdim * sizeof(datatype));
      }
    }
  if (@bottom_edge_index != -1LL)
    {
#pragma unroll(edge_length)
      for (int e = 0; e < edge_length; e++)
      {
        memcpy(sublattice[@numa_id][(t + 1) % 2] + znum * (@subzdim - edge_length) + e * znum + (@ydim_st) * xdim, sublattice[@numa_id + 1][(t + 1) % 2] + @bottom_edge_index + e * znum + (@ydim_st) * xdim, ((@ydim_ed) - (@ydim_st)) * xdim * sizeof(datatype));
      }
    }
  }
pthread_barrier_wait(&barrierEnd);
  if (@numa_in_id == 0)
  {
    int offset_subzdim = @numa_id == num_sublattices - 1 ? @subzdim : (@subzdim - 2 * edge_length);
    datatype *offset_lattice = @numa_id == 0 ? data->gh : data->gh + znum * (@avg_subzdim * @numa_id - edge_length);
    memcpy(offset_lattice, sublattice[@numa_id][data->ntimes % 2], sizeof(datatype) * znum * offset_subzdim);
    numa_free(sublattice[@numa_id][0], znum * @subzdim * sizeof(datatype));
    numa_free(sublattice[@numa_id][1], znum * @subzdim * sizeof(datatype));
  }
  pthread_exit(NULL);
}
'''

func_join_pthread_numa_stencil=r'''
void Stencil3d_@stencilName(datatype *gh, int ntimes)
{
  pthread_barrier_init(&barrierStart, NULL, num_sublattices * num_procs_sublattices);
  pthread_barrier_init(&barrier1, NULL, num_sublattices * num_procs_sublattices);
  pthread_barrier_init(&barrier2, NULL, num_sublattices * num_procs_sublattices);
  pthread_barrier_init(&barrierEnd, NULL, num_sublattices * num_procs_sublattices);

  pthread_t threads[num_sublattices * num_procs_sublattices];

  ThreadData threadData;
  threadData.gh = gh;
  threadData.ntimes = ntimes;
  
  pthread_attr_t attr;
  cpu_set_t cpuset;
  @pthread_attr_set_kernel

  for (int i = 0; i < num_sublattices * num_procs_sublattices; ++i)
  {
    pthread_join(threads[i], NULL);
  }

  pthread_barrier_destroy(&barrierStart);
  pthread_barrier_destroy(&barrierEnd);
  pthread_barrier_destroy(&barrier1);
  pthread_barrier_destroy(&barrier2);
}
'''

set_pthread_numa_stencil=r'''
pthread_attr_init(&attr);
CPU_ZERO(&cpuset);
CPU_SET(@cpu_id, &cpuset);
pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
if (pthread_create(&threads[@thread_id], &attr, @thread_func, &threadData) != 0)
{
  perror("Failed to create thread");
  exit(EXIT_FAILURE);
}
'''