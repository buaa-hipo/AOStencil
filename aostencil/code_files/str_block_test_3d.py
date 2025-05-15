head_pthread_numa_stencil=r'''
#define _GNU_SOURCE

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include <assert.h>
#include <sched.h>
#include <unistd.h>
#include <pthread.h>

#include <numa.h>
#include <omp.h>
#include <arm_neon.h>
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
#define num_procs_sublattices @num_procs_sublattices
#define num_procs_per_numa @num_procs_per_numa

typedef @datatype datatype;

static datatype * sublattice[2];
static pthread_barrier_t barrierStart,barrier1,barrier2,barrierEnd;
'''

func_pthread_numa_stencil=r'''
static void *thread_func_@cpu_id(void *arg)
{
  pthread_barrier_wait(&barrierStart);
  const int ntimes = *(int *)arg;

  if (@numa_in_id == 0)
  {
    sublattice[0] = (datatype *)numa_alloc_onnode(sizeof(datatype) * znum*zdim, @numa_id);
    sublattice[1] = (datatype *)numa_alloc_onnode(sizeof(datatype) * znum*zdim, @numa_id);

    if (sublattice[0] == NULL || sublattice[1] == NULL)
    {
      puts("Running out of memory!");
      printf("sublattice[%d] xdim:%d, ydim:%d, zdim:%d", @numa_id, xdim, ydim, zdim);
      exit(-1);
    }
  }

  for (int i = 0; i < ntimes; i++)
  {
    pthread_barrier_wait(&barrier1);
    const datatype *lattice = sublattice[(i+1)%2];
    datatype *latticeNext = sublattice[i%2];

    @pthread_for_kernel
    pthread_barrier_wait(&barrier2);
   
  }
  pthread_barrier_wait(&barrierEnd);
  if (@numa_in_id == 0)
  {
    numa_free(sublattice[0], sizeof(datatype) * znum*zdim);
    numa_free(sublattice[1], sizeof(datatype) * znum*zdim);
  }
  return NULL;
}
'''

func_join_pthread_numa_stencil=r'''
double Stencil2d_test_@stencilName(int ntimes)
{
  pthread_barrier_init(&barrierStart, NULL,  num_procs_sublattices);
  pthread_barrier_init(&barrier1, NULL,  num_procs_sublattices);
  pthread_barrier_init(&barrier2, NULL,  num_procs_sublattices);
  pthread_barrier_init(&barrierEnd, NULL,  num_procs_sublattices);
  
  pthread_attr_t attr;
  cpu_set_t cpuset;
  pthread_t threads[num_procs_sublattices];
  double start,end;
  start=omp_get_wtime();

  @pthread_attr_set_kernel

  for (int i = 0; i <  num_procs_sublattices; ++i)
  {
    pthread_join(threads[i], NULL);
  }

  end=omp_get_wtime();
  pthread_barrier_destroy(&barrierStart);
  pthread_barrier_destroy(&barrierEnd);
  pthread_barrier_destroy(&barrier1);
  pthread_barrier_destroy(&barrier2);
  return end-start;
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    perror("arg is blank\n");
    return 1;
  }
  int ntimes = atoi(argv[1]);
  double exec_time;
  exec_time = Stencil2d_test_@stencilName(ntimes);
  printf("%f\n", exec_time);
  return 0;
}
'''

set_pthread_numa_stencil=r'''
pthread_attr_init(&attr);
CPU_ZERO(&cpuset);
CPU_SET(@cpu_id, &cpuset);
pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
if (pthread_create(&threads[@thread_id], &attr, @thread_func, &ntimes) != 0)
{
  perror("Failed to create thread");
  exit(EXIT_FAILURE);
}
'''
