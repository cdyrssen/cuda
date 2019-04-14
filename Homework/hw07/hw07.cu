//nvcc hw07.cu -o hw07 -lm -lcurand
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.141592654
#define NO 0
#define YES 1

//constants globals
const int NUMBER_OF_BODIES = 6;
const int NUMBER_OF_RUNS = 1000;

const float STOP_TIME = 25.0;
const float DT = 0.0001;

const float CENTRAL_ATTRACTION_FORCE  = 0.1;
const float REPULSIVE_SLOPE = 50000.0;

const float DIAMETER_PS = 1.0; // Diameter of polystyrene spheres 1 micron
const float DIAMETER_NIPAM = 0.08; // Diameter of polyNIPAM microgel particles 80 nanometers

const float MASS = 1.0; //estimate with density 1.05g per cm cubed

const float START_RADIUS_OF_INVIRONMENT = 5.0;
const float MAX_INITIAL_VELOCITY = 1.0;
const float START_SEPERATION_TOL = 1.1;

//globals
__constant__ float g_drag = 15.25714286; //should be a function of temperature;
__constant__ float g_max_attraction = 376.5; //should be a function of temperature;

void CUDAErrorCheck(const char *message){
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess){
    printf("\n CUDA ERROR in: %s -> %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

struct cuda_rng{
    unsigned long seed;
    curandState s;

    __device__ float get_rand_val(unsigned long offset){
        curand_init(seed, 0, offset, &s);
        return curand_uniform(&s);
    }
};

__device__ void set_initial_conditions(float4 *run_pos, float3 *run_vel){
	float angle1, angle2, rad, dx, dy, dz;
    cuda_rng rng;
    rng.seed=(blockIdx.x*blockDim.x)+threadIdx.x;

    int offset = 0;
    do{ // Repeat the following process until a valid configuration is generated.
	    for(int i=0; i<NUMBER_OF_BODIES; i++){
            rad = START_RADIUS_OF_INVIRONMENT*rng.get_rand_val(i+offset+0);
            angle1 = PI*rng.get_rand_val(i+offset+1);
            angle2 = 2.0*PI*rng.get_rand_val(i+offset+2);

            run_pos[i].x = rad*sinf(angle1)*cosf(angle2);
            run_pos[i].y = rad*sinf(angle1)*sinf(angle2);
            run_pos[i].z = rad*cosf(angle1);
	    }

        for(int i=0; i<NUMBER_OF_BODIES-1; i++){
            for(int j=i+1; j<NUMBER_OF_BODIES; j++){
                dx = run_pos[i].x-run_pos[j].x;
                dy = run_pos[i].y-run_pos[j].y;
                dz = run_pos[i].z-run_pos[j].z;
            }
        }

        offset++;
    }while(sqrt(dx*dx + dy*dy + dz*dz) <= START_SEPERATION_TOL);
    offset += NUMBER_OF_BODIES;

    for(int i=0; i<NUMBER_OF_BODIES; i++){
        rad = MAX_INITIAL_VELOCITY*rng.get_rand_val(i+offset+0);
        angle1 = PI*rng.get_rand_val(i+offset+1);
        angle2 = 2.0*PI*rng.get_rand_val(i+offset+2);

        run_vel[i].x = rad*sinf(angle1)*cosf(angle2);
        run_vel[i].y = rad*sinf(angle1)*sinf(angle2);
        run_vel[i].z = rad*cosf(angle1);
        run_pos[i].w = MASS;
    }
}

__device__ void get_forces(float4 *run_pos, float3 *run_force){
    float dx, dy, dz, r, r2, total_force, d;

    for(int i=0; i<NUMBER_OF_BODIES-1; i++){
        for(int j=i+1; j<NUMBER_OF_BODIES; j++){
            dx = run_pos[j].x-run_pos[i].x;
            dy = run_pos[j].y-run_pos[i].y;
            dz = run_pos[j].z-run_pos[i].z;

            r2 = dx*dx + dy*dy + dz*dz;
            r = sqrt(r2);

			if(r < DIAMETER_PS){
				total_force =  REPULSIVE_SLOPE*r - REPULSIVE_SLOPE*DIAMETER_PS + g_max_attraction;
			}else if (r < DIAMETER_PS + DIAMETER_NIPAM){
                float tmp = g_max_attraction/DIAMETER_NIPAM;
				total_force =  -tmp*r + tmp*(DIAMETER_PS + DIAMETER_NIPAM);
			}else{
                total_force = 0.0;
            }

			run_force[i].x += total_force*dx/r;
			run_force[i].y += total_force*dy/r;
			run_force[i].z += total_force*dz/r;
			run_force[j].x -= total_force*dx/r;
			run_force[j].y -= total_force*dy/r;
			run_force[j].z -= total_force*dz/r;
        }
    }

    for(int i=0; i<NUMBER_OF_BODIES; i++){
        d = sqrt(run_pos[i].x*run_pos[i].x + run_pos[i].y*run_pos[i].y + run_pos[i].z*run_pos[i].z);
        run_force[i].x += -CENTRAL_ATTRACTION_FORCE*g_max_attraction*run_pos[i].x/d;
		run_force[i].y += -CENTRAL_ATTRACTION_FORCE*g_max_attraction*run_pos[i].y/d;
		run_force[i].z += -CENTRAL_ATTRACTION_FORCE*g_max_attraction*run_pos[i].z/d;
    }
}

__device__ void update_positions_and_velocities(float4 *run_pos, float3 *run_vel, float3 *run_force){
    for(int i=0; i<NUMBER_OF_BODIES; i++){
        run_vel[i].x += DT*(run_force[i].x - g_drag*run_vel[i].x)/run_pos[i].w;
		run_vel[i].y += DT*(run_force[i].y - g_drag*run_vel[i].y)/run_pos[i].w;
		run_vel[i].z += DT*(run_force[i].z - g_drag*run_vel[i].z)/run_pos[i].w;

		run_pos[i].x += DT*run_vel[i].x;
		run_pos[i].y += DT*run_vel[i].y;
		run_pos[i].z += DT*run_vel[i].z;
    }
}

__global__ void nbody(float4 *pos, float3 *vel, float3 *force, unsigned int *bins){
	unsigned long id = (blockIdx.x*blockDim.x)+threadIdx.x;

    if(id == 0){ bins[0] = bins[1] = bins[2] = 0; }

	float4 *run_pos = pos + id*NUMBER_OF_BODIES;
	float3 *run_vel = vel + id*NUMBER_OF_BODIES;
	float3 *run_force = force + id*NUMBER_OF_BODIES;

    set_initial_conditions(run_pos, run_vel);

    float t = 0.0;
    while(t < STOP_TIME){
        for(int i=0; i<NUMBER_OF_BODIES; i++){
            run_force[i].x = 0.0;
            run_force[i].y = 0.0;
            run_force[i].z = 0.0;
        }

        get_forces(run_pos, run_force);
        update_positions_and_velocities(run_pos, run_vel, run_force);

        t += DT;
    }

    float dx, dy, dz;
    float total_body_to_body_distance = 0.0;
    for(int i=0; i<NUMBER_OF_BODIES-1; i++){
        for(int j=i+1; j<NUMBER_OF_BODIES; j++){
            dx = run_pos[j].x-run_pos[i].x;
            dy = run_pos[j].y-run_pos[i].y;
            dz = run_pos[j].z-run_pos[i].z;

            total_body_to_body_distance += sqrt(dx*dx + dy*dy + dz*dz);
        }
    }

    if(total_body_to_body_distance < 16.7) { atomicAdd(bins+0, 1); }
    else if(total_body_to_body_distance < 16.9) { atomicAdd(bins+1, 1); }
    else{ atomicAdd(bins+2, 1); }
}

int main(int argc, char** argv){
	float4 *gpu_pos;
	float3 *gpu_vel, *gpu_force;

    unsigned int *cpu_bins, *gpu_bins;

	cudaMalloc(&gpu_pos, NUMBER_OF_RUNS*NUMBER_OF_BODIES*sizeof(float4));
	cudaMalloc(&gpu_vel, NUMBER_OF_RUNS*NUMBER_OF_BODIES*sizeof(float3));
	cudaMalloc(&gpu_force, NUMBER_OF_RUNS*NUMBER_OF_BODIES*sizeof(float3));
	cudaMalloc(&gpu_bins, 3*sizeof(unsigned int));

	dim3 gridDim, blockDim;

	gridDim.x = NUMBER_OF_RUNS;
	gridDim.y = 1;
	gridDim.z = 1;

	blockDim.x = 1;
	blockDim.y = 1;
	blockDim.z = 1;

	nbody<<<gridDim, blockDim>>>(gpu_pos, gpu_vel, gpu_force, gpu_bins);
    CUDAErrorCheck("kernel error...\n");

	// Copy results
	cpu_bins = (unsigned int*)malloc(3*sizeof(unsigned int));
	cudaMemcpyAsync(cpu_bins, gpu_bins, 3*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Octahedron: %d\t\t->\t%.2f%\n", cpu_bins[0], (float)cpu_bins[0]/(float)NUMBER_OF_RUNS*100);
    printf("Poly-tetrahedron: %d\t->\t%.2f%\n", cpu_bins[1], (float)cpu_bins[1]/(float)NUMBER_OF_RUNS*100);
    printf("Other: %d\t\t->\t%.2f%\n", cpu_bins[2], (float)cpu_bins[2]/(float)NUMBER_OF_RUNS*100);

	return 0;
}
