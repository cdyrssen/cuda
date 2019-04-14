// nvcc hw05.cu -o hw05 -lglut -lGL -lm

#include "cuda.h"
#include <GL/glut.h>
#include <stdio.h>

#define INF 2e10f

#define rnd(x)          (x*rand()/RAND_MAX)
#define SPHERES         10
#define WINDOW_WIDTH    1000
#define WINDOW_HEIGHT   500

struct Sphere {
    float r, g, b;
    float x, y, z;
    float radius;
};

Sphere *s;
__constant__ Sphere gpu_spheres[SPHERES];
const int PIXELS = WINDOW_WIDTH*WINDOW_HEIGHT;

__device__ float hit(float ox, float oy, float *n, Sphere s) {
    float dx = ox-s.x;
    float dy = oy-s.y;

    if(dx*dx + dy*dy < s.radius*s.radius){
        float dz = sqrtf(s.radius*s.radius - (dx*dx + dy*dy));
        *n = dz/s.radius;
        return dz+s.z;
    }

    return -INF;
}

__global__ void ray_tracer_const(float *output_render){
    unsigned long id = 3*(threadIdx.x+blockDim.x*blockIdx.x);
    int x = threadIdx.x;
    int y = blockIdx.x;
    float maxz = -INF;

    for(int i=0; i<SPHERES; i++){
        float color_scale;
        float t = hit(x, y, &color_scale, gpu_spheres[i]);

        if(t > maxz){
            output_render[id+0] = gpu_spheres[i].r*color_scale;
            output_render[id+1] = gpu_spheres[i].g*color_scale;
            output_render[id+2] = gpu_spheres[i].b*color_scale;
        }
    }
}

__global__ void ray_tracer(float *output_render, Sphere *s){
    unsigned long id = 3*(threadIdx.x+blockDim.x*blockIdx.x);
    int x = threadIdx.x;
    int y = blockIdx.x;
    float maxz = -INF;

    for(int i=0; i<SPHERES; i++){
        float color_scale;
        float t = hit(x, y, &color_scale, s[i]);

        if(t > maxz){
            output_render[id+0] = s[i].r*color_scale;
            output_render[id+1] = s[i].g*color_scale;
            output_render[id+2] = s[i].b*color_scale;
        }
    }
}

void CUDAErrorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

void display(void){
    float *cpu_render, *gpu_render;
    Sphere *cpu_spheres;


    // allocate memory for raytracing on GPU
    cudaMalloc(&gpu_render, PIXELS*3*sizeof(float));
    CUDAErrorCheck("cuda malloc gpu render\n");

    // allocate memory for spheres on GPU
    cudaMalloc(&s, SPHERES*sizeof(Sphere));
    CUDAErrorCheck("cuda malloc global spheres\n");

    // generate and copy spheres to gpu
    cpu_spheres = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
    for(int i=0; i<SPHERES; i++){
        cpu_spheres[i].r = rnd(1.0f);
        cpu_spheres[i].g = rnd(1.0f);
        cpu_spheres[i].b = rnd(1.0f);
        cpu_spheres[i].x = rnd((float)WINDOW_WIDTH);
        cpu_spheres[i].y = rnd((float)WINDOW_HEIGHT);
        cpu_spheres[i].z = rnd(1000.0f);
        cpu_spheres[i].radius = rnd(100.0f)+20;
    }
    cudaMemcpyToSymbol(gpu_spheres, cpu_spheres, sizeof(Sphere)*SPHERES);
    CUDAErrorCheck("cuda copy spheres to constant memory\n");
    cudaMemcpyAsync(s, cpu_spheres, SPHERES*sizeof(Sphere), cudaMemcpyHostToDevice);
    CUDAErrorCheck("cuda memcpy spheres to global memory\n");
    free(cpu_spheres);

    cudaEvent_t start,  stop;

    // Global
    cudaEventCreate(&start);
    CUDAErrorCheck("cuda event create start for global ray tracing\n");
    cudaEventCreate(&stop);
    CUDAErrorCheck("cuda event create stop for global ray tracing\n");
    cudaEventRecord(start, 0);
    CUDAErrorCheck("cuda event record start for global ray tracing\n");

    ray_tracer<<<WINDOW_HEIGHT, WINDOW_WIDTH>>>(gpu_render, s);

    cudaEventRecord(stop, 0);
    CUDAErrorCheck("cuda event record stop for global ray tracing\n");
    cudaEventSynchronize(stop);
    CUDAErrorCheck("cuda event synchronization for global ray tracing\n");
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    CUDAErrorCheck("cuda event elapsed time for global ray tracing\n");
    printf("Render time for global ray tracing: %3.3f ms\n", elapsed_time);
    /*cudaEventDestroy(start);
    CUDAErrorCheck("cuda event destroy start for global ray tracing\n");
    cudaEventDestroy(stop);
    CUDAErrorCheck("cuda event destroy stop for global ray tracing\n");*/

    // Constant
    cudaEventCreate(&start);
    CUDAErrorCheck("cuda event create start for constant ray tracing\n");
    cudaEventCreate(&stop);
    CUDAErrorCheck("cuda event create stop for constant ray tracing\n");
    cudaEventRecord(start, 0);
    CUDAErrorCheck("cuda event record start for constant ray tracing\n");

    ray_tracer_const<<<WINDOW_HEIGHT, WINDOW_WIDTH>>>(gpu_render);

    cudaEventRecord(stop, 0);
    CUDAErrorCheck("cuda event record stop for constant ray tracing\n");
    cudaEventSynchronize(stop);
    CUDAErrorCheck("cuda event synchronization for constant ray tracing\n");
    cudaEventElapsedTime(&elapsed_time, start, stop);
    CUDAErrorCheck("cuda event elapsed time for constant ray tracing\n");
    printf("Render time constant ray tracing: %3.3f ms\n", elapsed_time);
    cudaEventDestroy(start);
    CUDAErrorCheck("cuda event destroy start for constant ray tracing\n");
    cudaEventDestroy(stop);
    CUDAErrorCheck("cuda event destroy stop for constant ray tracing\n");

    cpu_render = (float*)malloc(PIXELS*3*sizeof(float));
    cudaMemcpyAsync(cpu_render, gpu_render, PIXELS*3*sizeof(float), cudaMemcpyDeviceToHost);
    CUDAErrorCheck("cuda memcpy render from device to host\n");
    cudaFree(gpu_render);
    CUDAErrorCheck("cuda free gpu render bitmap\n");
    cudaFree(s);

    glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, cpu_render);
    glFlush();

    free(cpu_render);
}

int main(int argc, char** argv){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Raytracing");
    glutDisplayFunc(display);
    glutMainLoop();
}
