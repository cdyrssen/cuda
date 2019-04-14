//nvcc fractalSimpleCPU.cu -o temp -lglut -lGL -lm
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define A  -0.984    //real
#define B  -0.2911   //imaginary

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);

__device__ double cuda_color (double x, double y){
	float mag,maxMag,t1;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;

	while (mag < maxMag && count < maxCount){
		t1 = x;
		x = x*x - y*y + A;
		y = (2.0 * t1 * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}

	if(count < maxCount) return(count/maxCount);
	else return(mag/maxMag);
}

__global__ void fractal(float *pixels){
    unsigned long id = (blockIdx.x * blockDim.x) + threadIdx.x;
    id *= 3;
    double x_map = (threadIdx.x-512.0)*(4.0/1024.0);
    double y_map = (blockIdx.x-512.0)*(4.0/1024.0);
    pixels[id] = cuda_color(x_map, y_map);
    pixels[id+1] = 3*cuda_color(x_map, y_map);
    pixels[id+2] = 2*cuda_color(x_map, y_map);
}

void display(void){
	float *pixels, *gpu_pixels;
    dim3 dimBlock, dimGrid;

    dimBlock.x = window_width;
    dimBlock.y = 1;
    dimBlock.z = 1;
    dimGrid.x = window_height;
    dimGrid.y = 1;
    dimGrid.z = 1;

    cudaMalloc(&gpu_pixels, window_width*window_height*3*sizeof(float));
    fractal<<<dimGrid, dimBlock>>>(gpu_pixels);
	pixels = (float *)malloc(window_width*window_height*3*sizeof(float));
    cudaMemcpyAsync(pixels, gpu_pixels, window_width*window_height*3*sizeof(float), cudaMemcpyDeviceToHost);

	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels);
	glFlush();
}

int main(int argc, char** argv){
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractals man, fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}
