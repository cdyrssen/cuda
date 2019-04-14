// Optimized using shared memory and on chip memory
// nvcc hw09.cu -o hw09 -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 32768
#define halfN	N/2
#define BLOCK 256

#define XWindowSize 2500
#define YWindowSize 2500

#define DRAW 10
#define DAMP 1.0

#define DT 0.001
#define STOP_TIME 10.0

#define G 1.0
#define H 1.0

#define EYE 45.0
#define FAR 90.0

// Globals
float4 p[N];
float3 v[N], f[N];
float4 *pos;
float3 *vel, *force;
FILE *data_file, *data_file1, *data_file2;
dim3 block, grid;
int NumberOfGpus, Gpu0Access, Gpu1Access;
const bool UseMultipleGPU = 1;

void CUDAerrorCheck(const char *message){
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess){
    printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

void set_initail_conditions(){
	int i,j,k,num,particles_per_side;
	float position_start, temp;
	float initial_seperation;

	temp = pow((float)N,1.0/3.0) + 0.99999;
	particles_per_side = temp;
	printf("\n cube root of N = %d \n", particles_per_side);
	position_start = -(particles_per_side -1.0)/2.0;
	initial_seperation = 2.0;

	for(i=0; i<N; i++) p[i].w = 1.0;

	num = 0;
	for(i=0; i<particles_per_side; i++){
		for(j=0; j<particles_per_side; j++){
			for(k=0; k<particles_per_side; k++){
				if(N <= num) break;
				p[num].x = position_start + i*initial_seperation;
				p[num].y = position_start + j*initial_seperation;
				p[num].z = position_start + k*initial_seperation;
				v[num].x = 0.0;
				v[num].y = 0.0;
				v[num].z = 0.0;
				num++;
			}
		}
	}

	block.x = BLOCK;
	block.y = 1;
	block.z = 1;

	grid.x = (N-1)/block.x + 1;
	grid.y = 1;
	grid.z = 1;

	cudaMalloc( (void**)&pos, N *sizeof(float4) );
	cudaMalloc( (void**)&vel, N *sizeof(float3) );
	cudaMalloc( (void**)&force, N *sizeof(float3) );
}

void draw_picture(){
	int i;

	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++){
		glPushMatrix();
		glTranslatef(p[i].x, p[i].y, p[i].z);
		glutSolidSphere(0.1,20,20);
		glPopMatrix();
	}

	glutSwapBuffers();
}

__device__ float3 getBodyBodyForce(float4 p0, float4 p1){
	float3 f;
	float dx = p1.x - p0.x;
	float dy = p1.y - p0.y;
	float dz = p1.z - p0.z;
	float r2 = dx*dx + dy*dy + dz*dz;
	float r = sqrt(r2);

	float force  = (G*p0.w*p1.w)/(r2) - (H*p0.w*p1.w)/(r2*r2);

	f.x = force*dx/r;
	f.y = force*dy/r;
	f.z = force*dz/r;

	return(f);
}

__global__ void getForcesCollisionDoubleGPU(float4 *pos, float3 *vel, float3 *force){
	int j,ii;
	float3 force_mag, forceSum;
	float4 posMe;
	__shared__ float4 shPos[BLOCK];
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	forceSum.x = 0.0;
	forceSum.y = 0.0;
	forceSum.z = 0.0;

	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	posMe.w = pos[id].w;

	for(j=0; j<gridDim.x; j++){
		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
		__syncthreads();

		#pragma unroll 32
		for(int i=0; i<blockDim.x; i++){
			ii = i + blockDim.x*j;
			if(ii != id && ii < N){
				force_mag = getBodyBodyForce(posMe, shPos[i]);
				forceSum.x += force_mag.x;
				forceSum.y += force_mag.y;
				forceSum.z += force_mag.z;
			}
		}
	}

	if(id < N){
		force[id].x = forceSum.x;
		force[id].y = forceSum.y;
		force[id].z = forceSum.z;
	}
}

__global__ void moveBodiesCollisionDoubleGPU(float4 *pos, float3 *vel, float3 *force){
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id<N){
		vel[id].x += ((force[id].x-DAMP*vel[id].x)/pos[id].w)*DT;
		vel[id].y += ((force[id].y-DAMP*vel[id].y)/pos[id].w)*DT;
		vel[id].z += ((force[id].z-DAMP*vel[id].z)/pos[id].w)*DT;

		pos[id].x += vel[id].x*DT;
		pos[id].y += vel[id].y*DT;
		pos[id].z += vel[id].z*DT;
	}
}

void getNumberOfGPUs(){
	cudaGetDeviceCount(&NumberOfGpus);
	printf("\n***** You have %d GPUs available\n", NumberOfGpus);
}

void checkPeerToPeerAccess(){
	if(1 < NumberOfGpus && UseMultipleGPU == 1){
		cudaDeviceCanAccessPeer(&Gpu0Access,0,1);
		cudaDeviceCanAccessPeer(&Gpu1Access,1,0);

		printf("\n***** You will be using %d GPUs\n", NumberOfGpus);

		if(Gpu0Access == 0) printf("\nTSU Error: Device0 can not do peer to peer\n");
		if(Gpu1Access == 0) printf("\nTSU Error: Device1 can not do peer to peer\n");

		cudaDeviceEnablePeerAccess(1,0);
	}
}

void n_body(){
	float dt;
	int   tdraw = 0;
	float time = 0.0;
	float elapsedTime;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dt = DT;

    float4 *pos_gpu0, *pos_gpu1, *pos0, *pos1;
    float3 *vel_gpu0, *vel_gpu1, *vel0, *vel1;

    pos_gpu0 = pos;
    pos_gpu1 = pos+N/2;
    vel_gpu0 = vel;
    vel_gpu1 = vel+N/2;

	cudaSetDevice(0);
	cudaMemcpy(pos_gpu0, pos0, N/2*sizeof(float4), cudaMemcpyHostToDevice);
	//CUDAerrorCheck("gpu position copy...\n");
	cudaMemcpy(vel_gpu0, vel0, N/2*sizeof(float3), cudaMemcpyHostToDevice);
	//CUDAerrorCheck("gpu velocity copy...\n");

	cudaSetDevice(1);
	cudaMemcpy(pos_gpu1, pos1+N/2, (N-N/2)*sizeof(float4), cudaMemcpyHostToDevice);
	//CUDAerrorCheck("gpu position copy...\n");
	cudaMemcpy(vel_gpu1, vel1+N/2, (N-N/2)*sizeof(float3), cudaMemcpyHostToDevice);
	//CUDAerrorCheck("gpu velocity copy...\n");

	block.x = BLOCK;
	block.y = 1;
	block.z = 1;

	grid.x = (N-1)/block.x + 1;
	grid.y = 1;
	grid.z = 1;

	while(time < STOP_TIME){
		cudaSetDevice(0);
		getForcesCollisionDoubleGPU<<<grid, block>>>(pos_0, vel_0, force);
		//CUDAerrorCheck("gpu0 force kernel...\n");
		moveBodiesCollisionDoubleGPU<<<grid, block>>>(pos_0, vel_0, force);
		//CUDAerrorCheck("gpu0 move kernel...\n");

		cudaSetDevice(1);
		getForcesCollisionDoubleGPU<<<grid, block>>>(pos_1, vel_1, force);
		//CUDAerrorCheck("gpu1 force kernel...\n");
		moveBodiesCollisionDoubleGPU<<<grid, block>>>(pos_1, vel_1, force);
		//CUDAerrorCheck("gpu1 move kernel...\n");

		cudaDeviceSynchronize();

		cudaSetDevice(0);
		cudaMemcpyPeerAsync(pos_1,1,pos_0,0,(N/2)*sizeof(float4));
		cudaMemcpyPeerAsync(vel_1,1,vel_0,0,(N/2)*sizeof(float4));

		cudaDeviceSynchronize();

		//To kill the draw comment out the next 7 lines.
		/*if(tdraw == DRAW) {
			cudaMemcpy(p, p_GPU, N *sizeof(float4), cudaMemcpyDeviceToHost);
			draw_picture();
			tdraw = 0;
		}
		tdraw++;*/

		time += dt;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %3.1f milliseconds\n", elapsedTime);
	//cudaMemcpy( p, p_GPU, N *sizeof(float4), cudaMemcpyDeviceToHost );
}

void control(){
	set_initail_conditions();
	draw_picture();
	n_body();
	draw_picture();

	printf("\n DONE \n");
	while(1);
}

void Display(void){
	gluLookAt(EYE, EYE, EYE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	control();
}

void reshape(int w, int h){
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, FAR);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv){
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
