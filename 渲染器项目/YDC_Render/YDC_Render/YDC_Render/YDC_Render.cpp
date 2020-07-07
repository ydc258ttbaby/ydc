// YDC_Render.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <windows.h>
#include <tchar.h>

#include "mathYdc.h"
#include "loadOBJ.h"
#include "winScreen.h"
#include <string.h>

# define PI 3.141592653589793238462643383279

using namespace ydc_ns;
int lastMouseX = -1, lastMouseY = -1;
vector_t up = { 0,1,0,1 };
vector_t eye = { 1,-1,15,1 }, target = { 0,0,0,1 };
vector_t lightSource = {-2,2,10,1};
float intensity = 200;
float DiffuseReflectionCoefficient = 1;
float SpecularReflectionCoefficient = 2.0;
float Sharpness = 1.0f;

//=================================
// 主程序
//=================================

void ComputeBlimPhongLight(vertex_t* p, const vector_t& n)
{
	color_t FinalColor = { 0.1f,0.4f,0.7f };
	
	// 环境分量
	float EnvironmentLight = 0.5;

	// 漫反射分量
	vector_t L;
	vector_sub(&L, &lightSource ,&p->pos);
	float R = vector_length(&L);
	vector_normalize(&L);
	float DiffuseLight = DiffuseReflectionCoefficient*(intensity/R/R) * (max(0.0, vector_dotProduct(&n, &L)));

	// 镜面反射分量
	vector_t lookAt,h;
	vector_sub(&lookAt,&target,&eye);
	vector_add(&lookAt,&lookAt,&p->pos);
	vector_normalize(&lookAt);
	vector_add(&h,&L,&lookAt);
	vector_normalize(&h);
	float SpecularLight = SpecularReflectionCoefficient * (intensity/R/R)* pow(max(0.0,vector_dotProduct(&n,&h)),Sharpness);

	color_scale(&FinalColor, EnvironmentLight + SpecularLight+ DiffuseLight);
	color_clamp(&FinalColor);
	p->color = FinalColor;

}
//画读进来的mesh
void draw_mesh(device_t *device, TriangleMesh_t *triangleMesh) {
	for (auto IB : triangleMesh->index_buffers) {
		vertex_t p1 = triangleMesh->vertexs[IB.p1 - 1], p2 = triangleMesh->vertexs[IB.p2 - 1], p3 = triangleMesh->vertexs[IB.p3 - 1];
		p1.tc.u = 0, p1.tc.v = 0;
		p2.tc.u = 0, p2.tc.v = 1;
		p3.tc.u = 1, p3.tc.v = 1;

		vector_t side1,side2,vectorN;
		vector_sub(&side1,&p1.pos,&p2.pos);
		vector_sub(&side2,&p1.pos,&p3.pos);
		vector_crossProduct(&vectorN,&side1,&side2);
		vector_normalize(&vectorN);

		ComputeBlimPhongLight(&p1, vectorN);
		ComputeBlimPhongLight(&p2, vectorN);
		ComputeBlimPhongLight(&p3, vectorN);
		device_draw_primitive_triangle(device, &p1, &p2, &p3);
	}
}


//设置相机,传入eye
void camera_at_zero(device_t *device, float x, float y, float z) {
	vector_t eye = { x,y,z,1 }, target = { 0,0,0,1 }, up = { 0,1,0,1 };
	matrix_set_lookat(&device->transform.m_view, &eye, &target, &up);
	transform_update(&device->transform);
}
//将eye的球坐标beta，alpha，r转换为xyz

void eye_coord_transform(vector_t *eye, const vector_t *target,float &alpha, float &beta) {
	matrix_t m_rotate;
	vector_t v1, v2, v3, v5, n, n1;
	
	vector_sub(&v1, eye, target);
	if (alpha != 0.0) {
		vector_crossProduct(&v3, &up, &v1);
		vector_crossProduct(&n, &v1, &v3);
		vector_normalize(&n);
		matrix_set_rotate(&m_rotate, &n, alpha);
		vector_matrix_multipy(&v2, &v1, &m_rotate);
		vector_add(eye, target, &v2);
		alpha = 0.0;
	}
	vector_sub(&v1, eye, target);
	if (beta != 0.0) {

		vector_crossProduct(&n, &v1, &up);
		vector_normalize(&n);
		matrix_set_rotate(&m_rotate, &n,beta);
		vector_matrix_multipy(&v2, &v1, &m_rotate);
		vector_add(eye, target, &v2);
		beta = 0.0;
	}
	//cout << "n1: "; printVector(&n1);
}
void MoveViewAndTarget(vector_t *eye, vector_t *target)
{
	
	vector_t v1,v3, v2;
	vector_sub(&v1, eye, target);
	vector_crossProduct(&v3, &up, &v1);
	vector_crossProduct(&v2, &v1, &v3);
	vector_normalize(&v1);
	vector_normalize(&v2);
	vector_normalize(&v3);
	float MoveSpeed = 0.1f;
	vector_scale(&v1, MoveSpeed);
	vector_scale(&v2, MoveSpeed);
	vector_scale(&v3, MoveSpeed);
	vector_t eyeTemp, targetTemp,upTemp;
	vector_copy(&upTemp, &up);
	vector_copy(&eyeTemp, eye);
	vector_copy(&targetTemp, target);
	vector_scale(&upTemp, MoveSpeed);
	if (screen_keys[VK_A]) vector_add(&eyeTemp, eye, &v3), vector_add(&targetTemp, target, &v3), cout << 'd' << endl;
	if (screen_keys[VK_D]) vector_sub(&eyeTemp, eye, &v3), vector_sub(&targetTemp, target, &v3), cout << 'a' << endl;
	if (screen_keys[VK_Q]) vector_add(&eyeTemp, eye, &v2), vector_add(&targetTemp, target, &v2), cout << 'w' << endl;
	if (screen_keys[VK_E]) vector_sub(&eyeTemp, eye, &v2), vector_sub(&targetTemp, target, &v2), cout << 'S' << endl;
	if (screen_keys[VK_S]) vector_add(&eyeTemp, eye, &v1), vector_add(&targetTemp, target, &v1), cout << 'q' << endl;
	if (screen_keys[VK_W]) vector_sub(&eyeTemp, eye, &v1), vector_sub(&targetTemp, target, &v1), cout << 'e' << endl;

	vector_copy(eye, &eyeTemp);
	vector_copy(target, &targetTemp);

}
//漫游参数修改
void device_transform_update(device_t *device, const vector_t *eye, const vector_t *target,const vector_t *up) {
	matrix_set_lookat(&device->transform.m_view, eye, target, up);
	transform_update(&device->transform);
}
//初始化纹理，设置纹理
void init_texture(device_t * device) {
	//static修饰的局部变量，实际属于全局变量，在此函数退出时，其值不会被改变
	static IUINT32 texture[256][256];
	for (int i = 0; i < 256; i++) {
		for (int j = 1; j < 256; j++) {
			int x = i / 32, y = j / 32;//int的除法，会被舍去小数
			//判断x+y的奇偶性
			texture[i][j] = ((x + y) & 1) ? 0xfffb2f : 0x3fbcef;
		}
	}
	device_set_texture(device, texture, 256 * 4, 256, 256);
}
float curveSmooth(float x)
{
	return (x != 0)?min(pow(x, 3.0f),100.0f):0;
}
int main()
{
	printTest();
	
	//读取mesh数据
	TriangleMesh_t triangleMesh1, triangleMesh2, triangleMesh3, triangleMesh4;
	string filename1 = ".\\resource\\cone.obj";
	string filename2 = ".\\resource\\cube.obj";
	string filename3 = ".\\resource\\sphere.obj";
	string filename4 = ".\\resource\\cylinder.obj";
	loadOBJ(filename1, &triangleMesh1);
	loadOBJ(filename2, &triangleMesh2);
	loadOBJ(filename3, &triangleMesh3);
	loadOBJ(filename4, &triangleMesh4);

	//声明定义
	device_t device;
	int states[] = { RENDER_STATE_TEXTURE,RENDER_STATE_COLOR,RENDER_STATE_WIREFRAME };
	int index = 0;
	int kbhit = 0;
	

	float alpha = 0.0;
	float beta = 0.0;
	float speed = 0.1f;
	float rotateSpeed = 0.016f;

	TCHAR *title = _T("ydc render ")
		_T(" test ");

	//初始化
	if (screen_init(800, 600, title))return -1;
	device_init(&device, 800, 600, screen_fb);
	init_texture(&device);
	device.render_state = RENDER_STATE_COLOR;

	//死循环渲染
	while (screen_exit == 0 && screen_keys[VK_ESCAPE] == 0) {
		screen_dispatch();
		//最开始清理
		device_clear(&device, 0);

		//手动改变 渲染状态
		if (screen_keys[VK_SPACE]) {
			if (kbhit == 0) {
				kbhit = 1;
				if (++index >= 3)index = 0;
				device.render_state = states[index];
			}
		}
		else {
			kbhit = 0;
		}

		//手动改变旋转角度 alpha
		if (buttons & MK_LBUTTON)
		{
			//cout << mouse_x << endl;
			//cout << mouse_y << endl;
			cout << "x: " << mouse_x << " y: " << mouse_y << endl;
			cout << "Lx: " << lastMouseX << " Ly: " << lastMouseX << endl;
			if (lastMouseX >= 0 && lastMouseY >= 0)
			{
				alpha += curveSmooth(mouse_x - lastMouseX)*rotateSpeed;
				beta += curveSmooth(mouse_y - lastMouseY)*rotateSpeed;
			}
		}
		//cout << "wz: " << wzDelta << endl;


		/*if (screen_keys[VK_RIGHT]) alpha -= rotateSpeed;
		if (screen_keys[VK_LEFT]) alpha += rotateSpeed;
		if (screen_keys[VK_DOWN]) beta -= rotateSpeed;
		if (screen_keys[VK_UP]) beta += rotateSpeed;
		*/MoveViewAndTarget(&eye, &target);


		//更新transform
		//cout << "beta:" << beta << endl;
		//cout << "eye:"; printVector(&eye);
		eye_coord_transform(&eye, &target,alpha, beta);
		device_transform_update(&device, &eye, &target, &up);

		//渲染
		draw_mesh(&device, &triangleMesh1);
		draw_mesh(&device, &triangleMesh2);
		draw_mesh(&device, &triangleMesh3);
		draw_mesh(&device, &triangleMesh4);
		screen_update();
		Sleep(1);

		lastMouseX = mouse_x;
		lastMouseY = mouse_y;
	}
	getchar();
	return 0;
}
