#pragma once
#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <windows.h>
#include <tchar.h>
using namespace std;
# define PI 3.1415926
# define RENDER_STATE_WIREFRAME 1 //��Ⱦ�߿�
# define RENDER_STATE_TEXTURE 2   //��Ⱦ����
# define RENDER_STATE_COLOR 4     //��Ⱦ��ɫ
typedef unsigned int IUINT32;
namespace ydc_ns {
	//======================
	//��ѧ��
	//======================
	//4*4������ �� 1*4��������
	typedef struct  { float m[4][4]; } matrix_t;
	typedef struct { float x, y, z, w; } vector_t;
	void printVector(const vector_t *v);
	void printMatrix(const matrix_t *m);

	void printTest();
	int T_clamp(int x, int min, int max);
	float F_clamp(float x,float min,float max);
	float interp(float x1, float x2, float t);
	void vector_copy(vector_t *v, const vector_t *v1);
	float vector_length(const vector_t *v);
	void vector_add(vector_t *v, const vector_t *v1, const vector_t *v2);
	void vector_sub(vector_t *v, const vector_t *v1, const vector_t *v2);
	float vector_dotProduct(const vector_t *v1, const vector_t *v2);
	void vector_crossProduct(vector_t *v, const vector_t *v1, const vector_t *v2);
	void vector_interp(vector_t *v, const vector_t *v1, const vector_t *v2, float t);
	void vector_normalize(vector_t *v);
	void vector_scale(vector_t *v, float k);
	
	void matrix_copy(matrix_t *m, const matrix_t *m1);
	void matrix_identity(matrix_t *m);
	void matrix_zeros(matrix_t *m);
	void matrix_add(matrix_t *m, const matrix_t *m1, const matrix_t *m2);
	void matrix_sub(matrix_t *m, const matrix_t *m1, const matrix_t *m2);
	void matrix_multiply(matrix_t *m, const matrix_t *m1, const matrix_t *m2);
	void matrix_scale(matrix_t *m, matrix_t *m1, float k);
	void vector_matrix_multipy(vector_t *v, const vector_t *v1, const matrix_t *m1);
	void matrix_set_translate(matrix_t *m, float x, float y, float z);
	void matrix_set_scale(matrix_t *m, float sx, float sy, float sz);
	void matrix_set_rotate(matrix_t *m, const vector_t *v, float theta);
	void matrix_set_lookat(matrix_t *m, const vector_t *eye, const vector_t *target, const vector_t *up);
	void matrix_set_perspective(matrix_t *m, float fov_y, float wOverH, float n, float f);

	//================
	//����任
	//================
	typedef struct {
		matrix_t m_world;		//��������任
		matrix_t m_view;		//���������任
		matrix_t m_projection;  //ͶӰ����任
		matrix_t m_transform;	//transform = world * view * projection
		float w, h;				//��Ļ��С
	}	transform_t;
	void transform_update(transform_t *ts);
	void transform_init(transform_t *ts, int w, int h);
	void transform_apply(vector_t *y, const vector_t *x, const transform_t *ts);
	int transform_check_cvv(const vector_t *v);
	void transform_to_UV(vector_t *uv, const vector_t *p, const transform_t *ts);

	//=====================================================================
	// ���μ��㣺���㡢ɨ���ߡ���Ե�����Ρ���������
	//=====================================================================
	typedef struct { float r, g, b; } color_t;	
	void color_scale(color_t* c, float k);
	void color_add(color_t* c,const color_t * c1,color_t * c2);
	void color_clamp(color_t* c);
	typedef struct { float u, v; }texcoord_t;
	typedef struct { vector_t pos; texcoord_t tc; color_t color; float rhw;} vertex_t;//tcΪ��������

	typedef struct { vertex_t v, v1, v2; }edge_t;
	typedef struct { float top, bottom; edge_t left, right; }trapezoid_t;
	typedef struct { vertex_t v, step; int x, y, w; }scanline_t;

	void vertex_init_thw(vertex_t *v);
	void vertex_interp(vertex_t *v, const vertex_t *v1, const vertex_t *v2, float t);
	void vertex_add(vertex_t *y, const vertex_t *x);
	void vertex_division(vertex_t *y, const vertex_t *x1, const vertex_t *x2, float w);
	int trapezoid_init_triangle(trapezoid_t *trap, const vertex_t *p1,const vertex_t *p2, const vertex_t *p3);
	void trapezoid_edge_interp(trapezoid_t *trap, float y);
	void trapezoid_init_scanline(const trapezoid_t *trap, scanline_t *scanline, int y);
	//=====================
	// ��Ⱦ�豸
	//=====================
	typedef struct {
		transform_t transform;      // ����任����
		int width;                  // ���ڿ��
		int height;                 // ���ڸ߶�
		IUINT32 **framebuffer;      // ���ػ��棺framebuffer[y] ����� y��
		float **zbuffer;            // ��Ȼ��棺zbuffer[y] Ϊ�� y��ָ��
		IUINT32 **texture;          // ����ͬ����ÿ������
		int tex_width;              // ������
		int tex_height;             // ����߶�
		float max_u;                // ��������ȣ�tex_width - 1
		float max_v;                // �������߶ȣ�tex_height - 1
		int render_state;           // ��Ⱦ״̬
		IUINT32 background;         // ������ɫ
		IUINT32 foreground;         // �߿���ɫ
	}	device_t;
	void device_init(device_t *device, int width, int height, void *fb);
	void device_destroy(device_t *device);
	void device_set_texture(device_t *device, void *bits, long sizePerRow, int w, int h);
	void device_clear(device_t *device, int mode);
	void draw_pixel(device_t *device, int u, int v, IUINT32 color);
	void draw_line(device_t *device, int x1, int y1, int x2, int y2, IUINT32 color);
	IUINT32 device_texture_read(const device_t *device, float u, float v);
	void device_draw_scanline(device_t *device, scanline_t *scanline);
	void device_draw_trap(device_t *device, trapezoid_t *trap);
	void device_draw_primitive_triangle(device_t *device, const vertex_t *v1, vertex_t *v2, vertex_t *v3);
}
