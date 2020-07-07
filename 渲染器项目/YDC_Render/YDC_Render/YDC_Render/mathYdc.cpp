#include "stdafx.h"
#include "mathYdc.h"
#include <iostream>
using namespace std;
using namespace ydc_ns;
//测试
void ydc_ns::printTest() {
	cout << "hello world!" << endl;
}

//======================
//数学库
//======================
// clamp函数
int ydc_ns::T_clamp(int x, int min, int max) {
	if (x < min)
		return min;
	else if (x > max)
		return max;
	else
		return x;
}
float ydc_ns::F_clamp(float x, float min, float max)
{
	if (x < min)
		return min;
	else if (x > max)
		return max;
	else
		return x;
}
// 插值interp函数
float ydc_ns::interp(float x1, float x2, float t) {
	return x1 + (x2 - x1)*t;
}
//向量拷贝
void ydc_ns::vector_copy(vector_t *v, const vector_t *v1) {
	v->x = v1->x;
	v->y = v1->y;
	v->z = v1->z;
	v->w = v1->w;
}
//计算向量的模，只考虑x,y,z，不考虑w
float ydc_ns::vector_length(const vector_t *v) {
	return sqrt(v->x * v->x + v->y * v->y + v->z*v->z);
}
//计算向量加法 v = v1 + v2
void ydc_ns::vector_add(vector_t *v, const vector_t *v1, const vector_t *v2) {
	v->x = v1->x + v2->x;
	v->y = v1->y + v2->y;
	v->z = v1->z + v2->z;
	v->w = 1.0f;
}
//计算向量减法 v = v1 - v2
void ydc_ns::vector_sub(vector_t *v, const vector_t *v1, const vector_t *v2) {
	v->x = v1->x - v2->x;
	v->y = v1->y - v2->y;
	v->z = v1->z - v2->z;
	v->w = 1.0f;
}
//向量点积 v1 ·v2
float ydc_ns::vector_dotProduct(const vector_t *v1, const vector_t *v2) {
	return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}
//向量叉积 v = v1 x v2
void ydc_ns::vector_crossProduct(vector_t *v, const vector_t *v1, const vector_t *v2) {
	v->x = v1->y * v2->z - v2->y * v1->z;
	v->y = v1->z * v2->x - v2->z * v1->x;
	v->z = v1->x * v2->y - v2->x * v1->y;
	v->w = 1.0f;
}
//向量插值 v = v1 + (v2 - v1)*t 
void ydc_ns::vector_interp(vector_t *v, const vector_t *v1, const vector_t *v2,float t) {
	v->x = v1->x + (v2->x - v1->x)*t;
	v->y = v1->y + (v2->y - v1->y)*t;
	v->z = v1->z + (v2->z - v1->z)*t;
	v->w = 1.0f;
}
//向量归一化 norm(*v)
void ydc_ns::vector_normalize(vector_t *v) {
	float l = ydc_ns::vector_length(v);
	if (l > 0.000001) {
		v->x = v->x / l;
		v->y = v->y / l;
		v->z = v->z / l;
	}
	v->w = 1.0f;
}
void ydc_ns::vector_scale(vector_t *v,float k)
{
	v->x = v->x *k;
	v->y = v->y *k;
	v->z = v->z *k;
}
// 颜色缩放
void ydc_ns::color_scale(color_t* c, float k)
{
	c->r = c->r * k;
	c->g = c->g * k;
	c->b = c->b * k;
}
// 颜色相加
void ydc_ns::color_add(color_t* c, const color_t* c1, color_t* c2)
{
	c->r = c1->r + c2->r;
	c->g = c1->g + c2->g;
	c->b = c1->b + c2->b;
}
void ydc_ns::color_clamp(color_t* c)
{
	c->r = F_clamp(c->r, 0.0f, 1.0f);
	c->g = F_clamp(c->g, 0.0f, 1.0f);
	c->b = F_clamp(c->b, 0.0f, 1.0f);
}
//矩阵拷贝
void ydc_ns::matrix_copy(matrix_t *m, const matrix_t *m1) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j];
		}
	}
}
//矩阵置为单位矩阵 identity(*m)
void ydc_ns::matrix_identity(matrix_t *m) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (i == j)
				m->m[i][j] = 1;
			else
				m->m[i][j] = 0;
		}
	}
}
//矩阵置0 zeros(*m)
void ydc_ns::matrix_zeros(matrix_t *m) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = 0;
		}
	}
}
//矩阵加法 m1 + m2
void ydc_ns::matrix_add(matrix_t *m, const matrix_t *m1, const matrix_t *m2) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j] + m2->m[i][j];
		}
	}
}
//矩阵减法 m1 - m2
void ydc_ns::matrix_sub(matrix_t *m, const matrix_t *m1, const matrix_t *m2) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j] - m2->m[i][j];
		}
	}
}
//矩阵乘法 m1 * m2
void ydc_ns::matrix_multiply(matrix_t *m, const matrix_t *m1, const matrix_t *m2) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][0] * m2->m[0][j]
						+ m1->m[i][1] * m2->m[1][j]
						+ m1->m[i][2] * m2->m[2][j]
						+ m1->m[i][3] * m2->m[3][j];
		}
	}
}
//矩阵数乘 k*m1
void ydc_ns::matrix_scale(matrix_t *m, matrix_t *m1, float k) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j] * k;
		}
	}
}
//向量与矩阵相乘 v1 * m1
void ydc_ns::vector_matrix_multipy(vector_t *v, const vector_t *v1, const matrix_t *m1) {
	v->x = v1->x * m1->m[0][0] + v1->y * m1->m[1][0] + v1->z * m1->m[2][0] + v1->w * m1->m[3][0];
	v->y = v1->x * m1->m[0][1] + v1->y * m1->m[1][1] + v1->z * m1->m[2][1] + v1->w * m1->m[3][1];
	v->z = v1->x * m1->m[0][2] + v1->y * m1->m[1][2] + v1->z * m1->m[2][2] + v1->w * m1->m[3][2];
	v->w = v1->x * m1->m[0][3] + v1->y * m1->m[1][3] + v1->z * m1->m[2][3] + v1->w * m1->m[3][3];
}
//平移变换矩阵 
void ydc_ns::matrix_set_translate(matrix_t *m, float x, float y, float z) {
	ydc_ns::matrix_identity(m);//先置为单位阵
	m->m[3][0] = x;
	m->m[3][1] = y;
	m->m[3][2] = z;
}
//缩放变换矩阵
void ydc_ns::matrix_set_scale(matrix_t *m, float sx, float sy, float sz) {
	ydc_ns::matrix_identity(m);//先置为单位阵
	m->m[0][0] = sx;
	m->m[1][1] = sy;
	m->m[2][2] = sz;
}
//旋转变换矩阵，沿v方向旋转theta
void ydc_ns::matrix_set_rotate(matrix_t *m,const vector_t *v, float theta) {
	vector_t normV;
	ydc_ns::vector_copy(&normV, v);
	ydc_ns::vector_normalize(&normV);
	ydc_ns::matrix_identity(m);
	float c = cos(theta), s = sin(theta);
	m->m[0][0] = normV.x * normV.x * (1 - c) + c;
	m->m[0][1] = normV.x * normV.y * (1 - c) + normV.z * s;
	m->m[0][2] = normV.x * normV.z * (1 - c) - normV.y * s;
	m->m[1][0] = normV.x * normV.y * (1 - c) - normV.z * s;
	m->m[1][1] = normV.y * normV.y * (1 - c) + c;
	m->m[1][2] = normV.y * normV.z * (1 - c) + normV.x * s;
	m->m[2][0] = normV.x * normV.z * (1 - c) + normV.y * s;
	m->m[2][1] = normV.y * normV.z * (1 - c) - normV.x * s;
	m->m[2][2] = normV.z * normV.z * (1 - c) + c;
}
void ydc_ns::printVector(const vector_t *v) {
	std::cout << ": " << v->x << " " << v->y << " " << v->z << " " << v->w << endl;
}
void ydc_ns::printMatrix(const matrix_t *m) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			std::cout << m->m[i][j] << " ";
		}
		std::cout << endl;
	}
}
//摄像机的lookat矩阵(从世界坐标系到摄像机坐标系的变换矩阵)，输入 eye,target,up(target-eye为z,x=up x z,y=z x x)
void ydc_ns::matrix_set_lookat(matrix_t *m, const vector_t *eye, const vector_t *target, const vector_t *up) {
	vector_t xAxis, yAxis, zAxis;
	ydc_ns::vector_sub(&zAxis, target, eye);
	ydc_ns::vector_normalize(&zAxis);
	ydc_ns::vector_crossProduct(&xAxis, up, &zAxis);
	ydc_ns::vector_normalize(&xAxis);
	ydc_ns::vector_crossProduct(&yAxis, &zAxis, &xAxis);
	
	
	matrix_t m_rotate_inv, m_translate_inv;
	ydc_ns::matrix_identity(&m_rotate_inv);
	ydc_ns::matrix_identity(&m_translate_inv);

	m_rotate_inv.m[0][0] = xAxis.x;
	m_rotate_inv.m[1][0] = xAxis.y;
	m_rotate_inv.m[2][0] = xAxis.z;
	m_rotate_inv.m[0][1] = yAxis.x;
	m_rotate_inv.m[1][1] = yAxis.y;
	m_rotate_inv.m[2][1] = yAxis.z;
	m_rotate_inv.m[0][2] = zAxis.x;
	m_rotate_inv.m[1][2] = zAxis.y;
	m_rotate_inv.m[2][2] = zAxis.z;

	m_translate_inv.m[3][0] = -eye->x;
	m_translate_inv.m[3][1] = -eye->y;
	m_translate_inv.m[3][2] = -eye->z;

	ydc_ns::matrix_multiply(m, &m_translate_inv, &m_rotate_inv);
}
// D3DX Matrix Perspective yFov LH
void ydc_ns::matrix_set_perspective(matrix_t *m, float fov_y, float wOverH, float n, float f) {
	float t = tan(0.5f * fov_y);
	ydc_ns::matrix_zeros(m);
	m->m[0][0] = (float)t/wOverH;
	m->m[1][1] = (float)t;
	m->m[2][2] = f / (f - n);
	m->m[3][2] = -n*f / (f - n);
	m->m[2][3] = 1;
}

//================
//坐标变换
//================

//更新transform = world * view * projection
void ydc_ns::transform_update(transform_t *ts) {
	matrix_t m;
	ydc_ns::matrix_multiply(&m, &(ts->m_world), &(ts->m_view));
	ydc_ns::matrix_multiply(&(ts->m_transform), &m, &(ts->m_projection));
}
//初始化transform
void ydc_ns::transform_init(transform_t *ts, int w, int h) {
	float wOverH = ((float)w) / ((float)h);
	ydc_ns::matrix_identity(&(ts->m_world));
	ydc_ns::matrix_identity(&(ts->m_view));
	ydc_ns::matrix_set_perspective(&(ts->m_projection), PI*0.5f, wOverH, 1.0f, 500.0f);
	ts->w = (float)w;
	ts->h = (float)h;
	ydc_ns::transform_update(ts);
}


//将一个向量进行transform： x * transform
void ydc_ns::transform_apply(vector_t *y, const vector_t *x, const transform_t *ts) {
	ydc_ns::vector_matrix_multipy(y, x, &(ts->m_transform));
}
//视锥裁剪，通过位运算记录六个条件
int ydc_ns::transform_check_cvv(const vector_t *v) {
	float w = v->w;
	int check = 0;
	if (abs(v->x) > abs(w)) check |= 1;
	if (abs(v->y) > abs(w)) check |= 2;
	if (v->z / w <  0.0f) check |= 4;
	if (v->z >  w) check |= 8;
	return check;
}
//由CVV得到屏幕坐标
void ydc_ns::transform_to_UV(vector_t *uv, const vector_t *p, const transform_t *ts) {
	float rhw = (1.0f / p->w);
	uv->x = (p->x *rhw + 1.0f) * ts->w * 0.5f;
	uv->y = (1.0f - p->y * rhw) * ts->h * 0.5f;
	uv->z = p->z * rhw;
	uv->w = 1.0f;
}
//=====================================================================
// 几何计算：顶点、扫描线、边缘、矩形、步长计算
//=====================================================================

//顶点：齐次坐标w的初始化
void ydc_ns::vertex_init_thw(vertex_t *v) {
	float rhw = 1.0f / (v->pos.w);
	v->rhw = rhw;
	v->tc.u *= rhw;
	v->tc.v *= rhw;
	v->color.r *= rhw;
	v->color.g *= rhw;
	v->color.b *= rhw;
}
//顶点插值
void ydc_ns::vertex_interp(vertex_t *v, const vertex_t *v1, const vertex_t *v2, float t) {
	ydc_ns::vector_interp(&v->pos, &v1->pos, &v2->pos, t);
	v->tc.u = interp(v1->tc.u, v2->tc.u, t);
	v->tc.v = interp(v1->tc.v, v2->tc.v, t);
	v->color.r = interp(v1->color.r, v2->color.r, t);
	v->color.g = interp(v1->color.g, v2->color.g, t);
	v->color.b = interp(v1->color.b, v2->color.b, t);
	v->rhw = interp(v1->rhw, v2->rhw,t);
}
//顶点相加，在自身基础上相加，所有成员相加
void ydc_ns::vertex_add(vertex_t *y, const vertex_t *x) {
	y->pos.x += x->pos.x;
	y->pos.y += x->pos.y;
	y->pos.z += x->pos.z;
	y->pos.w += x->pos.w;
	y->rhw += x->rhw;
	y->tc.u += x->tc.u;
	y->tc.v += x->tc.v;
	y->color.r += x->color.r;
	y->color.g += x->color.g;
	y->color.b += x->color.b;
}
//顶点相减（所有成员相减），并处以一个系数，得到一个新的顶点 y = (x2 - x1)/w
void ydc_ns::vertex_division(vertex_t *y, const vertex_t *x1, const vertex_t *x2, float w) {
	float inv = 1.0f / w;
	y->pos.x = (x2->pos.x - x1->pos.x) * inv;
	y->pos.y = (x2->pos.y - x1->pos.y) * inv;
	y->pos.z = (x2->pos.z - x1->pos.z) * inv;
	y->pos.w = (x2->pos.w - x1->pos.w) * inv;
	y->tc.u = (x2->tc.u - x1->tc.u) * inv;
	y->tc.v = (x2->tc.v - x1->tc.v) * inv;
	y->color.r = (x2->color.r - x1->color.r) * inv;
	y->color.g = (x2->color.g - x1->color.g) * inv;
	y->color.b = (x2->color.b - x1->color.b) * inv;
	y->rhw = (x2->rhw - x1->rhw) * inv;
}
//将三角形划分为0~2个梯形，并且返回合法个数
int ydc_ns::trapezoid_init_triangle(trapezoid_t *trap, const vertex_t *p1,const vertex_t *p2, const vertex_t *p3) {
	const vertex_t *p;
	float k, x;

	//保证 p3.y > p2.y > p1.y
	if (p1->pos.y > p2->pos.y) p = p1, p1 = p2, p2 = p;
	if (p1->pos.y > p3->pos.y) p = p1, p1 = p3, p3 = p;
	if (p2->pos.y > p3->pos.y) p = p2, p2 = p3, p3 = p;
	//y或者x共线，则无法构成梯形
	if (p1->pos.y == p2->pos.y && p1->pos.y == p3->pos.y) return 0;
	if (p1->pos.x == p2->pos.x && p1->pos.x == p3->pos.x) return 0;

	//三角形底边平
	if (p1->pos.y == p2->pos.y) {	
		if (p1->pos.x > p2->pos.x) p = p1, p1 = p2, p2 = p;
		trap[0].top = p1->pos.y;
		trap[0].bottom = p3->pos.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;
		trap[0].right.v1 = *p2;
		trap[0].right.v2 = *p3;
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	//三角形顶边平
	if (p2->pos.y == p3->pos.y) {	
		if (p2->pos.x > p3->pos.x) p = p2, p2 = p3, p3 = p;
		trap[0].top = p1->pos.y;
		trap[0].bottom = p3->pos.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		return (trap[0].top < trap[0].bottom) ? 1 : 0;
	}

	//一般情况
	trap[0].top = p1->pos.y;
	trap[0].bottom = p2->pos.y;
	trap[1].top = p2->pos.y;
	trap[1].bottom = p3->pos.y;

	k = (p3->pos.y - p1->pos.y) / (p2->pos.y - p1->pos.y);
	x = p1->pos.x + (p2->pos.x - p1->pos.x) * k;

	//判断三角形朝向
	if (x <= p3->pos.x) {		// triangle left
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		trap[1].left.v1 = *p2;
		trap[1].left.v2 = *p3;
		trap[1].right.v1 = *p1;
		trap[1].right.v2 = *p3;
	}
	else {					// triangle right
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p2;
		trap[1].left.v1 = *p1;
		trap[1].left.v2 = *p3;
		trap[1].right.v1 = *p2;
		trap[1].right.v2 = *p3;
	}

	return 2;
}
// 给一个y，求出左右两边纵坐标等于y的点，存到trap->left,right.v
void ydc_ns::trapezoid_edge_interp(trapezoid_t *trap, float y) {
	float s1 = trap->left.v2.pos.y - trap->left.v1.pos.y;
	float s2 = trap->right.v2.pos.y - trap->right.v1.pos.y;
	float t1 = (y - trap->left.v1.pos.y) / s1;
	float t2 = (y - trap->right.v1.pos.y) / s2;
	ydc_ns::vertex_interp(&trap->left.v, &trap->left.v1, &trap->left.v2, t1);
	ydc_ns::vertex_interp(&trap->right.v, &trap->right.v1, &trap->right.v2, t2);
}
// 根据左右两边的端点，计算扫描线 起点：v 步长：step （x y w均为int）
void ydc_ns::trapezoid_init_scanline(const trapezoid_t *trap, scanline_t *scanline, int y) {
	float width = trap->right.v.pos.x - trap->left.v.pos.x;
	scanline->x = (int)(trap->left.v.pos.x + 0.5f);
	scanline->w = (int)(trap->right.v.pos.x + 0.5f) - scanline->x;//扫描线的整数宽度
	scanline->y = y;
	scanline->v = trap->left.v;
	if (trap->left.v.pos.x >= trap->right.v.pos.x) scanline->w = 0;
	vertex_division(&scanline->step, &trap->left.v, &trap->right.v, width);
}

//=====================
// 渲染设备
//=====================

// 设备初始化，fb为外部帧缓存，非 NULL 将引用外部帧缓存（每行 4字节对齐）
void ydc_ns::device_init(device_t *device, int width, int height, void *fb) {
	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;//计算所需空间
	char *ptr = (char*)malloc(need + 64);// +64是额外的纹理空间
	char *framebuf, *zbuf;
	int j;
	assert(ptr);
	device->framebuffer = (IUINT32**)ptr; //framebuffer有h行，所以其指针需要 sizeof(void*)*h字节
	device->zbuffer = (float**)(ptr + sizeof(void*) * height);//zbuffer有h行，所以其指针需要 sizeof(void*)*h字节
	ptr += sizeof(void*) * height * 2;
	device->texture = (IUINT32**)ptr;//分配给纹理指针 sizeof(void*)*1024字节
	ptr += sizeof(void*) * 1024;
	framebuf = (char*)ptr;
	zbuf = (char*)ptr + width * height * 4;
	ptr += width * height * 8;
	if (fb != NULL) framebuf = (char*)fb;//如果有外部fb，则使用外部fb
	for (j = 0; j < height; j++) {
		//给每一行分配 4*w字节，即每一个元素为4字节，刚好可以存下IUINT32和float
		device->framebuffer[j] = (IUINT32*)(framebuf + width * 4 * j);
		device->zbuffer[j] = (float*)(zbuf + width * 4 * j);
	}
	device->texture[0] = (IUINT32*)ptr;
	device->texture[1] = (IUINT32*)(ptr + 16);
	memset(device->texture[0], 0, 64);//将texture[0]当前位置后的64个字节全部置为0
	device->tex_width = 2;//纹理宽高为2，刚好一个纹理就是 2*2*4=16字节
	device->tex_height = 2;
	device->max_u = 1.0f;
	device->max_v = 1.0f;
	device->width = width;
	device->height = height;
	device->background = 0x1e1e1e;//给背景指定颜色
	device->foreground = 0;//给线框指定颜色
	transform_init(&device->transform, width, height);//初始化transform
	device->render_state = RENDER_STATE_WIREFRAME;//初始化渲染模式为 线框渲染
}
// 删除设备
void ydc_ns::device_destroy(device_t *device) {
	if (device->framebuffer)
		free(device->framebuffer);//framebuffer的指针直接用的第一个ptr，是直接malloc分配的，所以需要free
	device->framebuffer = NULL;//指定为空指针，避免野指针
	device->zbuffer = NULL;
	device->texture = NULL;
}

// 设置当前纹理,使device->texture指向texture
void ydc_ns::device_set_texture(device_t *device, void *texture, long sizePerRow, int w, int h) {
	char *ptr = (char*)texture;
	int j;
	assert(w <= 1024 && h <= 1024);
	for (j = 0; j < h; ptr += sizePerRow, j++) 	// 每一行指针遍历
		device->texture[j] = (IUINT32*)ptr;
	device->tex_width = w;
	device->tex_height = h;
	device->max_u = (float)(w - 1);
	device->max_v = (float)(h - 1);
}
// 清空 framebuffer(设置背景为渐变) 和 zbuffer
void ydc_ns::device_clear(device_t *device, int mode) {
	int y, x, height = device->height;
	//遍历每一行
	for (y = 0; y < device->height; y++) {
		IUINT32 *dst = device->framebuffer[y];
		//
		IUINT32 cc = (height - 1 - y) * 230 / (height - 1);
		cc = (cc << 16) | (cc << 8) | cc;
		if (mode == 0) cc = device->background;
		//
		//遍历每一行的每一个指针，其中用x作为循环结束，实际指针为dst
		for (x = device->width; x > 0; dst++, x--) {
			dst[0] = cc;	//cc为背景渐变色
			//dst[0] = 0.0f;	//背景置为黑色
		}
	}
	for (y = 0; y < device->height; y++) {
		float *dst = device->zbuffer[y];
		for (x = device->width; x > 0; dst++, x--) dst[0] = 0.0f;
	}
}
// 画点,即改变framebuffer[x][y]
void ydc_ns::draw_pixel(device_t *device,int u,int v,IUINT32 color){
	//判断x，y的大小
	if ((IUINT32)u < (IUINT32)device->width && (IUINT32)v < (IUINT32)device->height) {
		device->framebuffer[v][u] = color;
	}
}
//画线段
void ydc_ns::draw_line(device_t *device,int x1,int y1,int x2,int y2,IUINT32 color){
	
	if (x1 == x2 && y1 == y2) {
		draw_pixel(device, x1, y1, color);
	}
	else if (x1 == x2) {
		int stepY = (y1 > y2) ? -1 : 1;
		for (int y = y1; y != y2; y += stepY) {
			draw_pixel(device, x1, y, color);
		}
		draw_pixel(device, x2, y2, color);
	}
	else if (y1 == y2) {
		int stepX = (x1 > x2) ? -1 : 1;
		for (int x = x1; x != x2; x += stepX) {
			draw_pixel(device, x, y1, color);
		}
		draw_pixel(device, x2, y2, color);
	}
	else {
		int dx = (x1 > x2) ? (x1 - x2) : (x2 - x1);
		int dy = (y1 > y2) ? (y1 - y2) : (y2 - y1);
		if (dx > dy) {
			if (x1 > x2) {
				int x = x1, y = y1;
				x1 = x2, y1 = y2, x2 = x, y2 = y;
			}
			int tempStep = 0;
			int stepY = (y1 > y2) ? -1 : 1;
			for (int x = x1, y = y1; x <= x2;++x) {
				draw_pixel(device, x, y, color);
				tempStep += dy;
				if (tempStep >= dx) {
					tempStep -= dx;
					y += stepY;
					draw_pixel(device, x, y, color);
				}
			}
			draw_pixel(device, x2, y2, color);
		}
		else if (dx <= dy) {
			if (y2 < y1) {
				int x = x1, y = y1;
				x1 = x2, y1 = y2, x2 = x, y2 = y;
			}
			int tempStep = 0;
			int stepX = (x1 > x2) ? -1 : 1;
			for (int x = x1, y = y1; y <= y2; ++y) {
				draw_pixel(device, x, y, color);
				tempStep += dx;
				if (tempStep >= dy) {
					tempStep -= dy;
					x += stepX;
					draw_pixel(device, x, y, color);
				}
			}
			draw_pixel(device, x2, y2, color);
		}
	}
}
// 根据坐标读取纹理,此处 0 < u,v < 1 
IUINT32 ydc_ns::device_texture_read(const device_t *device, float u, float v) {
	int x, y;
	u = u * device->max_u;
	v = v * device->max_v;
	x = (int)(u + 0.5f);
	y = (int)(v + 0.5f);
	x = T_clamp(x, 0, device->tex_width - 1);
	y = T_clamp(y, 0, device->tex_height - 1);
	return device->texture[y][x];
}

//=====================================================================
// 渲染实现
//=====================================================================

//画扫描线，只对texture和color模式有效，对于线框无效
void ydc_ns::device_draw_scanline(device_t *device, scanline_t *scanline) {
	//指针拷贝,可修改原来的内存位置（framebuffer与zbuffer）
	IUINT32 *framebuffer = device->framebuffer[scanline->y];
	float *zbuffer = device->zbuffer[scanline->y];
	int width = device->width;
	//画一个点，x+1,w-1，更新 v = v + step（v:扫描线起点）
	for (int w = scanline->w,  x = scanline->x; w > 0; x++, w--) {
		//限制x，只画窗口里的部分
		if (x >= 0 && x < width) {
			float rhw = scanline -> v.rhw;
			//深度测试，rhw越大，z越小，时刻更新zbuffer为最大的rhw，即最小的z
			if (rhw >= zbuffer[x]) {
				float w = 1.0f / rhw;//此处的w即为世界系中的点经过transform后得到的w坐标，即CVV中的z
				zbuffer[x] = rhw;
				if (device->render_state & RENDER_STATE_COLOR) {
					float r = scanline->v.color.r * w;
					float g = scanline->v.color.g * w;
					float b = scanline->v.color.b * w;
					int R = (int)(r * 255.0f);
					int G = (int)(g * 255.0f);
					int B = (int)(b * 255.0f);
					framebuffer[x] = (R << 16) | (G << 8) | (B);//用24位来存颜色，存一个f(15)需要4位
				}
				if (device->render_state & RENDER_STATE_TEXTURE) {
					float u = scanline->v.tc.u * w;
					float v = scanline->v.tc.v * w;
					IUINT32 cc = device_texture_read(device, u, v);
					framebuffer[x] = cc;
				}
			}
		}
		vertex_add(&scanline->v, &scanline->step);
		if (x >= width) break;
	}
}
//绘制多边形
void ydc_ns::device_draw_trap(device_t *device, trapezoid_t *trap) {
	scanline_t scanline;
	int top, bottom;
	top = (int)(trap->top + 0.5f);
	bottom = (int)(trap->bottom + 0.5f);
	for (int y = top; y < bottom; y++) {
		if (y >= 0 && y < device->height) {
			trapezoid_edge_interp(trap, (float)y + 0.5f);	//给定y,求此时的两个端点
			trapezoid_init_scanline(trap, &scanline, y);	//给定y，求出扫描线
			device_draw_scanline(device, &scanline);		//画出扫描线
		}
		if (y >= device->height) break;
	}
}
//根据渲染模式，绘制三角形
void ydc_ns::device_draw_primitive_triangle(device_t *device, const vertex_t *v1, vertex_t *v2, vertex_t *v3) {
	vector_t p1, p2, p3, c1, c2, c3;

	// c = v->pos * transform,从世界系到投影后
	transform_apply(&c1, &v1->pos, &device->transform);
	transform_apply(&c2, &v2->pos, &device->transform);
	transform_apply(&c3, &v3->pos, &device->transform);
	//裁剪
	if (transform_check_cvv(&c1) != 0)return;
	if (transform_check_cvv(&c2) != 0)return;
	if (transform_check_cvv(&c3) != 0)return;
	//CVV to 屏幕系
	transform_to_UV(&p1, &c1, &device->transform);
	transform_to_UV(&p2, &c2, &device->transform);
	transform_to_UV(&p3, &c3, &device->transform);

	//texture 或者 color 模式
	if (device->render_state & (RENDER_STATE_COLOR | RENDER_STATE_TEXTURE)) {
		vertex_t t1 = *v1, t2 = *v2, t3 = *v3;
		trapezoid_t traps[2];
		int numOfTrap;
		//把p的(x,y)给顶点t的pos
		t1.pos = p1;
		t2.pos = p2;
		t3.pos = p3;
		//把c的w给顶点t的pos，并初始化w
		t1.pos.w = c1.w;
		t2.pos.w = c2.w;
		t3.pos.w = c3.w;
		vertex_init_thw(&t1);
		vertex_init_thw(&t2);
		vertex_init_thw(&t3);
		//拆分三角形为多边形
		numOfTrap = trapezoid_init_triangle(traps, &t1, &t2, &t3);
		if (numOfTrap >= 1)device_draw_trap(device, &traps[0]);
		if (numOfTrap >= 2)device_draw_trap(device, &traps[1]);
	}
	//wireframe模式
	if (device->render_state & RENDER_STATE_WIREFRAME) {
		
		draw_line(device, (int)p1.x, (int)p1.y, (int)p2.x, (int)p2.y, device->foreground);
		draw_line(device, (int)p1.x, (int)p1.y, (int)p3.x, (int)p3.y, device->foreground);
		draw_line(device, (int)p3.x, (int)p3.y, (int)p2.x, (int)p2.y, device->foreground);
	}
}


