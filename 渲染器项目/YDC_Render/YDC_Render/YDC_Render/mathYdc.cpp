#include "stdafx.h"
#include "mathYdc.h"
#include <iostream>
using namespace std;
using namespace ydc_ns;
//����
void ydc_ns::printTest() {
	cout << "hello world!" << endl;
}

//======================
//��ѧ��
//======================
// clamp����
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
// ��ֵinterp����
float ydc_ns::interp(float x1, float x2, float t) {
	return x1 + (x2 - x1)*t;
}
//��������
void ydc_ns::vector_copy(vector_t *v, const vector_t *v1) {
	v->x = v1->x;
	v->y = v1->y;
	v->z = v1->z;
	v->w = v1->w;
}
//����������ģ��ֻ����x,y,z��������w
float ydc_ns::vector_length(const vector_t *v) {
	return sqrt(v->x * v->x + v->y * v->y + v->z*v->z);
}
//���������ӷ� v = v1 + v2
void ydc_ns::vector_add(vector_t *v, const vector_t *v1, const vector_t *v2) {
	v->x = v1->x + v2->x;
	v->y = v1->y + v2->y;
	v->z = v1->z + v2->z;
	v->w = 1.0f;
}
//������������ v = v1 - v2
void ydc_ns::vector_sub(vector_t *v, const vector_t *v1, const vector_t *v2) {
	v->x = v1->x - v2->x;
	v->y = v1->y - v2->y;
	v->z = v1->z - v2->z;
	v->w = 1.0f;
}
//������� v1 ��v2
float ydc_ns::vector_dotProduct(const vector_t *v1, const vector_t *v2) {
	return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}
//������� v = v1 x v2
void ydc_ns::vector_crossProduct(vector_t *v, const vector_t *v1, const vector_t *v2) {
	v->x = v1->y * v2->z - v2->y * v1->z;
	v->y = v1->z * v2->x - v2->z * v1->x;
	v->z = v1->x * v2->y - v2->x * v1->y;
	v->w = 1.0f;
}
//������ֵ v = v1 + (v2 - v1)*t 
void ydc_ns::vector_interp(vector_t *v, const vector_t *v1, const vector_t *v2,float t) {
	v->x = v1->x + (v2->x - v1->x)*t;
	v->y = v1->y + (v2->y - v1->y)*t;
	v->z = v1->z + (v2->z - v1->z)*t;
	v->w = 1.0f;
}
//������һ�� norm(*v)
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
// ��ɫ����
void ydc_ns::color_scale(color_t* c, float k)
{
	c->r = c->r * k;
	c->g = c->g * k;
	c->b = c->b * k;
}
// ��ɫ���
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
//���󿽱�
void ydc_ns::matrix_copy(matrix_t *m, const matrix_t *m1) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j];
		}
	}
}
//������Ϊ��λ���� identity(*m)
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
//������0 zeros(*m)
void ydc_ns::matrix_zeros(matrix_t *m) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = 0;
		}
	}
}
//����ӷ� m1 + m2
void ydc_ns::matrix_add(matrix_t *m, const matrix_t *m1, const matrix_t *m2) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j] + m2->m[i][j];
		}
	}
}
//������� m1 - m2
void ydc_ns::matrix_sub(matrix_t *m, const matrix_t *m1, const matrix_t *m2) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j] - m2->m[i][j];
		}
	}
}
//����˷� m1 * m2
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
//�������� k*m1
void ydc_ns::matrix_scale(matrix_t *m, matrix_t *m1, float k) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m->m[i][j] = m1->m[i][j] * k;
		}
	}
}
//������������ v1 * m1
void ydc_ns::vector_matrix_multipy(vector_t *v, const vector_t *v1, const matrix_t *m1) {
	v->x = v1->x * m1->m[0][0] + v1->y * m1->m[1][0] + v1->z * m1->m[2][0] + v1->w * m1->m[3][0];
	v->y = v1->x * m1->m[0][1] + v1->y * m1->m[1][1] + v1->z * m1->m[2][1] + v1->w * m1->m[3][1];
	v->z = v1->x * m1->m[0][2] + v1->y * m1->m[1][2] + v1->z * m1->m[2][2] + v1->w * m1->m[3][2];
	v->w = v1->x * m1->m[0][3] + v1->y * m1->m[1][3] + v1->z * m1->m[2][3] + v1->w * m1->m[3][3];
}
//ƽ�Ʊ任���� 
void ydc_ns::matrix_set_translate(matrix_t *m, float x, float y, float z) {
	ydc_ns::matrix_identity(m);//����Ϊ��λ��
	m->m[3][0] = x;
	m->m[3][1] = y;
	m->m[3][2] = z;
}
//���ű任����
void ydc_ns::matrix_set_scale(matrix_t *m, float sx, float sy, float sz) {
	ydc_ns::matrix_identity(m);//����Ϊ��λ��
	m->m[0][0] = sx;
	m->m[1][1] = sy;
	m->m[2][2] = sz;
}
//��ת�任������v������תtheta
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
//�������lookat����(����������ϵ�����������ϵ�ı任����)������ eye,target,up(target-eyeΪz,x=up x z,y=z x x)
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
//����任
//================

//����transform = world * view * projection
void ydc_ns::transform_update(transform_t *ts) {
	matrix_t m;
	ydc_ns::matrix_multiply(&m, &(ts->m_world), &(ts->m_view));
	ydc_ns::matrix_multiply(&(ts->m_transform), &m, &(ts->m_projection));
}
//��ʼ��transform
void ydc_ns::transform_init(transform_t *ts, int w, int h) {
	float wOverH = ((float)w) / ((float)h);
	ydc_ns::matrix_identity(&(ts->m_world));
	ydc_ns::matrix_identity(&(ts->m_view));
	ydc_ns::matrix_set_perspective(&(ts->m_projection), PI*0.5f, wOverH, 1.0f, 500.0f);
	ts->w = (float)w;
	ts->h = (float)h;
	ydc_ns::transform_update(ts);
}


//��һ����������transform�� x * transform
void ydc_ns::transform_apply(vector_t *y, const vector_t *x, const transform_t *ts) {
	ydc_ns::vector_matrix_multipy(y, x, &(ts->m_transform));
}
//��׶�ü���ͨ��λ�����¼��������
int ydc_ns::transform_check_cvv(const vector_t *v) {
	float w = v->w;
	int check = 0;
	if (abs(v->x) > abs(w)) check |= 1;
	if (abs(v->y) > abs(w)) check |= 2;
	if (v->z / w <  0.0f) check |= 4;
	if (v->z >  w) check |= 8;
	return check;
}
//��CVV�õ���Ļ����
void ydc_ns::transform_to_UV(vector_t *uv, const vector_t *p, const transform_t *ts) {
	float rhw = (1.0f / p->w);
	uv->x = (p->x *rhw + 1.0f) * ts->w * 0.5f;
	uv->y = (1.0f - p->y * rhw) * ts->h * 0.5f;
	uv->z = p->z * rhw;
	uv->w = 1.0f;
}
//=====================================================================
// ���μ��㣺���㡢ɨ���ߡ���Ե�����Ρ���������
//=====================================================================

//���㣺�������w�ĳ�ʼ��
void ydc_ns::vertex_init_thw(vertex_t *v) {
	float rhw = 1.0f / (v->pos.w);
	v->rhw = rhw;
	v->tc.u *= rhw;
	v->tc.v *= rhw;
	v->color.r *= rhw;
	v->color.g *= rhw;
	v->color.b *= rhw;
}
//�����ֵ
void ydc_ns::vertex_interp(vertex_t *v, const vertex_t *v1, const vertex_t *v2, float t) {
	ydc_ns::vector_interp(&v->pos, &v1->pos, &v2->pos, t);
	v->tc.u = interp(v1->tc.u, v2->tc.u, t);
	v->tc.v = interp(v1->tc.v, v2->tc.v, t);
	v->color.r = interp(v1->color.r, v2->color.r, t);
	v->color.g = interp(v1->color.g, v2->color.g, t);
	v->color.b = interp(v1->color.b, v2->color.b, t);
	v->rhw = interp(v1->rhw, v2->rhw,t);
}
//������ӣ��������������ӣ����г�Ա���
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
//������������г�Ա�������������һ��ϵ�����õ�һ���µĶ��� y = (x2 - x1)/w
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
//�������λ���Ϊ0~2�����Σ����ҷ��غϷ�����
int ydc_ns::trapezoid_init_triangle(trapezoid_t *trap, const vertex_t *p1,const vertex_t *p2, const vertex_t *p3) {
	const vertex_t *p;
	float k, x;

	//��֤ p3.y > p2.y > p1.y
	if (p1->pos.y > p2->pos.y) p = p1, p1 = p2, p2 = p;
	if (p1->pos.y > p3->pos.y) p = p1, p1 = p3, p3 = p;
	if (p2->pos.y > p3->pos.y) p = p2, p2 = p3, p3 = p;
	//y����x���ߣ����޷���������
	if (p1->pos.y == p2->pos.y && p1->pos.y == p3->pos.y) return 0;
	if (p1->pos.x == p2->pos.x && p1->pos.x == p3->pos.x) return 0;

	//�����εױ�ƽ
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

	//�����ζ���ƽ
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

	//һ�����
	trap[0].top = p1->pos.y;
	trap[0].bottom = p2->pos.y;
	trap[1].top = p2->pos.y;
	trap[1].bottom = p3->pos.y;

	k = (p3->pos.y - p1->pos.y) / (p2->pos.y - p1->pos.y);
	x = p1->pos.x + (p2->pos.x - p1->pos.x) * k;

	//�ж������γ���
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
// ��һ��y����������������������y�ĵ㣬�浽trap->left,right.v
void ydc_ns::trapezoid_edge_interp(trapezoid_t *trap, float y) {
	float s1 = trap->left.v2.pos.y - trap->left.v1.pos.y;
	float s2 = trap->right.v2.pos.y - trap->right.v1.pos.y;
	float t1 = (y - trap->left.v1.pos.y) / s1;
	float t2 = (y - trap->right.v1.pos.y) / s2;
	ydc_ns::vertex_interp(&trap->left.v, &trap->left.v1, &trap->left.v2, t1);
	ydc_ns::vertex_interp(&trap->right.v, &trap->right.v1, &trap->right.v2, t2);
}
// �����������ߵĶ˵㣬����ɨ���� ��㣺v ������step ��x y w��Ϊint��
void ydc_ns::trapezoid_init_scanline(const trapezoid_t *trap, scanline_t *scanline, int y) {
	float width = trap->right.v.pos.x - trap->left.v.pos.x;
	scanline->x = (int)(trap->left.v.pos.x + 0.5f);
	scanline->w = (int)(trap->right.v.pos.x + 0.5f) - scanline->x;//ɨ���ߵ��������
	scanline->y = y;
	scanline->v = trap->left.v;
	if (trap->left.v.pos.x >= trap->right.v.pos.x) scanline->w = 0;
	vertex_division(&scanline->step, &trap->left.v, &trap->right.v, width);
}

//=====================
// ��Ⱦ�豸
//=====================

// �豸��ʼ����fbΪ�ⲿ֡���棬�� NULL �������ⲿ֡���棨ÿ�� 4�ֽڶ��룩
void ydc_ns::device_init(device_t *device, int width, int height, void *fb) {
	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;//��������ռ�
	char *ptr = (char*)malloc(need + 64);// +64�Ƕ��������ռ�
	char *framebuf, *zbuf;
	int j;
	assert(ptr);
	device->framebuffer = (IUINT32**)ptr; //framebuffer��h�У�������ָ����Ҫ sizeof(void*)*h�ֽ�
	device->zbuffer = (float**)(ptr + sizeof(void*) * height);//zbuffer��h�У�������ָ����Ҫ sizeof(void*)*h�ֽ�
	ptr += sizeof(void*) * height * 2;
	device->texture = (IUINT32**)ptr;//���������ָ�� sizeof(void*)*1024�ֽ�
	ptr += sizeof(void*) * 1024;
	framebuf = (char*)ptr;
	zbuf = (char*)ptr + width * height * 4;
	ptr += width * height * 8;
	if (fb != NULL) framebuf = (char*)fb;//������ⲿfb����ʹ���ⲿfb
	for (j = 0; j < height; j++) {
		//��ÿһ�з��� 4*w�ֽڣ���ÿһ��Ԫ��Ϊ4�ֽڣ��պÿ��Դ���IUINT32��float
		device->framebuffer[j] = (IUINT32*)(framebuf + width * 4 * j);
		device->zbuffer[j] = (float*)(zbuf + width * 4 * j);
	}
	device->texture[0] = (IUINT32*)ptr;
	device->texture[1] = (IUINT32*)(ptr + 16);
	memset(device->texture[0], 0, 64);//��texture[0]��ǰλ�ú��64���ֽ�ȫ����Ϊ0
	device->tex_width = 2;//������Ϊ2���պ�һ��������� 2*2*4=16�ֽ�
	device->tex_height = 2;
	device->max_u = 1.0f;
	device->max_v = 1.0f;
	device->width = width;
	device->height = height;
	device->background = 0x1e1e1e;//������ָ����ɫ
	device->foreground = 0;//���߿�ָ����ɫ
	transform_init(&device->transform, width, height);//��ʼ��transform
	device->render_state = RENDER_STATE_WIREFRAME;//��ʼ����ȾģʽΪ �߿���Ⱦ
}
// ɾ���豸
void ydc_ns::device_destroy(device_t *device) {
	if (device->framebuffer)
		free(device->framebuffer);//framebuffer��ָ��ֱ���õĵ�һ��ptr����ֱ��malloc����ģ�������Ҫfree
	device->framebuffer = NULL;//ָ��Ϊ��ָ�룬����Ұָ��
	device->zbuffer = NULL;
	device->texture = NULL;
}

// ���õ�ǰ����,ʹdevice->textureָ��texture
void ydc_ns::device_set_texture(device_t *device, void *texture, long sizePerRow, int w, int h) {
	char *ptr = (char*)texture;
	int j;
	assert(w <= 1024 && h <= 1024);
	for (j = 0; j < h; ptr += sizePerRow, j++) 	// ÿһ��ָ�����
		device->texture[j] = (IUINT32*)ptr;
	device->tex_width = w;
	device->tex_height = h;
	device->max_u = (float)(w - 1);
	device->max_v = (float)(h - 1);
}
// ��� framebuffer(���ñ���Ϊ����) �� zbuffer
void ydc_ns::device_clear(device_t *device, int mode) {
	int y, x, height = device->height;
	//����ÿһ��
	for (y = 0; y < device->height; y++) {
		IUINT32 *dst = device->framebuffer[y];
		//
		IUINT32 cc = (height - 1 - y) * 230 / (height - 1);
		cc = (cc << 16) | (cc << 8) | cc;
		if (mode == 0) cc = device->background;
		//
		//����ÿһ�е�ÿһ��ָ�룬������x��Ϊѭ��������ʵ��ָ��Ϊdst
		for (x = device->width; x > 0; dst++, x--) {
			dst[0] = cc;	//ccΪ��������ɫ
			//dst[0] = 0.0f;	//������Ϊ��ɫ
		}
	}
	for (y = 0; y < device->height; y++) {
		float *dst = device->zbuffer[y];
		for (x = device->width; x > 0; dst++, x--) dst[0] = 0.0f;
	}
}
// ����,���ı�framebuffer[x][y]
void ydc_ns::draw_pixel(device_t *device,int u,int v,IUINT32 color){
	//�ж�x��y�Ĵ�С
	if ((IUINT32)u < (IUINT32)device->width && (IUINT32)v < (IUINT32)device->height) {
		device->framebuffer[v][u] = color;
	}
}
//���߶�
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
// ���������ȡ����,�˴� 0 < u,v < 1 
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
// ��Ⱦʵ��
//=====================================================================

//��ɨ���ߣ�ֻ��texture��colorģʽ��Ч�������߿���Ч
void ydc_ns::device_draw_scanline(device_t *device, scanline_t *scanline) {
	//ָ�뿽��,���޸�ԭ�����ڴ�λ�ã�framebuffer��zbuffer��
	IUINT32 *framebuffer = device->framebuffer[scanline->y];
	float *zbuffer = device->zbuffer[scanline->y];
	int width = device->width;
	//��һ���㣬x+1,w-1������ v = v + step��v:ɨ������㣩
	for (int w = scanline->w,  x = scanline->x; w > 0; x++, w--) {
		//����x��ֻ��������Ĳ���
		if (x >= 0 && x < width) {
			float rhw = scanline -> v.rhw;
			//��Ȳ��ԣ�rhwԽ��zԽС��ʱ�̸���zbufferΪ����rhw������С��z
			if (rhw >= zbuffer[x]) {
				float w = 1.0f / rhw;//�˴���w��Ϊ����ϵ�еĵ㾭��transform��õ���w���꣬��CVV�е�z
				zbuffer[x] = rhw;
				if (device->render_state & RENDER_STATE_COLOR) {
					float r = scanline->v.color.r * w;
					float g = scanline->v.color.g * w;
					float b = scanline->v.color.b * w;
					int R = (int)(r * 255.0f);
					int G = (int)(g * 255.0f);
					int B = (int)(b * 255.0f);
					framebuffer[x] = (R << 16) | (G << 8) | (B);//��24λ������ɫ����һ��f(15)��Ҫ4λ
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
//���ƶ����
void ydc_ns::device_draw_trap(device_t *device, trapezoid_t *trap) {
	scanline_t scanline;
	int top, bottom;
	top = (int)(trap->top + 0.5f);
	bottom = (int)(trap->bottom + 0.5f);
	for (int y = top; y < bottom; y++) {
		if (y >= 0 && y < device->height) {
			trapezoid_edge_interp(trap, (float)y + 0.5f);	//����y,���ʱ�������˵�
			trapezoid_init_scanline(trap, &scanline, y);	//����y�����ɨ����
			device_draw_scanline(device, &scanline);		//����ɨ����
		}
		if (y >= device->height) break;
	}
}
//������Ⱦģʽ������������
void ydc_ns::device_draw_primitive_triangle(device_t *device, const vertex_t *v1, vertex_t *v2, vertex_t *v3) {
	vector_t p1, p2, p3, c1, c2, c3;

	// c = v->pos * transform,������ϵ��ͶӰ��
	transform_apply(&c1, &v1->pos, &device->transform);
	transform_apply(&c2, &v2->pos, &device->transform);
	transform_apply(&c3, &v3->pos, &device->transform);
	//�ü�
	if (transform_check_cvv(&c1) != 0)return;
	if (transform_check_cvv(&c2) != 0)return;
	if (transform_check_cvv(&c3) != 0)return;
	//CVV to ��Ļϵ
	transform_to_UV(&p1, &c1, &device->transform);
	transform_to_UV(&p2, &c2, &device->transform);
	transform_to_UV(&p3, &c3, &device->transform);

	//texture ���� color ģʽ
	if (device->render_state & (RENDER_STATE_COLOR | RENDER_STATE_TEXTURE)) {
		vertex_t t1 = *v1, t2 = *v2, t3 = *v3;
		trapezoid_t traps[2];
		int numOfTrap;
		//��p��(x,y)������t��pos
		t1.pos = p1;
		t2.pos = p2;
		t3.pos = p3;
		//��c��w������t��pos������ʼ��w
		t1.pos.w = c1.w;
		t2.pos.w = c2.w;
		t3.pos.w = c3.w;
		vertex_init_thw(&t1);
		vertex_init_thw(&t2);
		vertex_init_thw(&t3);
		//���������Ϊ�����
		numOfTrap = trapezoid_init_triangle(traps, &t1, &t2, &t3);
		if (numOfTrap >= 1)device_draw_trap(device, &traps[0]);
		if (numOfTrap >= 2)device_draw_trap(device, &traps[1]);
	}
	//wireframeģʽ
	if (device->render_state & RENDER_STATE_WIREFRAME) {
		
		draw_line(device, (int)p1.x, (int)p1.y, (int)p2.x, (int)p2.y, device->foreground);
		draw_line(device, (int)p1.x, (int)p1.y, (int)p3.x, (int)p3.y, device->foreground);
		draw_line(device, (int)p3.x, (int)p3.y, (int)p2.x, (int)p2.y, device->foreground);
	}
}


