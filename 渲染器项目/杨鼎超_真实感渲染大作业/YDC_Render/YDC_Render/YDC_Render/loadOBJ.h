#pragma once
#include "stdafx.h"
#include "mathYdc.h"
#include <iostream>

#include <math.h>
#include <fstream>
#include <string>
#include <vector>

using namespace ydc_ns;

typedef struct  {
	int p1, p2, p3;
}index_buffer_t;
typedef struct {
	vector<vertex_t> vertexs = {};
	vector<index_buffer_t> index_buffers = {};
} TriangleMesh_t;

void printIB(const index_buffer_t *IB) {
	std::cout << ": " << IB->p1 << " " << IB->p2 << " " << IB->p3 << endl;
}
void loadOBJ(const std::string filename, TriangleMesh_t *mesh) {
	std::ifstream in(filename.c_str());
	if (!in.good()) {
		cout << "ERROR: file is not good!!!" << endl;
		exit(0);
	}
	char buffer[256], str[255];
	float f1, f2, f3;
	int p1, p2, p3;

	while (!in.getline(buffer, 255).eof())
	{
		buffer[255] = '\0';

		sscanf_s(buffer, "%s", str, 255);

		// reading a vertex
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			if (sscanf_s(buffer, "v %f %f %f", &f1, &f2, &f3) == 3)
			{
				vertex_t tempV = { {f1,f2,f3,1} };
				mesh->vertexs.push_back(tempV);
			}
			else
			{
				cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}
		// reading FaceMtls 
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			int nt = sscanf_s(buffer, "f %d %d %d", &p1, &p2, &p3);
			if (nt != 3)
			{
				cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}
			index_buffer_t tempIB = { p1,p2,p3 };
			mesh->index_buffers.push_back(tempIB);
		}
	}
}
void printTriangleMesh(const TriangleMesh_t *mesh) {
	for (auto vertex : mesh->vertexs) {
		std::cout << "v ";
		printVector(&vertex.pos);
	}
	for (auto IB : mesh->index_buffers) {
		std::cout << "f ";
		printIB(&IB);
	}
}

