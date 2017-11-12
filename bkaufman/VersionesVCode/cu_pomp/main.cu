#include "cu_algoritmo_1.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void){
	srand(time(NULL));

	typedef float T;
	int M = 1;
	int nPars = 10000;
	int nIntervalos = 700;

	FILE * archivo_txt = fopen("archivo_init_fin.txt", "w");
	
	parametros<T> init, fin;
	PropagoMC<T> p_mc = PropagoMC<T>("./test_data/test_data.txt", nPars, nIntervalos, 1);
	for(int i=0; i<10; i++){
		std::cout << i << std::endl;
		init.b = 1.0 + ((float) rand())/((float) RAND_MAX);
		init.g = 0.5;
		p_mc.setParams(init);
		p_mc.iteroNVeces(M, false);
		fin = p_mc.getParams();

		fprintf(archivo_txt, "%f, %f, %f, %f\n", init.b, init.g, fin.b, fin.g);
	}
	fclose(archivo_txt);

	return 0;
}
