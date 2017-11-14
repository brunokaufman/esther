#include "cu_algoritmo_1.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void){
	srand(time(NULL));

	typedef float T;
	int M = 30;
	int nPars = 10000;
	int nIntervalos = 700;

	FILE * archivo_txt = fopen("archivo_init_fin.txt", "w");
	
	parametros<T> init, fin;
	PropagoMC<T> p_mc = PropagoMC<T>("./test_data/test_data.txt", nPars, nIntervalos, 1);
	p_mc.setIntensidadPerturbaciones(0.1);
	for(int i=0; i<1; i++){
		std::cout << "Ensayo " << i << std::endl;
		init.b = 1.0 + ((float) rand())/((float) RAND_MAX);
		init.g = ((float) rand())/((float) RAND_MAX);
		p_mc.setParams(init);
		p_mc.iteroNVeces(M, true);
		fin = p_mc.getParams();

		fprintf(archivo_txt, "%f, %f, %f, %f\n", init.b, init.g, fin.b, fin.g);
	}
	fclose(archivo_txt);

	FILE * gnuplotPipe = popen("gnuplot -persist", "w");

	fprintf(gnuplotPipe, "set term pngcairo\n set output 'init_fin_mismos.png'\n set xlabel 'Valor Inicial del Parámetro'\n set ylabel 'Valor Final del Parámetro\n");
	fprintf(gnuplotPipe, "plot 'archivo_init_fin.txt' u 1:3 title 'Parámetro de Transmisión'\\\n, 'archivo_init_fin.txt' u 2:4 title 'Parámetro de Recuperación'\\\n, x title 'Linea Identidad'\n");
	
	fprintf(gnuplotPipe, "set output 'init_trans_fin_rec.png'\n set xlabel 'Valor Inicial del Parámetro de Transmisión'\n set ylabel 'Valor Final del Parámetro de Recuperación\n");
	fprintf(gnuplotPipe, "plot 'archivo_init_fin.txt' u 1:4 notitle\n");

	fprintf(gnuplotPipe, "set output 'init_rec_fin_trans.png'\n set xlabel 'Valor Inicial del Parámetro de Recuperación'\n set ylabel 'Valor Final del Parámetro de Transmisión\n");
	fprintf(gnuplotPipe, "plot 'archivo_init_fin.txt' u 2:3 notitle\n");
	
	pclose(gnuplotPipe);

	return 0;
}
