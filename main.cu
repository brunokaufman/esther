#include "cu_algoritmo_1.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void){
	srand(time(NULL));

	typedef float T;
	int M = 100;
	int nPars = 4000;
	int nIntervalos = 700;
	FILE * archivo_comparacion_init_fin = fopen("archivo_init_fin.txt", "w");

	FILE * archivo_medias_incertezas = fopen("archivo_medias_incertezas.txt", "w");//Abro este archivo para borrar los contenidos previos.
	fclose(archivo_medias_incertezas);
	
	parametros<T> init, fin;
	PropagoMC<T> p_mc = PropagoMC<T>("./test_data/FluNetInteractiveReport_USA_2016_2017_36.csv", nPars, nIntervalos, 1);
	p_mc.setIntensidadPerturbaciones(0.1);
	for(int i=0; i<1; i++){
		std::cout << "Ensayo " << i << std::endl;
		init.b0 = -0.73;
		init.b1 = -1.3;
		init.sig = 20.0;
		init.g = 0.33;
		p_mc.setParams(init);
		p_mc.iteroNVeces(M, true, false, true);
		fin = p_mc.getParams();

		fprintf(archivo_comparacion_init_fin, "%f, %f, %f, %f, %f, %f, %f, %f\n", init.b0, init.b1, init.g, init.sig, fin.b0, fin.b1, fin.g, fin.sig);
	}
	p_mc.printParams();

	fclose(archivo_comparacion_init_fin);

	FILE * gnuplotPipe = popen("gnuplot -persist", "w");

	fprintf(gnuplotPipe, "set term pngcairo\n set output 'init_fin_mismos.png'\n set xlabel 'Valor Inicial del Parámetro'\n set ylabel 'Valor Final del Parámetro\n");
	fprintf(gnuplotPipe, "plot 'archivo_init_fin.txt' u 1:5 title 'Parámetro de Transmisión Base'\\\n, 'archivo_init_fin.txt' u 2:6 title 'Parámetro de Transmisión Estacional'\\\n, 'archivo_init_fin.txt' u 3:7 title 'Parámetro de Recuperación'\\\n, 'archivo_init_fin.txt' u 4:8 title 'Parámetro de Ancho de la Estacionalidad'\\\n, x title 'Linea Identidad'\n");
	
	fprintf(gnuplotPipe, "set term pngcairo\n set output 'evolucion_pasadas.png'\n set xlabel 'Numero de Pasada'\n set ylabel 'Valor Ajustado del Parámetro'\n set y2label 'Verosimilitud del Ajuste'\n set ytics nomirror\n set y2tics\n");
	fprintf(gnuplotPipe, "plot 'archivo_medias_incertezas.txt' u 1:2:3 with yerrorbars title 'Parámetro de Transmisión Base'\\\n, 'archivo_medias_incertezas.txt' u 1:4:5 with yerrorbars title 'Parámetro de Transmisión Estacional'\\\n, 'archivo_medias_incertezas.txt' u 1:6:7 with yerrorbars title 'Parámetro de Recuperación'\\\n, 'archivo_medias_incertezas.txt' u 1:8:9 with yerrorbars title 'Parámetro de Ancho de la Estacionalidad'\\\n, 'archivo_medias_incertezas.txt' u 1:10 axis x1y2 title 'Verosimilitud'\n");
	
	/*fprintf(gnuplotPipe, "set output 'init_trans_fin_rec.png'\n set xlabel 'Valor Inicial del Parámetro de Transmisión'\n set ylabel 'Valor Final del Parámetro de Recuperación\n");
	fprintf(gnuplotPipe, "plot 'archivo_init_fin.txt' u 1:4 notitle\n");

	fprintf(gnuplotPipe, "set output 'init_rec_fin_trans.png'\n set xlabel 'Valor Inicial del Parámetro de Recuperación'\n set ylabel 'Valor Final del Parámetro de Transmisión\n");
	fprintf(gnuplotPipe, "plot 'archivo_init_fin.txt' u 2:3 notitle\n");*/
	
	pclose(gnuplotPipe);

	return 0;
}
