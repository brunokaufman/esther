#include<stdio.h>
#include<stdlib.h>

int main(){
	FILE * gnuplotPipe = popen("gnuplot -persist", "w");
	fprintf(gnuplotPipe, "set term pngcairo\n set output 'evolucion_parametros.png'\n set xlabel 'Número de Pasada'\n set logscale x\n set ylabel 'Valor del Parámetro'\n");
	fprintf(gnuplotPipe, "plot 'archivo_medias_incertezas.txt' u 1:2:3 w yerrorbars title 'Parámetro de Transmisión'\\\n, 'archivo_medias_incertezas.txt' u 1:4:5 w yerrorbars title 'Parámetro de Recuperación'\n");
	
	fprintf(gnuplotPipe, "set term pngcairo\n set output 'evolucion_verosimilitud.png'\n set xlabel 'Número de Pasada'\n set logscale x\n set ylabel 'Valor de la Verosimilitud'\n");
	fprintf(gnuplotPipe, "plot 'archivo_medias_incertezas.txt' u 1:6 notitle\n");
	
	pclose(gnuplotPipe);
	return 0;
}
