#include<stdio.h>
#include<stdlib.h>

int main(){
	FILE * gnuplotPipe = popen("gnuplot -persist", "w");
	
	int i;
	fprintf(gnuplotPipe, "set term pngcairo\n set output 'evolucion_parametros.png'\n set xlabel 'Número de Pasada'\n set ylabel 'Valor del Parámetro'\n");
	fprintf(gnuplotPipe, "plot 'archivo_medias_incertezas.txt' u 1:2:3 w yerrorbars title 'Parámetro de Transmisión Constante'\\\n, 'archivo_medias_incertezas.txt' u 1:4:5 w yerrorbars title 'Parámetro de Transmisión Estacional'\\\n, 'archivo_medias_incertezas.txt' u 1:6:7 w yerrorbars title 'Parámetro de Recuperación'\\\n, 'archivo_medias_incertezas.txt' u 1:8:9 w yerrorbars title 'Parámetro de Ancho Estacional'\n");
	
	fprintf(gnuplotPipe, "set term pngcairo\n set output 'evolucion_verosimilitud.png'\n set xlabel 'Número de Pasada'\n set ylabel 'Valor de la Verosimilitud'\n");
	fprintf(gnuplotPipe, "plot 'archivo_medias_incertezas.txt' u 1:10 notitle\n");
	
	pclose(gnuplotPipe);
	return 0;
}
