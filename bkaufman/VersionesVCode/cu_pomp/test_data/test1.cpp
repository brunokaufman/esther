#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <math.h>

template <typename T>
struct parametros_prueba{//Defino mis parametros.
	T b;
	T g;
};

template <typename T>
struct variables_estado{
	T t;
	T s;
	T i;
	T r;
	
	T poblacion = 100000.0;
};	

template <typename T>
struct propago_sistema_prueba{
	T h;
	
	propago_sistema_prueba<T>(T _h){
		h = _h;
	}
	
	variables_estado<T> propagar(parametros_prueba<T> p, variables_estado<T> v){//Defino la relacion de recurrencia.
		variables_estado<T> result;
		
		//Estas son mis relaciones de recurrencia para cada variable de estado.		
		result.s = v.s - h * p.g * v.i * p.b * v.s / v.poblacion;
		result.i = v.i + h * p.g * v.i * (p.b * v.s / v.poblacion - 1.0);
		result.r = v.r + h * p.g * v.i;
		result.t = v.t + h;
		
		if(result.s < 0.0){
			result.s = 0.0;
		}
		if(result.i < 0.0){
			result.i = 0.0;
		}
		if(result.r < 0.0){
			result.r = 0.0;
		}
		
		T total = result.s + result.i + result.r;//Mantengo el total en 10000.0, si no se aparta por error computacional.
		result.s *= v.poblacion/total;
		result.i *= v.poblacion/total;
		result.r *= v.poblacion/total;
		
		return result;
	}
};

template <typename T>
struct mido_sistema_prueba{
	float prob_deteccion = 0.01;
	
	float prob_normal(T i, int inf){//p es la probabilidad de un exito.
		float var = i * prob_deteccion * (1.0 - prob_deteccion);
		float diff = (i * prob_deteccion - inf);
		float result = exp(-diff * diff / (2 * var));
		
		if(result < 0.0){
			return 0.0;
		}
		else{
			return result;
		}
	}
	
	int gen_int_aleatorio(T maximo){//Con la distribucion que elija.
		int result;
		if(maximo <= 0.5){
			result = 0;
		}
		else{
			int moda_binom = (int) ((float)(maximo + 1) * prob_deteccion);
			bool aceptado = false;
			int random = 0;
			int count = 0;
			while(!aceptado && count<10000){
				random = std::rand() % ((int) maximo);
				float prob_aceptacion = ((float) std::rand())/((float) RAND_MAX);
				if(prob_normal(maximo, random) > prob_aceptacion){
					aceptado = true;
				}
				count++;
				if(count == 10000){
					std::cout << count << std::endl;
				}
			}
			result = random;
		}
		return result;
	}
	
	int infectados(variables_estado<T> v){
		return gen_int_aleatorio(v.i);
	}
};

int main(void){
	srand(time(NULL));
	typedef float T;
	int deltaSampleo = 100;
	int nSampleos = 100;
	
	parametros_prueba<T> p;
	variables_estado<T> v;
	p.b = 1.3;
	p.g = 0.5;
	v.t = 0.0;
	v.s = 99999.0;
	v.i = 1.0;
	v.r = 0.000;
	
	propago_sistema_prueba<T> prop = propago_sistema_prueba<T>(0.01);
	mido_sistema_prueba<T> mid;
	
	std::ofstream odata ("test_data.txt");
	for(int t=0; t<nSampleos * deltaSampleo; t++){
		v = prop.propagar(p, v);
		if(t % deltaSampleo == 0){
			std::cout << "Iteracion " << t / deltaSampleo << ", " << v.i * mid.prob_deteccion << std::endl;
			odata << t / deltaSampleo << "," << mid.infectados(v) << std::endl;
		}
	}
	return 0;
}
