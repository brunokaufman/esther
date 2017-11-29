#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>

#include <curand_kernel.h>

#include <time.h>
#include <math.h>

#include "cu_userdef.cuh"
#include "cu_stat.cuh"
#include "cpp_import.h"

struct curandstate_seeder{
	int tiempo;

	curandstate_seeder(int _tiempo){
		tiempo = _tiempo;
	}

	__device__ curandState operator()(int &idx){
		unsigned int semilla = idx + tiempo;
		curandState s;
		curand_init(semilla, 0, 0, &s);
		return s;
	}
};

template <typename T>
struct generador_parametros_aleatorios{
	T rango;
	T mult;
	
	generador_parametros_aleatorios(T _rango, T _mult){
		rango = _rango;
		mult = _mult;
    }
    
    generador_parametros_aleatorios(){//Constructor default.
		rango = 10.0;
		mult = 0.1;
    }
	
	__device__ parametros<T> operator()(curandState &s, parametros<T> &p){
		bool aceptado = false;
		parametros<T> perturbacion;
		while(!aceptado){
			for(int i=0; i<p.dim; i++){
				perturbacion.set_parametro(i, rango * 2.0 * (curand_uniform(&s) - 0.5));
			}
			float probabilidad_aceptacion = curand_uniform(&s);
			
			if(perturbacion.dist_pert_params() > probabilidad_aceptacion){
				aceptado = true;
			}
		}
		
		parametros<T> result;
		for(int k=0; k<p.dim; k++){
			result.set_parametro(k, (mult * perturbacion.get_parametro(k) + 1.0) * p.get_parametro(k));
		}
		result.err_chk();
		
		return result;
	}
};

template <typename T>
struct generador_estado_aleatorio{
	T rango;
	T mult;
	
	generador_estado_aleatorio(T _rango, T _mult){
		rango = _rango;
		mult = _mult;
    }
    
    generador_estado_aleatorio(){//Constructor default.
		rango = 10.0;
		mult = 0.1;
    }
	
	__device__ variables_estado<T> operator()(curandState &s, variables_estado<T> &v){
		bool aceptado = false;
		variables_estado<T> perturbacion;
		while(!aceptado){
			for(int i=0; i<v.dim; i++){
				perturbacion.set_variable(i, rango * 2.0 * (curand_uniform(&s) - 0.5));
			}
			float probabilidad_aceptacion = curand_uniform(&s);
			
			if(perturbacion.dist_pert_var() > probabilidad_aceptacion){
				aceptado = true;
			}
		}
		
		variables_estado<T> result;
		for(int k=0; k<v.dim; k++){
			result.set_variable(k, (mult * perturbacion.get_variable(k) + 1.0) * v.get_variable(k));
		}
		result.err_chk();
		
		return result;
	}
};

template <typename T>
struct parametros_a_vector{
	__device__ T operator()(int &i, parametros<T> p){
		int rest = i % p.dim;
		return p.get_parametro(rest);
	}
};

template <typename T>
struct variables_a_vector{	
	__device__ T operator()(int &i, variables_estado<T> v){
		int rest = i % v.dim;
		return v.get_variable(rest);
	}
};

template <typename T>
struct nuevos_params{
	cooling_factor<T> mult;
	
	nuevos_params(cooling_factor<T> _mult){
		mult = _mult;
	}
	
	parametros<T> p_nuevos(parametros<T> promedios, parametros<T> viejos, int n_dato, int n_pasada){
		parametros<T> result;

		for(int k=0; k<viejos.dim; k++){
			result.set_parametro(k, mult.c_factor(n_dato, n_pasada, k) * (promedios.get_parametro(k) - viejos.get_parametro(k)) + viejos.get_parametro(k));
		}
		
		result.err_chk();
		
		return result;
	}
};

template <typename T>
struct filtro{
	T a;

	filtro(T _a){
		a = _a;
	}

	__device__ T operator()(T presente, T pasado){
		return pow(pasado, a) * pow(presente, a);//a es la importancia relativa del presente respecto al pasado.
	}
};

template <typename T>
class PropagoMC{//Clase para generar distintas perturbaciones a los parametros y simular los resultados hasta la proxima medicion.
	private:
	thrust::device_vector<parametros<T> > p;
	int nPars;
	int dimPars;
	
	thrust::device_vector<variables_estado<T> > v;
	thrust::device_vector<variables_estado<T> > v_iniciales;
	int nVars;
	int dimVars;
	
	T delta_tiempo;
	int n_intervalos;
	
	parametros<T> p_viejo;
	variables_estado<T> v_viejo;
	variables_estado<T> v_medio_sin_peso;
	variables_estado<T> v_init;
	
	std::vector<variables_medicion<T> > datos = {};
	int total_datos;
	int dato_counter = 0;
	int pasada_counter = 0;
	
	thrust::device_vector<float> probabilidades_variables;
	float verosimilitud = 1.0;
	
	propagador_variables<T> prop_s;
	cooling_factor<T> c_f;

	T mult = 1.0;//Multiplicador que potencia o atenua las perturbaciones aleatorias. Default: las deja iguales.

	calculo_medias_y_varianzas<T> calc_est;

	thrust::device_vector<curandState> c_state;

	void seed_curand(){
		c_state.resize(nPars);
		curandstate_seeder c_seed(time(NULL));
		thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(nPars), c_state.begin(), c_seed);
	}
	
	void generoRandoms(){
		thrust::transform(thrust::device, c_state.begin(), c_state.end(), p.begin(), p.begin(), generador_parametros_aleatorios<T>(10.0, mult));
	}
	
	void propagoPuntos(){//Avanzo hasta el proximo punto de datos.
		if(dato_counter < total_datos){
			T diff_t = datos[dato_counter + 1].t - datos[dato_counter].t;
			delta_tiempo = diff_t / ((T) n_intervalos);
			prop_s.h = delta_tiempo;
			
			for(int i=0; i<n_intervalos; i++){
				thrust::transform(thrust::device, p.begin(), p.end(), v.begin(), v.begin(), prop_s);
			}
			
			dato_counter++;
		}
	}
	
	void calculoProbPuntos(){
		//Calculo las probabilidades nuevas de estar en cierto estado del sistema con mi modelo de medicion.
		thrust::device_vector<float> probabilidades_variables_actuales(nPars);
		thrust::transform(thrust::device, v.begin(), v.end(), p.begin(), probabilidades_variables_actuales.begin(), datos.data()[dato_counter]);

		//"Normalizo" dichas probabilidades, buscando que queden centradas alrededor de 1.0 (evito numeros excesivamente chicos cuando multiplico).
		//Primero calculo la "Normalizacion": el promedio de los pesos.
		float normalizacion = thrust::reduce(thrust::device, probabilidades_variables_actuales.begin(), probabilidades_variables_actuales.end(), 0.0, thrust::maximum<float>());
		verosimilitud = pow(verosimilitud, ((float) (dato_counter - 1))/((float) dato_counter)) * pow(normalizacion, 1.0/((float) dato_counter));
		//Ahora divido todas mis probabilidades (pesos) por su promedio.
		if(normalizacion <= 0.0 || ::isnan(normalizacion)){//Si hubo un error procesando la suma, o es muy chico, reinicializo los pesos a 1.0 (identidad).
			thrust::fill(thrust::device, probabilidades_variables_actuales.begin(), probabilidades_variables_actuales.end(), 1.0);
			std::cout << "Aviso: probabilidades reinicializadas en pasada " << pasada_counter << ", iteracion " << dato_counter << "." << std::endl;
		}
		else{
			thrust::transform(thrust::device, probabilidades_variables_actuales.begin(), probabilidades_variables_actuales.end(), thrust::make_constant_iterator<float>(normalizacion), probabilidades_variables_actuales.begin(), thrust::divides<float>());
		}

		//Ahora aplico una funcion que me de los pesos nuevos en funcion de los actuales y el pasado inmediato.
		thrust::transform(thrust::device, probabilidades_variables_actuales.begin(), probabilidades_variables_actuales.end(), probabilidades_variables.begin(), probabilidades_variables.begin(), filtro<float>(0.5));
	}
	
	void reduzcoParametros(){
		int n = nPars;
		int dim = dimPars;
		
		//Primera parte: paso al vector de parametros a un formato compatible con mi funcion en cu_stat.h
		thrust::device_vector<T> structvec_hecho_vector(n * dim);
		
		thrust::device_vector<int> count(n * dim);
		thrust::sequence(thrust::device, count.begin(), count.end(), 0);
		
		thrust::device_vector<int> map(n * dim);
		thrust::transform(thrust::device, count.begin(), count.end(), thrust::make_constant_iterator(dim), map.begin(), thrust::divides<int>());
		
		thrust::device_vector<parametros<T> > structvec_padded(n * dim);
		thrust::gather(thrust::device, map.begin(), map.end(), p.begin(), structvec_padded.begin());
		
		parametros_a_vector<T> psv_a_v;
		thrust::transform(thrust::device, count.begin(), count.end(), structvec_padded.begin(), structvec_hecho_vector.begin(), psv_a_v);
		
		//Siempre se reduce con la probabilidad de la particula (sea en el espacio de variables o en el espacio de fases) como peso.
		thrust::device_vector<float> probs_hechas_vector_padded(n * dim);
		thrust::gather(thrust::device, map.begin(), map.end(), probabilidades_variables.begin(), probs_hechas_vector_padded.begin());
		
		//Segunda parte: reduzco usando mi clase de calculo de promedio pesado y varianza en cu_stat.h
		calc_est = calculo_medias_y_varianzas<T>(n, dim, structvec_hecho_vector, probs_hechas_vector_padded);
		calc_est.realizo_calculos();
		
		//Tercera parte: transformo el vector media de parametros en un nuevo vector de parametros iniciales (todos iguales).
		parametros<T> p_promedio(calc_est.get_media());//Creo e inicializo los parametros promedio con el vector de la media.
		
		nuevos_params<T> n_p(c_f);//Amplifico las perturbaciones por el cooling factor.
		parametros<T> nuevo_p = n_p.p_nuevos(p_promedio, p_viejo, dato_counter, pasada_counter);//Coloco el nuevo centro de mis parametros.
		/*for(int i=0; i<dimPars; i++){//Coloco la nueva covarianza: la inversa de la covarianza obtenida en los puntos MC (para intentar compensar direcciones preferenciales).
			for(int j=0; j<=i; j++){
				nuevo_p.set_matriz(i, j, calc_est.get_covarianza_fraccional(i, j));
			}
		}*/
		thrust::fill(thrust::device, p.begin(), p.end(), nuevo_p);
		p_viejo = nuevo_p;
	}

	void reduzcoVariablesSinPeso(){//Solo se usa para estimar un "promedio" de por donde esta quedando el ajuste (si se desea imprimir).
		int n = nVars;
		int dim = dimVars;
		
		//Primera parte: paso al vector de parametros a un formato compatible con mi funcion en cu_stat.h
		thrust::device_vector<T> structvec_hecho_vector(n * dim);
		
		thrust::device_vector<int> count(n * dim);
		thrust::sequence(thrust::device, count.begin(), count.end(), 0);
		
		thrust::device_vector<int> map(n * dim);
		thrust::transform(thrust::device, count.begin(), count.end(), thrust::make_constant_iterator(dim), map.begin(), thrust::divides<int>());
		
		thrust::device_vector<variables_estado<T> > structvec_padded(n * dim);
		thrust::gather(thrust::device, map.begin(), map.end(), v.begin(), structvec_padded.begin());
		
		variables_a_vector<T> vsv_a_v;
		thrust::transform(thrust::device, count.begin(), count.end(), structvec_padded.begin(), structvec_hecho_vector.begin(), vsv_a_v);
		
		//Segunda parte: reduzco usando mi clase de calculo de promedio pesado y varianza en cu_stat.h
		calculo_medias_y_varianzas<T> calc_est_local(n, dim, structvec_hecho_vector);//Sin peso esta vez.
		calc_est_local.realizo_calculos();
		
		//Tercera parte: transformo el vector media de parametros en un nuevo vector de parametros iniciales (todos iguales).
		v_medio_sin_peso = variables_estado<T>(calc_est_local.get_media());//Creo e inicializo el estado promedio del sistema con el vector de la media.
	}

	void reduzcoVariablesIniciales(){//Solo se usa para estimar un "promedio" de por donde esta quedando el ajuste (si se desea imprimir).
		int n = nVars;
		int dim = dimVars;
		
		//Primera parte: paso al vector de parametros a un formato compatible con mi funcion en cu_stat.h
		thrust::device_vector<T> structvec_hecho_vector(n * dim);
		
		thrust::device_vector<int> count(n * dim);
		thrust::sequence(thrust::device, count.begin(), count.end(), 0);
		
		thrust::device_vector<int> map(n * dim);
		thrust::transform(thrust::device, count.begin(), count.end(), thrust::make_constant_iterator(dim), map.begin(), thrust::divides<int>());
		
		thrust::device_vector<variables_estado<T> > structvec_padded(n * dim);
		thrust::gather(thrust::device, map.begin(), map.end(), v_iniciales.begin(), structvec_padded.begin());
		
		variables_a_vector<T> vsv_a_v;
		thrust::transform(thrust::device, count.begin(), count.end(), structvec_padded.begin(), structvec_hecho_vector.begin(), vsv_a_v);
		
		//Segunda parte: reduzco usando mi clase de calculo de promedio pesado y varianza en cu_stat.h
		calculo_medias_y_varianzas<T> calc_est_local;
		if(verosimilitud > 0.0){
			calc_est_local = calculo_medias_y_varianzas<T>(n, dim, structvec_hecho_vector, probabilidades_variables);//Filtro por verosimilitud.
		}
		else{
			calc_est_local = calculo_medias_y_varianzas<T>(n, dim, structvec_hecho_vector);//Filtro sin peso.
		}
		calc_est_local.realizo_calculos();
		
		//Tercera parte: transformo el vector media de parametros en un nuevo vector de parametros iniciales (todos iguales).
		v_init = variables_estado<T>(calc_est_local.get_media());//Creo e inicializo el estado promedio del sistema con el vector de la media.
	}
	
	public:
	PropagoMC(std::string _filename, int _nPars, int _n_intervalos, int _dimMes){//Constructor en funcion de los parametros iniciales, los valores iniciales de las variables de estado, la cantidad de particulas hechas por MC, y la cantidad de intervalos deseados entre mediciones.
		//Inicializacion del vector de parametros (antes de randomizar en la primera perturbacion).
		p.resize(_nPars);
		thrust::fill(thrust::device, p.begin(), p.end(), p_viejo);

		probabilidades_variables.resize(_nPars);
		thrust::fill(thrust::device, probabilidades_variables.begin(), probabilidades_variables.end(), 1.0);
		
		nPars = _nPars;
		nVars = _nPars;//Son lo mismo.
		dimPars = p_viejo.dim;
		dimVars = v_init.dim;
		
		delta_tiempo = 1.0/((T) _n_intervalos);
		n_intervalos = _n_intervalos;
		prop_s.h = delta_tiempo;
		
		//Leo los datos. _dimMes es la dimension de mis mediciones sin contar al tiempo, asi que le agrego uno para darme el numero de columnas en mi archivo de datos.
		TimeSeries ts(_filename, ',', _dimMes + 1);
		std::vector<std::string> linea;
		linea.resize(_dimMes + 1);
		int i = 0;
		linea = ts.leo_linea();
		while(linea.size() != 0){
			variables_medicion<T> vmed_aux(linea);
			if(vmed_aux.notdato){//Sale del while si no leyo nada.
				break;
			}
			datos.push_back(vmed_aux);
			i++;
			linea = ts.leo_linea();
		}
		total_datos = i - 1;
		
		v.resize(nPars);
		thrust::fill(thrust::device, v.begin(), v.end(), v_init);//Inicializo todas mis simulaciones en el valor inicial de los datos.
	}
	
	void iteracionAlgoritmo(){
		generoRandoms();
		propagoPuntos();
		calculoProbPuntos();
		reduzcoVariablesSinPeso();
		reduzcoParametros();
	}

	void iteroNVeces(int N, bool graficoEvolucion, bool ajusto_estado_inicial, bool graboMediasIncertezas){
		seed_curand();//Seedeo los generadores que voy a necesitar.

		FILE * archivo_medias_incertezas;
		if(graboMediasIncertezas){
			archivo_medias_incertezas = fopen("archivo_medias_incertezas.txt", "a");
		}

		FILE * gnuplotPipeDatos, * gnuplotPipeParams;
		std::ofstream archivoDatos, archivoParams;

		if(ajusto_estado_inicial){
			v_iniciales.resize(nVars);
		}

		if(graficoEvolucion){//Abro dos pipes a gnuplot e inicializo la terminal en gif. Agrego comandos que necesite.
			gnuplotPipeDatos = popen("gnuplot -persist", "w");
			gnuplotPipeParams = popen("gnuplot -persist", "w");

			fprintf(gnuplotPipeDatos, "set term pngcairo\n");
			fprintf(gnuplotPipeParams, "set term pngcairo\n");
		}

		std::stringstream filename_datos;
		std::stringstream filename_parametros;

		pasada_counter = 0;
		for(int j=0; j<N; j++){
			if(graficoEvolucion){
				fprintf(gnuplotPipeDatos, "set output './graficos/datos_%d.png'\n set xlabel 'Tiempo (Semanas)'\n set ylabel 'Casos Reportados'\n set y2label 'Verosimilitud'\n set ytics nomirror\n set y2tics\n", pasada_counter);
				fprintf(gnuplotPipeParams, "set output './graficos/parametros_%d.png'\n set xlabel 'Iteracion de Filtrado (semana)'\n set ylabel 'Valor de Parametro'\n", pasada_counter);
		
				fprintf(gnuplotPipeDatos, "set title 'Pasada %d'\n", pasada_counter + 1);
				fprintf(gnuplotPipeParams, "set title 'Pasada %d'\n", pasada_counter + 1);	

				filename_datos.str(std::string());
				filename_parametros.str(std::string());
				filename_datos << "./archivos_intermedios/archivo_datos_" << j << ".txt";
				filename_parametros << "./archivos_intermedios/archivo_parametros_" << j << ".txt";

				archivoDatos.open(filename_datos.str(), std::ofstream::out);//Cada iteracion vuelvo a abrirlos. Al final los cierro.
				archivoParams.open(filename_parametros.str(), std::ofstream::out);		
			}

			thrust::fill(thrust::device, v.begin(), v.end(), v_init);//Inicializo todas mis simulaciones en el valor inicial de los datos.
			if(ajusto_estado_inicial){
				thrust::transform(thrust::device, c_state.begin(), c_state.end(), v.begin(), v.begin(), generador_estado_aleatorio<T>(10.0, 0.01 * pow(0.999, (float) pasada_counter)));//Perturbo la condicion inicial una cantidad aleatoria.
				thrust::copy(thrust::device, v.begin(), v.end(), v_iniciales.begin());
			}
			
			thrust::fill(thrust::device, probabilidades_variables.begin(), probabilidades_variables.end(), 1.0);//Cada vez tengo que volver a inicializar mis pesos a 1.0.

			for(int i=0; i<total_datos; i++){
				iteracionAlgoritmo();
				if(graficoEvolucion){
					//Calculo el "estado medio" del sistema para graficar (sin preferencia hacia los datos "verdaderos", o sea, sin pesos).
					datos[i].printMedicionConAjuste(archivoDatos, v_medio_sin_peso);
					archivoDatos << ", " << verosimilitud << std::endl;
					
					//Grafico los parametros, con el tiempo en el cual fueron calculados.
					archivoParams << datos[i].t << ", ";
					p_viejo.printParams(archivoParams);
				}
			}

			if(graficoEvolucion){
				archivoDatos.close();//Cierro para luego reabrirlos, asi siempre se sobreescriben.
				archivoParams.close();
				
				fprintf(gnuplotPipeDatos, "plot '%s' u 1:2 title 'Datos Registrados'\\\n, '%s' u 1:3 w l title 'Ajuste'\\\n, '%s' u 1:4 axis x1y2 w l title 'Verosimilitud'\n", filename_datos.str().c_str(), filename_datos.str().c_str(), filename_datos.str().c_str());
				fprintf(gnuplotPipeParams, "plot '%s' u 1:2 w l title 'Parametro de Transmision 1'\\\n, '%s' u 1:3 w l title 'Parametro de Transmision 2'\\\n, '%s' u 1:4 w l title 'Parametro de Recuperacion'\\\n, '%s' u 1:5 w l title 'Parametro de Ancho Estacional'\n", filename_parametros.str().c_str(), filename_parametros.str().c_str(), filename_parametros.str().c_str(), filename_parametros.str().c_str());			
			}

			if(ajusto_estado_inicial){
				reduzcoVariablesIniciales();
			}

			if(graboMediasIncertezas){
				fprintf(archivo_medias_incertezas, "%d, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", pasada_counter, p_viejo.b0, calc_est.get_covarianza(0, 0), p_viejo.b1, calc_est.get_covarianza(1, 1), p_viejo.g, calc_est.get_covarianza(2, 2), p_viejo.sig, calc_est.get_covarianza(3, 3), verosimilitud);
			}

			dato_counter = 0;//Reseteo luego de cada pasada por los datos.
			pasada_counter++;
		}

		if(graficoEvolucion){
			pclose(gnuplotPipeDatos);
			pclose(gnuplotPipeParams);
		}

		if(graboMediasIncertezas){
			pclose(archivo_medias_incertezas);
		}
	}

	void setParams(parametros<T> _p){
		p_viejo = _p;
		thrust::fill(thrust::device, p.begin(), p.end(), _p);
	}

	parametros<T> getParams(){
		return p_viejo;
	}

	void printParams(){
		std::cout << "Parametros Finales:" << std::endl;
		p_viejo.printParams();
		calc_est.print_covarianzas();
	}

	void setIntensidadPerturbaciones(T _mult){
		mult = _mult;
	}
};
