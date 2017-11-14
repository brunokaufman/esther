#include <fstream>
#include <stdio.h>

//AQUI COMIENZAN LAS COSAS QUE SON DEFINIDAS POR EL USUARIO.
template <typename T>
struct parametros{//Defino la estructura de mis parametros.
	int dim;//La cantidad de parametros (o sea, la dimension del espacio de parametros).
	
	//Mis parametros.
	T b;
	T g;
	
	//Cualquiera que quiero fijar tiene su valor fijo aca.
	
	//Vector auxiliar para una operacion.
	T * vector_params;
	
	//Defino una matriz inversa a la covarianza para la distribucion normal que voy a usar para b y g. Se almacena de forma no redundante (6 elementos, ya que es simetrica 3x3).
	float m_bb, m_bg, m_gg;
	
	__host__ __device__ parametros(){//Constructor default.
		//Incluyo valores de inicializacion.
		dim = 2;
		
		//Los valores iniciales que le doy a mis parametros.
		b = 1.5;
		g = 0.3;
		
		//Defino la inversa de la covarianza, de forma diagonal.
		m_bb = 1.0;
		m_bg = 0.0;
		m_gg = 1.0;
	}
	
	parametros(T * params){//Constructor desde un vector. El orden de los parametros es arbitrario.
		dim = 2;
		
		b = params[0];
		g = params[1];
		
		//Defino la inversa de la covarianza, de forma diagonal.
		m_bb = 1.0;
		m_bg = 0.0;
		m_gg = 1.0;
	}
	
	__host__ __device__ T get_parametro(int k){
		T result;
		switch(k){
			case 1:
				result = g;
				break;
			default:
				result = b;
		}
		return result;
	}
	
	__host__ __device__ void set_parametro(int k, T x){
		switch(k){
			case 1:
				g = x;
				break;
			default:
				b = x;
		}
	}
	
	__host__ void set_matriz(int i, int j, T x){
		int index = ((i + 1) * i) / 2 + j;
		switch(index){
			case 1:
				m_bg = x;
				break;
			case 2: 
				m_gg = x;
				break;
			default:
				m_bb = x;
		}
	}
	
	__device__ float dist_pert_params(){//Distribucion de probabilidad de tener una PERTURBACION en los parametros, representado en esta estructura. ESTO NO REPRESENTA AL VALOR COMPLETO DE LOS PARAMETROS.
		float exponente = m_bb * b * b + m_gg * g * g;//Las componentes diagonales.
		exponente += 2.0 * (m_bg * b * g);//Agrego las componentes no diagonales (multiplicadas por dos por simetria).
		
		return exp(-exponente/2.0);//NO NORMALIZO para que el maximo siempre sea 1.0 (me conviene para mi generador de numeros aleatorios).
	}
	
	__host__ __device__ void err_chk(){//Para chequear valores invalidos de los parametros. Puedo fijar parametros aca tambien.
		if(b < 1.0){
			b = 1.0;
		}
		if(g < 0.0){
			g = 0.0;
		}

		//Puedo fijar variables aca.
		//b = 1.3;
		//g = 0.5;
	}
	
	void printParams(std::ofstream& stream, int tiempo){
		stream << tiempo << ", " << b << ", " << g << std::endl;
		stream.flush();
	}
	
	void printParams(){
		std::cout << b << ", " << g << std::endl;
		std::cout.flush();
	}
};

template <typename T>
struct variables_estado{//Defino la estructura de mis variables de estado.
	int dim;//La cantidad de variables (o sea, la dimension de mi espacio de fases).
	
	//Mis variables.
	T s;
	T i;
	T r;
	
	T t;//El tiempo se usa para controlar.

	//Constantes que puedo querer tener por comodidad.
	T poblacion = 100000.0;
	T inf_iniciales = 1.0;
	
	__host__ __device__ variables_estado(){//Constructor default.
		dim = 3;
		
		//Incluyo valores iniciales.
		s = poblacion - inf_iniciales;
		i = inf_iniciales;
		r = 0.0;
	}
	
	variables_estado(T * params){//Constructor desde un vector. El orden de las variables es arbitrario.
		dim = 3;
		
		s = params[0];
		i = params[1];
		r = params[2];
	}
	
	__device__ T get_variable(int k){
		T result;
		switch(k){
			case 1:
				result = i;
				break;
			case 2:
				result = r;
				break;
			default:
				result = s;
				break;
		}
		return result;
	}
};


template <typename T>
struct variables_medicion{//Defino mis variables de medicion.
	T t;
	int inf;
	
	bool notdato = false;//Control por si no se leyo ningun dato correctamente. Si hay un error, se coloca en verdadero, que permite un control desde afuera.
	
	//Constantes que puedo llegar a querer.
	T prob_deteccion = 0.01;//Uno en cien.

	variables_medicion(){//Constructor default para inicializar la struct sin inicializar sus componentes. Utilizar con cuidado!
	}
	
	variables_medicion(std::vector<std::string> linea){
		try{
			t = std::stod(linea.data()[0]);//Agrega el string auxiliar del tiempo a un doble en el vector mismo.
			inf = std::stoi(linea.data()[1]);//Agrega el string auxiiliar de la cantidad de casos su vector correspondiente.
		}
		catch(const std::invalid_argument& ia){//Atrapa un error si el string pasado a alguno no era un numero, y no tiene efecto sobre el vector correspondiente (como en la primera fila).
			std::cout << "Argumento invalido al traducir caracter." << std::endl;
			notdato = true;
		}
		catch(const std::out_of_range& oor){//Atrapa un error si algun doble es demasiado grande.
			std::cout << "Numero fuera del rango." << std::endl;
			notdato = true;
		}
	}
	
	void printMedicionConAjuste(std::ofstream& stream, variables_estado<T> v_print){
		stream << t << ", " << inf << ", " << prob_deteccion * v_print.i << std::endl;
		stream.flush();
	}

	__device__ float prob_normal(float i, int real){//p es la probabilidad de un exito.
		float var = i * prob_deteccion * (1.0 - prob_deteccion);
		float diff = (i * prob_deteccion - real);
		float result = exp(-diff * diff / (2 * var));
		
		if(result < 0.0){
			return 0.0;
		}
		else{
			return result;
		}
	}
	
	__device__ float operator()(variables_estado<T> v_sistema, parametros<T> p){//Aca defino mi modelo de medicion: me toma mis mediciones, compara con las variables del sistema, y utilizando los parametros como se deba (si es que debe), me retorna la probabilidad de que esa medicion haya sido realizada. Se llama operator() porque se utiliza en paralelo, como operador THRUST de esta estructura. Operador binario obligatoriamente (por el funcionamiento del codigo de los algoritmos estadisticos).
		return prob_normal(v_sistema.i, inf);//La probabilidad de que, de i infectados, se reporten inf. Cada uno es independiente del otro, dada una probabilidad de reportaje.
	}
};

template <typename T>
struct propagador_variables{
	T h;
	
	propagador_variables(T _h){
		h = _h;
	}
	
	propagador_variables(){//Constructor default.
		h = 0.01;//Para no dejar desinicializada.
	}
	
	__device__ variables_estado<T> operator()(parametros<T> p, variables_estado<T> v){//Defino la relacion de recurrencia. Se le llama operator() porque es el operador THRUST de esta estructura. Operador binario obligatoriamente.
		variables_estado<T> result;
		
		//Estas son mis relaciones de recurrencia para cada variable de estado.		
		result.s = v.s - h * p.g * p.b * v.s * v.i / v.poblacion;
		result.i = v.i + h * p.g * v.i * (p.b * v.s / v.poblacion - 1.0);
		result.r = v.r + h * p.g * v.i;
		result.t = v.t + h;
		
		if(::isnan(result.s) || result.s < 0.0){
			result.s = 0.0;
		}
		if(::isnan(result.i) || result.i < 0.0){
			result.i = 0.0;
		}
		if(::isnan(result.r) || result.r < 0.0){
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
struct cooling_factor{
	T pow_iteracion = 1.0;
	T pow_pasada = 0.4;
	T a = 0.1;
	
	T c_factor(int iteracion, int pasada){
		T result_it = 1.0;
		T result_pas = 1.0;
		int i = iteracion;
		int j = pasada;
		while(i > 0){
			result_it *= pow_iteracion;
			i--;
		}
		while(j > 0){
			result_pas *= pow_pasada;
			j--;
		}
		return a * result_pas * result_it;
	}
};
//AQUI TERMINAN LAS COSAS QUE SON DEFINIDAS POR EL USUARIO.
