#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

//Estructura que contiene un valor dado, con su peso estadistico en punto flotante.
template <typename T>
struct dato{
	T valor;
	float peso;
};

//Estructura operador binario, transforma un valor y un peso en una sola estructura de tipo dato.
template <typename T>
struct vector_a_dato{
	__device__  void dato_chk(dato<T> dat){//Chequeo de nans y negativos absurdos.
		if(isnan(dat.valor)){
			dat.valor = (T) 0;
		}
		
		if(isnan(dat.peso) || dat.peso < 0.0){
			dat.peso = 0.0;
		}
	}
	
	__device__ dato<T> operator()(const T& _valor, const float& _peso){//El operador contenido.
		dato<T> resultado;
		resultado.peso = _peso;
		resultado.valor = _valor;
		
		dato_chk(resultado);
		
		return resultado;
	}
};

//Estructura operador binario responsable por reducir dos datos en uno. Promedia sus valores por sus pesos, y suma sus pesos como la significancia del dato nuevo.
template <typename T>
struct promedio_pesado{
	__device__  void dato_chk(dato<T> dat){//Chequeo de nans y pesos negativos absurdos.
		if(isnan(dat.valor)){
			dat.valor = (T) 0;
		}
		
		if(isnan(dat.peso) || dat.peso < 0.0){
			dat.peso = 0.0;
		}
	}
	
	__device__ dato<T> operator()(const dato<T> &a, const dato<T> &b){
		if(a.peso == 0.0){//Criterio de exclusion de datos insignificantes.
			return b;
		}
		else{
			if(b.peso == 0.0){//Criterio de exclusion de datos insignificantes.
				return a;
			}
			else{
				dato<T> resultado;
				resultado.peso = a.peso + b.peso;
				resultado.valor = (a.valor * ((T) a.peso) + b.valor * ((T) b.peso)) / ((T) resultado.peso);
				
				dato_chk(resultado);
				
				return resultado;
			}
		}
	}
};

struct mapeo_transpose{
	int n, dim;
	mapeo_transpose(int _n, int _dim){
		n = _n;
		dim = _dim;
	}
	__device__ int operator()(const int& i){
		return (i % n) * dim + i / n;
	}
};

struct mapeo_claves{
	int n;
	mapeo_claves(int _n){
		n = _n;
	}
	__device__ int operator()(const int &i){
		return i / n;
	}
};

struct indices_triangulares_1{
	int dim;
	int num_triang_dim;
	indices_triangulares_1(int _dim){
		dim = _dim;
		num_triang_dim = ((_dim + 1)*(_dim)) / 2;
	}
	__device__ int operator()(const int &i){
		int rest = i % num_triang_dim;
		int div = i / num_triang_dim;
		return div * dim + ((int) (((sqrt(1.0 + 8.0 * (float)rest)) - 1.0)/2.0));
	}
};

struct indices_triangulares_2{
	int num_triang_dim;
	int dim;
	indices_triangulares_2(int _dim){
		dim = _dim;
		num_triang_dim = ((_dim + 1)*(_dim)) / 2;
	}
	__device__ int operator()(const int &i){
		int div = i / num_triang_dim;
		int rest = i % num_triang_dim;
		int j = ((int) (((sqrt(1.0 + 8.0 * (float)(rest))) - 1.0)/2.0));
		return i + div * (dim - num_triang_dim) - ((j + 1) * j)/2;
	}
};

template <typename T>
struct resta_binaria_dato{//Resta los valores de dos datos, y retorna dicha resta.
	__device__ T operator()(const dato<T> &M2, const dato<T> &M1C){
		return M2.valor - M1C.valor;
	}
};

//Saca el valor correspondiente a un dato.
template <typename T>
struct dato_a_vector{
	__device__ T operator()(const dato<T> &a){
		return a.valor;
	}
};

struct modulo{
	int mod;

	modulo(int _mod){
		mod = _mod;
	}

	__device__ int operator()(const int &i){
		return i % mod;
	}
};

//Input en formato {{dato1x, dato1y, ...}, {dato2x, datos2y, ...}, ...}.
template <typename T>
class valor_medio{//Esta clase calcula la media de los datos ingresados en el input visto arriba. Con las transformaciones adecuadas, se puede usar para calcular cualquier momento.
	private:
	int n;
	int dim;
	int N;
	
	thrust::device_vector<T> datos_originales;
	thrust::device_vector<float> pesos_originales;
	
	thrust::device_vector<T> datos_transpuestos;
	thrust::device_vector<T> pesos_transpuestos;
	thrust::device_vector<dato<T> > medias;
	
	void traspongo_datos(){//Arreglo a los datos para que esten en formato {dato1x, dato2x, ..., dato1y, dato2y, ..., ...}. Necesario para la reduccion por clave adyacente.
		//Creo un arreglo que tiene la forma en la que mapeare la transpuesta.
		thrust::device_vector<int> map(N);//Almacenara la informacion del mapeo de transpuesta.
		mapeo_transpose mapear_transpuesta = mapeo_transpose(n, dim);
		thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), map.begin(), mapear_transpuesta);//Pasa esos indices a los correspondientes al mapeo de transpuesta.
		
		//Finalmente traspongo los datos.
		datos_transpuestos.resize(N);//Almacenara los datos transpuestos.
		pesos_transpuestos.resize(N);
		thrust::gather(thrust::device, map.begin(), map.end(), datos_originales.begin(), datos_transpuestos.begin());//Transpone los datos.
		thrust::gather(thrust::device, map.begin(), map.end(), pesos_originales.begin(), pesos_transpuestos.begin());//Transpone los pesos con el mismo mapeo.
	}
	
	void calculo_media(){//Calculo la media en todas las dimensiones, con una reduccion por clave.
		//Meto los datos, ya traspuestos en un array de estructuras dato, que tambien contendra la significancia del valor que contiene, como un peso.
		thrust::device_vector<dato<T> > datos_organizados(N);//Falta meterlos en la estructura que tambien contendra el peso n del dato, y cualquier otra informacion estadistica asociada eventualmente.
		vector_a_dato<T> transformo_a_struct;
		thrust::transform(thrust::device, datos_transpuestos.begin(), datos_transpuestos.end(), pesos_transpuestos.begin(), datos_organizados.begin(), transformo_a_struct);//Pasa los datos transpuestos al array estructura datos_organizados[]
		
		//Ahora usare una reduccion por clave para promediar los valores que toma cada dimension de la V.A., por separado.
		thrust::device_vector<int> C(dim);//Claves de salida para la reduccion.
		thrust::device_vector<dato<T> > D(dim);//Valores de salida de la reduccion con clave.
		thrust::equal_to<int> binary_pred;//El predicado de comparacion de claves es una operacion de igual.
		promedio_pesado<T> binary_op;//La operacion de reduccion es un promedio pesado por los pesos n que tienen las estructuras de dato estadistico.
		
		//Primero defino a las claves del estilo de {1, 1, 1, ..., 2, 2, 2, ..., ...}, para que se reduzcan solo los n (numero de V.A. en el espacio muestral) adyacentes.
		thrust::device_vector<int> A(N);
		mapeo_claves mapear_claves = mapeo_claves(n);
		thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), A.begin(), mapear_claves);
		
		//Finalmente reduzco.
		thrust::reduce_by_key(thrust::device, A.begin(), A.end(), datos_organizados.begin(), C.begin(), D.begin(), binary_pred, binary_op);//Realizo la reduccion. D es mi nuevo vector de medias.
		
		medias.resize(dim);
		thrust::copy(thrust::device, D.begin(), D.end(), medias.begin());
	}
	
	public:
	valor_medio(int _n, int _dim, thrust::device_vector<T> _datos_originales){
		n = _n;
		dim = _dim;
		N = _n * _dim;
		datos_originales = _datos_originales;
		pesos_originales.resize(n * dim);
		thrust::fill(thrust::device, pesos_originales.begin(), pesos_originales.end(), 1.0);
	}
	
	valor_medio(int _n, int _dim, thrust::device_vector<T> _datos_originales, thrust::device_vector<float> _pesos_originales){
		n = _n;
		dim = _dim;
		N = _n * _dim;
		datos_originales = _datos_originales;
		pesos_originales = _pesos_originales;
	}
	
	void realizo_calculos(){
		traspongo_datos();
		calculo_media();
	}
	
	int get_dim(){//Getter de la dimension.
		return dim;
	}
	
	thrust::device_vector<dato<T> > get_medias(){//Getter de la media calculada.
		return medias;
	}
};

template <typename T>
class calculo_medias_y_varianzas{
	private:
	int n;
	int dim;
	int N;
	int num_triang_dim;
	
	thrust::device_vector<int> map1;
	thrust::device_vector<int> map2;
	
	thrust::device_vector<T> datos_originales;
	thrust::device_vector<T> pesos_originales;
	thrust::device_vector<T> desviaciones_originales;
	thrust::device_vector<T> expandido1;
	thrust::device_vector<T> expandido2;
	thrust::device_vector<T> expandido1_pesos;
	thrust::device_vector<T> expandido2_pesos;
	
	thrust::device_vector<dato<T> > medias_calculadas;		
	thrust::device_vector<dato<T> > m2_calculadas;
	
	//Estos son los resultados finales.
	thrust::device_vector<T> vector_medias;
	thrust::device_vector<T> matriz_covarianzas;//IMPORTANTE: En formato triangular superior: {1,1; 2,1; 2,2; 3,1; 3,2; 3,3; ...}, ya que es simetrica la matriz.
	T norm_covarianza = (T) 0;

	private:
	void calculo_mapeos_triangulares(){//Mapeos a usar en el proximo metodo.
		map1.resize(N);
		map2.resize(N);
		
		thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), map1.begin(), indices_triangulares_1(dim));
		thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), map2.begin(), indices_triangulares_2(dim));
	}
	
	void expando_productos(){//Expande, usando los mapeos previos, a los datos en la forma que multiplicar a un componente de uno por uno del otro, componente a componente, se obtengan todos los productos posibles entre dos elementos cualquiera de una sola V.A.
		thrust::device_vector<int> map_media_expand(n * dim);
		thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n * dim), map_media_expand.begin(), modulo(dim));

		thrust::device_vector<T>  media_expandida(n * dim);
		thrust::gather(thrust::device, map_media_expand.begin(), map_media_expand.end(), vector_medias.begin(), media_expandida.begin());

		desviaciones_originales.resize(n * dim);
		thrust::transform(thrust::device, datos_originales.begin(), datos_originales.end(), media_expandida.begin(), desviaciones_originales.begin(), thrust::minus<T>());

		expandido1.resize(N);
		expandido2.resize(N);
		
		thrust::gather(thrust::device, map1.begin(), map1.end(), desviaciones_originales.begin(), expandido1.begin());
		thrust::gather(thrust::device, map2.begin(), map2.end(), desviaciones_originales.begin(), expandido2.begin());
		
		expandido1_pesos.resize(N);
		expandido2_pesos.resize(N);
		
		thrust::gather(thrust::device, map1.begin(), map1.end(), pesos_originales.begin(), expandido1_pesos.begin());
		thrust::gather(thrust::device, map2.begin(), map2.end(), pesos_originales.begin(), expandido2_pesos.begin());
	}
	
	void calculo_desviaciones(){//Calcula para cada dato, la distancia a la media, y multiplica a cada par dentro de una V.A.
		//Realizo el calculo de todas las combinaciones de productos entre cada par de componentes de cada V.A.
		thrust::device_vector<T> vector_productos(N);
		thrust::device_vector<T> vector_pesos_productos(N);
		thrust::transform(thrust::device, expandido1.begin(), expandido1.end(), expandido2.begin(), vector_productos.begin(), thrust::multiplies<T>());
		thrust::transform(thrust::device, expandido1_pesos.begin(), expandido1_pesos.end(), expandido2_pesos.begin(), vector_pesos_productos.begin(), thrust::multiplies<T>());
		
		//Uso mi clase de arriba, de calculo de valores medios para calcular la media del valor de cada producto (o sea, los segundos momentos M2).
		valor_medio<T> calc_M2 = valor_medio<T>(n, num_triang_dim, vector_productos, vector_pesos_productos);//Como dimension de este nuevo vector de V.A.s, uso el numero triangular correspondiente.
		calc_M2.realizo_calculos();
		m2_calculadas = calc_M2.get_medias();
	}
	
	void calculo_medias(){//Calcula las medias de los datos mismos.
		valor_medio<T> m_est = valor_medio<T>(n, dim, datos_originales, pesos_originales);
		m_est.realizo_calculos();
		medias_calculadas = m_est.get_medias();
		
		//Lo paso a mi vector final de medias.
		vector_medias.resize(dim);
		thrust::transform(thrust::device, medias_calculadas.begin(), medias_calculadas.end(), vector_medias.begin(), dato_a_vector<T>());
	}
	
	void calculo_varianzas(){//Calcula las varianzas restando los segundos momentos M2 de las M1 cuadradas.
		matriz_covarianzas.resize(num_triang_dim);
		thrust::transform(thrust::device, m2_calculadas.begin(), m2_calculadas.end(), matriz_covarianzas.begin(), dato_a_vector<T>());
		calculo_norm_covarianza();//Calculo la "normalizacion" que le aplico (para que los elementos significativos sean del orden de 1.0, por razones computacionales).
	}
	
	public:
	calculo_medias_y_varianzas(int _n, int _dim, thrust::device_vector<T> _datos_originales, thrust::device_vector<T> _pesos_originales){//Constructor con la posibilidad de distintos pesos para distintos datos.
		n = _n;
		dim = _dim;
		datos_originales = _datos_originales;
		pesos_originales = _pesos_originales;
		
		num_triang_dim = ((_dim + 1)*(_dim)) / 2;
		N = num_triang_dim * n;
	}
	
	calculo_medias_y_varianzas(int _n, int _dim, thrust::device_vector<T> _datos_originales){//Constructor para datos con pesos iguales.
		n = _n;
		dim = _dim;
		datos_originales = _datos_originales;
		pesos_originales.resize(_datos_originales.size());
		thrust::fill(pesos_originales.begin(), pesos_originales.end(), 1.0);
		
		num_triang_dim = ((_dim + 1)*(_dim)) / 2;
		N = num_triang_dim * n;
	}

	calculo_medias_y_varianzas(){
		n = 1;
		dim = 1;
		thrust::fill(datos_originales.begin(), datos_originales.end(), 1.0);
		thrust::fill(pesos_originales.begin(), pesos_originales.end(), 1.0);

		num_triang_dim = 1;
		N = 1;
	}
	
	void realizo_calculos(){//Contenedor de todos los calculos en su orden adecuado.
		//Este calcula la media.
		calculo_medias();

		//Estos primeros dos manejan la multiplicacion de los pares de variables aleatorias, para asi poder calcular el valor medio de su producto (ej. <x1x2>).
		calculo_mapeos_triangulares();
		expando_productos();

		//Ahora calculo las restas desde los datos a las medias, y las multiplico juntas.
		calculo_desviaciones();

		//Este finalmente resta el valor cuadrado de la media de los segundos momentos para obtener las covarianzas.
		calculo_varianzas();
	}
	
	T * get_media(){//Devuelve el vector de las medias.
		T * h_medias_ptr = (T *) malloc(dim * sizeof(T));
		thrust::copy(vector_medias.begin(), vector_medias.end(), h_medias_ptr);
		return h_medias_ptr;
	}
	
	T * get_covarianzas(){//Devuelve el vector de las covarianzas.
		T * h_covar_ptr = (T *) malloc(N * sizeof(T));
		thrust::copy(matriz_covarianzas.begin(), matriz_covarianzas.end(), h_covar_ptr);
		return h_covar_ptr;
	}

	T get_covarianza(int i, int j){//Devuelve el elemento del vector de covarianzas correspondiente al elemento (i, j) de la matriz simetrica de covarianzas.
		int index;
		if(j < i){
			index = ((i + 1) * i) / 2 + j;
		}
		else{
			index = ((j + 1) * j) / 2 + i;
		}
		return matriz_covarianzas.data()[index];
	}

	T get_covarianza_fraccional(int i, int j){//Devuelve el elemento de la matriz de covarianzas, dividido por el valor medio de las variables aleatorias correspondientes.
		int index;
		if(j < i){
			index = ((i + 1) * i) / 2 + j;
		}
		else{
			index = ((j + 1) * j) / 2 + i;
		}
		return matriz_covarianzas.data()[index] / abs(vector_medias.data()[i] * vector_medias.data()[j]);
	}
	
	int dim_media(){//La dimension del vector que contiene a las medias.
		return dim;
	}
	
	int dim_covarianzas(){//La dimension del vector que contiene a las covarianzas.
		return num_triang_dim;
	}
	
	void print_media(){//Imprime el valor medio obtenido para las variables aleatorias promediadas.
		std::cout << "Media: (";
		for(int i=0; i<dim-1; i++){
			std::cout << vector_medias.data()[i] << ", ";
		}
		std::cout << vector_medias.data()[dim-1] << ")" << std::endl << std::endl;
	}

	void calculo_norm_covarianza(){
		norm_covarianza = (T) 0;
		for(int j=0; j<dim; j++){
			for(int i=0; i<=j; i++){
				norm_covarianza += abs(get_covarianza_fraccional(i, j));
			}
		}
	}
	
	void print_covarianzas(){//Imprime las covarianzas de las variables aleatorias promediadas.
		std::cout << "Matriz Covarianzas:" << std::endl;
		for(int j=0; j<dim; j++){
			std::cout << "(";
			for(int i=0; i<dim-1; i++){
				std::cout << get_covarianza(i, j) << " ";
			}
			std::cout << get_covarianza(dim-1, j) << ")" << std::endl;
		}
		std::cout << std::endl;
	}
	
	void print_covarianzas_fraccional(){//Imprime la variacion que representan las covarianzas respecto a los valores de sus variables aleatorias.
		std::cout << "Matriz Covarianzas Fraccionales:" << std::endl;
		for(int j=0; j<dim; j++){
			std::cout << "(";
			for(int i=0; i<dim-1; i++){
				std::cout << get_covarianza_fraccional(i, j) << " ";
			}
			std::cout << get_covarianza_fraccional(dim-1, j) << ")" << std::endl;
		}
		std::cout << std::endl;
	}
};
