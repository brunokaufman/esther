#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

class TimeSeries{//Clase responsable de importar y almacenar los datos experimentales.
	private://Hago privados los campos por buena practica en POO.
	std::vector<double> vector_tiempo;//Uso vectores por simplicidad de implementacion.
	std::vector<double> vector_casos;
	std::vector<std::string> vector_datos;
	
	std::string nombre_logfile;//Nombre del log asociado a la clase.
	std::ifstream inFile;//Abro el archivo de donde importare mis datos.
	std::ofstream logFile;
	
	int n_variables;
	char delim;
	
	public:
	TimeSeries(std::string nombre_datafile, char _delim, int _n_variables, std::string nombre_log){//Constructor que especifica el delimitador y al nombre del archivo log.
		n_variables = _n_variables;
		delim = _delim;
		
		nombre_logfile = nombre_log;
		
		inFile.open(nombre_datafile.c_str(), std::ifstream::in);
		logFile.open(nombre_logfile.c_str(), std::ofstream::out);
		
		if (!inFile) {//Chequeo que haya sido abierto correctamente.
			logFile << "Constructor TimeSeries: No se pudo abrir el archivo." << std::endl;
			logFile.flush();
			exit(1);//Si no, termina en error.
		}
		
		vector_datos.resize(n_variables);
	}
	
	TimeSeries(std::string nombre_datafile, char delim, int _n_variables) : TimeSeries(nombre_datafile, delim, _n_variables, "./libs/cpp_import/log_import.txt"){}
	
	TimeSeries(std::string nombre_datafile, std::string nombre_log, int _n_variables) : TimeSeries(nombre_datafile, ',', _n_variables, nombre_log){}
	
	TimeSeries(std::string nombre_datafile, int _n_variables) : TimeSeries(nombre_datafile, ',', _n_variables, "./libs/cpp_import/log_import.txt"){}//Constructor default usa una coma como delimitador. Delega al constructor completo (C++11 requerido!).
	
	std::vector<std::string> leo_linea(){
		if(inFile.good()){
			std::vector<std::string> result;
			result.resize(n_variables);
			for(int j=0; j<n_variables-1; j++){
				getline(inFile, result[j], delim);
			}
			getline(inFile, result[n_variables - 1]);
			logFile << "Importador de lineas TimeSeries: Linea importada exitosamente." << std::endl;//Si no hay interrupciones, retorna un mensaje para avisar.
			logFile.flush();
			return result;
		}
		else{
			std::vector<std::string> result;
			result.resize(0);
			return result;
		}
	}
	
	std::string get_log_file(){//Getter para el nombre del archivo log (por si otras clases quieren copiarlo, editarlo, etc.).
		return nombre_logfile;
	}
	
	std::vector<double> get_tiempos(){//Getter para el vector de los tiempos.
		return vector_tiempo;
	}
	      
	std::vector<double> get_casos(){//Getter para el vector de la cantidad de casos infectados.
		return vector_casos;
	}
	
	void printDatos(){//Imprime los datos a la consola (usado para debuggear).
		for(int i=0; i<vector_tiempo.size(); i++){
			std::cout << vector_tiempo.data()[i] << " " << vector_casos.data()[i] << endl;
		}
	}
	
	~TimeSeries(){
		logFile << "Destructor TimeSeries: Vectores liberados exitosamente." << std::endl;
		logFile.flush();
		logFile.close();
	}
};
