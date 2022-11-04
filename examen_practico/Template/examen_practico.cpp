////////////////////////////////Cabeceras/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
/////////////////////////////////////////////////////////////////////////////

///////////////////////////////Espacio de nombres////////////////////////////
using namespace cv;
using namespace std;
/////////////////////////////////////////////////////////////////////////////

///////////////////////////////Espacio de constantes/////////////////////////
const float e = 2.718281828459;
const double M_PI = 3.14159265358979323846;
/////////////////////////////////////////////////////////////////////////////


///////////////////////////////Definición de funciones///////////////////////
vector<double>crearKernelDinamico(int sigmaRecibido, int tamanioKernel);
void mostrarKernel(vector<double>vectorRecibido, int tamanioKernel);
vector<double>promedioDelKernel(vector<double>vectorRecibido, int tamanioKernel);
Mat escala_grises_imagen(Mat original, int tamanioKernel);
Mat filtro_gaussiano(Mat imagenEnGrises, vector<double>vectorRecibido, int tamanio);
void calcularHistograma(Mat imagenRecibida, int histogramaRecibido[]);
Mat sobel_aplicado_enX(Mat imagen_recibida);
Mat sobel_aplicado_enY(Mat imagen_recibida);
Mat filtro_sobel(Mat imagen_recibida);
Mat supresionNoMaxima(Mat imagenRecibida, Mat anguloRecibido);
/////////////////////////////////////////////////////////////////////////////


/////////////////////////Inicio de la funcion principal///////////////////
int main()
{

	/********Declaracion de variables generales*********/
	char NombreImagen[] = "./lena.png";
	Mat imagen; // Matriz que contiene nuestra imagen sin importar el formato
	/************************/

	/*********Lectura de la imagen*********/
	imagen = imread(NombreImagen);

	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	/************************/

	int sigma = 0;
	int tamanioDelKernel = 0;
	int tamanioFilas_original = imagen.rows;
	int tamanioColumnas_original = imagen.cols;
	Mat imagen_gris_NTSC(imagen.rows, imagen.cols, CV_8UC1); // Permite cambiar la imagen por un uchar de un solo canal
	Mat imagen_GaussBlur(imagen.rows, imagen.cols, CV_8UC1);


	cout << "Por favor intoduce un valor para sigma: ";
	cin >> sigma;
	bool bandera = true;
	cout << "Por favor introduce el tamanio del kernel: ";
	cin >> tamanioDelKernel;
	do
	{

		if (tamanioDelKernel % 2 == 0 || tamanioDelKernel < 1)
		{
			cout << "Error tamanio par, introduce un numero impar: ";
			cin >> tamanioDelKernel;
		}
		else
		{
			bandera = false;
		}
	} while (bandera != false);
	cout << "\n";
	vector<double> kernelCreado;
	kernelCreado = crearKernelDinamico(sigma, tamanioDelKernel);
	cout << "Kernel creado" << endl;
	mostrarKernel(kernelCreado, tamanioDelKernel);
	vector<double> kernelPromediado;
	kernelPromediado = promedioDelKernel(kernelCreado, tamanioDelKernel);
	cout << "\n";
	cout << "Kernel normalizado" << endl;
	mostrarKernel(kernelPromediado, tamanioDelKernel);

	cout << "Creando imagen a grises" << endl;
	imagen_gris_NTSC = escala_grises_imagen(imagen, tamanioDelKernel);

	cout << "Aplicando flitro gaussiano a la imagen" << endl;
	imagen_GaussBlur = filtro_gaussiano(imagen_gris_NTSC, kernelPromediado, tamanioDelKernel);
	//EQUALIZANDO LA IMAGEN
	int histograma[256];
	int histogramaAuxiliar[256];
	int ecualizacion[256];
	int ecualizar = 0;
	int sumatorias = 0;
	//INICALIZAMOS EL HISTOGRAMA CON 0
	for (int i = 0; i < 256; i++)//HASTA 256 PORQUE SON 256 TONALIDADES DE GRIS
	{
		histograma[i] = 0;
	}
	// A CADA VALOR LO METEMOS EN UN indice, Y EN CADA INDICE VAMOS A IRLE SUMANDO PARA SABER CUANTOS VALORES TENEMOS
	int indice = 0;
	for (int i = 0; i < imagen_gris_NTSC.rows; i++)
	{
		for (int j = 0; j < imagen_gris_NTSC.cols; j++)
		{
			indice = imagen_gris_NTSC.at<Vec3b>(Point(j, i)).val[0];
			histograma[indice] += 1; // INDEXAMOS
		}
	}
	ecualizar = 255 / (imagen_gris_NTSC.rows * imagen_gris_NTSC.cols);
	for (int i = 0; i < imagen_gris_NTSC.rows; i++)
	{
		for (int j = 0; j < imagen_gris_NTSC.cols; j++)
		{
			for (int k = 0; k < 256; k++)
			{
				sumatorias += histograma[k];

			}
			histogramaAuxiliar[i] = ecualizar * sumatorias;
		}
	}
	Mat imagenEqualizada(imagen_gris_NTSC.rows, imagen_gris_NTSC.cols, CV_8UC1);
	for (int i = 0; i < imagen_gris_NTSC.rows; i++)
	{
		for (int j = 0; j < imagen_gris_NTSC.cols; j++)
		{
			imagenEqualizada.at<uchar>(Point(j, i)) = uchar(imagen_gris_NTSC.at<uchar>(Point(j, i)));
		}
	}

	
	
	Mat imagen_Sobel_Gx(imagen.rows, imagen.cols, CV_8UC1);
	Mat imagen_Sobel_Gy(imagen.rows, imagen.cols, CV_8UC1);
	Mat imagen_Sobel(imagen.rows, imagen.cols, CV_8UC1);

	cout << "Aplicando ecualización" << endl;
	
	cout << "Aplicando sobel en Gx" << endl;
	imagen_Sobel_Gx = sobel_aplicado_enX(imagenEqualizada);

	cout << "Aplicando Sobel en Gy" << endl;
	imagen_Sobel_Gy = sobel_aplicado_enY(imagenEqualizada);

	cout << "Aplicando filtro sobel" << endl;
	imagen_Sobel = filtro_sobel(imagenEqualizada);

	//SACAMOS EL TAN^-1
	Mat anguloSobel(imagenEqualizada.rows, imagenEqualizada.cols, CV_8UC1);
	double angulo = 0.0;
	for (int i = 0; i < imagenEqualizada.rows; i++)
	{
		for (int j = 0; j < imagenEqualizada.cols; j++)
		{
			//LO DA EN GRADOS, LO CONVERTIMOS A RAD
			angulo = atan2(double(imagen_Sobel_Gy.at<uchar>(Point(j, i))),double(imagen_Sobel_Gx.at<uchar>(Point(j, i)))) * (180/ M_PI);
			anguloSobel.at<uchar>(Point(j, i)) = angulo;
		}
		
	}

	//Comienza el CAnny
	//REALIZAMOS SUPRESIÓN NO MÁXIMA
	Mat supresionNM(imagen_Sobel.rows, imagen_Sobel.cols, CV_8UC1);
	supresionNM = supresionNoMaxima(imagen_Sobel, anguloSobel);

	namedWindow("Imagen Original", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Original", imagen);

	namedWindow("Imagen Grises", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Grises", imagen_gris_NTSC);

	namedWindow("Imagen Blur Gaussiano", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Blur Gaussiano", imagen_GaussBlur);

	namedWindow("Imagen Equalizada", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Equalizada", imagenEqualizada);

	namedWindow("Imagen Sobel Gx", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Sobel Gx", imagen_Sobel_Gx);

	namedWindow("Imagen Sobel Gy", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Sobel Gy", imagen_Sobel_Gy);

	namedWindow("Imagen Filtro Sobel", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Filtro Sobel", imagen_Sobel);

	namedWindow("Imagen Angulo sobel", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen Filtro angulo sobel", anguloSobel);

	namedWindow("Imagen NM", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Imagen NM", supresionNM);

	cout << "Filas y columnas de la original [ " << imagen.rows << " , " << imagen.cols << " ]" << endl;
	cout << "Filas y columnas de la Gris [ " << imagen_gris_NTSC.rows << " , " << imagen_gris_NTSC.cols << " ]" << endl;
	cout << "Filas y columnas de la Gauss [ " << imagen_GaussBlur.rows << " , " << imagen_GaussBlur.cols << " ]" << endl;
	cout << "Filas y columnas de la Gauss [ " << imagen_Sobel.rows << " , " << imagen_Sobel.cols << " ]" << endl;
	waitKey(0);
	destroyAllWindows();
	system("pause");
	return 0;
	
	/*
	//SE CREA UN HISTOGRAMA
	int histograma[256];
	calcularHistograma(imagen_GaussBlur, histograma);
	int tamanioImagen = imagen.rows * imagen.cols;
	//CALCULAMOS LA INTESIDAD PARA CADA VALOR
	int probatilidadIntensidad[256];

	for (int i = 0; i < 256; i++)
	{
		probatilidadIntensidad[i] = (double)histograma[i] / tamanioImagen;
	}

	float sumr = 0;
	float sumrx = 0;
	int negro = 0;
	int valorGris = 255;
	for (int i = negro; i <= valorGris; i++)
	{
		sumr += (probatilidadIntensidad[i]);
		sumrx = negro + (valorGris - negro) * sumr;
		int valr = (int)(sumrx);
		if (valr > 255)
		{
			probatilidadIntensidad[i] = 255;
		}
		else
		{
			probatilidadIntensidad[i] = valr;
		}
	}
	Mat imagenEqualizada(imagen.rows, imagen.cols, CV_8UC1);
	//CALCULCAMOS EL NÚMERO LA INTENSIDAD DE CADA PIXEL
	for (int i = 0; i < imagen.rows; i++)
	{
		for (int j = 0; j < imagen.cols; j++)
		{

			imagenEqualizada.at<uchar>(Point(j, i)) =(probatilidadIntensidad[i]);

		}
	}
	*/

	
}


//////////////////////////// Declaración de funciones///////////////////////
/*
double** crearKernelDinamico(int sigmaRecibido, int tamanioKernelRecibido)
{

}*/
vector<double>crearKernelDinamico(int sigmaRecibido, int tamanioKernel)
{
	int centroKernel = (tamanioKernel - 1) / 2;
	float sigmaKernel = 0.0;
	vector<double> kernel(tamanioKernel * tamanioKernel, 0.0); // CREACIÓN DE UN VECTOR kernel INICIALIZADO CON 0.01

	if (sigmaRecibido < 0)
	{
		sigmaKernel = 0.3 * ((tamanioKernel - 1) * 0.5 - 1) + 0.8; //SEGÚN OPENCV SI NUESTRO sigma ES MENOR A 0 USAMOS ESTA FORMULA
	}
	else
	{
		sigmaKernel = sigmaRecibido;
	}
	double pi = M_PI;
	double media = tamanioKernel / 2;
	double normalizar = 0;
	double valor = 0;
	for (int i = 0; i < tamanioKernel; i++)
	{
		for (int j = 0; j < tamanioKernel; j++)
		{
			// SE MUEVE ENTRE LAS POSICIONES 
			// PRIMERA ITERACIÓN => i = 0, tamanio = 5, j = 0, valor = (0*5) + 0 = 0
			// SEGUNDA ITERACIÓN => i = 0, tamanio = 5, j = 1, valor = (0*5) + 1 = 1
			// exp() ES EL MÉTODO QUE RETORNA EL RESULTADO DE EXPONENCIAL DE e
			kernel[(i * tamanioKernel) + j] = exp(-0.5 * (pow(i - centroKernel, 2.0)) + pow(j - centroKernel, 2.0)) / (2 * pi * sigmaKernel * sigmaKernel);
			// cout << "Valor i: " << i << " Valor j: " << j << "\t
			// cout << "Posicion :" << (i * tamanioKernel) + j << "\t";
			valor = kernel[(i * tamanioKernel) + j];
			//cout << valor << "\t";
		}
		//cout << "\n";
	}

	return kernel;

}

//IMPRIME EL VECTOR
void mostrarKernel(vector<double>vectorRecibido, int tamanioKernel)
{
	for (int i = 0; i < tamanioKernel; i++)
	{
		for (int j = 0; j < tamanioKernel; j++)
		{
			cout << vectorRecibido[(i * tamanioKernel) + j] << " ";
		}
		cout << "\n";
	}
}

//NORMALIZAMOS EL VECTOR
vector<double>promedioDelKernel(vector<double>vectorRecibido, int tamanioKernel)
{
	double normalizar = 0;
	for (int i = 0; i < tamanioKernel; i++)
	{
		for (int j = 0; j < tamanioKernel; j++)
		{
			normalizar += vectorRecibido[(i * tamanioKernel) + j];
		}

	}
	for (int i = 0; i < vectorRecibido.size(); i++)
	{
		//vectorRecibido[i] = 0;
		vectorRecibido[i] /= normalizar;
		//cout <<vectorRecibido[i] <<"\t";
		//cout << "\n";

	}
	return vectorRecibido;
}

//CONVERTIMOS LA IMAGEN A GRISES
Mat escala_grises_imagen(Mat imagenRecibida, int tamanioKernel)
{

	Mat imagenNTSC(imagenRecibida.rows, imagenRecibida.cols, CV_8UC1);

	int i, j;
	double azul, verde, rojo, gris_n;
	int centro = (tamanioKernel - 1) / 2;

	for (i = 0; i < imagenRecibida.rows; i++)
	{
		for (j = 0; j < imagenRecibida.cols; j++)
		{

			azul = imagenRecibida.at<Vec3b>(Point(j, i)).val[0];
			verde = imagenRecibida.at<Vec3b>(Point(j, i)).val[1];
			rojo = imagenRecibida.at<Vec3b>(Point(j, i)).val[2];

			gris_n = (rojo * 0.299 + verde * 0.587 + azul * 0.114);
			// cout << "Valor de gris en la posicion [" << i << " , " << j << "]" << gris_n<< endl;
			// CHECA LA POSICIÓN , CHECANDO DE IZQUIERDA A DERECHA, VE SI SOBRESALE DE LA IMAGEN O NO
			if (i - tamanioKernel < 0 || j - tamanioKernel < 0 || i + tamanioKernel > imagenRecibida.rows || j + tamanioKernel > imagenRecibida.cols)
			{
				imagenNTSC.at<uchar>(Point(j, i)) = uchar(0);
			}
			else
			{
				imagenNTSC.at<uchar>(Point(j, i)) = uchar(gris_n);
				// cout << imagenNTSC.at<uchar>(Point(j, i)) << endl;
			}


		}
	}

	return imagenNTSC;

}

//APLICAMOS EL FILTRO
Mat filtro_gaussiano(Mat imagenEnGrises, vector<double>vectorRecibido, int tamanio)
{
	Mat imagenConFiltro(imagenEnGrises.rows, imagenEnGrises.cols, CV_8UC1);
	int centro = (tamanio - 1) / 2;
	int i = 0;
	int j = 0;
	int a = 0;
	int b = 0;
	int filas = 0;
	int columnas = 0;
	// NOS MOVEMOS/ITERAMOS POR LA IMAGEN
	for (i = centro, filas = 0; i < imagenConFiltro.rows - centro; i++, filas++)
	{
		for (j = centro, columnas = 0; j < imagenConFiltro.cols - centro; j++, columnas++)
		{

			double suma = 0;
			//NOS MOVEMOS/ITERAMOS EN EL KERNEL
			for (a = 0; a < tamanio; a++)
			{
				for (b = 0; b < tamanio; b++)
				{
					// GUARDA LOS VALORES OBTENIDOS DE LOS PUNTOS DE LA IMAGEN POR EL kernel
					suma += imagenEnGrises.at<uchar>(j - (centro + a), i - (centro + b)) * vectorRecibido[(a * tamanio) + b];
					//cout << suma << endl; 
				}
			}
			// ASIGNA LOS VALORES OBTENIDOS DE LA SUMATORIA EN LA imagen COMO EN LA CONVERSIÓN EN GRIS
			imagenConFiltro.at<uchar>(columnas, filas) = uchar(suma);

		}
	}
	return imagenConFiltro;
}
void calcularHistograma(Mat imagenRecibida, int histogramaRecibido[])
{
	int histogramaAuxiliar[256];
	//INICIALIZAMOS EL HISTOGRAMA CON 0
	for (int i = 0; i < 256; i++)
	{
		histogramaRecibido[i] = 0;
		histogramaAuxiliar[i] = 0;
	}

	//CALCULCAMOS EL NÚMERO LA INTENSIDAD DE CADA PIXEL
	for (int i = 0; i < imagenRecibida.rows; i++)
	{
		for (int j = 0; j < imagenRecibida.cols; j++) 
		{
			
			//cout << histogramaRecibido[(int)imagenRecibida.at<uchar>(j, i)] << endl;
			histogramaRecibido[i] = histogramaAuxiliar[(int)imagenRecibida.at<uchar>(j, i)]++;
			//cout << histogramaRecibido[i]<<endl;
		}
	}


}
// OBSERVANDO LA APLICACIÓN DE SOBEL ES PARECIDA A COMO SE APLICÓ GAUSS
Mat sobel_aplicado_enY(Mat imagen_recibida)
{
	vector<int>gaussiano_enY({1,2,1,0,0,0,-1,-2,-3}); //CREAMOS NUESTRO KERNEL PARA x
	int kernel = 3;
	Mat imagenGY(imagen_recibida.rows, imagen_recibida.cols, CV_8UC1);
	int i = 0;
	int j = 0;
	// NOS MOVEMOS/ITERAMOS POR LA IMAGEN
	for (i = 0; i < imagenGY.rows; i++)
	{
		for (j = 0; j < imagenGY.cols; j++)
		{
			double suma = 0.0;
			//double valor_gris = 0.0;
			//valor_gris = imagen_recibida.at<Vec3b>(Point(j, i)).val[0];
			//cout << valor_gris << endl;
			//NOS MOVEMOS/ITERAMOS EN EL KERNEL
			for (int a = 0; a < kernel; a++)
			{
				for (int b = 0; b < kernel; b++)
				{
					suma += imagen_recibida.at<uchar>(Point(j+a, i+b)) * gaussiano_enY[(a * kernel) + b];
					//cout << "Valor en la posicion " << (a * kernel) + b << " :" << gaussiano_enX[(a * kernel) + b];
					//cout << suma ;
					//cout << "Posicion Kernel"<<(a*kernel)+b;
					
				}
				//cout << endl;
			}

			imagenGY.at<uchar>(Point(j, i)) = uchar(suma);
			//
			
	
			
		}
	}
	return imagenGY;
}

Mat sobel_aplicado_enX(Mat imagen_recibida)
{
	vector<int>gaussiano_enX({ 1, 0, -1, 2, 0, -2, 1, 0, -1 }); //CREAMOS NUESTRO KERNEL PARA x
	int kernel = 3;
	Mat imagenGx(imagen_recibida.rows, imagen_recibida.cols, CV_8UC1);
	int i = 0;
	int j = 0;
	// NOS MOVEMOS/ITERAMOS POR LA IMAGEN
	for (i = 0; i < imagenGx.rows; i++)
	{
		for (j = 0; j < imagenGx.cols; j++)
		{
			double suma = 0.0;
			//double valor_gris = 0.0;
			//valor_gris = imagen_recibida.at<Vec3b>(Point(j, i)).val[0];
			//cout << valor_gris << endl;
			//NOS MOVEMOS/ITERAMOS EN EL KERNEL
			for (int a = 0; a < kernel; a++)
			{
				for (int b = 0; b < kernel; b++)
				{
					suma += imagen_recibida.at<uchar>(Point(j + a, i + b)) * gaussiano_enX[(a * kernel) + b];
					//cout << "Valor en la posicion " << (a * kernel) + b << " :" << gaussiano_enX[(a * kernel) + b];
					//cout << suma ;
					//cout << "Posicion Kernel"<<(a*kernel)+b;

				}
				//cout << endl;
			}

			imagenGx.at<uchar>(Point(j, i)) = uchar(suma);
			//



		}
	}
	return imagenGx;
}

Mat filtro_sobel(Mat imagen_recibida)
{
	vector<int>gaussiano_enX({ 1, 0, -1, 2, 0, -2, 1, 0, -1 }); //CREAMOS NUESTRO KERNEL PARA x
	int kernel = 3;
	Mat imagenFiltro(imagen_recibida.rows, imagen_recibida.cols, CV_8UC1);
	vector<int>gaussiano_enY({ 1,2,1,0,0,0,-1,-2,-3 }); //CREAMOS NUESTRO KERNEL PARA y
	int i = 0;
	int j = 0;
	// NOS MOVEMOS/ITERAMOS POR LA IMAGEN
	for (i = 0; i < imagen_recibida.rows; i++)
	{
		for (j = 0; j < imagen_recibida.cols; j++)
		{
			double sumaY = 0.0;
			double sumaX = 0.0;
			double magnitud = 0.0;
			//double valor_gris = 0.0;
			//valor_gris = imagen_recibida.at<Vec3b>(Point(j, i)).val[0];
			//cout << valor_gris << endl;
			// NOS MOVEMOS/ITERAMOS POR LA IMAGEN
			for (int a = 0; a < kernel; a++)
			{
				for (int b = 0; b < kernel; b++)
				{
					sumaY += imagen_recibida.at<uchar>(Point(j + a, i + b)) * gaussiano_enY[(a * kernel) + b];
					sumaX += imagen_recibida.at<uchar>(Point(j + a, i + b)) * gaussiano_enX[(a * kernel) + b];
					//cout << "Valor en la posicion " << (a * kernel) + b << " :" << gaussiano_enX[(a * kernel) + b];
					//cout << suma ;
					//cout << "Posicion Kernel"<<(a*kernel)+b;
					magnitud = sqrt(pow(sumaX, 2) + pow(sumaY, 2));
				}
				//cout << endl;
			}

			(imagenFiltro.at<uchar>(Point(j, i)) = uchar((magnitud)));
			//



		}
	}
	return imagenFiltro;

		
}

Mat supresionNoMaxima(Mat imagenRecibida, Mat anguloRecibido)
{
	
		Mat bordeDetectado(imagenRecibida.rows, imagenRecibida.cols, CV_8UC1);
		int vecinoX = 0;
		int vecinoY = 0;
		double angulo = 0;//

		//CHECA SI LA INTENSIDAD DEL PIXEL EN LA MIMA DIRECCIÓN TIENE UN VALOR MAYOR AL PIXEL QUE SE ESTÁ PROCESANDO
		for (int i = 0; i < imagenRecibida.rows; i++)
		{
			for (int j = 0; j < imagenRecibida.cols; j++)
			{
				//cERO GRADOS
				if ((0 <= angulo < 22.5) || (157.5 <= angulo <= 180))
				{
					vecinoX = imagenRecibida.at<uchar>(Point(j , i ));
					vecinoY = imagenRecibida.at<uchar>(Point(j , i ));
				}
				//Noventa grados
				else if ((67.5 <= angulo < 112.5)) 
				{
					vecinoX = imagenRecibida.at<uchar>(Point(j , i ));
					vecinoY = imagenRecibida.at<uchar>(Point(j , i ));
				}
				//cuarenta y cinco grados
				else if ((22.5 <= angulo < 67.5)) 
				{
					vecinoX = imagenRecibida.at<uchar>(Point(j , i ));
					vecinoY = imagenRecibida.at<uchar>(Point(j , i ));
				}
				//135 grados
				else if ((112.5 <= angulo < 157.5)) 
				{
					vecinoX = imagenRecibida.at<uchar>(Point(j , i ));
					vecinoY = imagenRecibida.at<uchar>(Point(j , i ));
				}

				if (imagenRecibida.at<uchar>(Point(j , i )) >= vecinoX && imagenRecibida.at<uchar>(Point(j , i )) >= vecinoY)
				{
					bordeDetectado.at<uchar>(Point(j, i)) = imagenRecibida.at<uchar>(Point(j , i ));
				}

				angulo = anguloRecibido.at<uchar>(Point(j, i));
				bordeDetectado.at<uchar>(Point(j, i)) = uchar(0);
			}
		}
		return bordeDetectado;
}


	





/////////////////////////////////////////////////////////////////////////////
