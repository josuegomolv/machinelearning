# machinelearning
Proyecto de ML para el taller de TIC UTXJ

PASO 1. Instalar python 
  Revisar que tenga la versión de Python 3.8.10

PASO 2. Instalar Sickit Learn (SKLearn) a partir del siguiente enlace: 
  https://scikit-learn.org/stable/install.html
  No es necesario descargar nada para instalarlo, en la terminal de vsstudio o cmd se deben ejecutar las lineas que vienen en la página:
    python -m venv sklearn-env
    sklearn-env\Scripts\activate  # activate
    pip install -U scikit-learn

  Las siguientes líneas son para verificar que la instalación ses correcta:
    python -m pip show scikit-learn  # show scikit-learn version and location
    python -m pip freeze             # show all installed packages in the environment
    python -c "import sklearn; sklearn.show_versions()"

Paso 3. Descargar PANDAS
    En la terminal se debe ejecutar el siguiente comando:
    pip install pandas

PASO 4. Descargar el repositorio del proyecto
    Colocar el proyecto en la raíz del ordenador (C:).
    Después de descargar el proyecto, se debe abrir la carpeta de machinelearning en VsStudio.

PASO 5. Crear el entorno virtual (.venv)
  Abrir la terminal ejecutar el comando:
    python -m venv .venv 
  La ejecución del comando debió crear la carpeta (.venv)

Paso 6. Ejecución de pruebas
  Usando la terminal de vsstudio, acceder a la carpeta de algoritmos:  cd .\algoritmos\

  Se deberá testear los 5 archivos, para probar que funcionen correctamente y descartar algun error.
  Por cada ejecución del archivo, siempre se obtendrá como salida una ventana con un grafico (colores, puntos y lineas) y ademas en consola se imprimirán algunos recuadros como se muestra a continuación:
                    
                    Accuracy: 0.5217391304347826
                   Precision: 0.5555555555555556
                       -------------------------------------
                            Positivo            Negativo
                       -------------------------------------
                  Posi |                |                   |
                  tivo |       25       |         20       |
                       |                |                   |
                       -------------------------------------
                  Nega |                |                   |
                  tivo |       35       |         35       |
                       |                |                   |
                       -------------------------------------
  el despliegue de las ventanas y la impresión en consola indican que la salida fue correcta, pero favor de revisar que no se haya impreso algun error en consola, estas son las lineas de comando para ejecutar cada test:
    python .\KMeans.py

    python .\lineal-regression2.py

    python .\lineal-regression.py

    python .\NeuralNetworks.py
    
    python .\multipleSVM.py

Gracias, eso es todo. Cualquier duda favor de contactarme por Whatsapp 7641235074