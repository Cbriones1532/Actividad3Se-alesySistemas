# Diseño y análisis de filtros digitales

Resumen:
- Se diseñaron y compararon filtros digitales: Butterworth (LP), Chebyshev I (HP) e FIR por ventana (PB).
- Se aplicaron a una señal compuesta (50, 300, 700 Hz) contaminada con ruido blanco.
- Se muestran señales en tiempo y frecuencia, respuesta en frecuencia y coeficientes de los filtros.

Cómo ejecutar:
1. Crear entorno virtual e instalar dependencias:
   pip install -r requirements.txt
2. Ejecutar:
   python filtros_signales.py
3. Las figuras y resultados se guardan en la carpeta `figs/`.

   Pre requisitos
   Tener instalado Pyhton y las siguientes librerias:
   python -m venv venv
source venv/bin/activate   # linux/mac  (windows: venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt

Codigo para ejcutar:
python filtros_signales.py
# Esto generará la carpeta ./figs con imágenes y coeficientes.

Autor: Cristofer Alejandro Briones Arreaga
Fecha: 8/11/2025
