# **Mantenimiento Predictivo para Sistemas Avanzados de Ósmosis Inversa con Inteligencia Artificial**

**Nivel de Madurez**: TRL 2-3

## **Descripción**
Este proyecto tiene como objetivo desarrollar un sistema de inteligencia artificial para el mantenimiento predictivo de sistemas avanzados de ósmosis inversa. Utilizando técnicas de aprendizaje supervisado y modelos de aprendizaje automático, busca predecir el desgaste de las membranas y optimizar el proceso.

Nuestro planeta contiene 1386 millones de km³ de agua, cantidad que no ha cambiado en los últimos 2 mil millones de años. El 97% de esa agua es salada, y solo el 2.5% es dulce. En México, la población ha crecido significativamente, alcanzando 129 millones en 2024, lo que agrava la escasez de agua. Entre las soluciones emergentes está la ósmosis inversa, proceso que purifica agua eliminando sales y contaminantes mediante una membrana semipermeable, ideal para desalinizar agua de mar. Este proyecto busca optimizar su alto costo y mantenimiento por medio de un modelo de inteligencia artificial, que permita predecir el desgaste de componentes clave mediante el análisis de datos históricos, como la presión, el tamaño del caudal, temperatura y salinidad, permitiendo una intervención preventiva.

## **Características**
- Predicción de la condición física de las membranas semipermeables.
- Optimización del proceso de desalinización.
- Uso de datos históricos operativos: presión, caudal, temperatura y salinidad.
- Modelos de clasificación y regresión basados en Random Forest.

## **Requisitos**
- Python 3.8+
- Librerías: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## **Instalación**
1. Clona el repositorio:  
    ```bash
   git clone https://github.com/mcortezv/Predictive-Maintenance-for-Advanced-Reverse-Osmosis-Systems-with-Artificial-Intelligence
   ```

2. Instala las dependencias:
   
    ```bash
   pip install -r requirements.txt
    ```

## **Uso**
1. Carga el conjunto de datos en formato CSV en el directorio `data/`.
2. Ejecuta el script principal para entrenar y evaluar los modelos:
    ```bash
   python main.py
    ```

## **Contribución**
Por favor revisa el archivo [CONTRIBUTING](./CONTRIBUTING.md) para más detalles.

## **Licencia**
Este proyecto está licenciado bajo la [Licencia MIT](./LICENSE.md).