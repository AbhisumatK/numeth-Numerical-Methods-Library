

<p align="center">
    <img src="https://raw.githubusercontent.com/AbhisumatK/numeth-Numerical-Methods-Library/main/numeth.jpg" alt="numeth Logo" width="300">

<a href="https://www.producthunt.com/products/numeth?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-numeth" target="_blank" rel="noopener noreferrer"><img alt="numeth - Python package to use numerical method algorithms in 1 line | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1114847&amp;theme=light&amp;t=1775282970787"></a>

[![PyPI version](https://badge.fury.io/py/numeth.svg)](https://badge.fury.io/py/numeth)
[![PyPI downloads](https://img.shields.io/pypi/dm/numeth.svg)](https://pypistats.org/packages/numeth)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AbhisumatK/numeth-Numerical-Methods-Library/blob/main/LICENSE)

# numeth

Un paquete de Python totalmente funcional que implementa métodos numéricos fundamentales para ingeniería y matemáticas aplicadas. Diseñado para la usabilidad y la claridad educativa.

## Instalación

Instalar mediante pip:

```
pip install numeth
```

## Inicio Rápido

Este es un ejemplo sencillo que utiliza el método de Newton-Raphson para encontrar la raíz cuadrada de 2:

```python
from numeth import newton_raphson

def f(x):
    return x**2 - 2

def df(x):
    return 2 * x

root, iterations, converged = newton_raphson(f, df, x0=1.0, tol=1e-6, max_iter=100)
print(f"Root: {root}, Iterations: {iterations}, Converged: {converged}")
# Output: Root: 1.414213562373095, Iterations: 4, Converged: True
```

## Visualización

Puede visualizar fácilmente la convergencia o los resultados de cualquier método numérico utilizando el método `.graph()`.

```python
import numeth

# Integration visualization
tr = numeth.trapezoidal(lambda x: x**2, 0, 1)
tr.graph()

# Root finding visualization
sol = numeth.bisection(lambda x: x**2 - 2, 0, 2)
sol.graph()
```

El método `.graph()` proporciona una representación visual de cómo funciona el algoritmo, incluyendo gráficos de funciones, áreas de integración, marcadores de raíces y líneas tangentes para diferenciación.

### Métodos compatibles para visualización
Actualmente, la visualización es compatible con casi todos los módulos:
- **Integración**: Todos los métodos compatibles.
- **Búsqueda de raíces**: Todos los métodos compatibles.
- **Diferenciación**: Todos los métodos compatibles.
- **Interpolación**: Todos los métodos compatibles.
- **Optimización**: Todos los métodos compatibles.

*Nota: Los métodos de Álgebra Lineal (Eliminación de Gauss, Descomposición LU, Jacobi, Gauss-Seidel) actualmente no son compatibles con `.graph()` porque operan sobre vectores/matrices en lugar de funciones de una sola variable.*



## Métodos compatibles

### Integración
- Regla del trapecio (simple y compuesta)
- Regla de Simpson 1/3 (simple y compuesta)
- Regla de Simpson 3/8
- Cuadratura de Gauss (de 2 y 3 puntos)

### Diferenciación
- Diferencia hacia adelante (primera derivada)
- Diferencia hacia atrás (primera derivada)
- Diferencia central (primera derivada)
- Diferencia central (segunda derivada)
- Extrapolación de Richardson (primera derivada)

### Búsqueda de raíces
- Método de bisección
- Método de Newton-Raphson
- Método de la secante
- Método de la posición falsa

### Interpolación
- Interpolación lineal
- Interpolación de Lagrange
- Interpolación por diferencias divididas de Newton

### Álgebra lineal
- Eliminación de Gauss con pivoteo parcial
- Descomposición LU (método de Doolittle)
- Método iterativo de Jacobi
- Método iterativo de Gauss-Seidel

### Optimización
- Búsqueda de la sección áurea (minimización)
- Método de Newton para optimización (1D)

## Cómo contribuir o reportar problemas

¡Las contribuciones son bienvenidas! Por favor, envíe solicitudes de extracción (pull requests) o abra issues en el [repositorio de GitHub](https://github.com/AbhisumatK/numeth-Numerical-Methods-Library).

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulte el archivo [LICENSE](https://github.com/AbhisumatK/numeth-Numerical-Methods-Library/blob/main/LICENSE) para obtener más detalles.
