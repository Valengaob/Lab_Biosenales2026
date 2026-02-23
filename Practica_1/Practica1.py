import numpy as np
import matplotlib.pyplot as plt

print("PRACTICA 1 - PROCEDIMIENTO \n")

# -------------------------------------------------
# a) Crear vectores a y b
# -------------------------------------------------
a = np.array([3.1, 1, -0.5, -3.2, 6])
b = np.array([1, 3, 2.2, 5.1, 1])

print("a) Vectores:")
print("a =", a)
print("b =", b)
print()

# -------------------------------------------------
# b) Multiplicación escalar a·b (producto punto)
# -------------------------------------------------
# Para poder multiplicarlos (producto punto), deben tener la MISMA longitud.
if a.shape == b.shape:
    prod_escalar = np.dot(a, b)  # también sirve: a @ b
    print("b) Producto escalar (a·b):", prod_escalar)
    print("   Condición: a y b deben tener la misma longitud.\n")
else:
    print("b) No se puede calcular a·b: a y b tienen diferente longitud.\n")

# -------------------------------------------------
# c) Multiplicación punto a punto (element-wise)
# -------------------------------------------------
# Esto devuelve un vector del mismo tamaño
if a.shape == b.shape:
    prod_punto_punto = a * b
    print("c) Producto punto a punto (a*b):", prod_punto_punto, "\n")
else:
    print("c) No se puede calcular a*b: a y b tienen diferente longitud.\n")

# -------------------------------------------------
# d) Construir matriz A
# -------------------------------------------------
A = np.array([
    [2,  -1,  -3],
    [4,  1.5, -2.5],
    [7.3, -0.9, 0.2]
])

print("d) Matriz A:\n", A, "\n")

# -------------------------------------------------
# e) Transpuesta A^T
# -------------------------------------------------
AT = A.T
print("e) Transpuesta A^T:\n", AT, "\n")

# -------------------------------------------------
# f) Ejemplificar ones, round, ceil, floor
# -------------------------------------------------
print("f) Ejemplos de funciones:")

# ones
unos = np.ones((2, 4))  # matriz 2x4 de unos
print("np.ones((2,4)):\n", unos)

# round, ceil, floor con un vector de ejemplo
x = np.array([2.3, 2.5, 2.7, -1.2, -1.7])
print("\nVector ejemplo x =", x)

print("np.round(x) =", np.round(x))       # redondeo al entero más cercano
print("np.round(x, 1) =", np.round(x, 1)) # redondeo a 1 decimal
print("np.ceil(x)  =", np.ceil(x))        # techo
print("np.floor(x) =", np.floor(x))       # piso
print()

# -------------------------------------------------
# g) Primera fila, tercera columna de A
# -------------------------------------------------
valor_g = A[0, 2]  # fila 0 (primera), col 2 (tercera)
print("g) A[primera fila, tercera columna] = A[0,2] =", valor_g, "\n")

# -------------------------------------------------
# h) Segunda fila completa de A
# -------------------------------------------------
fila_h = A[1, :]  # fila 1 (segunda), todas las columnas
print("h) Segunda fila de A (A[1,:]) =", fila_h, "\n")

# -------------------------------------------------
# i) Dimensiones de A
# -------------------------------------------------
print("i) Dimensiones de A (A.shape) =", A.shape, "\n")

# -------------------------------------------------
# j) y[n] = sin(pi*0.12n), 0<=n<=100
# -------------------------------------------------
n = np.arange(0, 101)  # 0 a 100 inclusive
y = np.sin(np.pi * 0.12 * n)

print("j) y[n] creada. Primeros 5 valores:", y[:5], "\n")

# -------------------------------------------------
# k) y2[n] = cos(2pi*0.03n)
# -------------------------------------------------
y2 = np.cos(2 * np.pi * 0.03 * n)

print("k) y2[n] creada. Primeros 5 valores:", y2[:5], "\n")

# -------------------------------------------------
# l) s[n] = y[n] + y2[n] y t[n] = y[n]*y2[n]
# -------------------------------------------------
s = y + y2
t = y * y2

print("l) s[n] (suma) primeros 5:", s[:5])
print("   t[n] (producto) primeros 5:", t[:5], "\n")

# -------------------------------------------------
# m) Graficar y[n] y y2[n] en la misma figura
# -------------------------------------------------
plt.figure()
plt.plot(n, y, label="y[n] = sin(pi*0.12n)")
plt.plot(n, y2, label="y2[n] = cos(2pi*0.03n)")
plt.title("Señales y[n] y y2[n]")
plt.xlabel("n (muestras)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------------------------
# n) Graficar s[n] y t[n] en la misma figura
# -------------------------------------------------
plt.figure()
plt.plot(n, s, label="s[n] = y[n] + y2[n]")
plt.plot(n, t, label="t[n] = y[n] * y2[n]")
plt.title("Señales s[n] y t[n]")
plt.xlabel("n (muestras)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------------------------
print("PRACTICA 1 - REPASO PANDAS \n")
# -------------------------------------------------
# 1) Función solicitada
# ------------------------------------------------
import pandas as pd

def estadisticas_notas(dic_notas):
    serie = pd.Series(dic_notas)
    
    resultado = pd.Series({
        "Minima": serie.min(),
        "Maxima": serie.max(),
        "Media": serie.mean(),
        "Desviacion": serie.std()
    })
    
    return resultado


# ejemplo de prueba
notas = {"Ana": 4.5, "Luis": 3.2, "Maria": 4.8, "Juan": 2.9}

print("1) Estadísticas de notas:\n")
print(estadisticas_notas(notas),"\n")

# -------------------------------------------------
print("2) IMC con Pandas. \n")
# -------------------------------------------------
# a) Cargar archivo datos.csv
# -------------------------------------------------
df = pd.read_csv("datos.csv", sep=";")
print("a) Datos cargados correctamente.\n")
print(df.columns, "\n")

# -------------------------------------------------
# b) Mostrar primeras y últimas filas
# -------------------------------------------------
print("b) Primeras 5 filas:")
print(df.head(), "\n")

print("b) Últimas 5 filas:")
print(df.tail(), "\n")

# -------------------------------------------------
# c) Eliminar columna 'Unnamed: 0'
# -------------------------------------------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("c) Columna 'Unnamed: 0' eliminada.\n")

# -------------------------------------------------
# d y e) Convertir unidades y calcular IMC
# -------------------------------------------------
# convertir altura de pulgadas a metros
df["Height_m"] = df["Height"] * 0.0254

# convertir peso de libras a kg
df["Weight_kg"] = df["Weight"] * 0.453592

# calcular IMC
df["BMI"] = df["Weight_kg"] / (df["Height_m"] ** 2)

df["BMI"] = df["BMI"].round(2)

print("d y e) IMC calculado correctamente:")
print(df[["Weight_kg", "Height_m", "BMI"]].head(), "\n")

# -------------------------------------------------
# f) Clasificar IMC según categoría
# -------------------------------------------------

def categoria_imc(imc):
    if imc < 18.5:
        return "Bajo peso"
    elif imc < 25:
        return "Normal"
    elif imc < 30:
        return "Sobrepeso"
    else:
        return "Obesidad"

df["Categoria_IMC"] = df["BMI"].apply(categoria_imc)

print("f) Categorías agregadas:")
print(df[["BMI", "Categoria_IMC"]].head())
