def generar_triangulo_pascal(n):
    # Inicializa el triángulo con la primera fila
    triangulo = [[1]]

    for i in range(1, n):
        nueva_fila = list(range(1, i + 2))
        triangulo.append(nueva_fila)

    return triangulo


def imprimir_triangulo(triangulo):
    # Impresión del triángulo para que sea más visual
    max_width = len(" ".join(map(str, triangulo[-1]))) 
    for fila in triangulo:
        fila_str = " ".join(map(str, fila))
        print(fila_str.center(max_width))  # Centra cada fila


# Se solicita al usuario la profundidad del triángulo
profundidad = int(input("Ingresa la profundidad del triángulo de Pascal: "))
triangulo = generar_triangulo_pascal(profundidad)
imprimir_triangulo(triangulo)