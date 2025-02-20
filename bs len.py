# Definición de tokens y lexemas
TOKENS = {
    "PERSONAJE": ["Shelly", "Colt", "Spike", "Bull", "Jessie", "Brock", "Dynamike", "Bo", "Tara"],
    "HABILIDAD": ["Super", "Ataque básico", "Gadget", "Star Power"],
    "MODO_JUEGO": ["Atrapagemas", "Balón Brawl", "Atraco", "Supervivencia", "Zona Restringida"],
    "ACCION": ["Atacar", "Defender", "Esquivar", "Usar Super", "Recargar", "Curar"],
    "OBJETIVO": ["Gema", "Balón", "Caja fuerte", "Zona", "Enemigo", "Aliado"],
    "MAPA": ["Cañón Furioso", "Mina Abandonada", "Estación de Energía"],
    "ESTADO": ["Vivo", "Muerto", "Cargando Super", "Con Balón", "Con Gema"],
}

# Definición de patrones válidos
PATRONES = [
    ["PERSONAJE", "HABILIDAD", "OBJETIVO"],  # Ej: "Shelly usa Super en Enemigo"
    ["PERSONAJE", "ACCION", "OBJETIVO"],     # Ej: "Colt ataca Balón"
    ["MODO_JUEGO", "MAPA"],                  # Ej: "Atrapagemas en Cañón Furioso"
    ["PERSONAJE", "ESTADO"],                 # Ej: "Spike está Muerto"
]

# Función para identificar tokens en una oración
def identificar_tokens(oracion):
    palabras = oracion.split()
    tokens_identificados = []
    
    while palabras:
        encontrado = False
        for tipo_token, valores in TOKENS.items():
            for valor in valores:
                valor_palabras = valor.split()
                if palabras[:len(valor_palabras)] == valor_palabras:
                    tokens_identificados.append((tipo_token, valor))
                    palabras = palabras[len(valor_palabras):]
                    encontrado = True
                    break
            if encontrado:
                break
        if not encontrado:
            return None  # Si no se encuentra un token válido, la oración es inválida
    
    return tokens_identificados

# Función para validar la oración según los patrones
def validar_oracion(oracion):
    tokens = identificar_tokens(oracion)
    if tokens is None:
        return f"La oración '{oracion}' no es válida."
    
    tipos_detectados = [tipo for tipo, _ in tokens]
    
    for patron in PATRONES:
        if tipos_detectados == patron:
            return f"La oración '{oracion}' es válida. Sigue el patrón: {patron}"
    
    return f"La oración '{oracion}' no es válida."

# Ejemplos de uso
print(validar_oracion("Shelly Super Enemigo"))  # Válido
print(validar_oracion("Colt Atacar Balón"))     # Válido
print(validar_oracion("Atrapagemas Cañón Furioso"))  # Válido
print(validar_oracion("Spike Muerto"))          # Válido
print(validar_oracion("Jessie Defender Zona"))  # Válido
print(validar_oracion("Shelly usa Super en Enemigo"))  # Inválido, no es un patrón válido
