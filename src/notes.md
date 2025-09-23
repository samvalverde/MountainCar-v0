# TC2 - Algoritmos Genéticos (MountainCar-v0)

#### Bitácora de toma de decisiones y resultados progresivos con el algoritmo genético

Notas de desarrollo y experimentación
-------------------------------------

1. Implementación base
   - Política lineal (obs_size → action_size).
   - Fitness = recompensa promedio en N episodios.
   - Problema inicial: fitness se estancaba en -200.

2. Observaciones iniciales
   - Best fitness en -200 casi fijo desde el inicio.
   - A veces un "spike" en primeras generaciones, pero se perdía → sin elitismo.
   - Sospecha: políticas demasiado simples + sin preservación de mejores cromosomas.

3. Ajustes aplicados
   - Se guardan los logs en logs.txt para revisar evolución detallada en la ejecución del algoritmo.
   - plot_fitness ahora guarda PNG en experiments/results (evita bloqueo de plt.show()).
   - Agregado elitismo (al menos 1 mejor individuo sobrevive cada generación).
   - Propuesta más robusta: elitismo proporcional (ej. 5% de mejores).
   - Mating pool proporcional (ej. 25% de población).
   - Con esto la evolución depende del tamaño de población, no de valores hardcodeados.

4. Próximos pasos
   - Probar diferentes configuraciones (baseline, high_mutation, large_population).
   - Comparar fitness con/sin elitismo.
   - Revisar si política lineal es suficiente o si hace falta una red con capa oculta.
   - Preparar gráficas para análisis crítico (fitness promedio y máximo por generación).

5. Cambio en la representación de la política
   - Antes: política lineal (2 → 3).
   - Problema: incapaz de aprender estrategias no lineales (ej. ir hacia atrás para tomar impulso).
   - Ajuste: añadimos una hidden layer (2 → 8 → 3).
   - Activación: ReLU entre capas.
   - Cromosoma ahora incluye los pesos de W1 (2×8) y W2 (8×3), en total 40 genes.
   - Ventaja: mayor expresividad de la política → más chances de mejorar fitness.


Checklist de Experimentos
-------------------------

[ ] Baseline
    - Población = 30
    - Mutación = 0.1
    - Crossover = 0.7
    - Elitismo = OFF

[ ] Baseline + Elitismo simple
    - Igual que baseline
    - Elitismo = 1 mejor individuo

[ ] Elitismo proporcional
    - Población = 30
    - Mutación = 0.1
    - Crossover = 0.7
    - Elitismo = 5% mejores
    - Mating pool = 25%

[ ] High Mutation
    - Población = 30
    - Mutación = 0.3
    - Crossover = 0.7
    - Elitismo proporcional

[ ] Large Population
    - Población = 50
    - Mutación = 0.1
    - Crossover = 0.7
    - Elitismo proporcional

[ ] Comparativa final
    - Graficar fitness promedio y máximo
    - Guardar gráficas en experiments/results
    - Guardar datos en CSV para análisis crítico


Resultados con hidden layer y elitismo proporcional
---------------------------------------------------

- Con la nueva representación de la política (2 → 8 → 3 con ReLU) y la estrategia de elitismo/mating pool proporcional, el GA logró alcanzar la cima en MountainCar-v0 en apenas 50 generaciones.
- Esto representa una mejora sustancial respecto al modelo lineal inicial, que se quedaba estancado en -200.
- Los mejores individuos alcanzaron recompensas cercanas a -114, lo cual implica que el agente logra resolver la tarea en menos de 100 pasos.
- La convergencia rápida sugiere que la arquitectura y parámetros actuales son bastante expresivos y eficientes para este entorno.
- Reflexión crítica: que el GA haya resuelto el problema tan rápido puede indicar que:
  1. La política con hidden layer es demasiado poderosa para un entorno relativamente simple.
  2. Los parámetros del GA (elitismo, tasa de mutación, tamaño de población) están muy bien calibrados para este caso.
  3. El entorno MountainCar-v0, en su versión estándar (goal_velocity=0), es menos desafiante de lo esperado.
- Como trabajo futuro, sería interesante:
  - Aumentar la dificultad del entorno (`goal_velocity > 0`).
  - Reducir la capacidad de la hidden layer para analizar el impacto en la convergencia.
  - Aumentar el número de generaciones y evaluar la estabilidad de los promedios.
