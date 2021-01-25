# Modelo-de-Arbol-de-Clasificacion-con-RStudio
author: "Naren Castellon"
date: "01/24/2021"

## 3. **Construcción, Diagramas y evaluación: árboles de clasificación**

Los árboles de decisión (DT) son un método de aprendizaje supervisado no paramétrico que se utiliza para clasificación y regresión . El objetivo es crear un modelo que prediga el valor de una variable objetivo aprendiendo reglas de decisión simples inferidas de las características de los datos. Un árbol puede verse como una aproximación constante por partes.



![](imagenes/arbol_decision1.png)

#### **Algunas ventajas de los árboles de decisión son:**

* Sencillo de entender e interpretar. Los árboles se pueden visualizar.

* Requiere poca preparación de datos. Otras técnicas a menudo requieren la normalización de datos, es necesario crear variables ficticias y eliminar valores en blanco. Sin embargo, tenga en cuenta que este módulo no admite valores perdidos.

* El costo de usar el árbol (es decir, predecir datos) es logarítmico en el número de puntos de datos usados para entrenar el árbol.

* Capaz de manejar datos tanto numéricos como categóricos. Sin embargo, la implementación de scikit-learn no admite variables categóricas por ahora. Otras técnicas suelen estar especializadas en analizar conjuntos de datos que tienen un solo tipo de variable. Consulte los algoritmos para obtener más información.

* Capaz de manejar problemas de múltiples salidas.

* Utiliza un modelo de caja blanca. Si una situación dada es observable en un modelo, la explicación de la condición se explica fácilmente mediante lógica booleana. Por el contrario, en un modelo de caja negra (por ejemplo, en una red neuronal artificial), los resultados pueden ser más difíciles de interpretar.

* Posible validar un modelo mediante pruebas estadísticas. Eso permite tener en cuenta la fiabilidad del modelo.

* Funciona bien incluso si el modelo real a partir del cual se generaron los datos viola sus suposiciones.

#### **Las desventajas de los árboles de decisión incluyen:**

* Los aprendices de árboles de decisiones pueden crear árboles demasiado complejos que no generalizan bien los datos. Esto se llama sobreajuste. Mecanismos como la poda, el establecimiento del número mínimo de muestras necesarias en un nodo de la hoja o el establecimiento de la profundidad máxima del árbol son necesarios para evitar este problema.

* Los árboles de decisión pueden ser inestables porque pequeñas variaciones en los datos pueden resultar en la generación de un árbol completamente diferente. Este problema se mitiga mediante el uso de árboles de decisión dentro de un conjunto.

* Las predicciones de los árboles de decisión no son suaves ni continuas, sino aproximaciones constantes por partes, como se ve en la figura anterior. Por lo tanto, no son buenos para la extrapolación.

* Se sabe que el problema de aprender un árbol de decisión óptimo es NP-completo en varios aspectos de la optimalidad e incluso para conceptos simples. En consecuencia, los algoritmos prácticos de aprendizaje del árbol de decisiones se basan en algoritmos heurísticos como el algoritmo codicioso en el que se toman decisiones localmente óptimas en cada nodo. Dichos algoritmos no pueden garantizar la devolución del árbol de decisiones globalmente óptimo. Esto se puede mitigar entrenando varios árboles en un aprendiz de conjunto, donde las características y muestras se muestrean al azar con reemplazo.

* Hay conceptos que son difíciles de aprender porque los árboles de decisión no los expresan fácilmente, como XOR, problemas de paridad o multiplexor.

* Los aprendices del árbol de decisiones crean árboles sesgados si dominan algunas clases. Por lo tanto, se recomienda equilibrar el conjunto de datos antes de ajustarlo al árbol de decisiones.

* Puedemos usar un par de paquetes R para construir árboles de clasificación. 

* En este apartado mostraremos como usar los paquete `caret()`, `rpart()` y `rpart.plot()` para generar diagramas de árboles de clasificación.

**Paso 1.** Cargue los paquetes rpart, rpart.plot y caret:

```{r message=FALSE}
library(rpart)
library(rpart.plot)
library(caret)
```

**Paso 2.** Lea los datos:
```{r}
bn <- read.csv("data/banknote-authentication.csv")
```

**Paso 3.** Cree particiones de datos. Necesitamos dos particiones: formación y validación. En lugar de copiar los datos en las particiones, solo mantendremos los índices de los casos que representan los casos de entrenamiento y el subconjunto cuando sea necesario:
```{r}
set.seed(1000)
train.idx <- createDataPartition(bn$class, p = 0.8, list = FALSE)
```

En el paso 3, configuramos la semilla aleatoria para que sus resultados coincidan con los que mostramos.

**Paso 4.** Construye el árbol:
```{r}
mod <- rpart(class ~ ., data = bn[train.idx, ], method = "class", 
  control = rpart.control(minsplit = 20, cp = 0.01))
```

La función `rpart ()` construye el modelo de árbol basándose en lo siguiente:
  
* Fórmula que especifica las variables dependientes e independientes
* Conjunto de datos para usar
* Una especificación a través de `method="class"` que queremos construir un árbol de clasificación (en lugar de un árbol de regresión)
* Parámetros de control especificados a través de la configuración `control = rpart.control ()`; aquí hemos indicado que el árbol solo debe considerar nodos con al menos 20 casos para dividir y usar el valor del parámetro de complejidad de 0.01; estos dos valores representan los valores predeterminados y los hemos incluido solo para ilustración

**Paso 5.** Vea la salida de texto (su resultado podría diferir si no estableció la semilla aleatoria como en el paso 3):
```{r}
mod
```


**Paso 6.** Genere un diagrama del árbol (su árbol puede diferir si no estableció la semilla aleatoria como en el paso 3):

```{r}
prp(mod, type = 2, extra = 104, nn = TRUE, fallen.leaves = TRUE, main="Diagrama de árbol",
  faclen = 4, varlen = 8, shadow.col = "gray",box.palette = c("-Red", "palegreen3"))
```

El paso 6 usa la función `prp ()` del paquete rpart.plot para producir una gráfica del árbol con un aspecto agradable:

* use `type = 2` para obtener una gráfica con cada nodo etiquetado y con la etiqueta dividida debajo del nodo
* use `extra = 4` para mostrar la probabilidad de cada clase en el nodo (condicionado en el nodo y por lo tanto sumando a 1); agregue 100 (por lo tanto, `extra = 104`) para mostrar el número de casos en el nodo como un porcentaje del número total de casos
* use `nn = TRUE` para mostrar los números de nodo; el nodo raíz es el nodo número 1 y el nodo n tiene nodos secundarios numerados 2n y 2n + 1
* use `fallen.leaves = TRUE` para mostrar todos los nodos de hojas en la parte inferior del gráfico
* use `faclen` para abreviar los nombres de clases en los nodos a una longitud máxima específica
* usa `varlen` para abreviar nombres de variables
* use `shadow.col` para especificar el color de la sombra que proyecta cada nodo

**Paso 7.** Pode el árbol:
```{r}
# Primero vea el cptable
  # !! Nota !!: Su tabla puede ser diferente debido a la
  # aspecto aleatorio en validación cruzada
mod$cptable
```

```{r}
# Elija el valor de CP como el valor más alto cuyo
# xerror no es mayor que el mínimo xerror + xstd
# Con los datos anteriores, resulta que
# el quinto, 0.01182033
# Tus valores pueden ser diferentes debido a
# muestreo
# Reemplace 5 en la siguiente línea con el valor apropiado para sus datos
mod.pruned = prune(mod, mod$cptable[5, "CP"])
```

El paso 7 podar el árbol para reducir la posibilidad de que el modelo modele demasiado los datos de entrenamiento, es decir, para reducir el sobreajuste. Dentro de este paso, primero miramos la tabla de complejidad generada a través de la validación cruzada. Luego usamos la tabla para determinar el nivel de complejidad de corte como el valor más grande de xerror (error de validación cruzada) que no es mayor que una desviación estándar por encima del error mínimo de validación cruzada.

**Paso 8.** Vea el árbol podado (su árbol se verá diferente):
```{r}
prp(mod.pruned, type = 2, extra = 104, nn = TRUE, fallen.leaves = TRUE, main="Árbol Podado",
  faclen = 4, varlen = 8, shadow.col = "gray",  box.palette = "RdYlGn")
#box.palette = c("pink", "palegreen3","-RdYlGn") colores
```

**Paso 9.** Utilice el modelo podado para predecir la partición de validación (observe el signo menos antes de train.idx para considerar los casos en la partición de validación):
```{r}
pred.pruned1 <- predict(mod, bn[-train.idx,], type = "class")
```

**Paso 10.** Genere la matriz de error / clasificación-confusión:
```{r}
library(caret)
#table(bn[-train.idx,]$class, pred.pruned1, dnn = c("Actual", "Predicted"))

confusionMatrix(table(bn[-train.idx,]$class, pred.pruned1))
```


Los pasos del 8 al 10 muestran el árbol podado; use el árbol podado para predecir la clase para la partición de validación y luego genere la matriz de error para la partición de validación.

**A continuación, discutimos una variación importante en las predicciones utilizando árboles de clasificación.**

#### Calcular probabilidades.

Podemos generar probabilidades en lugar de clasificaciones especificando `type = "prob"`:

```{r}
pred.pruned2 <- predict(mod, bn[-train.idx,], type = "prob")

```

## **Crear el gráfico ROC**

Usando las probabilidades brutas anteriores y las etiquetas de clase, podemos generar un gráfico ROC. Consulte la receta Generación de gráficos ROC anteriormente en este capítulo para obtener más detalles:

```{r}
pred <- prediction(pred.pruned2[,2], bn[-train.idx,"class"])
perf <- performance(pred, "tpr", "fpr")
plot(perf, main="CURVA ROC", col="blue", lwd=3)
segments(0, 0, 1, 1, lty=2)
#plotROC(bn[-train.idx,]$class, pred.pruned)
```

El comando de segmentos dibuja la línea punteada que representa el clasificador sin valor predictivo.

Para medir esto cuantitativamente, necesitamos crear un nuevo objeto de rendimiento con medida = "auc" o área bajo la curva.

```{r}
roc_auc<-performance(pred, measure="auc")

```

Ahora el roc_auc se almacena como un objeto S4. Esto es bastante diferente al marco de datos y las matrices. Primero, podemos usar la función `str ()` para ver su estructura.

```{r}
str(roc_auc)
```
El objeto ROC tiene seis miembros. El valor de AUC se almacena en valores y. Para extraer eso, usamos el símbolo @ de acuerdo con la salida de la función str ().

```{r}
roc_auc@y.values # El área bajo curva

```
Así, el AUC obtenido = 97.21% lo que sugiere un clasificador justo, de acuerdo con el esquema de puntuación anterior.
