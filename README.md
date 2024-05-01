# BuscaPiso-NLP
Trabajo para NLP sobre un buscador de pisos mediante lenguaje natural. Recopilamos información de distintas webs de búsqueda de pisos para crear un modelo que sirva como buscador a través de un prompt en función de las consultas que realicemos. El modelo debe devolver las 5 mejores opciones ordenadas de mayor a menor.
----------
SEMANA 1
----------

¿QUÉ DATOS USAMOS?

Tras contactar con distintas plataformas, sólo podemos recopilar información de:

 - Indomio
 - Pisos.com
 
 Hacemos los primeros scraps de las plataformas, primeramente en Indomio y nos basamos en este para hacer el posterior de Pisos.com.
 Se crean diccionarios que almacenan como keys los links de los mismos anuncios, y sus items son: precio, descripción, características generales, otras características e información sobre eficiencia energética.
 Para no ejecutar el código en sucesivas ocasionies cada vez que trabajamos en los códigos, creamos y subimos al repositorio sucesivos .pkl que contienen las url de los barrios, de los anuncios y el diccionario final que contiene la información sobre cada casa.
