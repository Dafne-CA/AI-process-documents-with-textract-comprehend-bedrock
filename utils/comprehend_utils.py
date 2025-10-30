import boto3
import re
import os
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

class ClasificadorDocumentos:
    def __init__(self):
        """Inicializa el clasificador híbrido Comprehend + Reglas"""
        self.comprehend = boto3.client('comprehend')
        
    def clasificar_documento(self, texto: str) -> Dict:
        """
        Clasifica un documento usando estrategia en cascada:
        1. Reglas rápidas (gratis)
        2. Patrones avanzados (gratis) 
        3. Comprehend (solo si es necesario)
        """
        if not texto or len(texto.strip()) < 10:
            return {
                'clase': 'desconocido',
                'confianza': 0.0,
                'metodo': 'texto_insuficiente'
            }
        
        # Estrategia en cascada para optimizar costos
        resultado_rapido = self._clasificacion_rapida(texto)
        if resultado_rapido['clase'] != 'desconocido':
            return resultado_rapido
        
        resultado_patron = self._clasificacion_por_patrones(texto)
        if resultado_patron['clase'] != 'desconocido':
            return resultado_patron
        
        # Solo usar Comprehend si las reglas fallan
        return self._clasificar_con_comprehend(texto)
    
    def clasificar_lote(self, textos: List[str]) -> List[Dict]:
        """Clasifica múltiples documentos eficientemente"""
        resultados = []
        
        for texto in textos:
            resultado = self.clasificar_documento(texto)
            resultados.append(resultado)
        
        return resultados
    
    def _clasificacion_rapida(self, texto: str) -> Dict:
        """Reglas simples que cubren el 80% de los casos"""
        texto_lower = texto.lower()
        
        # Palabras clave por categoría con pesos
        keywords = {
            'contrato': {
                'keywords': [
                    'contrato de', 'contrato para', 'contrato n°', 'contrato número',
                    'contratación', 'convenio', 'acuerdo entre', 'cláusula',
                    'vigencia', 'partes contratantes', 'objeto del contrato'
                ],
                'peso': 1.0
            },
            'factura': {
                'keywords': [
                    'factura electronica', 'factura n°', 'factura número',
                    'ruc:', 'igv:', 'importe total', 'valor venta',
                    'gravada', 'inafecta', 'exonerada', 'detracción'
                ],
                'peso': 1.0
            },
            'boleta': {
                'keywords': [
                    'boleta de venta', 'boleta n°', 'boleta número',
                    'consumidor final', 'dni:', 'documento identidad',
                    'boletería', 'venta al contado'
                ],
                'peso': 0.9
            },
            'demanda': {
                'keywords': [
                    'demanda de', 'juzgado', 'demandante', 'demandado',
                    'proceso judicial', 'recurso', 'sentencia',
                    'juez', 'tribunal', 'proceso número'
                ],
                'peso': 0.9
            },
            'estado_cuenta': {
                'keywords': [
                    'estado de cuenta', 'extracto bancario', 'tarjeta crédito',
                    'movimientos', 'saldo disponible', 'banco',
                    'débitos', 'créditos', 'pago mínimo', 'fecha corte'
                ],
                'peso': 0.9
            },
            'recibo': {
                'keywords': [
                    'recibo de', 'pago de', 'servicio de', 'mes de',
                    'luz', 'agua', 'teléfono', 'internet',
                    'servicios públicos', 'suministro'
                ],
                'peso': 0.8
            },
            'carta_notarial': {
                'keywords': [
                    'carta notarial', 'notaría', 'notarial',
                    'fe pública', 'notificación', 'intimación',
                    'notario público', 'protocolo notarial'
                ],
                'peso': 0.8
            }
        }
        
        mejor_clase = 'desconocido'
        mejor_puntaje = 0
        
        for clase, config in keywords.items():
            puntaje = 0
            for keyword in config['keywords']:
                if keyword in texto_lower:
                    puntaje += config['peso']
            
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_clase = clase
        
        # Solo considerar válido si tiene al menos 2 coincidencias
        if mejor_puntaje >= 1.8:
            confianza = min(mejor_puntaje / 3.0, 0.95)  # Normalizar a 0-0.95
            return {
                'clase': mejor_clase,
                'confianza': round(confianza, 2),
                'metodo': 'reglas_rapidas',
                'puntaje': mejor_puntaje
            }
        
        return {
            'clase': 'desconocido',
            'confianza': 0.0,
            'metodo': 'reglas_rapidas'
        }
    
    def _clasificacion_por_patrones(self, texto: str) -> Dict:
        """Patrones regex para casos más específicos"""
        patrones = {
            'factura': [
                (r'RUC\s*:\s*\d{11}', 2.0),
                (r'FACTURA\s*ELECTRÓNICA\s*:\s*[Ff]\d{3}-\d{1,9}', 2.5),
                (r'IGV\s*\(\d+%\)\s*:\s*S/\.\s*\d+\.\d{2}', 1.5),
                (r'N°\s*DE\s*DOCUMENTO\s*:\s*\d{11}', 1.5),
                (r'OPERACIÓN\s*GRAVADA\s*:\s*S/\.\s*\d+\.\d{2}', 1.0)
            ],
            'contrato': [
                (r'CONTRATO\s*DE\s*[A-Z\s]+\s*N°\s*\d+', 2.0),
                (r'CLÁUSULA\s*(PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|SÉPTIMA|OCTAVA|NOVENA|DÉCIMA)', 1.5),
                (r'VIGENCIA\s*:\s*DEL\s*\d{2}/\d{2}/\d{4}\s*AL\s*\d{2}/\d{2}/\d{4}', 1.0),
                (r'ENTRE\s*[A-Z\s]+\s*Y\s*[A-Z\s]+', 1.0),
                (r'OBJETO\s*DEL\s*CONTRATO', 1.5)
            ],
            'boleta': [
                (r'BOLETA\s*DE\s*VENTA\s*ELECTRÓNICA\s*:\s*[Bb]\d{3}-\d{1,9}', 2.0),
                (r'DOCUMENTO\s*DE\s*IDENTIDAD\s*:\s*\d{8}', 1.5),
                (r'CONSUMIDOR\s*FINAL', 1.0),
                (r'BOLETA\s*N°\s*\d+', 1.0)
            ],
            'estado_cuenta': [
                (r'TARJETA\s*DE\s*CRÉDITO\s*:\s*\*+\d{4}', 2.0),
                (r'LÍMITE\s*DE\s*CRÉDITO\s*:\s*S/\.\s*\d+\.\d{2}', 1.5),
                (r'FECHA\s*DE\s*CORTE\s*:\s*\d{2}/\d{2}/\d{4}', 1.0),
                (r'PAGO\s*MÍNIMO\s*:\s*S/\.\s*\d+\.\d{2}', 1.0),
                (r'SALDO\s*ANTERIOR\s*:\s*S/\.\s*\d+\.\d{2}', 1.0)
            ],
            'recibo': [
                (r'RECIBO\s*DE\s*PAGO\s*N°\s*\d+', 1.5),
                (r'SERVICIO\s*DE\s*(LUZ|AGUA|TELEFONÍA|INTERNET)', 1.0),
                (r'PERÍODO\s*:\s*\w+\s*\d{4}', 1.0),
                (r'LECTURA\s*ANTERIOR\s*:\s*\d+', 0.8)
            ]
        }
        
        mejor_clase = 'desconocido'
        mejor_puntaje = 0
        
        for clase, lista_patrones in patrones.items():
            puntaje_clase = 0
            for patron, peso in lista_patrones:
                if re.search(patron, texto, re.IGNORECASE):
                    puntaje_clase += peso
            
            if puntaje_clase > mejor_puntaje:
                mejor_puntaje = puntaje_clase
                mejor_clase = clase
        
        if mejor_puntaje >= 1.5:
            confianza = min(mejor_puntaje / 4.0, 0.90)
            return {
                'clase': mejor_clase,
                'confianza': round(confianza, 2),
                'metodo': 'patrones_avanzados',
                'puntaje': mejor_puntaje
            }
        
        return {
            'clase': 'desconocido',
            'confianza': 0.0,
            'metodo': 'patrones_avanzados'
        }
    
    def _clasificar_con_comprehend(self, texto: str) -> Dict:
        """Usar Comprehend solo para casos difíciles"""
        try:
            # Limitar texto para optimizar costos
            texto_limite = texto[:2000]  # Primeros 2000 caracteres
            
            # Detectar entidades con Comprehend
            respuesta = self.comprehend.detect_entities(
                Text=texto_limite,
                LanguageCode='es'
            )
            
            # Inferir clase basado en entidades
            inferencia = self._inferir_clase_desde_entidades(respuesta['Entities'])
            
            return {
                'clase': inferencia['clase'],
                'confianza': inferencia['confianza'],
                'metodo': 'comprehend_entidades',
                'entidades': respuesta['Entities'][:5]  # Primeras 5 entidades
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'TextSizeLimitExceededException':
                # Intentar con texto más corto
                return self._clasificar_con_comprehend(texto[:1000])
            else:
                return {
                    'clase': 'error_comprehend',
                    'confianza': 0.0,
                    'metodo': 'comprehend_error',
                    'error': str(e)
                }
        except Exception as e:
            return {
                'clase': 'desconocido',
                'confianza': 0.0,
                'metodo': 'comprehend_exception',
                'error': str(e)
            }
    
    def _inferir_clase_desde_entidades(self, entidades: List[Dict]) -> Dict:
        """Inferir tipo de documento basado en entidades de Comprehend"""
        scores = {
            'contrato': 0.0,
            'factura': 0.0,
            'boleta': 0.0,
            'demanda': 0.0,
            'estado_cuenta': 0.0,
            'recibo': 0.0,
            'carta_notarial': 0.0
        }
        
        for entidad in entidades:
            texto_entidad = entidad['Text'].lower()
            score_entidad = entidad['Score']
            tipo_entidad = entidad['Type']
            
            # Lógica de inferencia basada en entidades y tipos
            if tipo_entidad == 'COMMERCIAL_ITEM':
                if 'factura' in texto_entidad:
                    scores['factura'] += score_entidad * 2.0
                elif 'boleta' in texto_entidad:
                    scores['boleta'] += score_entidad * 2.0
                elif 'contrato' in texto_entidad:
                    scores['contrato'] += score_entidad * 2.0
                elif 'recibo' in texto_entidad:
                    scores['recibo'] += score_entidad * 2.0
            
            elif tipo_entidad == 'ORGANIZATION':
                if 'notaría' in texto_entidad or 'notarial' in texto_entidad:
                    scores['carta_notarial'] += score_entidad * 1.5
                elif any(banco in texto_entidad for banco in ['banco', 'scotiabank', 'bcp', 'bbva', 'interbank']):
                    scores['estado_cuenta'] += score_entidad * 1.2
                elif any(tribunal in texto_entidad for tribunal in ['juzgado', 'tribunal', 'corte']):
                    scores['demanda'] += score_entidad * 1.5
            
            elif tipo_entidad == 'QUANTITY':
                if any(palabra in texto_entidad for palabra in ['contrato', 'cláusula']):
                    scores['contrato'] += score_entidad * 1.0
            
            elif tipo_entidad == 'OTHER':
                if 'ruc' in texto_entidad:
                    scores['factura'] += score_entidad * 1.5
                elif 'dni' in texto_entidad:
                    scores['boleta'] += score_entidad * 1.2
                elif 'tarjeta' in texto_entidad:
                    scores['estado_cuenta'] += score_entidad * 1.0
        
        # Encontrar la clase con mayor score
        clase_maxima = max(scores, key=scores.get)
        score_maximo = scores[clase_maxima]
        
        # Calcular confianza normalizada
        confianza = min(score_maximo * 2.0, 0.95) if score_maximo > 0 else 0.0
        
        return {
            'clase': clase_maxima if score_maximo > 0.3 else 'desconocido',
            'confianza': round(confianza, 2)
        }

# Instancia global para reutilizar
clasificador = ClasificadorDocumentos()

# Funciones de conveniencia
def clasificar_texto(texto: str) -> Dict:
    """Función simple para clasificar un texto"""
    return clasificador.clasificar_documento(texto)

def clasificar_multiple_textos(textos: List[str]) -> List[Dict]:
    """Función simple para clasificar múltiples textos"""
    return clasificador.clasificar_lote(textos)

# Ejemplo de uso
if __name__ == "__main__":
    # Test del clasificador
    textos_prueba = [
        "CONTRATO DE CONSULTORIA TECNOLÓGICA N° 12345",
        "FACTURA ELECTRÓNICA RUC: 20100066603",
        "BOLETA DE VENTA DNI: 87654321",
        "Estado de cuenta Banco de Crédito",
        "Texto genérico sin patrones específicos"
    ]
    
    for texto in textos_prueba:
        resultado = clasificar_texto(texto)
        print(f"Texto: {texto[:50]}...")
        print(f"Clase: {resultado['clase']} | Confianza: {resultado['confianza']} | Método: {resultado['metodo']}")
        print("-" * 60)