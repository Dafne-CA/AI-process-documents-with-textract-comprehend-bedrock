import boto3
import os
import time
from trp import Document
import pandas as pd
from botocore.exceptions import ClientError
import json

# Configurar session correctamente
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

s3 = session.client('s3')
textract = session.client('textract')

S3_BUCKET = os.getenv("BUCKET_NAME")

def upload_bytes_to_s3(bytes_data, key):
    """Sube bytes directamente a S3"""
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=bytes_data)
        return f"s3://{S3_BUCKET}/{key}"
    except Exception as e:
        raise Exception(f"Error subiendo a S3: {str(e)}")

def start_textract_analysis(bucket, key):
    """Inicia análisis de Textract para PDF"""
    try:
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}},
            FeatureTypes=['TABLES', 'FORMS']
        )
        return response['JobId']
    except Exception as e:
        raise Exception(f"Error iniciando análisis Textract: {str(e)}")

def wait_for_textract_job(job_id, delay=5):
    """Espera a que el job de Textract termine"""
    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        
        if status in ['SUCCEEDED', 'FAILED']:
            return response
            
        time.sleep(delay)

def detect_document_text(bytes_data):
    """Detección sincrónica para imágenes - SOLO TEXTO (para compatibilidad)"""
    try:
        response = textract.detect_document_text(
            Document={'Bytes': bytes_data}
        )
        return response
    except Exception as e:
        raise Exception(f"Error en detect_document_text: {str(e)}")

def analyze_document_with_tables(bytes_data):
    """Análisis sincrónico para imágenes CON SOPORTE DE TABLAS"""
    try:
        response = textract.analyze_document(
            Document={'Bytes': bytes_data},
            FeatureTypes=['TABLES', 'FORMS']
        )
        return response
    except Exception as e:
        raise Exception(f"Error en analyze_document: {str(e)}")

def parse_textract_blocks(blocks):
    """Parsea los bloques de Textract a texto estructurado - MEJORADO"""
    text_lines = []
    tables = []
    forms = {}
    
    # Primero procesar todos los bloques básicos
    for block in blocks:
        block_type = block['BlockType']
        
        if block_type == 'LINE':
            text_lines.append(block.get('Text', ''))
        
        elif block_type == 'KEY_VALUE_SET':
            # Procesar formularios
            if 'KEY' in block.get('EntityTypes', []):
                key_text = get_text_from_block(block, blocks)
                value_text = find_value_for_key(block, blocks)
                if key_text and key_text.strip():
                    forms[key_text] = value_text
    
    # Luego procesar tablas (necesitan todos los bloques disponibles)
    for block in blocks:
        if block['BlockType'] == 'TABLE':
            table_data = process_table_block(block, blocks)
            # Solo agregar tablas que tengan contenido
            if not table_data.empty and table_data.shape[0] > 0 and table_data.shape[1] > 0:
                tables.append(table_data)
    
    return {
        'text': '\n'.join(text_lines),
        'tables': tables,
        'forms': forms
    }

def find_value_for_key(key_block, blocks):
    """Encuentra el valor correspondiente para un bloque key"""
    try:
        if 'Relationships' in key_block:
            for relationship in key_block['Relationships']:
                if relationship['Type'] == 'VALUE':
                    for value_id in relationship['Ids']:
                        value_block = next((b for b in blocks if b['Id'] == value_id), None)
                        if value_block:
                            return get_text_from_block(value_block, blocks)
    except Exception:
        pass
    return ""

def get_text_from_block(block, blocks):
    """Extrae texto de un bloque y sus hijos"""
    text = ""
    if 'Text' in block:
        text += block['Text'] + " "
    
    if 'Relationships' in block:
        for relationship in block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = next((b for b in blocks if b['Id'] == child_id), None)
                    if child_block and 'Text' in child_block:
                        text += child_block['Text'] + " "
    
    return text.strip()
def process_table_block(table_block, blocks):
    """Procesa bloques de tabla a DataFrame - CORREGIDO"""
    try:
        if 'Relationships' not in table_block:
            return pd.DataFrame()
        
        # Encontrar TODAS las celdas de la tabla
        all_cells = []
        for relationship in table_block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = next((b for b in blocks if b['Id'] == child_id), None)
                    if child_block and child_block['BlockType'] == 'CELL':
                        all_cells.append(child_block)
        
        if not all_cells:
            return pd.DataFrame()
        
        # Encontrar dimensiones máximas de la tabla
        max_row = max(cell.get('RowIndex', 0) for cell in all_cells)
        max_col = max(cell.get('ColumnIndex', 0) for cell in all_cells)
        
        if max_row == 0 or max_col == 0:
            return pd.DataFrame()
        
        # Crear matriz para la tabla
        table_matrix = [['' for _ in range(max_col)] for _ in range(max_row)]
        
        # Llenar la matriz con el contenido de las celdas
        for cell in all_cells:
            row_idx = cell.get('RowIndex', 1) - 1
            col_idx = cell.get('ColumnIndex', 1) - 1
            cell_text = get_text_from_block(cell, blocks)
            
            if 0 <= row_idx < max_row and 0 <= col_idx < max_col:
                table_matrix[row_idx][col_idx] = cell_text
        
        # Crear DataFrame
        df = pd.DataFrame(table_matrix)
        
        # Limpiar filas y columnas completamente vacías
        df = df.replace('', pd.NA)
        df = df.dropna(how='all').dropna(axis=1, how='all').fillna('')
        
        return df
        
    except Exception as e:
        print(f"Error procesando tabla: {e}")
        return pd.DataFrame()

def process_files_with_textract(files):
    """Función principal mejorada - CON SOPORTE DE TABLAS PARA IMÁGENES"""
    results = []
    
    for file in files:
        try:
            file_bytes = file.read()
            filename = file.name
            
            # Generar key única en S3
            timestamp = int(time.time())
            key = f"textract-input/{timestamp}_{filename.replace(' ', '_')}"
            
            # Subir a S3
            s3_uri = upload_bytes_to_s3(file_bytes, key)
            
            # Procesar según tipo de archivo
            if filename.lower().endswith('.pdf'):
                # Análisis asíncrono para PDF
                job_id = start_textract_analysis(S3_BUCKET, key)
                textract_response = wait_for_textract_job(job_id)
                blocks = textract_response.get('Blocks', [])
            else:
                # PARA IMÁGENES: usar analyze_document para obtener tablas
                textract_response = analyze_document_with_tables(file_bytes)
                blocks = textract_response.get('Blocks', [])
            
            # Parsear resultados
            parsed_data = parse_textract_blocks(blocks)
            
            result = {
                "filename": filename,
                "s3_uri": s3_uri,
                "text": parsed_data['text'],
                "tables": parsed_data['tables'],
                "forms": parsed_data['forms'],
                "pages": len([b for b in blocks if b['BlockType'] == 'PAGE']) or 1
            }
            
            results.append(result)
            
        except Exception as e:
            # En caso de error, intentar extracción básica de texto
            try:
                # Fallback: extraer solo texto
                textract_response = detect_document_text(file_bytes)
                blocks = textract_response.get('Blocks', [])
                text_lines = [block.get('Text', '') for block in blocks if block['BlockType'] == 'LINE']
                
                error_result = {
                    "filename": file.name,
                    "s3_uri": "",
                    "text": '\n'.join(text_lines),
                    "tables": [],
                    "forms": {},
                    "pages": 1
                }
                results.append(error_result)
            except:
                # Último fallback
                error_result = {
                    "filename": file.name,
                    "s3_uri": "",
                    "text": f"Error procesando archivo: {str(e)}",
                    "tables": [],
                    "forms": {},
                    "pages": 1
                }
                results.append(error_result)
    
    return results

def extract_tables_from_result(result):
    """Extrae y formatea tablas de un resultado - MEJORADO"""
    tables_data = []
    
    for i, table_df in enumerate(result.get('tables', [])):
        # Verificar que el DataFrame no esté vacío y tenga contenido real
        if (not table_df.empty and 
            table_df.shape[0] > 0 and 
            table_df.shape[1] > 0 and
            not table_df.isna().all().all()):
            
            # Limpiar el DataFrame
            table_df_clean = table_df.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all').fillna('')
            
            # Solo incluir si después de limpiar todavía tiene contenido
            if not table_df_clean.empty and table_df_clean.shape[0] > 0 and table_df_clean.shape[1] > 0:
                tables_data.append({
                    'index': i + 1,
                    'dataframe': table_df_clean,
                    'rows': len(table_df_clean),
                    'columns': len(table_df_clean.columns)
                })
    
    return tables_data