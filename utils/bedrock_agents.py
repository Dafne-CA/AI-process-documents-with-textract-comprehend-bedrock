# utils/bedrock_agents.py
import boto3
import uuid
import json
from botocore.exceptions import ClientError
import os

def get_bedrock_agent_client(region_name="us-east-1"):
    """Obtiene el cliente de Bedrock Agent Runtime"""
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name
    )
    return session.client('bedrock-agent-runtime')

def invoke_agent_legacy(agent_id, agent_alias_id, session_id, input_text):
    """Versión que SÍ funciona - forzando respuesta en español"""
    try:
        bedrock_agent = get_bedrock_agent_client()
        
        # Añadir instrucción específica para español al inicio del input
        spanish_instruction = "Por favor, responde SOLAMENTE en español. "
        enhanced_input = spanish_instruction + input_text
        
        # Invocar el agente
        response = bedrock_agent.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=enhanced_input
        )
        
        # Procesar la respuesta
        full_response = ""
        for event in response['completion']:
            if 'chunk' in event and 'bytes' in event['chunk']:
                chunk_text = event['chunk']['bytes'].decode('utf-8')
                full_response += chunk_text
        
        return full_response
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return f"Error AWS ({error_code}): {error_message}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"