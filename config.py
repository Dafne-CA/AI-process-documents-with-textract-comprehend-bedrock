import os
from dotenv import load_dotenv

load_dotenv()

# AWS credentials y región
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # default por si no está en .env
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Comprehend endpoint ARN (si ya creaste el endpoint real-time)
COMPREHEND_ENDPOINT_ARN = os.getenv("COMPREHEND_ENDPOINT_ARN")

# Bedrock Supervisor Agent
SUPERVISOR_ALIAS_ID = os.getenv("SUPERVISOR_ALIAS_ID")  # ej: arn:aws:bedrock:us-east-1:607520774564:agent/alias/ATNHUVR3WV
SUPERVISOR_AGENT_ID = os.getenv("SUPERVISOR_AGENT_ID")    # ej: KTBA6VBCKD
