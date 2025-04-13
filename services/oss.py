from aiohttp.hdrs import CONTENT_DISPOSITION
from oss2.credentials import EnvironmentVariableCredentialsProvider
import oss2
import logging
import time
import os
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import logging
from typing import BinaryIO
import random
from itertools import islice
import datetime
from constant import (
    OSS_REGION,
    OSS_ENDPOINT,
    OSS_BUCKET_NAME,
    COS_BUCKET_NAME,
    COS_REGION,
    COS_SECRET_ID,
    COS_SECRET_KEY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

required_env_vars = ['OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET']
for var in required_env_vars:
    if var not in os.environ:
        logging.error(f"Environment variable {var} is not set.")
        exit(1)

auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
async def oss_upload(filename, data):
    try:
        bucket = oss2.Bucket(auth=auth, endpoint=OSS_ENDPOINT, bucket_name=OSS_BUCKET_NAME, region=OSS_REGION)
        result = bucket.put_object(key=f"images/{filename}", data=data, headers={
            'x-oss-object-acl': 'public-read',
            'x-oss-storage-class': 'Standard',
            'Content-Disposition': 'inline',
            'x-oss-forbid-overwrite': 'true'
        })
        return f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/images/{filename}"
    except Exception as e:
        raise e

cos_config = CosConfig(
    Region=COS_REGION,
    SecretId=COS_SECRET_ID,
    SecretKey=COS_SECRET_KEY,

)
client = CosS3Client(cos_config)
# 上传到腾讯云对象存储
async def cos_upload(filename: str, data:BinaryIO):
    try:
        current_datetime = datetime.date.today()
        year = current_datetime.year
        month = current_datetime.month
        path_key = f"{year}/{month}/{filename}"
        response = client.put_object(
            Bucket=COS_BUCKET_NAME,
            Key=path_key,
            Body=data,
            StorageClass='STANDARD',
            ContentDisposition="inline"
        )
        return f"https://{COS_BUCKET_NAME}.cos.{COS_REGION}.myqcloud.com/{path_key}"
    except Exception as e:
        raise e



