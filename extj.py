import boto3

import os

import tarfile

from botocore import UNSIGNED

from botocore.config import Config



# Define S3 bucket and prefix

bucket_name = 'openneuro'

prefix = 'ds000030/ds000030_R1.0.2/compressed/'



# Local directories

download_dir = 'downloads'

extract_dir = 'extracted'

os.makedirs(download_dir, exist_ok=True)

os.makedirs(extract_dir, exist_ok=True)



# Anonymous S3 client using botocore.UNSIGNED

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))



# List and download files

paginator = s3.get_paginator('list_objects_v2')

pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)



for page in pages:

    if 'Contents' in page:

        for obj in page['Contents']:

            key = obj['Key']

            filename = os.path.basename(key)

            if not filename:

                continue  # skip folders



            local_path = os.path.join(download_dir, filename)

            print(f"Downloading {filename}...")

            s3.download_file(bucket_name, key, local_path)



            # Extract the file

            print(f"Extracting {filename}...")

            try:

                with tarfile.open(local_path, 'r:gz') as tar:

                    tar.extractall(path=extract_dir)

            except tarfile.TarError as e:

                print(f"Error extracting {filename}: {e}")



print("✅ All files downloaded and extracted.")


