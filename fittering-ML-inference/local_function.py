import sys
from PIL import Image
from predict import predict
import boto3
import json
import pandas as pd

BUCKET_NAME = "fittering-measurements-images"
FRONT_FILE_NAME = "front.jpg"
SIDE_FILE_NAME = "side.jpg"

s3_key = pd.read_csv('/home/shin/Documents/aws_access_key/s3AccessUser_accessKeys.csv')

s3 = boto3.client('s3', 
	    aws_access_key_id=s3_key['Access key ID'][0],
        aws_secret_access_key=s3_key['Secret access key'][0])

def handler(event, context):
	front_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=FRONT_FILE_NAME)['Body'])
	side_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=SIDE_FILE_NAME)['Body'])
	meas = predict(front_image, side_image, 1.81, 
		model_cnn_ckpt_path='./weights/CNNForward.pth',
               segmentation_ckpt_path='./weights/SGHM-ResNet50.pth')
	return {
		'statusCode': 200,
		'body': str(meas)
	}

if __name__ == "__main__":
	print(handler(None, None))
