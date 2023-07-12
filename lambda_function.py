import sys
from PIL import Image
from predict import predict
import boto3
import json

BUCKET_NAME = "fittering-measurements-images"
FRONT_FILE_NAME = "front.jpg"
SIDE_FILE_NAME = "side.jpg"
s3 = boto3.client('s3')

def handler(event, context):
	front_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=FRONT_FILE_NAME)['Body'])
	side_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=SIDE_FILE_NAME)['Body'])
	height = event['body-json']['height']
	meas = predict(front_image, side_image, height)
	return {
		'statusCode': 200,
		'body': meas
	}

if __name__ == "__main__":
	print(handler(None, None))
