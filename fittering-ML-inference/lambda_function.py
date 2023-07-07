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
	# front_image = Image.open("./images/front.jpg")
	# side_image = Image.open("./images/side.jpg")
	front_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=FRONT_FILE_NAME)['Body'])
	side_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=SIDE_FILE_NAME)['Body'])
	meas = predict(front_image, side_image, 1.81, 
		model_cnn_ckpt_path='./weights/CNNForward.pth',
               segmentation_ckpt_path='./weights/SGHM-ResNet50.pth')
	return {
		'statusCode': 200,
		'body': str(meas)
	}


# def handler(event, context):
# 	data = s3.get_object(Bucket=BUCKET_NAME, Key=FRONT_FILE_NAME)
# 	res = data['Body']
# 	img = Image.open(res)
# 	return {
# 		'statusCode': 200,
# 		'body': img
# 	}

if __name__ == "__main__":
	print(handler(None, None))
