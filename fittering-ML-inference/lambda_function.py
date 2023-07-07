import sys
# from PIL import Image
# from predict import predict
# def handler2(event, context):
# 	front_image = Image.open("./images/front.jpg")
# 	side_image = Image.open("./images/side.jpg")
# 	meas = predict(front_image, side_image, 1.81, 
# 		model_cnn_ckpt_path='./weights/CNNForward.pth',
#                segmentation_ckpt_path='./weights/SGHM-ResNet50.pth')
# 	return str(meas['height'])

def handler(event, context):
	return 'hello'

if __name__ == "__main__":
	print(handler(None, None))
