FROM sinjy1203/fittering-meas:2.0

COPY ./ ${LAMBDA_TASK_ROOT}
RUN ls -la ${LAMBDA_TASK_ROOT}/*
# COPY requirements.serving.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
# RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
# RUN pip install -r requirements.serving.txt
# RUN yum update && yum install -y libglvnd-glx

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]