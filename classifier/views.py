# image_classifier_app/views.py
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import torch
import torchvision.transforms as transforms

class ImageClassification(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        # Load the model
        model = self.load_model()

        # Get the uploaded image
        uploaded_image = request.data['image']
        img = Image.open(uploaded_image)

        # Preprocess the image
        img = self.preprocess_image(img)

        # Make a prediction
        with torch.no_grad():
            output = model(img)

        # Post-process the output to get label and probability
        label, probability = self.postprocess_output(output)

        return Response({'label': label, 'probability': probability}, status=status.HTTP_200_OK)

    def load_model(self):
        model_path = 'my_model.pth'  # Adjust this to the actual path of your model
        model = torch.load(model_path)
        model.eval()
        return model

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)
        return image

    def postprocess_output(self, output):
        # Implement your post-processing logic here
        # For example, use softmax and get the class with the highest probability
        return "Class", 0.95  # Replace with actual label and probability

