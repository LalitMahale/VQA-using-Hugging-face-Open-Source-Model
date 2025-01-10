import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from io import BytesIO

class Multimodal:
    def __init__(self,model_name:str="dandelin/vilt-b32-finetuned-vqa"):
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.qa_model = ViltForQuestionAnswering.from_pretrained(model_name)

    def process(self,image,text):
        try:         
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))

            # Ensure that image is now in PIL format and text is a string
            if not isinstance(image, Image.Image):
                raise ValueError("Image is not a valid PIL.Image.Image object.")
            if not isinstance(text, str):
                raise ValueError("Text is not a valid string.") 
            
            encoding = self.processor(image,text,return_tensors="pt")

            outputs = self.qa_model(**encoding)

            logits = outputs.logits

            ids = logits.argmax(-1).item()

            answer = self.qa_model.config.id2label[ids]

            return answer



        except Exception as e:
            return str(e)
