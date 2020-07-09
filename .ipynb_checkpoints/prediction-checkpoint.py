from main import ImagePipeline

def load_model():
    #load the model
    model = FashionCNN()
    model.load_state_dict(torch.load('fashion_model'))

def classify(file):
    inp = ImagePipeline(file).transform()
    pytorch_input = Variable(torch.from_numpy(inp).view(1,1,28,28).float())
    prediction = int(torch.max(model(pytorch_input),1)[1][0])
    return output_mapping[prediction]