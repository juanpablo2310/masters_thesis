import torch


class Evaluator():
    
    def __init__(self, model, data, device, totalLabels = None,targets = None):
        self.model = model
        self.data = data
        self.device = device
        self.totalLabels = totalLabels
        self.targets = targets
    
    def evaluate(self):
        ...
    def predict(self):
        ...


class TensorflowEvaluator(Evaluator):  
    def __init__(self, model, data, device,totalLabels = None,targets = None):
        super().__init__(model, data, device,totalLabels,targets)  
    
    def evaluate(self):
        return self.model.evaluate(self.data, self.targets, verbose=0)

    def predict(self):
        predictions = self.model.predict(self.data)
        return predictions      

class PytorchEvaluator(Evaluator):

    def __init__(self, model, data, device,totalLabels = None,targets = None):
        super().__init__(model, data, device,totalLabels,targets) 
    
    def predict(self):
        self.model.eval()
        with torch.no_grad():
            # whole_predictions = []
            # for data, _ in self.data_loader:
            data = data.to(torch.float32).to(self.device)
            network = self.model.to(torch.float32).to(self.device)
            output = network(data)
                # whole_predictions.append(output)
        return output
    
    def evaluate(self):
        # predictions = self.predict()
        # eval = predictions - self.targets
        # return eval
        ...
    
    def __sub__(self, other):
        ...