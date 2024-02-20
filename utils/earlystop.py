import torch

class EarlyStopper:
    '''
    function for Early Stopping of learning epochs 
    
    if loss value is not step until patience value 
    your machine will stop after patience
    
    class args:
            patience: patience value of step -> int
            min_delta: min change amount for early stop -> int
    '''
    def __init__(self, patience:int=3, min_delta:int=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.save_counter = 0
        self.max_acc = 0.

    def early_stop(self, model, acc:float, name='test.pth', mode=True):
        if acc > self.max_acc:
            self.counter = 0
            self.max_acc = acc
            torch.save(model.state_dict(), 'output_model/'+name)
        elif acc < (self.max_acc + self.min_delta):
            self.counter += 1  
            if self.counter >= self.patience and mode:
                print('early stoper run!')
                return True
            
        return False