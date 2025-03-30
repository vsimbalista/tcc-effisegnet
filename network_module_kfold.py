import lightning as L                   # treinamento de modelos em pytorch
import torch                            # bib principal para tensores e rede neural     
from hydra.utils import instantiate     # utilitário do hydra, que permite criar objetos (otimizador, scheduler) com configs personalizadas
from monai import metrics as mm         # biblioteca MONAI para métricas específicas de segmentação médica (Dice, IoU)

# Implementação de classe Net que utiliza pytorch lightning para treinar, validar e testar modelo de segmentação
class Net(L.LightningModule):        
    def __init__(self, model, criterion, optimizer, lr, scheduler=None):
        super().__init__()
        self.model = model  # Modelo de segmentação

        # Métricas
        self.get_accuracy = mm.ConfusionMatrixMetric(include_background=False, metric_name="accuracy")
        self.get_recall = mm.ConfusionMatrixMetric(include_background=False, metric_name="sensitivity")
        self.get_precision = mm.ConfusionMatrixMetric(include_background=False, metric_name="precision")
        self.get_iou = mm.MeanIoU(include_background=False)
        
        self.get_train_accuracy = mm.ConfusionMatrixMetric(include_background=False, metric_name="accuracy")
        
        self.criterion = criterion  # Função de perda
        self.optimizer = optimizer  # Otimizador
        self.scheduler = scheduler  # Scheduler
        self.lr = lr                # Taxa de aprendizado

    def forward(self, x):       # Método forward pass: simplesmente aplica o modelo à entrada x
        return self.model(x)

    def configure_optimizers(self): 
        optimizer = instantiate(self.optimizer, self.parameters(), lr=self.lr)  # Cria o otimizador e o scheduler usando instantiate do Hydra.
        if self.scheduler:                                                      # permite criação dinamica com configurações definidas
            return {
                "optimizer": optimizer,
                "lr_scheduler": instantiate(self.scheduler, optimizer=optimizer),
                "monitor": "val_loss",
            }
        return optimizer

    def training_step(self, batch, batch_idx):  # batch: lote de dados (imagens e máscaras)
        x, y = batch                            # divide imagens (x) e máscaras (y)
        logits = self(x)
        
        loss = self.criterion(logits, y) # Perda
        self.log("train_loss", loss)
                
        preds = (torch.sigmoid(logits) > 0.5).long() #preds
        self.get_train_accuracy(preds, y)    # Acumula a Acurácia
        
        return loss
    
    def test_step(self, batch, batch_idx):  # Mesmo funcionamento do step de validação!
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        self.log("test_loss", loss)         # Registra perda

        preds = (torch.sigmoid(logits) > 0.5).long()    # Aplica func de ativação e binariza
        self.get_accuracy(preds, y)  # Acumula a Acurácia
        self.get_recall(preds, y)
        self.get_precision(preds, y)
        self.get_iou(preds, y)

        return loss
    
    def on_train_epoch_end(self):
        accuracy = self.get_train_accuracy.aggregate()[0].item()  # Agrega a Acurácia da época
        self.log("train_accuracy", accuracy)  # Registra Acurácia
        self.get_train_accuracy.reset()
    
    def on_test_epoch_end(self):
        accuracy = self.get_accuracy.aggregate()[0].item()    # Cálculo das métricas finais para a época
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()
        iou = self.get_iou.aggregate().item()
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        self.log("test_accuracy", accuracy)  # Registra Acurácia
        self.log("test_recall", recall)
        self.log("test_precision", precision)
        self.log("test_iou", iou)
        self.log("test_f1", f1_score)

        self.get_accuracy.reset()
        self.get_recall.reset()
        self.get_precision.reset()
        self.get_iou.reset()
