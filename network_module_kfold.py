import lightning as L                   # treinamento de modelos em pytorch
import torch                            # bib principal para tensores e rede neural     
from hydra.utils import instantiate     # utilitário do hydra, que permite criar objetos (otimizador, scheduler) com configs personalizadas
from monai import metrics as mm         # biblioteca MONAI para métricas específicas de segmentação médica (Dice, IoU)

# Implementação de classe Net que utiliza pytorch lightning para treinar, validar e testar modelo de segmentação
class Net(L.LightningModule):
    def __init__(self, model, criterion, optimizer, lr, scheduler=None):
        super().__init__()
        self.model = model                                          # O modelo de segmentação a ser treinado

        self.get_dice = mm.DiceMetric(include_background=False)     # Dice val/test
        self.get_iou = mm.MeanIoU(include_background=False)         # IoU
        
        self.get_recall = mm.ConfusionMatrixMetric(                 # Recall
            include_background=False, metric_name="sensitivity"
        )
        self.get_precision = mm.ConfusionMatrixMetric(              # Precisão
            include_background=False, metric_name="precision"
        )
        self.get_accuracy = mm.ConfusionMatrixMetric(               # Acurácia (val/test)
            include_background=False, metric_name="accuracy"
        )

        self.get_train_accuracy = mm.ConfusionMatrixMetric(         # Acurácia treino
            include_background=False, metric_name="accuracy"
        )
        self.get_train_dice = mm.DiceMetric(include_background=False)     # Dice train
        
        self.criterion = criterion  # Criterion: Função de perda (loss)
        self.optimizer = optimizer  # Otimizador
        self.scheduler = scheduler  # Scheduler de aprendizado
        self.lr = lr                # Learning rate

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

        if self.model.deep_supervision:         # Para modelos de supervisão profunda, a perda principal e auxiliar são combinadas para treinar o modelo
            logits, logits_aux = self(x)        # Logits: resultados da camada de saída da rede antes da aplicação da função de ativação final (confiança do modelo em cada classe)

            aux_loss = sum(self.criterion(z, y) for z in logits_aux)
            loss = (self.criterion(logits, y) + aux_loss) / (1 + len(logits_aux))   # Perda principal somada com auxiliar

            self.log("train_loss", loss)    # Registro da perda de treinamento
            return loss

        logits = self(x)
        
        loss = self.criterion(logits, y) # Perda
        self.log("train_loss", loss)
                
        preds = (torch.sigmoid(logits) > 0.5).long() #preds
        self.get_train_dice(preds, y)        # Acumula o Dice
        self.get_train_accuracy(preds, y)    # Acumula a Acurácia
        
        return loss

    def validation_step(self, batch, batch_idx):    # Calcula a perda e métricas de validação
        x, y = batch

        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)

        loss = self.criterion(logits, y)
        self.log("val_loss", loss)          # Registra perda

        preds = (torch.sigmoid(logits) > 0.5).long()    # Normaliza logits com sigmoid e faz predição binária !
        self.get_accuracy(preds, y)  # Acumula a Acurácia
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

        return loss
    
    def test_step(self, batch, batch_idx):  # Mesmo funcionamento do step de validação!
        x, y = batch

        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)

        loss = self.criterion(logits, y)
        self.log("test_loss", loss)         # Registra perda

        preds = (torch.sigmoid(logits) > 0.5).long()    # Aplica func de ativação e binariza
        self.get_accuracy(preds, y)  # Acumula a Acurácia
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

        return loss
    
    def on_train_epoch_end(self):
        # print("\nEnd of training epoch")  # Debugging
        accuracy = self.get_train_accuracy.aggregate()[0].item()  # Agrega a Acurácia da época
        dice = self.get_train_dice.aggregate().item()     # Agrega o Dice Score da época
        
        self.log("train_dice", dice)       # Registra Dice Score
        self.log("train_accuracy", accuracy)  # Registra Acurácia
    
        self.get_train_dice.reset()     # Reseta os valores acumulados para a próxima época
        self.get_train_accuracy.reset()

    def on_validation_epoch_end(self):
        accuracy = self.get_accuracy.aggregate()[0].item()
        dice = self.get_dice.aggregate().item()     # Cálculo das métricas finais para a época
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()

        self.log("val_accuracy", accuracy)  # Registra Acurácia
        self.log("val_dice", dice)      # Registra as métricas no log para a época
        self.log("val_iou", iou)
        self.log("val_recall", recall)
        self.log("val_precision", precision)
        self.log("val_f1", 2 * (precision * recall) / (precision + recall + 1e-8))

        self.get_accuracy.reset()
        self.get_dice.reset()       # Reseta as métricas para a próxima época
        self.get_iou.reset()
        self.get_recall.reset()
        self.get_precision.reset()
    
    def on_test_epoch_end(self):    # Mesmo funcionamento da anterior, mas para teste!
        accuracy = self.get_accuracy.aggregate()[0].item()    
        dice = self.get_dice.aggregate().item()
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()
        
        self.log("test_accuracy", accuracy)  # Registra Acurácia
        self.log("test_dice", dice)
        self.log("test_iou", iou)
        self.log("test_recall", recall)
        self.log("test_precision", precision)
        self.log("test_f1", 2 * (precision * recall) / (precision + recall + 1e-8))

        self.get_accuracy.reset()
        self.get_dice.reset()
        self.get_iou.reset()
        self.get_recall.reset()
        self.get_precision.reset()
