import hydra    # Configuração do experimento, gerenciando parâmetros em forma hierárquica
import lightning as L   # treinamento e validação de modelos pytorch
import torch    # Bib principal para computação em tensores
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner   # Encontra automaticamente maior tamanho de lote, otimiza hiperparâmetros, pode monitorar métricas durante treinamento
from monai.networks.nets.efficientnet import get_efficientnet_image_size    # MONAI: bib para imagens médicas: obtem tamanho da imagem adequada para EfficientNet

from sklearn.model_selection import KFold, train_test_split
import json

from datamodule_kfold import KvasirSEGDataset
from network_module import Net

L.seed_everything(42, workers=True)             # Semente global para garantir resultados reprodutíveis
torch.set_float32_matmul_precision("medium")    # Precisão de multiplicação de matrizes para melhor desempenho em GPUs


@hydra.main(config_path="config", config_name="config", version_base=None)  # Define a função principal do script, que serpa executada com a "config" fornecida
def main(cfg):                                                              # cfg: objeto estruturado de forma hierarquica
    logger = loggers.TensorBoardLogger("logs/", name=str(cfg.run_name))     # Cria um logger do TensorBoard para monitorar o treinamento e registrar métricas
    model = instantiate(cfg.model.object)   # cria uma instância do modelo definido na configuração. deve especificar o tipo e modelo a ser usado.
    if cfg.img_size == "derived":
        img_size = get_efficientnet_image_size(model.model_name)    # Obtem o tamanho adequado a partir do modelo se o tamanho for "derived"
    else:
        img_size = cfg.img_size     # Caso contrário, usa o tamanho especificado na configuração


    dataset = KvasirSEGDataset(batch_size=cfg.batch_size, img_size=img_size) # Instancia o DataSet definido em datamodule.py
    dataset.setup() # Force setup

    # Step 1: Split data into 90% (train + val) and 10% (test)
    train_val_indices, test_indices = train_test_split(
        list(range(len(dataset.dataset))),
        test_size=0.1,
        random_state=42)
    
    # Update dataset to hold the fixed test set
    # dataset.set_test_split(test_indices)

    # Save the test indices for later use
    with open("test_indices.json", "w") as f:
        json.dump(test_indices, f)
        
    # Step 2: Perform K-Fold on the remaining 90% (train + val)
    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        print(f"Starting fold {fold + 1}...")
        
        # Convert train/val indices back to the full dataset's indexing
        train_indices = [train_val_indices[i] for i in train_idx]
        val_indices = [train_val_indices[i] for i in val_idx]

        # Update dataset with the current fold's train and validation indices
        dataset.set_splits(train_indices, val_indices, test_indices) #test will be fixed
        
        net = Net(          # Instancia o modelo de treinamento do network_module
            model=model,
            criterion=instantiate(cfg.criterion),   # Criterion: função perda
            optimizer=cfg.optimizer,                # Otimizador
            lr=cfg.lr,                              # Learning rate
            scheduler=cfg.scheduler,                # Scheduler
        )
    
        trainer = instantiate(cfg.trainer, logger=logger)   # Cira objeto trainer para gerenciar o ciclo de train / val
    
        # if efficientnetb5, b6, or b7, use binsearch to find the largest batch size
        if cfg.model.object.model_name in ["efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]:    # Aplicando tuner para encontrar maior valor de lote adequado
            tuner = Tuner(trainer)
            tuner.scale_batch_size(net, dataset, mode="binsearch")
        
        trainer.fit(net, dataset)       # Inicia treinamento do modelo
        test_metrics = trainer.test(net, dataset, test_indices=test_indices)      # Avalia o modelo usando o conjunto de dados
        
        # Log metrics
        all_metrics.append({
            "fold": fold + 1,
            "test_metrics": test_metrics,
        })

    # Save all metrics to a file
    with open("cross_val_results.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Cross-validation completed! Results saved to cross_val_results.json")


if __name__ == "__main__":  # Finaliza a definição e chama função principal quando script é executado diretamente
    main()
