import hydra    # Configuração do experimento, gerenciando parâmetros em forma hierárquica
import lightning as L   # treinamento e validação de modelos pytorch
import torch    # Bib principal para computação em tensores
from hydra.utils import instantiate
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner   # Encontra automaticamente maior tamanho de lote, otimiza hiperparâmetros, pode monitorar métricas durante treinamento
from monai.networks.nets.efficientnet import get_efficientnet_image_size    # MONAI: bib para imagens médicas: obtem tamanho da imagem adequada para EfficientNet

from sklearn.model_selection import KFold, train_test_split
import json
import numpy as np  # Para conversão de arrays em listas

from datamodule_kfold import KvasirSEGDataset
from history_kfold_val.network_module_val import Net

L.seed_everything(42, workers=True)             # Semente global para garantir resultados reprodutíveis
torch.set_float32_matmul_precision("medium")    # Precisão de multiplicação de matrizes para melhor desempenho em GPUs


@hydra.main(config_path="config", config_name="config", version_base=None)  # Define a função principal do script, que serpa executada com a "config" fornecida
def main(cfg):                                                              # cfg: objeto estruturado de forma hierarquica
    logger = loggers.TensorBoardLogger("logs/", name=str(cfg.run_name))     # Cria um logger do TensorBoard para monitorar o treinamento e registrar métricas

    if cfg.img_size == "derived":
        img_size = get_efficientnet_image_size(cfg.model.object.model_name)    # Obtem o tamanho adequado a partir do modelo se o tamanho for "derived"
    else:
        img_size = cfg.img_size     # Caso contrário, usa o tamanho especificado na configuração

    dataset = KvasirSEGDataset(batch_size=cfg.batch_size, img_size=img_size) # Instancia o DataSet definido em datamodule.py
    dataset.setup() # Force setup to access dataset length
    
    indices = list(range(len(dataset.dataset)))

    # Step 1: Perform K-Fold on the remaining 90% (train + val)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_metrics = []
        
    for fold, (train_indices, val_test_indices) in enumerate(kf.split(indices)):
        print(f"Starting fold {fold + 1}...")

        # Step 2: Split val_test data in half (5% val + 5% test) vs 90% (train)
        val_indices, test_indices = train_test_split(
            val_test_indices,
            test_size=0.5,
            random_state=42)

        # Update dataset with the current fold's train and validation indices
        dataset.set_splits(train_indices, val_indices, test_indices) #test trasnform will be fixed
        
        # Instanciar modelo e dependências para cada fold
        model = instantiate(cfg.model.object)
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
        test_metrics = trainer.test(net, dataset)      # Avalia o modelo usando o conjunto de dados
        
        # Log metrics
        all_metrics.append({
            "fold": fold + 1,
            "test_metrics": test_metrics,
            "split": {
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "n_test": len(test_indices)
            },
            # "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices
        })

    # Converter ndarrays para listas antes de salvar no arquivo JSON
    for metrics in all_metrics:
        # metrics["train_indices"] = np.array(metrics["train_indices"]).tolist()
        metrics["val_indices"] = np.array(metrics["val_indices"]).tolist()
        metrics["test_indices"] = np.array(metrics["test_indices"]).tolist()
        metrics["test_metrics"] = [metric.tolist() if isinstance(metric, np.ndarray) else metric for metric in metrics["test_metrics"]]

    # Save all metrics to a file
    with open(f"results/kfold_results_{str(cfg.run_name)}.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Cross-validation completed! Results saved to cross_val_results.json")


if __name__ == "__main__":  # Finaliza a definição e chama função principal quando script é executado diretamente
    main()
