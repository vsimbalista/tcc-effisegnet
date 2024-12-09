import math

import torch
import torch.nn as nn   # Módulos para criação de camadas de redes neurais
from monai.networks.nets import EfficientNetBNFeatures                      # EfficientNet: carrega recursos de modelo pré-treinado
from monai.networks.nets.efficientnet import get_efficientnet_image_size    # EfficientNet: retorna tamanho da imagem adequado


class GhostModule(nn.Module):   # Inovação que reduz redundância de cálculos ao criar representações "fantasmas" (menos custosas) de convoluções mais complexas
    def __init__(               # É montada a partir de uma convolução padrão seguida por uma mais simples (depthwise)
        self,
        in_channels,            # n de canais de entrada da camada conv
        out_channels,           # n de canais de saída da camada conv
        kernel_size=1,          # tamanho do kernel para 1ª convolução
        ratio=2,                # controla quantos canais são produzidos pela convolução "barata" vs convolução primária (aqui 50%)
        dw_size=3,              # tamanho do kernel para convolução "barata" (depthwise)
        stride=1,               # passo da convolução
        relu=True,
    ):
        super(GhostModule, self).__init__()                 # Definindo as operações convolucionais
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)     # calcula qtd de canais que a convolução principal precisa gerar
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(      # Convolução 2D padrão
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), # Func de ativação ReLU
            nn.BatchNorm2d(init_channels),                      # Normalização em batches
        )

        self.cheap_operation = nn.Sequential(   # Convolução depthwise: gera "cópias baratas" (canais) da conv principal (economia de memória e computação)
            nn.Conv2d(                          # Na convolução depthwise, cada filtro convolui separadamente em um único canal
                init_channels,                  # O resultado são os "canais fantasmas"
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), # Func de ativação ReLU
            nn.BatchNorm2d(new_channels),                       # Normalização em batches
        )

    def forward(self, x):                       # Forward pass: 
        x1 = self.primary_conv(x)                   # Primeiro aplica convolução principal (x1)
        x2 = self.cheap_operation(x1)               # Depois aplica convolução "barata" (x2)
        out = torch.cat([x1, x2], dim=1)            # Concatena as duas saídas ao longo do eixo dos canais
        return out[:, : self.out_channels, :, :]    # Seleciona os primeiros canais, que são o número exato de out_channels


class EffiSegNetBN(nn.Module):  # Classe que define a arquitetura de segmentação baseada no EfficientNet (herda de Module)
    def __init__(       # método __init__ apenas prepara os objetos da camada para serem usados posteriormente
        self,
        ch=64,                          # Nº de canais de saída que a rede vai processar
        pretrained=True,                # Modelo sendo carregado com pesos PRÉ-TREINADOS !!!
        freeze_encoder=False,           # Não congelou os pesos da EfficientNet durante o treinamento, apenas foi inicializado pré-treinado!
        deep_supervision=False,         # Habilita supervisão profunda para usar múltiplas saídas intermediárias (útil em treinamento)
        model_name="efficientnet-b0",
    ):
        super(EffiSegNetBN, self).__init__()
        self.model_name = model_name
        self.encoder = EfficientNetBNFeatures(  # Aqui está usando o EfficientNetBNFeatures paracarregar recursos do modelo pré-treinado
            model_name=model_name,              # Esse é o ENCODER
            pretrained=pretrained,              # Envia o argumento pretrained=True
        )

        # remove unused layers
        del self.encoder._avg_pooling   # As camadas finais do EfficientNet são projetadas para tarefas de classificação
        del self.encoder._dropout       # Como o objetivo é segmentação, as camadas não necessárias são removidas
        del self.encoder._fc

        # extract the last number from the model name, example: efficientnet-b0 -> 0
        b = int(model_name[-1])

        num_channels_per_output = [     # Selecionando a configuração adequada de canais para diferentes estágios de saída da EfficientNet
            (16, 24, 40, 112, 320),     # efficientnet-b0 -> 0
            (16, 24, 40, 112, 320),     # efficientnet-b1 -> 1
            (16, 24, 48, 120, 352),     # efficientnet-b2 -> 2
            (24, 32, 48, 136, 384),     # ...
            (24, 32, 56, 160, 448),
            (24, 40, 64, 176, 512),
            (32, 40, 72, 200, 576),
            (32, 48, 80, 224, 640),
            (32, 56, 88, 248, 704),
            (72, 104, 176, 480, 1376),
        ]

        channels_per_output = num_channels_per_output[b]    # Cada camada da EfficientNet gera um n diferente de canais
                                                            # Aqui se definiu quantos serão usados para cada camada
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False                 # Se os pesos forem congelados, trava os parâmetros

        self.deep_supervision = deep_supervision

        upsampled_size = get_efficientnet_image_size(model_name)
        self.up1 = nn.Upsample(size=upsampled_size, mode="nearest")     # Para combinar saídas de diferentes escalas (resoluções),
        self.up2 = nn.Upsample(size=upsampled_size, mode="nearest")     # cada uma é aumentada para o tamanho original da imagem (ideal).
        self.up3 = nn.Upsample(size=upsampled_size, mode="nearest")     # Isso é feito com nn.Upsample, que usa interpolação do tipo "nearest neighbor".
        self.up4 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up5 = nn.Upsample(size=upsampled_size, mode="nearest")

        # Convoluções e Batch Normalization
        self.conv1 = nn.Conv2d(                                                             # Cada uma das saídas do encoder passa por uma convolução
            channels_per_output[0], ch, kernel_size=3, stride=1, padding=1, bias=False      
        )
        self.bn1 = nn.BatchNorm2d(ch)                                                       # Seguida de normalização em lotes para ajustar o numero de canais para ch

        self.conv2 = nn.Conv2d(
            channels_per_output[1], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(ch)

        self.conv3 = nn.Conv2d(
            channels_per_output[2], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(ch)

        self.conv4 = nn.Conv2d(
            channels_per_output[3], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(ch)

        self.conv5 = nn.Conv2d(
            channels_per_output[4], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(ch)

        self.relu = nn.ReLU(inplace=True)

        if self.deep_supervision:
            self.conv7 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn7 = nn.BatchNorm2d(ch)
            self.conv8 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn8 = nn.BatchNorm2d(ch)
            self.conv9 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn9 = nn.BatchNorm2d(ch)
            self.conv10 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn10 = nn.BatchNorm2d(ch)
            self.conv11 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn11 = nn.BatchNorm2d(ch)

        self.bn6 = nn.BatchNorm2d(ch)
        self.ghost1 = GhostModule(ch, ch)   # Definição dos módulos fantasma
        self.ghost2 = GhostModule(ch, ch)

        self.conv6 = nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0, bias=False)   # Convolução final (gera a máscara)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)    # Chama o ENCODER !
                                                # A EfficientNet gera múltiplas saídas em diferentes escalas (x0, x1, x2, x3, x4). Essas saídas correspondem a diferentes estágios da rede.
        x0 = self.conv1(x0)     # Convolução
        x0 = self.relu(x0)      # Ativação ReLU
        x0 = self.bn1(x0)       # Normaização de batches

        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.bn2(x1)

        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.bn3(x2)

        x3 = self.conv4(x3)
        x3 = self.relu(x3)
        x3 = self.bn4(x3)

        x4 = self.conv5(x4)
        x4 = self.relu(x4)
        x4 = self.bn5(x4)

        x0 = self.up1(x0)   # DECODER: upsamplings
        x1 = self.up2(x1)   # Saídas são aumentadas para a resolução ideal usando Upsample
        x2 = self.up3(x2)
        x3 = self.up4(x3)
        x4 = self.up5(x4)

        # Soma de todas as saídas para formar uma única saída X                            
        x = x0 + x1 + x2 + x3 + x4  
        x = self.bn6(x)             # Resultado é normalizado por lotes
        x = self.ghost1(x)          # Depois passa por dois módulos fantasma para reduzir complexidade computacional
        x = self.ghost2(x)
        x = self.conv6(x)           # Convolução final (conv6) que gera a máscara final de segmentação (1 canal de saída)

        if self.deep_supervision:
            x0 = self.bn7(x0)
            x0 = self.conv7(x0)

            x1 = self.bn8(x1)
            x1 = self.conv8(x1)

            x2 = self.bn9(x2)
            x2 = self.conv9(x2)

            x3 = self.bn10(x3)
            x3 = self.conv10(x3)

            x4 = self.bn11(x4)
            x4 = self.conv11(x4)

            return x, [x0, x1, x2, x3, x4]

        return x
