import os
from collections import Counter

# Função para listar imagens, processar nomes e identificar duplicatas
def analisar_imagens(pasta):
    nomes_transformados = []

    # Vasculhar a pasta e processar nomes de arquivos de imagens
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith(('.jpg', '.png', '.jpeg')):  # Verifica extensões de imagem
            # Pega o conteúdo antes do primeiro hífen
            nome_processado = arquivo.split('-')[0]
            nomes_transformados.append(nome_processado)

    # Contar duplicatas
    contagem = Counter(nomes_transformados)
    duplicatas = {k: v for k, v in contagem.items() if v > 1}
    unicos = {k: v for k, v in contagem.items() if v == 1}

    # Soma total de duplicatas
    soma_duplicatas = sum(duplicatas.values())

    # Resultado
    print(f"Total de registros únicos: {len(unicos)}")
    print(f"Total de duplicatas distintas: {len(duplicatas)}")
    print(f"Duplicatas detalhadas: {duplicatas}")
    print(f"Soma total das duplicatas: {soma_duplicatas}")
    return nomes_transformados, duplicatas, unicos

# Caminho para a pasta contendo as imagens
caminho_pasta = "imagens_cancer_boca/images"

# Chamar a função
nomes_transformados, duplicatas, unicos = analisar_imagens(caminho_pasta)
