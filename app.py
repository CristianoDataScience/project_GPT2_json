import json
from transformers import pipeline
import streamlit as st
import plotly.express as px
import pandas as pd

# Função para interpretar dados de casos de dengue em Goiânia
def interpretar_dados_dengue(caminho_json, modelo="t5-small", max_length=100):
    generator = pipeline("text-generation", model=modelo)

    # Carregue o arquivo JSON diretamente
    conteudo_json = caminho_json.read()
    data = json.loads(conteudo_json)

    anos = data["ano"]
    meses = data["mes"]
    casos = data["casos"]
    

    # Encontre o mês com a maior variação nos casos
    maior_variacao_mes = meses[casos.index(max(casos))]

    resultados = []

    for i in range(len(anos)):
        # Corrija o texto da frase
        texto_json = f"Em {meses[i]} de {anos[i]}, houve {casos[i]} casos de dengue em Goiânia."

        # Destaque o mês com a maior variação de casos
        if meses[i] == maior_variacao_mes:
            texto_json += " Este mês apresentou uma sazonalidade acentuada."

        # Remova caracteres extras
        texto_json = " ".join(texto_json.split())

        texto_natural = generator(texto_json, max_length=max_length, do_sample=True, temperature=0.7)
        resultado = texto_natural[0]["generated_text"]

        # Pós-processamento para corrigir erros nos textos
        resultado = resultado.replace("....", "...").replace("....", "...").replace("...", "..")
        resultado = resultado.replace("......", "...").replace(".....", "...").replace("..", ".")
        resultado = resultado.replace(" ,", ",").replace("  ", " ").replace(" .", ".")

        resultados.append(resultado)

    return resultados, meses, casos

def interpretar_dados_json(caminho_json):
    # Carregue o arquivo JSON diretamente
    conteudo_json = caminho_json.read()
    
    try:
        data = json.loads(conteudo_json)
    except json.JSONDecodeError as e:
        st.error("Erro ao carregar o JSON: " + str(e))
        return []

    if "metadados" in data:
        # É um JSON de outro tipo, como os exemplos 2, 3, ou 4
        titulo = data["metadados"]["titulo"]
        temporalidade = data["metadados"]["temporalidade"]
        eixo_x = data["eixo_x"]
        series = data["series"]

        resultados = []

        # Crie mensagens interpretativas com base nos dados
        for serie, valores in series.items():
            mensagem = f"No contexto de {titulo} ({temporalidade}), os valores são:"
            for i in range(len(eixo_x)):
                mensagem += f"\n- Em {eixo_x[i]}, a porcentagem foi {valores[i]}."
            resultados.append(mensagem)

        return resultados
    else:
        return ["JSON não reconhecido. Por favor, forneça um JSON válido."]

# Streamlit app
st.title("Interpretação de Dados")

caminho_arquivo_json = st.file_uploader("Carregar arquivo JSON", type=["json"])

if caminho_arquivo_json:
    resultados_interpretados = interpretar_dados_json(caminho_arquivo_json)

    if resultados_interpretados:
        st.subheader("Mensagens:")
        for resultado in resultados_interpretados:
            st.write(resultado)
