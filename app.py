import io
import streamlit as st
import pandas as pd
import json
import os
import time
import ast
import requests
import concurrent.futures
from anthropic import Anthropic
from anthropic._exceptions import OverloadedError
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)

st.set_page_config(page_title="Validação de Lista com Claude e Visualização Avançada", layout="wide")
st.title("Validação de Lista com Claude e Visualização Avançada")

# Função para parser de verificação (JSON ou string de dict)
def parse_verificacao_text(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}
    txt = x.replace('\\n', '')
    txt = txt.strip()
    try:
        return json.loads(txt)
    except Exception:
        try:
            return ast.literal_eval(txt)
        except Exception:
            return {}

# Cria abas: Processamento e Visualização
tab1, tab2 = st.tabs(["📤 Processar CSV", "📈 Visualizar Resultados"])

# ------------------ TAB 1: Processamento ------------------
with tab1:
    st.header("📤 Processar CSV: Validação de Lista com Claude")

    # Botão de download do XLSX de exemplo
    with open("validacao_teste.xlsx", "rb") as f:
        excel_bytes = f.read()

    st.download_button(
        label="📥 Baixar Excel de Exemplo",
        data=excel_bytes,
        file_name="validacao_teste.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    uploaded_file = st.file_uploader(
        "📤 Envie o arquivo de validação (CSV ou XLSX)",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        # Leitura do arquivo
        if uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Extrai prompts e modelos
        PROMPT_ESCOLHE_PRODUTOS = df["PROMPT_ESCOLHE_PRODUTOS"][0]
        PROMPT_ESTRUTURA_LISTA = df["PROMPT_ESTRUTURA_LISTA"][0]
        MODELO_ESCOLHE_PRODUTOS = df["MODELO_ESCOLHE_PRODUTOS"][0]
        PROMPT_AGENTE_VERIFICACAO = df["PROMPT_AGENTE_VERIFICACAO"][0]

        st.success("Arquivo carregado com sucesso. Iniciando o processamento...")
        progress_bar = st.progress(0)

        # Função de normalização com retry em caso de OverloadedError
def processar_linha(input_texto, retries=3):
    prompt = PROMPT_ESTRUTURA_LISTA.replace("{{input}}", input_texto)
    for attempt in range(retries):
        try:
            resposta = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(resposta.content[0].text).get("lista", [])
        except OverloadedError:
            wait = 2 ** attempt
            time.sleep(wait)
        except Exception:
            return []
    return []

        # Aplica normalização com barra de progresso
        standardized = []
        total = len(df)
        for idx, texto in enumerate(df["LISTA"]):
            standardized.append(processar_linha(texto))
            progress_bar.progress((idx + 1) / total)
            time.sleep(0.5)  # evita hits muito rápidos
        df["standardized_products"] = standardized
        df["unique_products"] = df["standardized_products"].apply(lambda prods: list(dict.fromkeys(prods)))

        # Função de busca de produtos no BFF GraphQL
def search_for_product(product):
    url = 'https://api.trela.com.br/emporium-bff/graphql'
    headers = {"Content-Type": "application/json"}
    graphql_query = f"""
    query searchProducts {{
        search(query: \"{product}\", hubId: 1) {{
            products {{
                nodes {{
                    productId
                    productName
                    supplierName
                }}
            }}
        }}
    }}
    """
    resp = requests.post(url, headers=headers, json={"query": graphql_query})
    if resp.status_code == 200:
        nodes = resp.json().get("data", {}).get("search", {}).get("products", {}).get("nodes", [])
        return (product.lower(), [
            {"name": n.get("productName"), "id": n.get("productId"), "supplier": n.get("supplierName")}
            for n in nodes
        ])
    return (product.lower(), [])

        # Construção da biblioteca de busca em paralelo
def get_products_search_library(unique_products):
    library = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(search_for_product, p) for p in unique_products]
        for fut in concurrent.futures.as_completed(futures):
            key, nodes = fut.result()
            library[key] = nodes
    return library

        df["resultado_search"] = df["unique_products"].apply(get_products_search_library)

        # Fluxo de escolha de IDs com retry
def run_flow_for_product(key, resultado_search, retries=3):
    prompt = PROMPT_ESCOLHE_PRODUTOS.replace("{key}", key)
    prompt = prompt.replace(
        "{json.dumps(resultado_search, indent=2, ensure_ascii=False)}",
        json.dumps(resultado_search, indent=2, ensure_ascii=False)
    )
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODELO_ESCOLHE_PRODUTOS,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            return json.loads(resp.content[0].text.strip())
        except OverloadedError:
            time.sleep(2 ** attempt)
        except Exception:
            break
    return ["NOT_FOUND"]

        def escolhe_produtos(resultado_search: dict):
    output = {}
    for key, res in resultado_search.items():
        output[key] = run_flow_for_product(key, res)
        time.sleep(1)
    return output

        # Aplica escolha de produtos
        id_escolhidos = []
        for res in df["resultado_search"]:
            id_escolhidos.append(escolhe_produtos(res))
            time.sleep(2)
        df["id_agente_escolhe_produtos"] = id_escolhidos

        # Mapeia IDs para nomes e fornecedores
df_info_products = pd.read_csv("info_products_with_brand_name.csv")

def map_ids_to_prod_fornecedor(id_dict):
    result = {}
    for key, id_list in id_dict.items():
        if not isinstance(id_list, list) or not all(isinstance(x, int) for x in id_list):
            result[key] = []
            continue
        subset = df_info_products[df_info_products['PRODUCT_ID'].isin(id_list)]
        result[key] = [
            {"produto": r['PRODUCT_NAME'], "fornecedor": r['BRAND_NAME']} for _, r in subset.iterrows()
        ]
    return result

        df["nome_fornecedor_agente_escolhe_produtos"] = df["id_agente_escolhe_produtos"].apply(map_ids_to_prod_fornecedor)

        # Função de verificação final
        def run_verificacao(input_obj):
    input_json = json.dumps(input_obj, ensure_ascii=False) if isinstance(input_obj, dict) else input_obj
    prompt = f"{PROMPT_AGENTE_VERIFICACAO}\nInput: {input_json}"
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content
        except OverloadedError:
            time.sleep(2 ** attempt)
        except Exception:
            break
    return ''

        df["output_agente_verificacao"] = df["nome_fornecedor_agente_escolhe_produtos"].apply(run_verificacao)

        # Extrai JSON limpo
def extrair_json(textblock):
    texto = getattr(textblock, "text", str(textblock))
    start = texto.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(texto[start:], start):
        if ch == '{': depth += 1
        elif ch == '}': depth -= 1
        if depth == 0:
            return texto[start:i+1]
    return None

        df["output_agente_verificacao_limpo"] = df["output_agente_verificacao"].apply(extrair_json)

        # Conta respostas sim/não
def conta_sim_nao(obj):
    if pd.isna(obj):
        return pd.Series({'quantidade_sim': 0, 'quantidade_nao': 0})
    try:
        d = json.loads(obj.replace('\\n', '')) if isinstance(obj, str) else obj
    except Exception:
        try:
            d = ast.literal_eval(obj.strip())
        except Exception:
            return pd.Series({'quantidade_sim': 0, 'quantidade_nao': 0})
    n_sim = n_nao = 0
    for listas in d.values():
        if isinstance(listas, list):
            for item in listas:
                resp = item.get('resposta')
                if resp == 'sim': n_sim += 1
                elif resp == 'não': n_nao += 1
    return pd.Series({'quantidade_sim': n_sim, 'quantidade_nao': n_nao})

        contagens = df["output_agente_verificacao_limpo"].apply(conta_sim_nao)
        df = pd.concat([df, contagens], axis=1)
        df['perc_acerto'] = df.apply(
            lambda r: round((r['quantidade_sim'] / (r['quantidade_sim'] + r['quantidade_nao']) * 100), 2)
            if (r['quantidade_sim'] + r['quantidade_nao']) > 0 else 0,
            axis=1
        )

        # Cria planilha de resultados
        resumo_df = pd.DataFrame({
            'Métrica': ['Total de SIM', 'Total de NÃO', 'Percentual geral de acerto (%)'],
            'Valor': [df['quantidade_sim'].sum(), df['quantidade_nao'].sum(), round((df['quantidade_sim'].sum() / (df['quantidade_sim'].sum() + df['quantidade_nao'].sum())) * 100, 2)]
        })

        df_excel = df.rename(columns={'nome_fornecedor_agente_escolhe_produtos': 'produtos_fornecedores'})[
            ['produtos_fornecedores', 'output_agente_verificacao_limpo', 'quantidade_sim', 'quantidade_nao', 'perc_acerto']
        ]

        # Download dos resultados
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_excel.to_excel(writer, sheet_name='Dados', index=False)
            resumo_df.to_excel(writer, sheet_name='Resumo', index=False)
        output.seek(0)

        st.subheader("✅ Resultado Final")
        st.dataframe(df_excel)
        st.download_button("📥 Baixar planilha de resultados (.xlsx)", data=output, file_name="resultados.xlsx")

# ------------------ TAB 2: Visualização ------------------
with tab2:
    st.header("📈 Visualização Avançada de Resultados")
    uploaded_xlsx = st.file_uploader(
        "Envie o arquivo de resultados (.xlsx)", type=["xlsx"]
    )
    if uploaded_xlsx:
        df_vis = pd.read_excel(uploaded_xlsx, sheet_name='Dados')
        if 'produtos_fornecedores' in df_vis.columns:
            df_vis['produtos_fornecedores'] = df_vis['produtos_fornecedores'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        ver_cols = [c for c in df_vis.columns if 'verificacao' in c.lower()]
        if ver_cols:
            df_vis['verificacao'] = df_vis[ver_cols[0]].apply(parse_verificacao_text)
        else:
            st.error("Coluna de verificação não encontrada.")

        total_sim = total_nao = 0
        for d in df_vis['verificacao']:
            if isinstance(d, dict):
                for resp_list in d.values():
                    for r in resp_list:
                        if r.get('resposta') == 'sim': total_sim += 1
                        elif r.get('resposta') == 'não': total_nao += 1
        perc = round((total_sim / (total_sim + total_nao) * 100), 2) if (total_sim + total_nao) > 0 else 0

        st.subheader("📊 Resumo de Resultados")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de SIM", total_sim)
        c2.metric("Total de NÃO", total_nao)
        c3.metric("% de Acerto", f"{perc}%")
        st.markdown("---")

        st.subheader("🔍 Detalhamento por Item")
        for _, row in df_vis.iterrows():
            pf = row['produtos_fornecedores'] or {}
            vr = row['verificacao'] or {}
            for item, prods in pf.items():
                st.markdown(f"**Item pedido: {item}**")
                for idx, prod in enumerate(prods):
                    nome = prod['produto']
                    status = vr.get(item, [])[idx].get('resposta') if idx < len(vr.get(item, [])) else None
                    exp = vr.get(item, [])[idx].get('explicacao', '') if idx < len(vr.get(item, [])) else ''
                    icon = '✅' if status == 'sim' else '❌'
                    line = f"- {nome} {icon}"
                    if exp:
                        line += f"  - Explicação: {exp}"
                    st.markdown(line)
                st.write("")