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

# Carrega variáveis de ambiente
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Funções utilitárias

def parse_verificacao_text(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}
    txt = x.replace('\\n', '').strip()
    try:
        return json.loads(txt)
    except Exception:
        try:
            return ast.literal_eval(txt)
        except Exception:
            return {}


def search_for_product(product):
    url = 'https://api.trela.com.br/emporium-bff/graphql'
    headers = {"Content-Type": "application/json"}
    query = f"""
    query searchProducts {{
      search(query: \"{product}\", hubId: 1) {{
        products {{ nodes {{ productId productName supplierName }} }}
      }}
    }}
    """
    resp = requests.post(url, headers=headers, json={"query": query})
    if resp.status_code == 200:
        nodes = resp.json().get("data", {}).get("search", {}).get("products", {}).get("nodes", [])
        return (product.lower(), [
            {"name": n["productName"], "id": n["productId"], "supplier": n["supplierName"]}
            for n in nodes
        ])
    return (product.lower(), [])


def processar_linha(texto, prompt_estrutura, retries=3):
    prompt = prompt_estrutura.replace("{{input}}", texto)
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(resp.content[0].text).get("lista", [])
        except OverloadedError:
            time.sleep(2 ** attempt)
        except Exception:
            break
    return []


def run_flow_for_product(key, resultado, prompt_escolhe, modelo, retries=3):
    filled = prompt_escolhe.replace("{key}", key)
    filled = filled.replace(
        "{json.dumps(resultado, indent=2, ensure_ascii=False)}",
        json.dumps(resultado, indent=2, ensure_ascii=False)
    )
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=modelo,
                max_tokens=256,
                messages=[{"role": "user", "content": filled}]
            )
            return json.loads(resp.content[0].text.strip())
        except OverloadedError:
            time.sleep(2 ** attempt)
        except Exception:
            break
    return ["NOT_FOUND"]


def run_verificacao(input_obj, prompt_geral, retries=3):
    input_json = json.dumps(input_obj, ensure_ascii=False) if isinstance(input_obj, dict) else input_obj
    prompt = f"{prompt_geral}\nInput: {input_json}"
    for attempt in range(retries):
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


def extrair_json(textblock):
    txt = getattr(textblock, "text", str(textblock))
    start = txt.find('{')
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(txt[start:], start):
        if ch == '{': depth += 1
        elif ch == '}': depth -= 1
        if depth == 0:
            return txt[start:i+1]
    return None


def conta_sim_nao(obj):
    if pd.isna(obj):
        return pd.Series({'quantidade_sim': 0, 'quantidade_nao': 0})
    try:
        data = json.loads(obj.replace('\\n', '')) if isinstance(obj, str) else obj
    except Exception:
        try:
            data = ast.literal_eval(obj)
        except Exception:
            return pd.Series({'quantidade_sim': 0, 'quantidade_nao': 0})
    sim = nao = 0
    for lst in data.values():
        if isinstance(lst, list):
            for item in lst:
                if item.get('resposta') == 'sim': sim += 1
                elif item.get('resposta') == 'não': nao += 1
    return pd.Series({'quantidade_sim': sim, 'quantidade_nao': nao})

# Configuração da página Streamlit
st.set_page_config(page_title="Validação com Claude", layout="wide")

# Layout de abas
tab1, tab2 = st.tabs(["📤 Processar CSV", "📈 Visualizar"])

with tab1:
    st.header("Processar e Validar Lista")

    # Download de exemplo
    with open("validacao_teste.xlsx", "rb") as f:
        st.download_button(
            "📥 Exemplo XLSX", f.read(), "validacao_teste.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    arquivo = st.file_uploader("Envie CSV ou XLSX", type=["csv","xlsx"])
    if arquivo:
        df = pd.read_excel(arquivo) if arquivo.name.endswith(".xlsx") else pd.read_csv(arquivo)

        # Lê prompts/modelos
        p_estrutura = df.at[0, "PROMPT_ESTRUTURA_LISTA"]
        p_escolhe = df.at[0, "PROMPT_ESCOLHE_PRODUTOS"]
        m_escolhe = df.at[0, "MODELO_ESCOLHE_PRODUTOS"]
        p_verifica = df.at[0, "PROMPT_AGENTE_VERIFICACAO"]

        st.success("Carregado. Processando...")
        prog = st.progress(0)

        # Normaliza lista
        df["standardized_products"] = df["LISTA"].apply(lambda x: processar_linha(x, p_estrutura))
        prog.progress(0.25)

        # Remover duplicados
        df["unique_products"] = df["standardized_products"].apply(lambda lst: list(dict.fromkeys(lst)))
        prog.progress(0.35)

        # Busca produtos
        df["resultado_search"] = df["unique_products"].apply(
            lambda ups: dict(concurrent.futures.ThreadPoolExecutor(max_workers=4)
                             .map(search_for_product, ups))
        )
        prog.progress(0.6)

        # Escolhe IDs
        df["id_agente_escolhe_produtos"] = df["resultado_search"].apply(
            lambda res: {k: run_flow_for_product(k, res[k], p_escolhe, m_escolhe) for k in res}
        )
        prog.progress(0.8)

        # Mapeia nomes/fornecedores
        info = pd.read_csv("info_products_with_brand_name.csv")
        df["nome_fornecedor_agente_escolhe_produtos"] = df["id_agente_escolhe_produtos"].apply(
            lambda d: [
                {"produto": info.loc[info.PRODUCT_ID==pid, "PRODUCT_NAME"].iat[0],
                 "fornecedor": info.loc[info.PRODUCT_ID==pid, "BRAND_NAME"].iat[0]}
                for lst in d.values() for pid in lst
            ]
        )
        prog.progress(0.9)

        # Verificação final
        df["output_agente_verificacao"] = df["nome_fornecedor_agente_escolhe_produtos"].apply(
            lambda o: run_verificacao(o, p_verifica)
        )
        df["output_agente_verificacao_limpo"] = df["output_agente_verificacao"].apply(extrair_json)
        df = pd.concat([df, df["output_agente_verificacao_limpo"].apply(conta_sim_nao)], axis=1)
        df['perc_acerto'] = df.apply(
            lambda r: round((r['quantidade_sim']/(r['quantidade_sim']+r['quantidade_nao'])*100),2)
            if (r['quantidade_sim']+r['quantidade_nao'])>0 else 0, axis=1
        )
        prog.progress(1.0)

        # Resultado
        st.subheader("✅ Resultado")
        cols = ['nome_fornecedor_agente_escolhe_produtos', 'output_agente_verificacao_limpo',
                'quantidade_sim', 'quantidade_nao', 'perc_acerto']
        st.dataframe(df[cols])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df[cols].to_excel(writer, sheet_name='Dados', index=False)
            pd.DataFrame({
                'Métrica': ['Total SIM','Total NÃO','% Acerto'],
                'Valor': [df.quantidade_sim.sum(), df.quantidade_nao.sum(), df.perc_acerto.mean()]
            }).to_excel(writer, sheet_name='Resumo', index=False)
        buf.seek(0)
        st.download_button("📥 Baixar resultados", buf, "resultados.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab2:
    st.header("Visualização Avançada")
    xls = st.file_uploader("Envie resultados .xlsx", type="xlsx")
    if xls:
        dv = pd.read_excel(xls, sheet_name='Dados')
        dv['produtos_fornecedores'] = dv['nome_fornecedor_agente_escolhe_produtos'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        dv['verificacao'] = dv['output_agente_verificacao_limpo'].apply(parse_verificacao_text)
        total_sim = total_nao = 0
        for d in dv.verificacao:
            if isinstance(d, dict):
                for lst in d.values():
                    for r in lst:
                        if r.get('resposta')=='sim': total_sim+=1
                        elif r.get('resposta')=='não': total_nao+=1
        perc = round((total_sim/(total_sim+total_nao)*100),2) if (total_sim+total_nao)>0 else 0
        c1,c2,c3 = st.columns(3)
        c1.metric("Total SIM", total_sim)
        c2.metric("Total NÃO", total_nao)
        c3.metric("% Acerto", f"{perc}%")
        st.markdown("---")
        for idx,row in dv.iterrows():
            st.markdown(f"**{idx+1}. Item**: {row.produtos_fornecedores}")
            st.write("")