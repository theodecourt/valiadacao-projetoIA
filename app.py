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
from dotenv import load_dotenv

# Carrega variáveis do .env (defina ANTHROPIC_API_KEY nas Streamlit Secrets)
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)

st.title("Validação de Lista com Claude e Visualização Avançada")

# Parser de verificação
def parse_verificacao_text(x):
    if isinstance(x, dict): return x
    if not isinstance(x, str): return {}
    txt = x.replace('\\n', '').strip()
    try:
        return json.loads(txt)
    except:
        try:
            return ast.literal_eval(txt)
        except:
            return {}

# Abas
tab1, tab2 = st.tabs(["📤 Processar CSV", "📈 Visualizar Resultados"])

# ------------------ TAB 1 ------------------
with tab1:
    st.header("Processar e Validar Lista")

    # Exemplo de Excel
    try:
        with open("validacao_teste.xlsx", "rb") as f:
            excel_bytes = f.read()
        st.download_button(
            "📥 Baixar Excel de Exemplo",
            data=excel_bytes,
            file_name="validacao_teste.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except FileNotFoundError:
        st.warning("Arquivo de exemplo 'validacao_teste.xlsx' não encontrado no servidor.")

    uploaded_file = st.file_uploader(
        "📤 Envie o arquivo de validação (CSV ou XLSX)",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        # Leitura
        if uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Debug colunas
        st.write("Colunas disponíveis:", df.columns.tolist())

        # Atenção ao nome das colunas: corrigido de 'PROMTP_' para 'PROMPT_'
        try:
            PROMPT_ESCOLHE_PRODUTOS = df["PROMPT_ESCOLHE_PRODUTOS"][0]
            PROMPT_ESTRUTURA_LISTA  = df["PROMPT_ESTRUTURA_LISTA"][0]
            MODELO_ESCOLHE_PRODUTOS = df["MODELO_ESCOLHE_PRODUTOS"][0]
            PROMPT_AGENTE_VERIFICACAO = df["PROMPT_AGENTE_VERIFICACAO"][0]
        except KeyError as e:
            st.error(f"Coluna não encontrada: {e}. Verifique os cabeçalhos do seu arquivo.")
            st.stop()

        st.success("Arquivo carregado. Iniciando processamento...")

        # Normalização das listas
        def processar_linha(texto):
            prompt = PROMPT_ESTRUTURA_LISTA.replace("{{input}}", texto)
            resp = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            try:
                return json.loads(resp.content[0].text).get("lista", [])
            except:
                return []

        df["standardized_products"] = df["LISTA"].apply(processar_linha)
        df["unique_products"] = df["standardized_products"].apply(lambda x: list(dict.fromkeys(x)))

        # Busca no BFF GraphQL
        def search_for_product(product):
            url = 'https://api.trela.com.br/emporium-bff/graphql'
            headers = {"Content-Type": "application/json"}
            query = f"""
            query searchProducts {{
                search(query: \"{product}\", hubId: 1) {{ products {{ nodes {{ productId productName supplierName }} }} }}
            }}"""
            resp = requests.post(url, headers=headers, json={"query": query})
            if resp.status_code == 200:
                nodes = resp.json().get("data",{}).get("search",{}).get("products",{}).get("nodes",[])
                return product.lower(), [{"name":n["productName"],"id":n["productId"],"supplier":n["supplierName"]} for n in nodes]
            return product.lower(), []

        df["resultado_search"] = df["unique_products"].apply(
            lambda ups: dict(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(search_for_product, ups))
        )

        # Escolha de IDs
        def run_flow_for_product(key, resultado):
            filled = PROMPT_ESCOLHE_PRODUTOS.replace("{key}", key)
            filled = filled.replace(
                "{json.dumps(resultado, indent=2, ensure_ascii=False)}",
                json.dumps(resultado, indent=2, ensure_ascii=False)
            )
            resp = client.messages.create(
                model=MODELO_ESCOLHE_PRODUTOS,
                max_tokens=256,
                messages=[{"role": "user", "content": filled}]
            )
            try:
                return json.loads(resp.content[0].text.strip())
            except:
                return ["NOT_FOUND"]

        df["id_agente_escolhe_produtos"] = df["resultado_search"].apply(lambda r: {k: run_flow_for_product(k, r) for k in r})

        # Mapeia para nomes/fornecedor
        df_info = pd.read_csv("info_products_with_brand_name.csv")
        def map_ids(d):
            out = {}
            for k, lst in d.items():
                out[k] = []
                if isinstance(lst, list):
                    for pid in lst:
                        row = df_info[df_info.PRODUCT_ID == pid]
                        if not row.empty:
                            out[k].append({
                                "produto": row.PRODUCT_NAME.values[0],
                                "fornecedor": row.BRAND_NAME.values[0]
                            })
            return out
        df["nome_fornecedor_agente_escolhe_produtos"] = df["id_agente_escolhe_produtos"].apply(map_ids)

        # Verificação final
        def run_verificacao(o):
            inp = json.dumps(o, ensure_ascii=False) if isinstance(o, dict) else o
            prompt = PROMPT_AGENTE_VERIFICACAO + "\nInput: " + inp
            resp = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content

        df["output_agente_verificacao"] = df["nome_fornecedor_agente_escolhe_produtos"].apply(run_verificacao)

        def extrair_json(tb):
            txt = getattr(tb, "text", str(tb))
            start = txt.find('{')
            if start < 0: return None
            depth = 0
            for i, ch in enumerate(txt[start:], start):
                if ch=='{': depth+=1
                elif ch=='}': depth-=1
                if depth==0: return txt[start:i+1]
            return None

        df["output_agente_verificacao_limpo"] = df["output_agente_verificacao"].apply(extrair_json)

        def conta_sim_nao(o):
            if pd.isna(o): return pd.Series({'quantidade_sim':0,'quantidade_nao':0})
            try: d = json.loads(o.replace('\\n',''))
            except: 
                try: d = ast.literal_eval(o)
                except: return pd.Series({'quantidade_sim':0,'quantidade_nao':0})
            s=n=0
            for lst in d.values():
                if isinstance(lst,list):
                    for it in lst:
                        if it.get('resposta')=='sim': s+=1
                        elif it.get('resposta')=='não': n+=1
            return pd.Series({'quantidade_sim':s,'quantidade_nao':n})

        cont = df["output_agente_verificacao_limpo"].apply(conta_sim_nao)
        df = pd.concat([df, cont], axis=1)
        df['perc_acerto'] = df.apply(lambda r: round((r.quantidade_sim/(r.quantidade_sim+r.quantidade_nao)*100),2)
                                     if r.quantidade_sim+r.quantidade_nao>0 else 0, axis=1)

        # resumo
        resumo = pd.DataFrame({
            'Métrica':['Total SIM','Total NÃO','% Acerto'],
            'Valor':[df.quantidade_sim.sum(), df.quantidade_nao.sum(), df.perc_acerto.mean()]
        })

        # prepara Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.rename(columns={'nome_fornecedor_agente_escolhe_produtos':'produtos_fornecedores'})[
                ['produtos_fornecedores','output_agente_verificacao_limpo','quantidade_sim','quantidade_nao','perc_acerto']
            ].to_excel(w, sheet_name='Dados', index=False)
            resumo.to_excel(w, sheet_name='Resumo', index=False)
        buf.seek(0)

        st.subheader("✅ Resultado Final")
        st.dataframe(df[['produtos_fornecedores','output_agente_verificacao_limpo','quantidade_sim','quantidade_nao','perc_acerto']])
        st.download_button("📥 Baixar resultados (.xlsx)", data=buf, file_name="resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------ TAB 2 ------------------
with tab2:
    st.header("Visualização Avançada de Resultados")
    uploaded_xlsx = st.file_uploader(
        "Envie o arquivo de resultados (.xlsx)", type=["xlsx"]
    )
    if uploaded_xlsx:
        df_vis = pd.read_excel(uploaded_xlsx, sheet_name='Dados')
        df_vis['produtos_fornecedores'] = df_vis['produtos_fornecedores'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x,str) else x
        )
        ver_cols = [c for c in df_vis.columns if 'verificacao' in c.lower()]
        if not ver_cols:
            st.error("Coluna de verificação não encontrada.")
            st.stop()
        df_vis['verificacao'] = df_vis[ver_cols[0]].apply(parse_verificacao_text)

        total_s=total_n=0
        for d in df_vis['verificacao']:
            if isinstance(d,dict):
                for lst in d.values():
                    for it in lst:
                        if it.get('resposta')=='sim': total_s+=1
                        elif it.get('resposta')=='não': total_n+=1
        perc = round((total_s/(total_s+total_n)*100),2) if total_s+total_n>0 else 0

        c1,c2,c3 = st.columns(3)
        c1.metric("Total de SIM", total_s)
        c2.metric("Total de NÃO", total_n)
        c3.metric("% de Acerto", f"{perc}%")
        st.markdown("---")
        st.subheader("🔍 Detalhamento por Item")
        for _, row in df_vis.iterrows():
            st.markdown(f"**Item pedido:** {row.produtos_fornecedores}")
            st.write("")
