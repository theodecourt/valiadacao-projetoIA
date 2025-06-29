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

# Carrega variáveis de ambiente do Streamlit Secrets
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error("Chave ANTHROPIC_API_KEY não encontrada em st.secrets. Configure em .streamlit/secrets.toml.")
    st.stop()
client = Anthropic(api_key=ANTHROPIC_API_KEY)

st.set_page_config(page_title="Validação de Lista com Claude", layout="wide")
st.title("Validação de Lista com Claude e Visualização Avançada")

# Função para parser de verificação (JSON ou string de dict)
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

# Cria abas
tab1, tab2 = st.tabs(["📤 Processar CSV", "📈 Visualizar Resultados"])

# ------------------ TAB 1: Processamento ------------------
with tab1:
    st.header("Processar e Validar Lista")
    
    # Botão de download do XLSX de exemplo
    try:
        with open("./validacao_teste.xlsx", "rb") as f:
            excel_bytes = f.read()
        st.download_button(
            label="📥 Baixar Excel de Exemplo", 
            data=excel_bytes, 
            file_name="validacao_teste.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except FileNotFoundError:
        st.warning("Arquivo de exemplo 'validacao_teste.xlsx' não encontrado. Faça upload de um arquivo válido.")

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

        # Verifica colunas necessárias
        cols = df.columns.tolist()
        expected = ["PROMTP_ESCOLHE_PRODUTOS","PROMPT_ESTRUTURA_LISTA","MODELO_ESCOLHE_PRODUTOS","PROMPT_AGENTE_VERIFICACAO","LISTA"]
        missing = [c for c in expected if c not in cols]
        if missing:
            st.error(f"Colunas faltando: {missing}. Verifique seu arquivo.")
            st.stop()

        PROMPT_ESCOLHE_PRODUTOS = df["PROMTP_ESCOLHE_PRODUTOS"][0]
        PROMPT_ESTRUTURA_LISTA = df["PROMPT_ESTRUTURA_LISTA"][0]
        MODELO_ESCOLHE_PRODUTOS = df["MODELO_ESCOLHE_PRODUTOS"][0]
        PROMPT_AGENTE_VERIFICACAO = df["PROMPT_AGENTE_VERIFICACAO"][0]

        st.success("Arquivo carregado com sucesso. Iniciando o processamento...")

        # Passo 1: padroniza produtos
        def processar_linha(input_texto):
            prompt = PROMPT_ESTRUTURA_LISTA.replace("{{input}}", input_texto)
            max_retries = 3
            base_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Add a small delay between attempts to avoid rate limiting
                    if attempt > 0:
                        time.sleep(base_delay * (2 ** attempt))
                        
                    resp = client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    try:
                        return json.loads(resp.content[0].text).get("lista", [])
                    except Exception as e:
                        st.warning(f"Erro ao processar resposta JSON: {str(e)}")
                        return []
                        
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        st.error(f"Erro após {max_retries} tentativas: {str(e)}")
                        return []
                    continue
            
            return []

        # Add rate limiting for the entire DataFrame processing
        def process_dataframe_with_rate_limit(df):
            results = []
            for idx, row in df.iterrows():
                result = processar_linha(row["LISTA"])
                results.append(result)
                # Add a small delay between rows to avoid rate limiting
                time.sleep(0.5)
            return results

        # Replace the direct apply with our rate-limited version
        df["standardized_products"] = process_dataframe_with_rate_limit(df)
        df["unique_products"] = df["standardized_products"].apply(lambda x: list(dict.fromkeys(x)))

        # Passo 2: busca GraphQL concorrente
        def search_for_product(product):
            url = 'https://api.trela.com.br/emporium-bff/graphql'
            headers = {"Content-Type": "application/json"}
            query = f"""
            query searchProducts {{
                search(query: \"{product}\", hubId: 1) {{ products {{ nodes {{ productId productName supplierName }} }} }}
            }}"""
            resp = requests.post(url, headers=headers, json={"query": query})
            if resp.status_code == 200:
                nodes = resp.json().get("data", {}).get("search", {}).get("products", {}).get("nodes", [])
                return product.lower(), [
                    {"name": n["productName"], "id": n["productId"], "supplier": n["supplierName"]}
                    for n in nodes
                ]
            return product.lower(), []

        def get_products_search_library(unique_products):
            library = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
                futures = {exe.submit(search_for_product, p): p for p in unique_products}
                for f in concurrent.futures.as_completed(futures):
                    k, v = f.result()
                    library[k] = v
            return library

        df["resultado_search"] = df["unique_products"].apply(get_products_search_library)

        # Passo 3: agente escolhe IDs com retries
        def run_flow_for_product(key, resultado, retries=3):
            filled = PROMPT_ESCOLHE_PRODUTOS.replace("{key}", key)
            filled = filled.replace(
                "{json.dumps(resultado, indent=2, ensure_ascii=False)}",
                json.dumps(resultado, indent=2, ensure_ascii=False)
            )
            for i in range(retries):
                try:
                    r = client.messages.create(
                        model=MODELO_ESCOLHE_PRODUTOS,
                        max_tokens=256,
                        messages=[{"role": "user", "content": filled}]
                    )
                    return json.loads(r.content[0].text.strip())
                except Exception:
                    time.sleep(2 ** i)
            return ["NOT_FOUND"]

        def escolhe_todos(result_search):
            return {k: run_flow_for_product(k, result_search) for k in result_search}

        df["id_agente_escolhe_produtos"] = df["resultado_search"].apply(escolhe_todos)

        # Passo 4: mapeia IDs para nomes e fornecedores
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

        # Passo 5: verificação final
        def run_verificacao(o):
            inp = json.dumps(o, ensure_ascii=False) if isinstance(o, dict) else o
            prompt = PROMPT_AGENTE_VERIFICACAO + "\nInput: " + inp
            try:
                resp = client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content
            except Exception:
                return None

        df["output_agente_verificacao"] = df["nome_fornecedor_agente_escolhe_produtos"].apply(run_verificacao)

        # Extrai JSON da resposta
        def extrair_json(tb):
            txt = getattr(tb, "text", str(tb))
            start = txt.find('{')
            if start < 0: return None
            depth = 0
            for i, ch in enumerate(txt[start:], start):
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
                if depth == 0: return txt[start:i+1]
            return None

        df["output_agente_verificacao_limpo"] = df["output_agente_verificacao"].apply(extrair_json)

        # Conta sim/nao e calcula percentual
        def conta_sim_nao(o):
            if pd.isna(o): return pd.Series({'quantidade_sim':0,'quantidade_nao':0})
            try:
                d = json.loads(o.replace('\\n',''))
            except:
                try: d = ast.literal_eval(o)
                except: return pd.Series({'quantidade_sim':0,'quantidade_nao':0})
            s = n = 0
            for lst in d.values():
                if isinstance(lst, list):
                    for it in lst:
                        if it.get('resposta')=='sim': s+=1
                        elif it.get('resposta')=='não': n+=1
            return pd.Series({'quantidade_sim':s,'quantidade_nao':n})

        cont = df["output_agente_verificacao_limpo"].apply(conta_sim_nao)
        df = pd.concat([df, cont], axis=1)
        df['perc_acerto'] = df.apply(
            lambda r: round((r.quantidade_sim/(r.quantidade_sim+r.quantidade_nao)*100),2)
            if (r.quantidade_sim+r.quantidade_nao)>0 else 0, axis=1
        )

        # Gera resumo e planilha
        resumo = pd.DataFrame({
            'Métrica':['Total SIM','Total NÃO','% Acerto'],
            'Valor':[df.quantidade_sim.sum(), df.quantidade_nao.sum(), df.perc_acerto.mean()]
        })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.rename(columns={'nome_fornecedor_agente_escolhe_produtos':'produtos_fornecedores'})[
                ['produtos_fornecedores','output_agente_verificacao_limpo','quantidade_sim','quantidade_nao','perc_acerto']
            ].to_excel(writer, sheet_name='Dados', index=False)
            resumo.to_excel(writer, sheet_name='Resumo', index=False)
        buf.seek(0)

        st.subheader("✅ Resultado Final")
        # Renomeia coluna para exibição correta
        df = df.rename(columns={'nome_fornecedor_agente_escolhe_produtos': 'produtos_fornecedores'})
        st.dataframe(df[[
            'produtos_fornecedores',
            'output_agente_verificacao_limpo',
            'quantidade_sim',
            'quantidade_nao',
            'perc_acerto'
        ]])
        st.download_button("📥 Baixar resultados (.xlsx)", data=buf, file_name="resultados.xlsx",  
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------ TAB 2: Visualização Avançada ------------------
with tab2:
    st.header("Visualização Avançada de Resultados")
    uploaded_xlsx = st.file_uploader("Envie o arquivo de resultados (.xlsx)", type=["xlsx"] )
    if uploaded_xlsx:
        df_vis = pd.read_excel(uploaded_xlsx, sheet_name='Dados')
        if 'produtos_fornecedores' in df_vis.columns:
            df_vis['produtos_fornecedores'] = df_vis['produtos_fornecedores'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        ver_cols = [c for c in df_vis.columns if 'verificacao' in c.lower()]
        if not ver_cols:
            st.error("Coluna de verificação não encontrada.")
            st.stop()
        df_vis['verificacao'] = df_vis[ver_cols[0]].apply(parse_verificacao_text)

        total_s = total_n = 0
        for d in df_vis['verificacao']:
            if isinstance(d, dict):
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
            for item, prods in (row['produtos_fornecedores'] or {}).items():
                st.markdown(f"**Item pedido:** {item}")
                vr = row['verificacao'].get(item, [])
                for idx, prod in enumerate(prods):
                    nome = prod['produto']
                    status = vr[idx].get('resposta') if idx < len(vr) else None
                    exp = vr[idx].get('explicacao','') if idx < len(vr) else ''
                    icon = '✅' if status=='sim' else '❌'
                    line = f"- {nome} {icon}"
                    if exp:
                        line += f"  - Explicação: {exp}"
                    st.markdown(line)
                st.write("")
