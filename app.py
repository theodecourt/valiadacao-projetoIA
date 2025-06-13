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
    if isinstance(x, dict): return x
    if not isinstance(x, str): return {}
    txt = x.replace('\\n','').strip()
    try: return json.loads(txt)
    except: 
        try: return ast.literal_eval(txt)
        except: return {}


def search_for_product(product):
    url = 'https://api.trela.com.br/emporium-bff/graphql'
    headers = {"Content-Type": "application/json"}
    query = f"""
    query searchProducts {{
      search(query: \"{product}\", hubId: 1) {{ products {{ nodes {{ productId productName supplierName }} }} }}
    }}"""
    resp = requests.post(url, headers=headers, json={"query": query})
    if resp.status_code==200:
        nodes = resp.json().get("data", {}).get("search", {}).get("products", {}).get("nodes", [])
        return product.lower(), [{"name":n["productName"],"id":n["productId"],"supplier":n["supplierName"]} for n in nodes]
    return product.lower(), []


def processar_linha(texto, prompt_estrutura, retries=3):
    prompt = prompt_estrutura.replace("{{input}}", texto)
    for i in range(retries):
        try:
            resp=client.messages.create(model="claude-3-5-haiku-20241022",max_tokens=1024,messages=[{"role":"user","content":prompt}])
            return json.loads(resp.content[0].text).get("lista", [])
        except OverloadedError:
            time.sleep(2**i)
        except:
            break
    return []


def run_flow_for_product(key, resultado, prompt_escolhe, modelo, retries=3):
    filled = prompt_escolhe.replace("{key}", key)
    filled = filled.replace("{json.dumps(resultado, indent=2, ensure_ascii=False)}", json.dumps(resultado, indent=2, ensure_ascii=False))
    for i in range(retries):
        try:
            resp=client.messages.create(model=modelo,max_tokens=256,messages=[{"role":"user","content":filled}])
            return json.loads(resp.content[0].text.strip())
        except OverloadedError:
            time.sleep(2**i)
        except:
            break
    return ["NOT_FOUND"]


def run_verificacao(input_obj, prompt_geral, retries=3):
    inp = json.dumps(input_obj,ensure_ascii=False) if isinstance(input_obj,dict) else input_obj
    prompt = f"{prompt_geral}\nInput: {inp}"
    for i in range(retries):
        try:
            resp=client.messages.create(model="claude-opus-4-20250514",max_tokens=1024,messages=[{"role":"user","content":prompt}])
            return resp.content
        except OverloadedError:
            time.sleep(2**i)
        except:
            break
    return ''


def extrair_json(textblock):
    txt=getattr(textblock,"text",str(textblock))
    start=txt.find('{')
    if start<0: return None
    depth=0
    for i,ch in enumerate(txt[start:],start):
        if ch=='{': depth+=1
        elif ch=='}': depth-=1
        if depth==0: return txt[start:i+1]
    return None


def conta_sim_nao(obj):
    import pandas as _pd
    if _pd.isna(obj): return _pd.Series({'quantidade_sim':0,'quantidade_nao':0})
    try: data=json.loads(obj.replace('\\n','')) if isinstance(obj,str) else obj
    except:
        try: data=ast.literal_eval(obj)
        except: return _pd.Series({'quantidade_sim':0,'quantidade_nao':0})
    sim=nao=0
    for lst in data.values():
        if isinstance(lst,list):
            for it in lst:
                if it.get('resposta')=='sim': sim+=1
                elif it.get('resposta')=='não': nao+=1
    return _pd.Series({'quantidade_sim':sim,'quantidade_nao':nao})

# Streamlit setup
st.set_page_config(page_title="Validação com Claude",layout="wide")

tab1,tab2=st.tabs(["📤 Processar CSV","📈 Visualizar"])

with tab1:
    st.header("Processar e Validar Lista")
    # download exemplo
    with open("validacao_teste.xlsx","rb") as f:
        st.download_button("📥 Exemplo XLSX",f.read(),"validacao_teste.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    arq=st.file_uploader("Envie CSV ou XLSX",type=["csv","xlsx"])
    if arq:
        df=pd.read_excel(arq) if arq.name.lower().endswith(".xlsx") else pd.read_csv(arq)
        cols=df.columns.tolist()
        # detecta colunas
        col_estrutura=next((c for c in cols if "ESTRUTURA" in c),None)
        col_escolhe=next((c for c in cols if "ESCOLHE_PRODUTOS" in c),None)
        col_modelo=next((c for c in cols if "MODELO_ESCOLHE_PRODUTOS" in c),None)
        col_verif=next((c for c in cols if "AGENTE_VERIFICACAO" in c),None)
        if not all([col_estrutura,col_escolhe,col_modelo,col_verif]):
            st.error("Colunas obrigatórias não encontradas: "+str(cols)); st.stop()
        p_estrutura,p_escolhe,m_escolhe,p_verif=(df.at[0,col_estrutura],df.at[0,col_escolhe],df.at[0,col_modelo],df.at[0,col_verif])
        st.success("Carregado. Processando...")
        prog=st.progress(0)
        # normalized
        df['standardized']=df['LISTA'].apply(lambda x: processar_linha(x,p_estrutura))
        prog.progress(0.2)
        df['unique']=df['standardized'].apply(lambda l:list(dict.fromkeys(l)))
        prog.progress(0.3)
        df['search']=df['unique'].apply(lambda ups: dict(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(search_for_product,ups)))
        prog.progress(0.5)
        df['ids']=df['search'].apply(lambda r:{k:run_flow_for_product(k,r[k],p_escolhe,m_escolhe) for k in r})
        prog.progress(0.7)
        info=pd.read_csv("info_products_with_brand_name.csv")
        df['mapped']=df['ids'].apply(lambda d:[{"produto":info.loc[info.PRODUCT_ID==i,"PRODUCT_NAME"].iat[0],"fornecedor":info.loc[info.PRODUCT_ID==i,"BRAND_NAME"].iat[0]} for lst in d.values() for i in lst])
        prog.progress(0.8)
        df['verif_raw']=df['mapped'].apply(lambda o: run_verificacao(o,p_verif))
        df['verif_clean']=df['verif_raw'].apply(extrair_json)
        df=pd.concat([df,df['verif_clean'].apply(conta_sim_nao)],axis=1)
        df['perc']=df.apply(lambda r:round((r.quantidade_sim/(r.quantidade_sim+r.quantidade_nao)*100),2) if r.quantidade_sim+r.quantidade_nao>0 else 0,axis=1)
        prog.progress(1.0)
        st.dataframe(df[['mapped','verif_clean','quantidade_sim','quantidade_nao','perc']])
        buf=io.BytesIO();
        with pd.ExcelWriter(buf,engine='openpyxl') as w:
            df[['mapped','verif_clean','quantidade_sim','quantidade_nao','perc']].to_excel(w,sheet_name='Dados',index=False)
            pd.DataFrame({'Métrica':['Total SIM','Total NÃO','% Acerto'],'Valor':[df.quantidade_sim.sum(),df.quantidade_nao.sum(),round(df.quantidade_sim.sum()/(df.quantidade_sim.sum()+df.quantidade_nao.sum())*100,2)]}).to_excel(w,sheet_name='Resumo',index=False)
        buf.seek(0)
        st.download_button("📥 Baixar resultados",buf,"resultados.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab2:
    st.header("Visualização Avançada")
    xls=st.file_uploader("Envie resultados .xlsx",type="xlsx")
    if xls:
        dv=pd.read_excel(xls,sheet_name='Dados')
        dv['verif']=dv['verif_clean'].apply(parse_verificacao_text)
        sim=nao=0
        for d in dv.verif:
            if isinstance(d,dict):
                for lst in d.values():
                    for it in lst:
                        if it.get('resposta')=='sim': sim+=1
                        elif it.get('resposta')=='não': nao+=1
        perc=round((sim/(sim+nao)*100),2) if sim+nao>0 else 0
        c1,c2,c3=st.columns(3)
        c1.metric("Total SIM",sim)
        c2.metric("Total NÃO",nao)
        c3.metric("% Acerto",f"{perc}%")
        st.markdown("---")
        for idx,row in dv.iterrows(): st.markdown(f"**Item {idx+1}:** {row.mapped}")
