import streamlit as st
import re, unicodedata, json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

st.set_page_config(page_title="Adam Kadmus – INSEPA+ALNULU", page_icon="🤖")

# Sessão
if 'role' not in st.session_state:      st.session_state.role = None
if 'arquivos' not in st.session_state:  st.session_state.arquivos = []
if 'token_counter' not in st.session_state: st.session_state.token_counter = 1

# Funções Auxiliares
def normalize_and_substitute(w):
    norm = unicodedata.normalize('NFKD', w).encode('ASCII','ignore').decode()
    subs = {'@':'a','0':'o','1':'l'}
    return ''.join(subs.get(c,c) for c in norm)

def calc_alnulu_value(w):
    mapping = {
        'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,
        'h':8,'i':9,'j':-10,'k':11,'l':12,'m':-13,'n':14,
        'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,
        'v':-22,'w':23,'x':24,'y':-25,'z':26,
        '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9
    }
    cleaned = re.sub(r'[^\w]', '', normalize_and_substitute(w).lower())
    return sum(mapping.get(ch,0) for ch in cleaned)

def tokenize_text(txt, start):
    toks=[]; c=start
    for w in txt.split():
        toks.append(f"{w} 0.{c}")
        c+=1
    return toks, c

def get_numeric_keys(toks):
    return [float(t.split()[1]) for t in toks if len(t.split())>1]

def is_numeric_key(s):
    return re.match(r'^\d+\.\d+(,\s*\d+\.\d+)*$', s.strip()) is not None

def compute_indices_filhos(file):
    ids=[]
    for b in file.get("Blocos", []):
        ids += b["Combinação"]
    return ids

def retrieve_response(msg, file):
    blocos = file["Blocos"]
    if not blocos:
        return "…", "", []
    entradas = [b["Entrada"] for b in blocos]
    vec = TfidfVectorizer().fit_transform(entradas)
    u = vec.transform([msg])
    idx = int(np.argmax((vec * u.T).toarray().ravel()))
    b = blocos[idx]
    return b["Saída"], b["Reação S"], b["Combinação S"]

def get_file_by_pref(pref):
    p=pref.lower().strip()
    candidates=[]
    for f in st.session_state.arquivos:
        if f["Índice mãe"]==0: continue
        score=0
        for b in f["Blocos"]:
            score += p in b["Entrada"].lower() or p in b["Saída"].lower()
        candidates.append((score,f))
    candidates.sort(key=lambda x:x[0], reverse=True)
    return candidates[0][1] if candidates and candidates[0][0]>0 else None

def train_model_for_file(file, epochs=10):
    B=file.get("Blocos",[])
    if not B: return None
    X=[b["Entrada"] for b in B]
    y=[1 if "olá" in e.lower() else 0 for e in X]
    if len(X)>1:
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)
    else:
        Xtr,Xte,ytr,yte = X,X,y,y
    vec=TfidfVectorizer()
    Xtr_v=vec.fit_transform(Xtr)
    Xte_v=vec.transform(Xte)
    clf=SGDClassifier(loss='log_loss',max_iter=1,warm_start=True,tol=None,random_state=42)
    hist=[]
    for ep in range(1, epochs+1):
        clf.partial_fit(Xtr_v,ytr,classes=np.array([0,1]))
        pred=clf.predict(Xte_v)
        acc=metrics.accuracy_score(yte,pred)*100
        hist.append({"Epoch":ep,"Acurácia (%)":round(acc,2)})
    return hist

# Interface
st.title("Adam Kadmus – INSEPA + ALNULU")

# Login
if st.session_state.role is None:
    pwd = st.text_input("Senha de criadora:", type="password")
    if st.button("Entrar"):
        st.session_state.role = "criadora" if pwd=="Lux15" else "usuario"

# Modo Criadora
if st.session_state.role == "criadora":
    st.header("👩‍💻 Modo Criadora")

    # 1. CRUD Arquivos
    c1,c2,c3 = st.columns(3)
    with c1:
        novo = st.text_input("Novo Arquivo:", key="new_arq")
        if st.button("Criar Arquivo"):
            st.session_state.arquivos.append({
                "Arquivo":novo,
                "Índice mãe":len(st.session_state.arquivos),
                "Blocos":[]
            })
            st.success("Arquivo criado! Role para baixo.")
    with c2:
        if st.session_state.arquivos:
            ex = st.selectbox("Excluir Arquivo:", options=range(len(st.session_state.arquivos)),
                              format_func=lambda i: st.session_state.arquivos[i]["Arquivo"])
    with c3:
        if st.session_state.arquivos and st.button("Excluir Arquivo"):
            st.session_state.arquivos.pop(ex)
            st.success("Arquivo excluído!")

    if not st.session_state.arquivos:
        st.info("Cadastre ao menos um arquivo.")
        st.stop()

    # 2. Selecione arquivo e veja índices filhos
    sel = st.selectbox("Selecione Arquivo:", options=range(len(st.session_state.arquivos)),
                       format_func=lambda i: st.session_state.arquivos[i]["Arquivo"])
    file = st.session_state.arquivos[sel]
    st.write("Índices filhos:", compute_indices_filhos(file))

    # 3. Criação de Blocos
    st.subheader("➕ Criar Bloco")
    ent  = st.text_input("Entrada:", key="in_ent")
    ctxE = st.text_input("Contexto E:", key="in_ctxE")
    reaE = st.text_input("Reação E:", key="in_reaE")
    if st.button("Gerar Combinação E"):
        te, nc = tokenize_text(ent, st.session_state.token_counter)
        st.session_state.te = te; st.session_state.token_counter = nc
        st.write("Combinação E:", get_numeric_keys(te))

    sai  = st.text_input("Saída:", key="in_sai")
    ctxS = st.text_input("Contexto S:", key="in_ctxS")
    reaS = st.text_input("Reação S:", key="in_reaS")
    if st.button("Gerar Combinação S"):
        ts, nc = tokenize_text(sai, st.session_state.token_counter)
        st.session_state.ts = ts; st.session_state.token_counter = nc
        nums = get_numeric_keys(ts)
        st.write("Combinação S:", nums)
        st.write("Ponto finalizador:", nums[-1])

    if st.button("Salvar Bloco"):
        te = st.session_state.get("te",[])
        ts = st.session_state.get("ts",[])
        if not all([ent,sai,te,ts]):
            st.warning("Complete todos os passos.")
        else:
            comb = get_numeric_keys(te)+get_numeric_keys(ts)
            bloco = {
                "Bloco": len(file["Blocos"])+1,
                "Combinação": comb,
                "Entrada": ent,
                "Combinação E": te,
                "Contexto E": ctxE,
                "Reação E": reaE,
                "Saída": sai,
                "Combinação S": get_numeric_keys(ts),
                "Contexto S": ctxS,
                "Reação S": reaS
            }
            file["Blocos"].append(bloco)
            st.success("Bloco salvo!")

    # 4. Simulação de Treinamento
    st.markdown("---")
    st.subheader("📈 Simular Treinamento")
    ep = st.slider("Épocas:",1,50,10)
    if st.button("Treinar"):
        hist = train_model_for_file(file, epochs=ep)
        if hist:
            df = pd.DataFrame(hist)
            st.table(df)
            st.line_chart(df.set_index("Epoch"))
        else:
            st.warning("Sem blocos para treinar.")

    # 5. Espaço de Teste (Criadora)
    st.markdown("---")
    st.subheader("🔍 Espaço de Teste")
    test_msg = st.text_input("Digite mensagem para testar:", key="test_msg")
    if st.button("Testar"):
        msg = test_msg.strip().lower()
        found=False
        for b in file["Blocos"]:
            if msg == b["Entrada"].lower():
                st.info("Corresponde à Entrada")
                st.write("Combinação E:", b["Combinação E"])
                st.write("Contexto E:", b["Contexto E"])
                st.write("Reação E:", b["Reação E"])
                found=True; break
            if msg == b["Saída"].lower():
                st.info("Corresponde à Saída")
                st.write("Combinação S:", b["Combinação S"])
                st.write("Contexto S:", b["Contexto S"])
                st.write("Reação S:", b["Reação S"])
                found=True; break
        if not found:
            resp, rea, combs = retrieve_response(test_msg, file)
            st.write("Resposta (por similaridade):", resp)
            st.write("Reação:", rea)
            st.write("Combinação S:", combs)

    # 6. Preview JSON ao vivo
    st.markdown("---")
    st.subheader("🔄 Preview JSON")
    live = [{
        "Arquivo":f["Arquivo"],
        "Índice mãe":f["Índice mãe"],
        "Índices filhos":compute_indices_filhos(f),
        "Blocos":f["Blocos"]
    } for f in st.session_state.arquivos]
    st.json(live)

    # 7. Exportar JSON no disco
    st.markdown("---")
    st.subheader("💾 Salvar JSON no diretório")
    if st.button("Salvar JSON no disco"):
        with open("adam_data.json","w",encoding="utf-8") as fp:
            json.dump(live, fp, ensure_ascii=False, indent=2)
        st.success("Arquivo 'adam_data.json' salvo em seu diretório.")

# Modo Usuário
else:
    st.header("👤 Modo Usuário")
    if not st.session_state.arquivos:
        st.write("Sem conteúdo disponível.")
        st.stop()

    name = st.text_input("Seu nome:")
    pref = st.text_input("Sua preferência:")
    msg  = st.text_input("Mensagem ou chave (ex.: '0.1, 0.2'):")

    if st.button("Enviar"):
        pub = [a for a in st.session_state.arquivos if a["Índice mãe"]!=0]
        if is_numeric_key(msg):
            keys=[float(x.strip()) for x in msg.split(',')]
            f = get_file_by_pref(pref) or pub[0]
            found=None
            for b in f["Blocos"]:
                if get_numeric_keys(b["Combinação E"])==keys:
                    found=b; break
            if found:
                st.write("Mensagem:",found["Saída"])
                st.write("Reação:",  found["Reação S"])
            else:
                st.write("Chave não encontrada.")
        else:
            f = get_file_by_pref(pref) or pub[0]
            resp,rea,_ = retrieve_response(msg,f)
            if name:
                resp=resp.replace("criadora",name)
            st.write("Mensagem:",resp)
            st.write("Reação:", rea)

st.write(f"Próximo token: 0.{st.session_state.token_counter}")
