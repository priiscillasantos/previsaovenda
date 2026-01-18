
# ==========================================
# Streamlit - Previsão de renda (CRISP-DM)
# Mais leve, com história + navegação
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import streamlit.components.v1 as components


# ------------------------------
# 1) Configurações gerais
# ------------------------------
st.set_page_config(
    page_title="Previsão de renda - Implantação (CRISP-DM)",
    page_icon="💰",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "input" / "previsao_de_renda.csv"
MODEL_PATH = BASE_DIR / "output" / "modelo_final_randomforest.pkl"


# ------------------------------
# 2) Funções (com cache)
# ------------------------------
@st.cache_data(show_spinner=False)
def carregar_dados(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Se veio a coluna "Unnamed: 0" (índice), eu removo pra não atrapalhar
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Garantindo data_ref como datetime e criando ano_ref/mes_ref (se existir data_ref)
    if "data_ref" in df.columns:
        df["data_ref"] = pd.to_datetime(df["data_ref"], errors="coerce")
        df["ano_ref"] = df["data_ref"].dt.year
        df["mes_ref"] = df["data_ref"].dt.month

    return df


@st.cache_resource(show_spinner=False)
def carregar_modelo(model_path: Path):
    return joblib.load(model_path)


def encontrar_html_relatorio(base_dir: Path) -> list[Path]:
    # Procura na raiz e também dentro do output
    candidatos = []
    candidatos += list(base_dir.glob("renda_analisys*.html"))
    candidatos += list((base_dir / "output").glob("renda_analisys*.html"))
    # Remove duplicados mantendo ordem
    vistos = set()
    final = []
    for p in candidatos:
        if p.resolve() not in vistos:
            final.append(p)
            vistos.add(p.resolve())
    return final


def aplicar_filtros(df: pd.DataFrame,
                    data_ini=None,
                    data_fim=None,
                    tipos=None,
                    educs=None,
                    sexos=None) -> pd.DataFrame:
    df_f = df.copy()

    # filtro por data_ref
    if "data_ref" in df_f.columns and (data_ini is not None) and (data_fim is not None):
        df_f = df_f[(df_f["data_ref"] >= pd.to_datetime(data_ini)) & (df_f["data_ref"] <= pd.to_datetime(data_fim))]

    # filtros categóricos
    if tipos and "tipo_renda" in df_f.columns:
        df_f = df_f[df_f["tipo_renda"].isin(tipos)]

    if educs and "educacao" in df_f.columns:
        df_f = df_f[df_f["educacao"].isin(educs)]

    if sexos and "sexo" in df_f.columns:
        df_f = df_f[df_f["sexo"].isin(sexos)]

    return df_f


def kpis_basicos(df: pd.DataFrame) -> dict:
    k = {}
    k["linhas"] = int(df.shape[0])
    k["colunas"] = int(df.shape[1])

    if "renda" in df.columns and df.shape[0] > 0:
        renda = df["renda"].dropna()
        k["media"] = float(renda.mean()) if len(renda) else np.nan
        k["mediana"] = float(renda.median()) if len(renda) else np.nan
        k["p90"] = float(renda.quantile(0.90)) if len(renda) else np.nan
        k["p10"] = float(renda.quantile(0.10)) if len(renda) else np.nan
    else:
        k["media"] = np.nan
        k["mediana"] = np.nan
        k["p90"] = np.nan
        k["p10"] = np.nan

    return k


def tabela_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return pd.DataFrame(columns=["coluna", "faltantes", "pct"])
    out = pd.DataFrame({
        "coluna": miss.index,
        "faltantes": miss.values,
        "pct": (miss.values / len(df) * 100).round(2)
    })
    return out


def histograma_simples(series: pd.Series, bins: int = 30) -> pd.DataFrame:
    s = series.dropna().astype(float)
    if s.empty:
        return pd.DataFrame({"faixa": [], "contagem": []})

    counts, edges = np.histogram(s, bins=bins)
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"{edges[i]:.0f}–{edges[i+1]:.0f}")
    return pd.DataFrame({"faixa": labels, "contagem": counts})


def garantir_ano_mes(df: pd.DataFrame) -> pd.DataFrame:
    # se não existir ano_ref/mes_ref, tento criar via data_ref
    df2 = df.copy()
    if ("ano_ref" not in df2.columns or "mes_ref" not in df2.columns) and "data_ref" in df2.columns:
        df2["ano_ref"] = pd.to_datetime(df2["data_ref"], errors="coerce").dt.year
        df2["mes_ref"] = pd.to_datetime(df2["data_ref"], errors="coerce").dt.month
    return df2


# ------------------------------
# 3) Carregamento com validação
# ------------------------------
st.title("💰 Previsão de renda - Implantação (CRISP-DM)")
st.caption("Aplicação em Streamlit para **contar a história dos dados** e usar o **modelo treinado** para previsões.")

with st.expander("📦 Arquivos esperados (clique para ver)"):
    st.code(f"CSV:   {DATA_PATH}\nMODELO:{MODEL_PATH}")

if not DATA_PATH.exists():
    st.error("Não encontrei o CSV em: ./input/previsao_de_renda.csv")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Não encontrei o modelo em: ./output/modelo_final_randomforest.pkl")
    st.stop()

df = carregar_dados(DATA_PATH)
df = garantir_ano_mes(df)

# ------------------------------
# 4) Sidebar (filtros + navegação)
# ------------------------------
st.sidebar.header("Filtros")

# datas
if "data_ref" in df.columns and df["data_ref"].notna().any():
    dmin = df["data_ref"].min().date()
    dmax = df["data_ref"].max().date()
    periodo = st.sidebar.date_input("Período (data_ref)", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(periodo, tuple) and len(periodo) == 2:
        data_ini, data_fim = periodo
    else:
        data_ini, data_fim = dmin, dmax
else:
    data_ini, data_fim = None, None

# filtros categóricos
tipos_opts = sorted(df["tipo_renda"].dropna().unique().tolist()) if "tipo_renda" in df.columns else []
educ_opts = sorted(df["educacao"].dropna().unique().tolist()) if "educacao" in df.columns else []
sexo_opts = sorted(df["sexo"].dropna().unique().tolist()) if "sexo" in df.columns else []

tipos_sel = st.sidebar.multiselect("Tipo de renda", options=tipos_opts, default=tipos_opts[:4] if len(tipos_opts) > 4 else tipos_opts)
educ_sel = st.sidebar.multiselect("Educação", options=educ_opts, default=educ_opts[:4] if len(educ_opts) > 4 else educ_opts)
sexo_sel = st.sidebar.multiselect("Sexo", options=sexo_opts, default=sexo_opts)

amostra_graficos = st.sidebar.slider("Amostra para gráficos (performance)", min_value=500, max_value=8000, value=3000, step=500)

st.sidebar.divider()
pagina = st.sidebar.radio(
    "Navegação",
    ["Visão geral", "Análises", "Relatório HTML", "Previsão"],
    index=0
)

df_f = aplicar_filtros(df, data_ini, data_fim, tipos_sel, educ_sel, sexo_sel)

# amostra para gráficos (pra não travar)
if len(df_f) > amostra_graficos:
    df_plot = df_f.sample(amostra_graficos, random_state=42)
else:
    df_plot = df_f.copy()


# ------------------------------
# 5) Páginas
# ------------------------------
if pagina == "Visão geral":
    st.subheader("1) O recorte atual em linguagem simples")

    k = kpis_basicos(df_f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", f"{k['linhas']:,}".replace(",", "."))
    c2.metric("Colunas", f"{k['colunas']:,}".replace(",", "."))
    c3.metric("Renda média", f"{k['media']:.2f}" if np.isfinite(k["media"]) else "—")
    c4.metric("Renda mediana", f"{k['mediana']:.2f}" if np.isfinite(k["mediana"]) else "—")

    if np.isfinite(k["p90"]):
        st.markdown(
            f"""
- A **renda mediana** é **{k['mediana']:.2f}** e a **média** é **{k['media']:.2f}** (média maior costuma indicar **cauda à direita**: alguns valores bem altos puxam a média).
- Os **10% maiores valores (p90)** ficam acima de **{k['p90']:.2f}**.
- Os **10% menores valores (p10)** ficam abaixo de **{k['p10']:.2f}**.
            """
        )

    # comparação rápida por tipo_renda (mediana)
    if "tipo_renda" in df_f.columns and "renda" in df_f.columns and len(df_f) > 0:
        med_por_tipo = df_f.groupby("tipo_renda")["renda"].median().sort_values(ascending=False)
        if len(med_por_tipo) >= 2:
            top_tipo = med_por_tipo.index[0]
            bot_tipo = med_por_tipo.index[-1]
            st.info(f"Comparando mediana por tipo de renda: **{top_tipo}** tem mediana maior (~{med_por_tipo.iloc[0]:.2f}) do que **{bot_tipo}** (~{med_por_tipo.iloc[-1]:.2f}).")

    with st.expander("Ver amostra dos dados (primeiras 15 linhas)"):
        st.dataframe(df_f.head(15), use_container_width=True, height=350)

    st.subheader("2) Qualidade dos dados (faltantes)")
    falt = tabela_faltantes(df_f)
    if falt.empty:
        st.success("✅ Neste recorte, não há valores faltantes.")
    else:
        st.dataframe(falt, use_container_width=True, height=220)
        st.warning(
            "Se tiver muita falta em uma coluna importante, isso pode afetar as análises e o modelo. "
            "Aqui a ideia é deixar isso claro logo no começo."
        )


elif pagina == "Análises":
    st.subheader("3) Distribuição da renda (visão rápida)")

    if "renda" not in df_plot.columns or df_plot["renda"].dropna().empty:
        st.warning("Não encontrei a coluna `renda` (ou ela está vazia) neste recorte.")
    else:
        usar_log = st.checkbox("Ver distribuição em log (log1p)", value=False)
        renda_s = df_plot["renda"].copy()

        if usar_log:
            renda_s = np.log1p(renda_s)

        hist = histograma_simples(renda_s, bins=30)
        if hist.empty:
            st.warning("Não consegui montar o histograma neste recorte.")
        else:
            # gráfico leve (sem matplotlib): bar_chart com dataframe pequeno
            st.bar_chart(hist.set_index("faixa")["contagem"], height=320)

        st.caption("Dica: log1p ajuda quando existe muita diferença entre rendas baixas e altas (cauda longa).")

    st.subheader("4) Renda por grupos (o que mais explica diferenças?)")

    colA, colB = st.columns(2)

    with colA:
        if "tipo_renda" in df_plot.columns and "renda" in df_plot.columns:
            tab_tipo = (df_plot.groupby("tipo_renda")["renda"]
                        .agg(contagem="count", media="mean", mediana="median")
                        .sort_values("mediana", ascending=False)
                        .round(2))
            st.markdown("**Por tipo_renda**")
            st.dataframe(tab_tipo, use_container_width=True, height=320)
        else:
            st.info("Não encontrei `tipo_renda` e/ou `renda` para esse resumo.")

    with colB:
        if "educacao" in df_plot.columns and "renda" in df_plot.columns:
            tab_edu = (df_plot.groupby("educacao")["renda"]
                       .agg(contagem="count", media="mean", mediana="median")
                       .sort_values("mediana", ascending=False)
                       .round(2))
            st.markdown("**Por educacao**")
            st.dataframe(tab_edu, use_container_width=True, height=320)
        else:
            st.info("Não encontrei `educacao` e/ou `renda` para esse resumo.")

    st.subheader("5) Evolução no tempo (se data_ref existir)")
    if "data_ref" in df_plot.columns and df_plot["data_ref"].notna().any() and "renda" in df_plot.columns:
        # agrego por mês para ficar leve
        tmp = df_plot.dropna(subset=["data_ref"]).copy()
        tmp["mes"] = tmp["data_ref"].dt.to_period("M").dt.to_timestamp()
        serie = tmp.groupby("mes")["renda"].median().sort_index()
        st.line_chart(serie, height=300)
        st.caption("Aqui eu uso a **mediana mensal**, porque ela é mais estável do que a média em dados com outliers.")
    else:
        st.info("Sem `data_ref` válida neste recorte, não dá pra mostrar a evolução temporal.")


elif pagina == "Relatório HTML":
    st.subheader("📄 Relatório HTML (opcional)")

    st.write(
        "Se você gerou um HTML no notebook (ex.: ydata-profiling/pandas-profiling), "
        "o ideal é salvar como **renda_analisys.html**. "
        "Este app procura tanto na **raiz do projeto** quanto dentro de **output/**."
    )

    htmls = encontrar_html_relatorio(BASE_DIR)
    if not htmls:
        st.warning("Não encontrei nenhum arquivo `renda_analisys*.html` na raiz nem em `output/`.")
        st.code("Sugestão: copie o HTML para a mesma pasta do .py OU mantenha em output/ com extensão .html.")
    else:
        escolha = st.selectbox("Escolha o HTML para exibir", options=htmls, format_func=lambda p: str(p.name))
        st.caption(f"Lendo arquivo: {escolha}")

        try:
            conteudo = escolha.read_text(encoding="utf-8", errors="ignore")
            # iframe interno com scroll: não trava a página
            components.html(conteudo, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Falha ao abrir o HTML: {e}")


elif pagina == "Previsão":
    st.subheader("🔮 Previsão de renda (usando o modelo treinado)")

    # carrego o modelo aqui pra não pesar nas outras páginas
    modelo = carregar_modelo(MODEL_PATH)

    # Features esperadas (padrão do seu projeto)
    FEATURES = [
        "sexo",
        "posse_de_veiculo",
        "posse_de_imovel",
        "qtd_filhos",
        "tipo_renda",
        "educacao",
        "estado_civil",
        "tipo_residencia",
        "idade",
        "tempo_emprego",
        "qt_pessoas_residencia",
        "ano_ref",
        "mes_ref"
    ]

    st.write("Preencha um perfil e eu retorno a renda prevista.")

    # opções dos selects
    tipo_renda_opts = sorted(df["tipo_renda"].dropna().unique().tolist()) if "tipo_renda" in df.columns else []
    educ_opts = sorted(df["educacao"].dropna().unique().tolist()) if "educacao" in df.columns else []
    estado_civil_opts = sorted(df["estado_civil"].dropna().unique().tolist()) if "estado_civil" in df.columns else []
    tipo_res_opts = sorted(df["tipo_residencia"].dropna().unique().tolist()) if "tipo_residencia" in df.columns else []
    sexo_opts = sorted(df["sexo"].dropna().unique().tolist()) if "sexo" in df.columns else ["F", "M"]

    # defaults de ano/mes pegando do dataset
    anos = sorted(df["ano_ref"].dropna().unique().tolist()) if "ano_ref" in df.columns else [2015]
    meses = sorted(df["mes_ref"].dropna().unique().tolist()) if "mes_ref" in df.columns else list(range(1, 13))

    with st.form("form_previsao"):
        col1, col2, col3 = st.columns(3)

        with col1:
            sexo = st.selectbox("sexo", options=sexo_opts)
            tipo_renda = st.selectbox("tipo_renda", options=tipo_renda_opts) if tipo_renda_opts else st.text_input("tipo_renda")
            educacao = st.selectbox("educacao", options=educ_opts) if educ_opts else st.text_input("educacao")
            estado_civil = st.selectbox("estado_civil", options=estado_civil_opts) if estado_civil_opts else st.text_input("estado_civil")

        with col2:
            tipo_residencia = st.selectbox("tipo_residencia", options=tipo_res_opts) if tipo_res_opts else st.text_input("tipo_residencia")
            idade = st.number_input("idade", min_value=14, max_value=100, value=30, step=1)
            qtd_filhos = st.number_input("qtd_filhos", min_value=0, max_value=15, value=0, step=1)
            qt_pessoas_residencia = st.number_input("qt_pessoas_residencia", min_value=1, max_value=20, value=2, step=1)

        with col3:
            posse_de_veiculo = st.checkbox("posse_de_veiculo", value=False)
            posse_de_imovel = st.checkbox("posse_de_imovel", value=False)

            # tempo_emprego: deixo opção de "não sei" pra virar NaN e o imputador do pipeline resolver
            nao_sei_tempo = st.checkbox("Não sei o tempo_emprego (deixar em branco)", value=False)
            if nao_sei_tempo:
                tempo_emprego = np.nan
                st.caption("Vou enviar `tempo_emprego = NaN` para o modelo (o pipeline deve imputar).")
            else:
                tempo_emprego = st.number_input("tempo_emprego (anos)", min_value=0.0, max_value=60.0, value=5.0, step=0.1)

            ano_ref = st.selectbox("ano_ref", options=anos, index=len(anos)-1 if len(anos) else 0)
            mes_ref = st.selectbox("mes_ref", options=meses, index=len(meses)-1 if len(meses) else 0)

        submit = st.form_submit_button("Prever renda")

    if submit:
        registro = pd.DataFrame([{
            "sexo": sexo,
            "posse_de_veiculo": posse_de_veiculo,
            "posse_de_imovel": posse_de_imovel,
            "qtd_filhos": qtd_filhos,
            "tipo_renda": tipo_renda,
            "educacao": educacao,
            "estado_civil": estado_civil,
            "tipo_residencia": tipo_residencia,
            "idade": idade,
            "tempo_emprego": tempo_emprego,
            "qt_pessoas_residencia": qt_pessoas_residencia,
            "ano_ref": int(ano_ref) if pd.notna(ano_ref) else np.nan,
            "mes_ref": int(mes_ref) if pd.notna(mes_ref) else np.nan
        }])

        # reordeno para as features esperadas
        for col in FEATURES:
            if col not in registro.columns:
                registro[col] = np.nan
        registro = registro[FEATURES]

        pred = float(modelo.predict(registro)[0])

        st.success(f"✅ Renda prevista: **{pred:,.2f}**".replace(",", "X").replace(".", ",").replace("X", "."))
        with st.expander("Ver os dados enviados para o modelo"):
            st.dataframe(registro, use_container_width=True)

