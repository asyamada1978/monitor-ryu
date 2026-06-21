#!/usr/bin/env python3
"""
Monitor de Vagas Ryu — radar automático (v2: dedup + Adzuna ampliada + Jooble).

Roda no GitHub Actions (semanal): busca vagas na Adzuna e na Jooble (Brasil),
remove duplicatas, ranqueia por similaridade com o perfil do Enzo (ML) e grava
em jobs.json. O painel (index.html) lê esse jobs.json e mostra as vagas sozinho.

Secrets (variáveis de ambiente):
  ADZUNA_APP_ID, ADZUNA_APP_KEY   (obrigatórias)
  JOOBLE_KEY                      (opcional — se ausente, usa só a Adzuna)
"""
import os
import json
import time
import unicodedata
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------ #
#  PERFIL DO ENZO (alvo do ML)                                       #
# ------------------------------------------------------------------ #
PROFILE = (
    "Estudante de Ciencias Economicas em formacao. Estagiario de financas em FP&A: "
    "planejamento financeiro, orcamento budget, forecast, analise de desempenho, "
    "pricing e analise de ASP. Desenvolve dashboards em Power BI, analise de dados "
    "de vendas, analise de credito e risco. Python e SQL. Ingles avancado. Setor "
    "farmaceutico. Interesse em FP&A, planejamento financeiro, controladoria, "
    "business intelligence, analise financeira, tesouraria, projecoes."
)
SKILLS = ["fpa", "planejamento financeiro", "orcamento", "budget", "forecast",
          "controladoria", "business intelligence", "power bi", "excel", "python",
          "sql", "pricing", "analise financeira", "dre", "tesouraria"]

# Buscas (ampliadas)
QUERIES = [
    "estagio financeiro", "estagio FP&A", "estagio planejamento financeiro",
    "estagio controladoria", "estagio business intelligence",
    "analista financeiro junior", "analista de planejamento financeiro", "analista FP&A",
    "controladoria junior", "planejamento financeiro junior",
    "analista de planejamento junior", "estagio planejamento",
    "estagio marketing digital", "analista de marketing junior",
    "estagio comercial", "analista comercial junior",
]

ROLE_KEYWORDS = ["estagio", "estagiario", "trainee", "analista financeiro",
                 "analista de planejamento", "planejamento financeiro", "fp&a", "fpa",
                 "controladoria", "orcamento", "budget", "forecast", "analise financeira",
                 "business intelligence", "analista junior", "analista jr", "tesouraria",
                 "inteligencia de negocios", "financeiro junior", "assistente financeiro",
                 "analista de custos"]
NEGATIVE_KEYWORDS = ["senior", "pleno", "gerente", "coordenador", "diretor",
                     "especialista", "supervisor", "head ", "manager", "lead "]
LOCATION_OK = ["sao paulo", "sp", "remoto", "remote", "hibrido", "brasil", "barueri",
               "osasco", "guarulhos", "embu", "santo amaro", "alphaville"]

# Precisão: exige nível certo no título + domínio financeiro no texto
LEVEL_TITLE = ["estag", "trainee", "junior", "jr", "assistente", "analista", "aprendiz", "intern"]
FINANCE_KW = ["financ", "fp&a", "fpa", "fp & a", "planejamento financ", "orcament", "budget",
              "forecast", "controladoria", "tesouraria", "custos", "analise financ",
              "business intelligence", "inteligencia de negocio", "projec", "contabil", "economia",
              "marketing", "comercial", "planejamento", "demanda"]
# domínios que NÃO são do Enzo — se aparecem no título sem termo-alvo, descarta
NEG_TITLE = ["recursos humanos", "juridic", "logistic", "suprimentos", "compras",
             "atendimento", "enfermagem", "laborator", "clinic", "design",
             "varejo", "manuten", "seguranca do trabalho", "produto", "ti ",
             "infraestrutura", "obras"]
FINANCE_TITLE = ["financ", "fp&a", "fpa", "fp & a", "planejamento", "orcament", "budget",
                 "forecast", "controladoria", "tesouraria", "custos", "contabil", "economia",
                 "business intelligence", "inteligencia de negocio", "projec",
                 "marketing", "comercial", "demanda"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "monitor-ryu/1.0"})

ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/br/search/{page}"
JOOBLE_URL = "https://jooble.org/api/{key}"
TOP_N = 40
MAX_DAYS_OLD = 21
PER_PAGE = 25
ADZUNA_PAGES = 2          # quantas páginas por busca na Adzuna
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs.json")


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def detect_tipo(title):
    t = norm(title)
    if "trainee" in t:
        return "Trainee"
    if "estag" in t:
        return "Estágio"
    if "junior" in t or " jr" in t or "analista" in t or "assistente" in t:
        return "Júnior"
    return "Outro"


def qualifies(job):
    title = norm(job["title"])
    text = norm(job["title"] + " " + job["company"] + " " + job["description"])
    # 1) precisa ter o NÍVEL certo no título (estágio/júnior/analista/assistente/trainee)
    if not any(k in title for k in LEVEL_TITLE):
        return False
    # 2) precisa ser do DOMÍNIO financeiro/BI
    if not any(k in text for k in FINANCE_KW):
        return False
    # 3) corta sênior/pleno/gestão
    if any(k in title for k in NEGATIVE_KEYWORDS):
        return False
    # 3b) corta outros domínios (marketing/vendas/RH/etc.) quando o título não é financeiro
    if any(k in title for k in NEG_TITLE) and not any(k in title for k in FINANCE_TITLE):
        return False
    # 4) localização compatível (SP/remoto/híbrido)
    loc = norm(job["location"])
    if loc and not any(k in loc for k in LOCATION_OK):
        return False
    return True


# ----------------------------- FONTES ----------------------------- #
def fetch_adzuna(app_id, app_key):
    out = []
    for q in QUERIES:
        for page in range(1, ADZUNA_PAGES + 1):
            try:
                r = SESSION.get(
                    ADZUNA_URL.format(page=page),
                    params={"app_id": app_id, "app_key": app_key, "what": q,
                            "where": "São Paulo", "results_per_page": PER_PAGE,
                            "max_days_old": MAX_DAYS_OLD, "content-type": "application/json"},
                    timeout=25,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"[adzuna] falha em '{q}' p{page}: {e}")
                break
            results = data.get("results", [])
            for item in results:
                url = item.get("redirect_url", "")
                if not url:
                    continue
                out.append({
                    "title": (item.get("title", "") or "").replace("<strong>", "").replace("</strong>", ""),
                    "company": (item.get("company") or {}).get("display_name", ""),
                    "location": (item.get("location") or {}).get("display_name", ""),
                    "description": item.get("description", ""),
                    "link": url, "created": item.get("created", ""), "source": "Adzuna",
                })
            if len(results) < PER_PAGE:
                break  # não há mais páginas
            time.sleep(0.3)
    print(f"[adzuna] {len(out)} vagas brutas coletadas")
    return out


def fetch_jooble(key):
    out = []
    kw = "estágio financeiro, FP&A, planejamento financeiro, analista financeiro júnior, controladoria, business intelligence"
    for page in ("1", "2"):
        try:
            r = SESSION.post(JOOBLE_URL.format(key=key),
                              json={"keywords": kw, "location": "São Paulo", "page": page},
                              timeout=25)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[jooble] falha p{page}: {e}")
            break
        for it in data.get("jobs", []):
            link = it.get("link", "")
            if not link:
                continue
            out.append({
                "title": it.get("title", ""), "company": it.get("company", ""),
                "location": it.get("location", ""), "description": it.get("snippet", ""),
                "link": link, "created": it.get("updated", ""), "source": "Jooble",
            })
        time.sleep(0.3)
    print(f"[jooble] {len(out)} vagas brutas coletadas")
    return out


def dedupe(jobs):
    """Remove repetidas: por link e por empresa+vaga."""
    seen_link, seen_key, uniq = set(), set(), []
    for j in jobs:
        key = norm(j["company"]) + "|" + norm(j["title"])
        if j["link"] in seen_link or key in seen_key:
            continue
        seen_link.add(j["link"])
        seen_key.add(key)
        uniq.append(j)
    return uniq


# ------------------------------ ML -------------------------------- #
def profile_document():
    return " ".join([norm(PROFILE)] + [norm(s) for s in SKILLS] * 3)


def score(jobs):
    if not jobs:
        return jobs
    texts = [norm(j["title"] + ". " + j["company"] + ". " + j["location"] + ". " + j["description"])
             for j in jobs]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
    tfidf = vec.fit_transform([profile_document()] + texts)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:]).ravel()
    if sim.max() > sim.min():
        sim = (sim - sim.min()) / (sim.max() - sim.min())
    # boost: termos-alvo no título + preferência leve por estágio (foco do Enzo agora)
    HIGH = ["fp&a", "fpa", "planejamento financ", "controladoria", "orcament", "budget",
            "forecast", "estagio financ", "analise financ", "business intelligence"]
    for i, j in enumerate(jobs):
        t = norm(j["title"])
        b = 0.12 if any(k in t for k in HIGH) else 0.0
        if "estag" in t:
            b += 0.05
        sim[i] = min(1.0, sim[i] + b)
    for j, s in zip(jobs, sim):
        j["score"] = float(round(s, 4))
    jobs.sort(key=lambda x: x["score"], reverse=True)
    return jobs


def fit_label(s):
    return "Alto" if s >= 0.60 else ("Médio-Alto" if s >= 0.35 else "Médio")


def elig_label(tipo):
    return {"Estágio": "Elegível (cursando)", "Júnior": "Confirmar elegibilidade",
            "Trainee": "Só após formar (2028)"}.get(tipo, "Confirmar elegibilidade")


def to_card(j):
    tipo = detect_tipo(j["title"])
    pct = int(round(j["score"] * 100))
    return {
        "id": str(abs(hash(j["link"])) % (10 ** 12)),
        "tipo": tipo, "empresa": j["company"] or "—", "vaga": j["title"],
        "area": "", "local": j["location"] or "—",
        "fit": fit_label(j["score"]), "elig": elig_label(tipo),
        "why": f"Aderência {pct}% ao perfil · radar ({j.get('source', 'fonte')}).",
        "prazo": "Recente (≤21 dias)", "link": j["link"], "star": False,
        "score": j["score"],
    }


def main():
    app_id = os.getenv("ADZUNA_APP_ID", "").strip()
    app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
    jooble_key = os.getenv("JOOBLE_KEY", "").strip()

    raw = []
    if app_id and app_key:
        raw += fetch_adzuna(app_id, app_key)
    else:
        print("[erro] defina ADZUNA_APP_ID e ADZUNA_APP_KEY nos Secrets.")
    if jooble_key:
        raw += fetch_jooble(jooble_key)
    else:
        print("[info] JOOBLE_KEY ausente — usando só a Adzuna (pode adicionar depois).")

    print(f"[radar] {len(raw)} vagas brutas no total")
    raw = dedupe(raw)
    print(f"[radar] {len(raw)} apos remover duplicatas")

    qualified = [j for j in raw if qualifies(j)]
    print(f"[radar] {len(qualified)} passaram no filtro de perfil")

    score(qualified)
    cards = [to_card(j) for j in qualified[:TOP_N]]

    brt = timezone(timedelta(hours=-3))
    payload = {"updated_at": datetime.now(brt).strftime("%d/%m/%Y %H:%M"),
               "count": len(cards), "jobs": cards}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[radar] {len(cards)} vagas gravadas em jobs.json")


if __name__ == "__main__":
    main()
