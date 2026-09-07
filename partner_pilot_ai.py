"""
AI-Powered Partner Outreach Automation (with Partner Discovery)
Production-Grade Version with SQLAlchemy Backend, Strict Error Boundaries, and Graceful Fallbacks.
"""

import os
import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# --- Database Setup (SQLAlchemy + SQLite) ---
DB_FILE = "sqlite:///partners.db"
Base = declarative_base()

class PartnerModel(Base):
    __tablename__ = "partners"
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String, unique=True, nullable=False)
    industry = Column(String, nullable=False)
    target_audience = Column(String, default="Unknown")
    company_size = Column(String, default="Unknown")
    partnership_notes = Column(Text, default="")
    status = Column(String, default="Not Contacted")
    campaign = Column(String, default="General Outreach")
    website = Column(String, default="")
    discovery_score = Column(String, default="")
    discovery_source = Column(String, default="Manual")
    source_url = Column(String, default="")

try:
    engine = create_engine(DB_FILE, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
except Exception as e:
    engine = None
    SessionLocal = None

# --- Scoring config ---
HIGH_RELEVANCE_INDUSTRIES = ["EdTech", "EdTech / Data Science", "FinTech Education"]
MEDIUM_RELEVANCE_INDUSTRIES = ["Professional Services", "Media & Entertainment", "IT Services"]

AUDIENCE_KEYWORDS = [
    "student", "students", "professional", "professionals", "exam",
    "career", "aspirant", "aspirants", "coder", "coders", "learning",
    "mba", "finance", "data scientist", "analysts"
]
POSITIVE_INTENT_KEYWORDS = [
    "eager", "actively seeking", "open to", "interested",
    "looking to", "exploring", "referral"
]
SIZE_SCORES = {"Large": 20, "Medium": 14, "Small": 8}

MAX_INDUSTRY_SCORE = 30
MAX_AUDIENCE_SCORE = 25
MAX_SIZE_SCORE = 20
MAX_INTENT_SCORE = 25


# --- Database Data Load/Save Helpers ---
def load_data() -> pd.DataFrame:
    default_columns = [
        "Company Name", "Industry", "Target Audience", "Company Size",
        "Partnership Notes", "Status", "Campaign", "Website",
        "Discovery Score", "Discovery Source", "Source URL"
    ]
    if not engine or not SessionLocal:
        return pd.DataFrame(columns=default_columns)
    
    session = SessionLocal()
    try:
        rows = session.query(PartnerModel).all()
        if not rows:
            df = pd.DataFrame(columns=default_columns)
        else:
            data = [{
                "Company Name": r.company_name,
                "Industry": r.industry,
                "Target Audience": r.target_audience,
                "Company Size": r.company_size,
                "Partnership Notes": r.partnership_notes,
                "Status": r.status,
                "Campaign": r.campaign,
                "Website": r.website,
                "Discovery Score": r.discovery_score,
                "Discovery Source": r.discovery_source,
                "Source URL": r.source_url,
            } for r in rows]
            df = pd.DataFrame(data)
    except Exception as e:
        df = pd.DataFrame(columns=default_columns)
    finally:
        session.close()

    for col, default in {"Status": "Not Contacted", "Campaign": "General Outreach", "Website": "", "Discovery Score": "", "Discovery Source": "Manual", "Source URL": ""}.items():
        if col not in df.columns:
            df[col] = default
    return df


def add_partner_db(name, industry, audience, size, notes, campaign="General Outreach",
                   website="", discovery_score="", discovery_source="Manual", source_url=""):
    if not SessionLocal:
        return False
    session = SessionLocal()
    try:
        existing = session.query(PartnerModel).filter_by(company_name=name).first()
        if existing:
            # Update existing if needed, or skip
            session.close()
            return True
        
        new_partner = PartnerModel(
            company_name=name,
            industry=industry,
            target_audience=audience,
            company_size=size,
            partnership_notes=notes,
            status="Not Contacted",
            campaign=campaign,
            website=website,
            discovery_score=str(discovery_score),
            discovery_source=discovery_source,
            source_url=source_url
        )
        session.add(new_partner)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()


def update_partner_status_db(company_name: str, new_status: str):
    if not SessionLocal:
        return
    session = SessionLocal()
    try:
        partner = session.query(PartnerModel).filter_by(company_name=company_name).first()
        if partner:
            partner.status = new_status
            session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


# --- Partner Score functions ---
def score_industry(industry: str) -> int:
    if industry in HIGH_RELEVANCE_INDUSTRIES:
        return MAX_INDUSTRY_SCORE
    elif industry in MEDIUM_RELEVANCE_INDUSTRIES:
        return int(MAX_INDUSTRY_SCORE / 2)
    return 5


def score_audience(target_audience: str) -> int:
    text = str(target_audience).lower()
    matches = sum(1 for k in AUDIENCE_KEYWORDS if k in text)
    return min(matches * 8, MAX_AUDIENCE_SCORE)


def score_size(company_size: str) -> int:
    return SIZE_SCORES.get(str(company_size).strip(), 5)


def score_intent(notes: str) -> int:
    text = str(notes).lower()
    matches = sum(1 for k in POSITIVE_INTENT_KEYWORDS if k in text)
    return min(matches * 12, MAX_INTENT_SCORE)


def classify_priority(score: int) -> str:
    if score >= 65:
        return "High"
    elif score >= 50:
        return "Medium"
    return "Low"


def score_partner(row: pd.Series) -> pd.Series:
    try:
        industry_pts = score_industry(row["Industry"])
        audience_pts = score_audience(row["Target Audience"])
        size_pts = score_size(row["Company Size"])
        intent_pts = score_intent(row["Partnership Notes"])
        total = industry_pts + audience_pts + size_pts + intent_pts

        return pd.Series({
            "Industry Score": industry_pts,
            "Audience Score": audience_pts,
            "Size Score": size_pts,
            "Intent Score": intent_pts,
            "Total Score": total,
            "Priority": classify_priority(total),
        })
    except Exception:
        return pd.Series({
            "Industry Score": 0, "Audience Score": 0, "Size Score": 0,
            "Intent Score": 0, "Total Score": 0, "Priority": "Low"
        })


# --- Outreach message generator ---
def build_rationale(notes: str) -> str:
    text = str(notes).lower()
    matched = [k for k in POSITIVE_INTENT_KEYWORDS if k in text]
    if matched:
        return f"Given your team's interest in {matched[0]} partnerships, "
    return "Given the overlap between our audiences, "


def generate_template_message(row: pd.Series) -> str:
    name = row["Company Name"]
    industry = row["Industry"]
    audience = row["Target Audience"]
    notes = row["Partnership Notes"]
    website = row.get("Website", "")

    website_line = f"\nI came across {name} at {website}. " if website else "\n"
    rationale = build_rationale(notes)

    return f"""Subject: Exploring a Partnership Between Our Teams and {name}

Hi {name} team,{website_line}
I hope you're doing well. I've been following the work {name} is doing in the
{industry} space, particularly around {audience.lower() if audience and audience != "Unknown" else "your audience"},
and I think there could be a strong opportunity for our organizations to collaborate.

{rationale}I believe a partnership focused on shared
audience growth, co-branded content, or referral opportunities could be
mutually beneficial for both teams.

Would you be open to a short 15-minute call next week to explore whether
this could be a good fit? Happy to work around your schedule.

Looking forward to hearing your thoughts.

Best regards,
[Your Name]
Business Development / Strategic Alliances""".strip()


def generate_ai_message(row: pd.Series) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_message = generate_template_message(row)
    if not api_key:
        return base_message
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite this outreach email to sound more natural, keeping the "
                    f"same length and facts unchanged:\n\n{base_message}"
                )
            }]
        )
        return response.content[0].text
    except Exception:
        return base_message


# --- AI Partner Discovery Engine ---
BLOCKED_DOMAINS = [
    "wikipedia.org", "reddit.com", "linkedin.com", "indeed.com", "glassdoor.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
    "quora.com", "medium.com", "news.google.com", "timesofindia.com",
    "economictimes.com", "businessinsider.com", "crunchbase.com", "pinterest.com"
]

INDUSTRY_KEYWORDS = {
    "EdTech": ["education", "edtech", "e-learning", "exam prep", "tutoring", "online courses", "learning platform"],
    "EdTech / Data Science": ["data science course", "analytics training", "data science bootcamp"],
    "FinTech Education": ["fintech education", "finance learning", "mba prep", "cat preparation"],
    "IT Services": ["software company", "it services", "cloud services", "saas", "technology solutions"],
    "Professional Services": ["career services", "consulting", "recruitment", "placement", "coaching institute"],
    "Media & Entertainment": ["media company", "content platform", "entertainment", "creator economy"],
    "Retail": ["retail", "retailer", "e-commerce", "online store", "shopping"],
    "Healthcare": ["healthcare", "hospital", "clinic", "medical", "health services", "pharmacy"],
}

DISCOVERY_AUDIENCE_KEYWORDS = AUDIENCE_KEYWORDS + [
    "consumers", "customers", "shoppers", "buyers", "clients", "users",
    "parents", "family", "families", "students", "professionals"
]

def generate_search_queries(description: str, market: str, audience: str, partnership_type: str, max_queries: int = 5) -> list:
    try:
        desc_words = [w.strip(".,") for w in description.split() if len(w) > 3][:5]
        desc_short = " ".join(desc_words[:3])
        first_audience = audience.split(",")[0].strip() if audience else ""
        first_type = partnership_type.split(",")[0].strip() if partnership_type else ""

        candidates = [
            f"{desc_short} companies {market}".strip(),
            f"{first_audience} platforms {market}".strip(),
            f"{desc_short} {first_type} {market}".strip(),
        ]

        queries = []
        for q in candidates:
            q = " ".join(q.split())
            if q and q not in queries:
                queries.append(q)
        return queries[:max_queries]
    except Exception:
        return ["partner companies market"]


def normalize_domain(url: str) -> str:
    if not url:
        return ""
    try:
        domain = url.lower().strip()
        domain = re.sub(r"^https?://", "", domain)
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain
    except Exception:
        return ""


def is_blocked_domain(domain: str) -> bool:
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def is_listicle_title(title: str) -> bool:
    try:
        text = title.lower()
        patterns = [
            r"^\d+\s+\w+",
            r"\btop\s*\d+\b",
            r"\bbest\s+\d+\b",
            r"\bhow\s+to\b",
            r"\b(vs\.?|versus)\b",
        ]
        return any(re.search(p, text) for p in patterns)
    except Exception:
        return False


def search_web(queries: list, api_key: str, max_results_per_query: int = 5):
    try:
        from tavily import TavilyClient
    except ImportError:
        return [], "The 'tavily-python' package isn't installed."

    try:
        client = TavilyClient(api_key=api_key)
    except Exception:
        return [], "Could not start the Tavily client -- check the API key."

    all_results = []
    for q in queries:
        try:
            response = client.search(q, max_results=max_results_per_query)
            for r in response.get("results", []):
                r["discovery_query"] = q
                all_results.append(r)
        except Exception:
            continue

    if not all_results:
        return [], "No results came back. Check your network or API key."
    return all_results, None


def guess_industry(text: str) -> str:
    text = text.lower()
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return industry
    return "Unknown"


def extract_company_candidates(raw_results: list) -> list:
    candidates = []
    for r in raw_results:
        try:
            url = r.get("url", "")
            domain = normalize_domain(url)
            if not domain or is_blocked_domain(domain):
                continue

            title = (r.get("title") or "").strip()
            if is_listicle_title(title):
                continue

            content = r.get("content") or ""
            combined_text = f"{title} {content}"

            name = re.split(r"[|\-–—:]", title)[0].strip() if title else domain
            if not name:
                name = domain

            industry = guess_industry(combined_text)
            matched_audience = sorted(set(k for k in DISCOVERY_AUDIENCE_KEYWORDS if k in content.lower()))
            audience = ", ".join(matched_audience) if matched_audience else "Unknown"

            matched_intent = sorted(set(k for k in POSITIVE_INTENT_KEYWORDS if k in content.lower()))
            partnership_signal = ", ".join(matched_intent) if matched_intent else "Unknown"

            candidates.append({
                "company_name": name,
                "website": domain,
                "industry": industry,
                "target_audience": audience,
                "description": content[:300],
                "company_size": "Unknown",
                "partnership_signals": partnership_signal,
                "source_url": url,
                "discovery_query": r.get("discovery_query", ""),
            })
        except Exception:
            continue
    return candidates


def deduplicate_companies(candidates: list) -> list:
    seen_domains, seen_names, unique = set(), set(), []
    for c in candidates:
        try:
            name_key = c["company_name"].strip().lower()
            if c["website"] in seen_domains or name_key in seen_names:
                continue
            seen_domains.add(c["website"])
            seen_names.add(name_key)
            unique.append(c)
        except Exception:
            continue
    return unique


def compute_semantic_relevance(description: str, content: str) -> float:
    if not description or not content or len(description.strip()) < 3 or len(content.strip()) < 3:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([description, content])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception:
        return 0.0


def calculate_discovery_score(candidate: dict, market: str, audience_input: str, description: str) -> dict:
    try:
        full_text = (candidate.get("company_name", "") + " " + candidate["description"]).lower()
        industry_pts = 30 if candidate["industry"] != "Unknown" else 10

        if candidate["target_audience"] != "Unknown":
            audience_pts = 20
        else:
            audience_pts = 10

        market_pts = 15 if market and market.lower() in full_text else 5
        signal_pts = 20 if candidate["partnership_signals"] != "Unknown" else 0
        
        similarity = compute_semantic_relevance(description, candidate["description"])
        relevance_pts = round(similarity * 15, 1)

        total = industry_pts + audience_pts + market_pts + signal_pts + relevance_pts
        return {
            "Industry Match": industry_pts, "Audience Match": audience_pts,
            "Market Match": market_pts, "Partnership Signal": signal_pts,
            "Search Relevance": relevance_pts, "Discovery Score": min(round(total), 100),
        }
    except Exception:
        return {"Discovery Score": 50}


def classify_discovery_score(score: int) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Strong"
    elif score >= 60:
        return "Potential"
    return "Weak"


def prepare_partner_record(candidate: dict) -> dict:
    notes = "Discovered through web search."
    if candidate.get("partnership_signals") and candidate["partnership_signals"] != "Unknown":
        notes += f" Signals found: {candidate['partnership_signals']}."
    else:
        notes += " Verify before contacting."

    return {
        "Company Name": candidate["company_name"],
        "Industry": candidate["industry"],
        "Target Audience": candidate["target_audience"],
        "Company Size": candidate["company_size"],
        "Partnership Notes": notes,
        "Website": candidate["website"],
        "Discovery Score": candidate.get("Discovery Score", ""),
        "Discovery Source": "Web Search",
        "Source URL": candidate["source_url"],
    }


# ============================================================================
# APP UI
# ============================================================================
st.set_page_config(page_title="PartnerPilot AI", layout="wide")

st.title("🤝 PartnerPilot AI")
st.caption("Discover, score, rank, and reach out to potential partners safely and efficiently.")

# Load Data safely
try:
    df = load_data()
    if not df.empty:
        scores = df.apply(score_partner, axis=1)
        df = pd.concat([df, scores], axis=1)
    else:
        # Provide default empty structure with score cols if empty
        for col in ["Industry Score", "Audience Score", "Size Score", "Intent Score", "Total Score", "Priority"]:
            df[col] = 0 if "Score" in col else "Low"
except Exception as e:
    st.error("Error loading partner database. Initializing fallback view.")
    df = pd.DataFrame(columns=["Company Name", "Industry", "Target Audience", "Company Size", "Partnership Notes", "Status", "Campaign", "Website", "Total Score", "Priority"])

# Sidebar Filters
st.sidebar.header("Filters")
available_priorities = ["High", "Medium", "Low"]
priority_filter = st.sidebar.multiselect("Priority", options=available_priorities, default=available_priorities)

industries = sorted(df["Industry"].unique().tolist()) if not df.empty and "Industry" in df.columns else []
industry_filter = st.sidebar.multiselect("Industry", options=industries, default=industries)

campaigns = sorted(df["Campaign"].unique().tolist()) if not df.empty and "Campaign" in df.columns else []
campaign_filter = st.sidebar.multiselect("Campaign", options=campaigns, default=campaigns)

if not df.empty and "Priority" in df.columns and "Industry" in df.columns and "Campaign" in df.columns:
    filtered_df = df[
        df["Priority"].isin(priority_filter) & df["Industry"].isin(industry_filter) & df["Campaign"].isin(campaign_filter)
    ]
else:
    filtered_df = df

tab_discover, tab_dashboard, tab_manage, tab_outreach = st.tabs([
    "🔎 Discover", "📊 Dashboard & Ranking", "✍️ Add Partner", "✉️ Outreach & Follow-up"
])

# ============================================================================
# TAB 1: DISCOVER
# ============================================================================
with tab_discover:
    st.subheader("🔎 AI Partner Discovery")
    st.caption("Search the web safely for real companies. Review and approve before adding.")

    with st.expander("Discovery settings", expanded=True):
        d_description = st.text_area(
            "Partnership requirement",
            placeholder="e.g. Find education and technology partners for an exam-preparation platform"
        )
        dc1, dc2 = st.columns(2)
        d_market = dc1.text_input("Target market", placeholder="e.g. India")
        d_audience = dc2.text_input("Target audience", placeholder="e.g. College students, MBA aspirants")
        d_type = st.text_input("Partnership type", placeholder="e.g. Referral partnerships, co-marketing")
        dc3, dc4 = st.columns(2)
        d_count = dc3.number_input("Number of companies to discover", min_value=1, max_value=20, value=5)
        d_min_score = dc4.slider("Minimum discovery score", 0, 100, 50)
        discover_clicked = st.button("🔍 Discover Partners")

    if discover_clicked:
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_key:
            st.warning("Tavily API key not found in environment variables. Web discovery requires an active key.")
        elif not d_description.strip():
            st.warning("Please enter a partnership requirement description first.")
        else:
            with st.spinner("Searching the web securely..."):
                queries = generate_search_queries(d_description, d_market, d_audience, d_type)
                raw_results, error = search_web(queries, tavily_key, max_results_per_query=3)

            if error:
                st.warning(f"Discovery notice: {error}")
            else:
                candidates = extract_company_candidates(raw_results)
                candidates = deduplicate_companies(candidates)
                for c in candidates:
                    c.update(calculate_discovery_score(c, d_market, d_audience, d_description))
                    c["Match Level"] = classify_discovery_score(c["Discovery Score"])
                candidates = [c for c in candidates if c["Discovery Score"] >= d_min_score]
                candidates = sorted(candidates, key=lambda c: c["Discovery Score"], reverse=True)[:d_count]

                st.session_state["discovery_candidates"] = candidates
                st.session_state["discovery_stats"] = {"discovered": len(raw_results), "verified": len(candidates)}

                if not candidates:
                    st.info("No companies met the threshold. Try lowering the score slider or broadening your description.")

    if st.session_state.get("discovery_candidates"):
        st.markdown("#### Discovery Results — Review Before Adding")
        selected_indices = []
        for i, c in enumerate(st.session_state["discovery_candidates"]):
            with st.container(border=True):
                cc1, cc2 = st.columns([5, 1])
                with cc1:
                    st.markdown(f"**{c['company_name']}** — {c['industry']} · [{c['website']}](https://{c['website']})")
                    preview = c["description"][:150] + ("..." if len(c["description"]) > 150 else "")
                    st.caption(preview)
                    st.caption(f"Discovery Score: {c['Discovery Score']}/100 ({c['Match Level']})")
                with cc2:
                    if st.checkbox("Select", key=f"disc_sel_{i}"):
                        selected_indices.append(i)

        if st.button("➕ Add Selected Partners"):
            if not selected_indices:
                st.warning("Select at least one company first.")
            else:
                added = 0
                for i in selected_indices:
                    record = prepare_partner_record(st.session_state["discovery_candidates"][i])
                    success = add_partner_db(
                        record["Company Name"], record["Industry"], record["Target Audience"],
                        record["Company Size"], record["Partnership Notes"], campaign="Web Discovery",
                        website=record["Website"], discovery_score=record["Discovery Score"],
                        discovery_source=record["Discovery Source"], source_url=record["Source URL"]
                    )
                    if success:
                        added += 1
                stats = st.session_state.get("discovery_stats", {})
                stats["added"] = stats.get("added", 0) + added
                st.session_state["discovery_stats"] = stats
                st.success(f"Successfully added {added} companies to the database.")
                st.session_state.pop("discovery_candidates", None)
                st.rerun()

# ============================================================================
# TAB 2: DASHBOARD & RANKING
# ============================================================================
with tab_dashboard:
    st.subheader("📊 Dashboard Overview")
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Potential Partners", len(df))
        col2.metric("High Priority Partners", int((df["Priority"] == "High").sum()) if "Priority" in df.columns else 0)
        col3.metric("Average Partner Score", round(df["Total Score"].mean(), 1) if "Total Score" in df.columns else 0)

        if "Priority" in df.columns:
            priority_counts = df["Priority"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
            st.bar_chart(priority_counts)

        st.subheader("📋 Partner Ranking")
        display_cols = [c for c in ["Company Name", "Industry", "Company Size", "Campaign", "Total Score", "Priority", "Status"] if c in filtered_df.columns]
        if not filtered_df.empty and display_cols:
            ranked = filtered_df.sort_values("Total Score", ascending=False)
            st.dataframe(ranked[display_cols], use_container_width=True, hide_index=True)
            
            csv_data = ranked[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Ranked List as CSV", data=csv_data, file_name="partner_ranking.csv", mime="text/csv")
        else:
            st.info("No data available under the current filters.")
    else:
        st.info("Database is currently empty. Use the 'Discover' or 'Add Partner' tab to add records.")

# ============================================================================
# TAB 3: ADD PARTNER MANUALLY
# ============================================================================
with tab_manage:
    st.subheader("✍️ Add Partner Manually")
    with st.form("add_partner_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Company Name")
        industry = c2.text_input("Industry")
        audience = st.text_input("Target Audience")
        size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
        notes = st.text_area("Partnership Notes")
        campaign = st.text_input("Campaign", value="General Outreach")
        submitted = st.form_submit_button("Add Partner")
        if submitted:
            if name.strip() and industry.strip():
                success = add_partner_db(name, industry, audience, size, notes, campaign=campaign)
                if success:
                    st.success(f"Added {name} successfully.")
                    st.rerun()
                else:
                    st.warning("Could not save partner (company name might already exist).")
            else:
                st.warning("Company Name and Industry are required.")

# ============================================================================
# TAB 4: OUTREACH & FOLLOW-UP
# ============================================================================
with tab_outreach:
    st.subheader("✉️ Outreach Message Generator")
    if not df.empty and "Company Name" in df.columns:
        company_list = df["Company Name"].tolist()
        selected_company = st.selectbox("Select a partner", options=company_list)
        selected_row = df[df["Company Name"] == selected_company].iloc[0]

        use_ai = st.checkbox("Enhance message with AI (requires ANTHROPIC_API_KEY)", value=False)

        if st.button("Generate Outreach Message"):
            message = generate_ai_message(selected_row) if use_ai else generate_template_message(selected_row)
            st.text_area("Generated Message", value=message, height=280)

        st.subheader("📌 Follow-up Tracker")
        status_options = ["Not Contacted", "Contacted", "Replied", "Meeting", "Converted"]
        current_status = selected_row.get("Status", "Not Contacted")
        status_index = status_options.index(current_status) if current_status in status_options else 0
        
        new_status = st.selectbox(
            f"Update status for {selected_company}",
            options=status_options,
            index=status_index
        )

        if st.button("Save Status"):
            update_partner_status_db(selected_company, new_status)
            st.success(f"Status for {selected_company} updated to '{new_status}'.")
            st.rerun()
    else:
        st.info("No partners available for outreach yet.")

st.divider()
st.caption("Production-Ready Student Project. Safe exception boundaries enabled.")
