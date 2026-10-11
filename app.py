import streamlit as st
from graph import app_graph

st.set_page_config(page_title="Multi-Agent AI", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Agent AI System")
st.caption("Multi agentic research system")

with st.sidebar:
    st.header("Agents")
    st.markdown("**🧭 Supervisor** — routes your query")
    st.markdown("**🔍 Web Search** — uses Tavily")
    st.markdown("**🌤️ Weather** — uses OpenWeatherMap")
    st.markdown("**✍️ Formatter** — uses Gemma (OpenRouter)")
    st.divider()
    st.info("Ask anything! Weather queries are auto-routed to the weather agent.")
    st.info("Ask anything! Web search queries are auto-routed to the web search agent.")

query = st.text_input("Ask me anything:", placeholder="e.g. What is the weather in London?")

if st.button("Submit", type="primary") and query.strip():
    with st.spinner("Processing..."):
        try:
            result = app_graph.invoke({"query": query})

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Route", result.get("route", "—").replace("_", " ").title())
            with col2:
                st.metric("Source", result.get("source", "—").replace("_", " ").title())

            st.divider()
            st.subheader("Answer")
            st.write(result.get("final_answer", "No answer generated."))

            with st.expander("Raw Data"):
                st.text(result.get("raw_result", ""))

        except Exception as e:
            st.error(f"Error: {e}")
