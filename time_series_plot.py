import altair as alt
import streamlit as st

@st.experimental_memo(ttl=60 * 60 * 24)
def get_chart(data):
    hover = alt.selection_single(
        fields=["Date-Time"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Portfolio Equity")
        .mark_line()
        .encode(
            alt.X("Date-Time", scale=alt.Scale(zero=False)),
            alt.Y("Equity", scale=alt.Scale(zero=False)),
            # x="Date-Time",
            # y="Equity",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            alt.X("Date-Time", scale=alt.Scale(zero=False)),
            alt.Y("Equity", scale=alt.Scale(zero=False)),
            # x="Date-Time",
            # y="Equity",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date-Time", title="Date-Time"),
                alt.Tooltip("Equity", title="Equity (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()