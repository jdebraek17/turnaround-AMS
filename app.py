from datetime import time as dtime
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Schiphol Gate Dashboard", layout="wide")
st.title("Schiphol Passenger Gate Dashboard")

TIER_ORDER = ["Early (< 0 min)", "On-Time (0 min)", "1 - 15 min (D15)", "16 - 30 min", "31 - 60 min", "> 60 min"]
DELAY_TIERS = {"LIGHT": (5, 30), "MODERATE": (30, 60), "HEAVY": (60, 120)}
SCENARIO_ORDER = ["BASELINE", "LIGHT", "MODERATE", "HEAVY"]
SCENARIO_COLORS = {"BASELINE": "#475569", "LIGHT": "#22a06b", "MODERATE": "#e69f00", "HEAVY": "#d62728"}


@st.cache_data
def prepare_data(file_bytes):
    with pd.ExcelFile(io.BytesIO(file_bytes)) as workbook:
        turnaround = pd.read_excel(workbook, sheet_name="cpdsp_20190710")
        times = pd.read_excel(workbook, sheet_name="Blad1")
    for sheet, frame, required in [
        ("cpdsp_20190710", turnaround, {"AC_TYPE", "GATE", "TOBT", "EOBT", "SOBT", "CSN", "DEST"}),
        ("Blad1", times, {"Type", "TRT Mins"}),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Sheet '{sheet}' is missing columns: {', '.join(sorted(missing))}")
    turnaround["AC_TYPE"] = turnaround["AC_TYPE"].astype("string").str.strip()
    turnaround["GATE"] = turnaround["GATE"].astype("string").str.strip().str.upper()
    times["Type"] = times["Type"].astype("string").str.strip()
    times["TRT Mins"] = pd.to_numeric(times["TRT Mins"], errors="coerce")
    times = times.dropna(subset=["Type", "TRT Mins"])
    times = times.loc[times["TRT Mins"] > 0].copy()
    if times["Type"].duplicated().any():
        raise ValueError("Blad1 must contain one valid turnaround-duration row per aircraft Type.")
    records = turnaround.merge(times[["Type", "TRT Mins"]], left_on="AC_TYPE", right_on="Type", how="left", validate="many_to_one")
    records["PIER"] = records["GATE"].str.extract(r"^([A-Z]+)")[0]
    records = records.loc[records["PIER"].notna() & ~records["PIER"].isin(["J", "R", "S"]) & ~records["GATE"].isin(["GD", "HG01"])].copy()
    missing_duration = int(records["TRT Mins"].isna().sum())
    records = records.dropna(subset=["TRT Mins"]).copy()
    records["Turnaround_Min"] = records["TRT Mins"]
    for column in ["TOBT", "EOBT", "SOBT"]:
        records[column] = pd.to_datetime(records[column], errors="coerce")
    records["End_Time"] = records["TOBT"].fillna(records["EOBT"]).fillna(records["SOBT"])
    missing_time = int(records["End_Time"].isna().sum())
    records = records.dropna(subset=["End_Time"]).copy()
    records["Start_Time"] = records["End_Time"] - pd.to_timedelta(records["Turnaround_Min"], unit="m")
    return records, missing_duration, missing_time


@st.cache_data
def calculate_occupancy(records, selected_date_text):
    day = pd.Timestamp(selected_date_text)
    grid = pd.date_range(day + pd.Timedelta(hours=5), day + pd.Timedelta(hours=23), freq="15min", inclusive="left")
    counts = records.groupby("PIER")["GATE"].nunique().to_dict()
    rows = []
    for t in grid:
        active = records.loc[(records["Start_Time"] <= t) & (records["End_Time"] > t)]
        for pier, total in counts.items():
            occupied = active.loc[active["PIER"] == pier, "GATE"].nunique()
            rows.append({"Time": t, "Pier": pier, "Occupied_Gates": occupied, "Total_Gates": total, "Occupancy_Pct": round(100 * occupied / total, 1)})
    return pd.DataFrame(rows)


def get_tier(minutes):
    if minutes < 0:
        return TIER_ORDER[0]
    if minutes == 0:
        return TIER_ORDER[1]
    if minutes <= 15:
        return TIER_ORDER[2]
    if minutes <= 30:
        return TIER_ORDER[3]
    if minutes <= 60:
        return TIER_ORDER[4]
    return TIER_ORDER[5]


def simulate_disruption(records, scenario, storm_start, storm_end, seed):
    """Assign an additional positive hold to every row in the baseline storm window."""
    low, high = DELAY_TIERS[scenario]
    sim = records.copy()
    mask = sim["End_Time"].between(storm_start, storm_end, inclusive="both")
    sim["Within_Storm_Window"] = mask
    sim["Disruption_Delay_Min"] = 0.0
    sim.loc[mask, "Disruption_Delay_Min"] = np.random.RandomState(seed).uniform(low, high, int(mask.sum())).round(1)
    sim["Sim_Start_Time"] = sim["Start_Time"]
    sim["Sim_End_Time"] = sim["End_Time"] + pd.to_timedelta(sim["Disruption_Delay_Min"], unit="m")
    sim["Total_Stand_Occupied_Min"] = sim["Turnaround_Min"] + sim["Disruption_Delay_Min"]
    return sim, int(mask.sum())


def flag_gate_conflicts(df, start_col, end_col):
    """Immediate-predecessor test, not an exhaustive count of overlapping pairs."""
    flagged = df.sort_values(["GATE", start_col]).copy()
    grouped = flagged.groupby("GATE")
    flagged["Prev_Flight_Clear_Time"] = grouped[end_col].shift(1)
    flagged["Previous_Flight"] = grouped["CSN"].shift(1)
    flagged["Gate_Conflict"] = flagged[start_col] < flagged["Prev_Flight_Clear_Time"]
    flagged["Overlap_Min"] = ((flagged["Prev_Flight_Clear_Time"] - flagged[start_col]).dt.total_seconds() / 60).clip(lower=0)
    return flagged


@st.cache_data
def compare_weather_scenarios(records, selected_date_text, storm_start, storm_end, seed):
    day = pd.Timestamp(selected_date_text)
    grid = pd.date_range(day + pd.Timedelta(hours=5), day + pd.Timedelta(hours=23), freq="15min", inclusive="left")
    scopes = {"All observed passenger stands": records["GATE"].nunique()}
    scopes.update({f"Pier {pier}": count for pier, count in records.groupby("PIER")["GATE"].nunique().items()})
    rows, summaries = [], []
    for name in SCENARIO_ORDER:
        if name == "BASELINE":
            sim = records.copy()
            sim["Sim_Start_Time"] = sim["Start_Time"]
            sim["Sim_End_Time"] = sim["End_Time"]
        else:
            sim, _ = simulate_disruption(records, name, storm_start, storm_end, seed)
        flagged = flag_gate_conflicts(sim, "Sim_Start_Time", "Sim_End_Time")
        for scope, capacity in scopes.items():
            subset = sim if scope == "All observed passenger stands" else sim.loc[sim["PIER"] == scope.removeprefix("Pier ")]
            scope_flags = flagged if scope == "All observed passenger stands" else flagged.loc[flagged["PIER"] == scope.removeprefix("Pier ")]
            scope_rows = []
            for t in grid:
                active = subset.loc[(subset["Sim_Start_Time"] <= t) & (subset["Sim_End_Time"] > t)]
                occupied, aircraft = active["GATE"].nunique(), len(active)
                row = {
                    "Time": t, "Scenario": name, "Scope": scope,
                    "Observed_Stands": int(capacity), "Occupied_Gates": int(occupied),
                    "Aircraft_Demand": int(aircraft), "Occupancy_Pct": 100 * occupied / capacity,
                    "Demand_Pct": 100 * aircraft / capacity,
                }
                rows.append(row)
                scope_rows.append(row)
            curve = pd.DataFrame(scope_rows)
            peak = curve["Demand_Pct"].max()
            summaries.append({
                "Scope": scope, "Scenario": name, "Observed stands": int(capacity),
                "Peak occupied gates": int(curve["Occupied_Gates"].max()),
                "Peak occupancy (%)": round(curve["Occupancy_Pct"].max(), 1),
                "Peak aircraft demand": int(curve["Aircraft_Demand"].max()),
                "Peak demand (%)": round(peak, 1),
                "Peak demand time(s)": ", ".join(curve.loc[curve["Demand_Pct"] == peak, "Time"].dt.strftime("%H:%M")),
                "Gate-conflict flags": int(scope_flags["Gate_Conflict"].sum()),
            })
    return pd.DataFrame(rows), pd.DataFrame(summaries)


uploaded_file = st.sidebar.file_uploader("Upload turnaround.xlsx", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload your Excel workbook using the sidebar.")
    st.stop()
try:
    all_records, missing_duration, missing_time = prepare_data(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Could not prepare the workbook: {exc}")
    st.stop()
if missing_time:
    st.warning(f"Excluded {missing_time} records without a usable departure timestamp.")
if all_records.empty:
    st.error("No valid passenger-gate records remain.")
    st.stop()

selected_date = st.sidebar.selectbox("Departure / readiness date", sorted(all_records["End_Time"].dt.date.unique()))
records = all_records.loc[all_records["End_Time"].dt.date == selected_date].copy()
selected_pier = st.sidebar.selectbox("Pier for the baseline Gantt chart", sorted(records["PIER"].unique()))
buffer_minutes = st.sidebar.number_input("Capacity buffer (minutes)", min_value=0, value=15, step=1)
OP_MINUTES = 18 * 60

gate_summary = records.groupby("GATE").agg(flights_handled=("CSN", "size"), total_trt_minutes=("Turnaround_Min", "sum"), avg_trt_minutes=("Turnaround_Min", "mean")).reset_index()
gate_summary["Utilization_Pct"] = (100 * gate_summary["total_trt_minutes"] / OP_MINUTES).round(1)
gate_summary["Theoretical_Capacity"] = np.floor(OP_MINUTES / (gate_summary["avg_trt_minutes"] + buffer_minutes)).astype(int)
gate_summary["Remaining_Slots"] = np.maximum(0, gate_summary["Theoretical_Capacity"] - gate_summary["flights_handled"])
records = records.merge(gate_summary, on="GATE", how="left")
temporal_occupancy = calculate_occupancy(records, str(selected_date))
total_passenger_stands = records["GATE"].nunique()
total_curve = temporal_occupancy.groupby("Time", as_index=False)["Occupied_Gates"].sum()
total_curve["Airport_Utilization_Pct"] = (100 * total_curve["Occupied_Gates"] / total_passenger_stands).round(1)

col1, col2, col3 = st.columns(3)
col1.metric("Flight records", len(records))
col2.metric("Observed passenger gates", total_passenger_stands)
col3.metric("Peak sampled occupancy", f"{total_curve['Airport_Utilization_Pct'].max():.1f}%")
tab_overview, tab_scenario = st.tabs(["Gate occupancy & punctuality", "Disruption scenario"])

with tab_overview:
    pier_subset = records.loc[records["PIER"] == selected_pier].sort_values("GATE")
    fig_gantt = px.timeline(
        pier_subset, x_start="Start_Time", x_end="End_Time", y="GATE", color="AC_TYPE",
        title=f"Estimated Stand Occupancy — Pier {selected_pier}", hover_name="CSN",
        hover_data={"AC_TYPE": True, "Turnaround_Min": ":.1f", "DEST": True, "Start_Time": "|%H:%M", "End_Time": "|%H:%M", "Utilization_Pct": True, "flights_handled": True, "Remaining_Slots": True},
    )
    fig_gantt.update_yaxes(autorange="reversed", title_text="Passenger Gate / Stand")
    fig_gantt.update_xaxes(title_text=f"Time ({selected_date})", dtick=3600000, tickformat="%H:%M")
    fig_gantt.update_layout(height=650)
    st.plotly_chart(fig_gantt, use_container_width=True)

    fig_temporal = px.area(temporal_occupancy, x="Time", y="Occupied_Gates", color="Pier", title="Estimated Passenger Gate Occupancy", labels={"Occupied_Gates": "Occupied Passenger Gates"}, hover_data={"Occupancy_Pct": ":.1f", "Total_Gates": True})
    fig_temporal.add_hline(y=total_passenger_stands, line_dash="dot", line_color="red", annotation_text=f"Observed gates ({total_passenger_stands})", annotation_position="top left")
    fig_temporal.update_xaxes(dtick=3600000, tickformat="%H:%M")
    fig_temporal.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig_temporal, use_container_width=True)

    fig_pct = px.line(total_curve, x="Time", y="Airport_Utilization_Pct", title="Estimated Gate Occupancy Ratio (%)", labels={"Airport_Utilization_Pct": "Observed Gate Occupancy (%)"})
    fig_pct.update_yaxes(range=[0, 100], ticksuffix="%")
    fig_pct.update_xaxes(dtick=3600000, tickformat="%H:%M")
    fig_pct.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig_pct, use_container_width=True)

    fig_pier_lines = px.line(temporal_occupancy, x="Time", y="Occupancy_Pct", color="Pier", title="Estimated Gate Occupancy by Pier (%)", labels={"Occupancy_Pct": "Gates Occupied (%)"})
    fig_pier_lines.update_yaxes(range=[0, 100], ticksuffix="%")
    fig_pier_lines.update_xaxes(dtick=3600000, tickformat="%H:%M")
    fig_pier_lines.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig_pier_lines, use_container_width=True)

    delay_records = records.copy()
    delay_records["Delay_Min"] = (delay_records["EOBT"] - delay_records["SOBT"]).dt.total_seconds() / 60
    excluded_delay = int(delay_records["Delay_Min"].isna().sum())
    delay_records = delay_records.dropna(subset=["Delay_Min"]).copy()
    if delay_records.empty:
        st.info("No records have both EOBT and SOBT.")
    else:
        delay_records["Delay_Tier"] = delay_records["Delay_Min"].apply(get_tier)
        delay_counts = delay_records["Delay_Tier"].value_counts().reindex(TIER_ORDER, fill_value=0).reset_index()
        delay_counts.columns = ["Delay_Tier", "Flights"]
        delay_counts["Percentage"] = (100 * delay_counts["Flights"] / delay_counts["Flights"].sum()).round(1)
        delay_counts["Bar_Label"] = delay_counts["Flights"].astype(str) + " (" + delay_counts["Percentage"].astype(str) + "%)"
        fig_delay = px.bar(delay_counts, x="Delay_Tier", y="Flights", text="Bar_Label", title="Estimated Schedule Slippage (EOBT vs SOBT)", labels={"Delay_Tier": "Schedule Slippage Tier", "Flights": "Flight Count"})
        fig_delay.update_traces(textposition="outside")
        fig_delay.update_layout(height=500)
        st.plotly_chart(fig_delay, use_container_width=True)

    with st.expander("Gate capacity summary", expanded=False):
        st.dataframe(gate_summary, use_container_width=True, hide_index=True)
        st.download_button("Download gate summary (CSV)", gate_summary.to_csv(index=False).encode("utf-8"), file_name=f"gate_summary_{selected_date}.csv", mime="text/csv")

    with st.expander("Methodology and limitations", expanded=False):
        st.markdown("""
- Records are selected by baseline **End_Time date**. Counts assume one row per flight.
- End_Time uses **TOBT → EOBT → SOBT**. Start_Time subtracts the aircraft-type turnaround duration; it is not an actual gate-arrival timestamp.
- Passenger-gate filtering excludes piers J, R, and S, and locations GD and HG01. Missing valid durations or departure timestamps are excluded.
- Gate totals are observed in the selected records, not a verified inventory of usable, compatible stands.
- Occupancy counts distinct active gates at 15-minute samples from **05:00 to 22:45**. Brief peaks may be missed, and adjacent-date flight records are not included.
- The sidebar pier selector changes only the baseline Gantt chart.
- EOBT minus SOBT is **estimated schedule slippage**, not actual departure punctuality or an independently validated D15 KPI.
- Capacity = floor(1,080 / (average turnaround + buffer)). Remaining slots are theoretical, not verified scheduling availability.
- The gate-summary Utilization_Pct sums turnaround durations without clipping to operating hours or removing overlaps; it can exceed 100%.
- Full date-and-time values in a consistent timezone are required. This is a historical-data prototype, not a live operational feed.
""")
        st.write(f"Excluded schedule-slippage records without both EOBT and SOBT: {excluded_delay}.")

with tab_scenario:
    st.subheader("Weather disruption scenario")
    sc1, sc2, sc3, sc4 = st.columns(4)
    scenario = sc1.selectbox("Scenario for the Gantt chart", list(DELAY_TIERS), index=1, format_func=lambda name: f"{name} ({DELAY_TIERS[name][0]}–{DELAY_TIERS[name][1]} min)")
    storm_start_time = sc2.time_input("Storm window start", value=dtime(11, 0))
    storm_end_time = sc3.time_input("Storm window end", value=dtime(17, 0))
    seed = int(sc4.number_input("Random seed", min_value=0, max_value=4294967295, value=42, step=1))
    piers = sorted(records["PIER"].unique())
    scenario_pier = st.selectbox("Pier for the simulated Gantt chart", piers, index=piers.index("B") if "B" in piers else 0)
    day = pd.Timestamp(selected_date)
    storm_start = day + pd.Timedelta(hours=storm_start_time.hour, minutes=storm_start_time.minute)
    storm_end = day + pd.Timedelta(hours=storm_end_time.hour, minutes=storm_end_time.minute)
    if storm_end <= storm_start:
        st.error("The storm window must end after it starts.")
    else:
        sim, flights_in_window = simulate_disruption(records, scenario, storm_start, storm_end, seed)
        flagged = flag_gate_conflicts(sim, "Sim_Start_Time", "Sim_End_Time")
        baseline_flagged = flag_gate_conflicts(sim, "Sim_Start_Time", "End_Time")
        conflicts = int(flagged["Gate_Conflict"].sum())
        baseline_conflicts = int(baseline_flagged["Gate_Conflict"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flights within storm window", flights_in_window)
        m2.metric("Conflict flags (scenario)", conflicts)
        m3.metric("Conflict flags (baseline)", baseline_conflicts)
        m4.metric("Additional flags", conflicts - baseline_conflicts)

        st.subheader("Weather impact — scenario comparison")
        comparison, summary = compare_weather_scenarios(records, str(selected_date), storm_start, storm_end, seed)
        control1, control2 = st.columns(2)
        scope = control1.selectbox("Comparison scope", ["All observed passenger stands"] + [f"Pier {pier}" for pier in piers])
        measure = control2.selectbox("Comparison measure", ["Estimated stand demand ratio", "Gate occupancy ratio"])
        metric = "Demand_Pct" if measure == "Estimated stand demand ratio" else "Occupancy_Pct"
        chart_data = comparison.loc[comparison["Scope"] == scope].copy()
        denominator = int(chart_data["Observed_Stands"].iloc[0])
        show_target = st.checkbox("Show illustrative planning target", value=False)
        target = st.slider("Illustrative target (%) — not a Schiphol standard", min_value=50, max_value=100, value=90, step=1) if show_target else None
        fig_compare = px.line(
            chart_data, x="Time", y=metric, color="Scenario",
            category_orders={"Scenario": SCENARIO_ORDER}, color_discrete_map=SCENARIO_COLORS,
            title=f"{measure} — {scope}",
            labels={metric: f"{measure} (%)", "Time": f"Time ({selected_date})"},
            hover_data={"Occupied_Gates": True, "Aircraft_Demand": True, "Observed_Stands": True, metric: ":.1f"},
        )
        fig_compare.add_vrect(x0=storm_start, x1=storm_end, fillcolor="#94a3b8", opacity=0.12, line_width=0, annotation_text="Storm window", annotation_position="top left")
        fig_compare.add_hline(y=100, line_dash="dash", line_color="#475569", annotation_text="100% of observed stands — not verified capacity", annotation_position="top left")
        if target is not None:
            fig_compare.add_hline(y=target, line_dash="dot", line_color="#7c3aed", annotation_text=f"Illustrative target ({target}%)", annotation_position="bottom left")
        fig_compare.update_yaxes(range=[0, max(110, float(chart_data[metric].max()) * 1.12)], ticksuffix="%")
        fig_compare.update_xaxes(dtick=3600000, tickformat="%H:%M")
        fig_compare.update_layout(height=600, hovermode="x unified", legend_title_text="Scenario")
        st.plotly_chart(fig_compare, use_container_width=True)

        summary_view = summary.loc[summary["Scope"] == scope].drop(columns="Scope").copy()
        if target is not None:
            exceedances = chart_data.assign(Above_Target=chart_data[metric] > target).groupby("Scenario")["Above_Target"].sum()
            summary_view["Samples above illustrative target"] = summary_view["Scenario"].map(exceedances).astype(int)
        with st.expander("Scenario results table and download", expanded=False):
            st.dataframe(summary_view, use_container_width=True, hide_index=True)
            st.download_button("Download scenario comparison (CSV)", chart_data.to_csv(index=False).encode("utf-8"), file_name=f"weather_comparison_{selected_date}.csv", mime="text/csv")

        subset = flagged.loc[flagged["PIER"] == scenario_pier].sort_values("GATE")
        low, high = DELAY_TIERS[scenario]
        fig_disrupted = px.timeline(
            subset, x_start="Sim_Start_Time", x_end="Sim_End_Time", y="GATE", color="Disruption_Delay_Min",
            title=f"Simulated Stand Occupancy — Pier {scenario_pier} — {scenario} ({low}–{high}m Holds)",
            color_continuous_scale="Reds", hover_name="CSN",
            hover_data={"AC_TYPE": True, "Turnaround_Min": ":.1f", "Disruption_Delay_Min": ":.1f", "Within_Storm_Window": True, "Gate_Conflict": True, "Sim_Start_Time": "|%H:%M", "Sim_End_Time": "|%H:%M"},
            labels={"Disruption_Delay_Min": "Simulated hold (min)"},
        )
        fig_disrupted.update_yaxes(autorange="reversed", title_text="Gate / Stand")
        fig_disrupted.update_xaxes(title_text=f"Time ({selected_date})", dtick=3600000, tickformat="%H:%M")
        fig_disrupted.update_layout(height=600)
        st.plotly_chart(fig_disrupted, use_container_width=True)

        with st.expander("Simulated conflict details", expanded=False):
            table = flagged.loc[flagged["Gate_Conflict"], ["GATE", "Previous_Flight", "CSN", "AC_TYPE", "Sim_Start_Time", "Prev_Flight_Clear_Time", "Overlap_Min", "Within_Storm_Window", "Disruption_Delay_Min", "Total_Stand_Occupied_Min"]]
            if table.empty:
                st.write("No gate-conflict flags detected.")
            else:
                st.dataframe(table, use_container_width=True, hide_index=True)

        with st.expander("Baseline overlap details", expanded=False):
            baseline_table = baseline_flagged.loc[baseline_flagged["Gate_Conflict"], ["GATE", "Previous_Flight", "CSN", "AC_TYPE", "Sim_Start_Time", "Prev_Flight_Clear_Time", "Overlap_Min"]]
            if baseline_table.empty:
                st.write("No baseline overlap flags detected.")
            else:
                st.dataframe(baseline_table, use_container_width=True, hide_index=True)

        with st.expander("Methodology and limitations", expanded=False):
            st.markdown(f"""
**Selection and simulation**
- Selected date: **{selected_date}**. Baseline End_Time window: **{storm_start:%H:%M}–{storm_end:%H:%M}**, including both boundaries.
- **Every one of the {flights_in_window} window records receives a positive simulated hold.** This is an assumption, not evidence that all would actually be delayed.
- The window count is independent of severity and seed. Baseline means no additional simulated delay, not verified ideal weather.
- Separate uniform-random delay draws use LIGHT 5–30, MODERATE 30–60, and HEAVY 60–120 minutes.
- The seed makes unchanged data and settings reproducible. The same seed is used for all three weather scenarios.
- Changing the Gantt scenario selector does not hide comparison curves. Changing the seed or storm window recalculates the weather scenarios.

**Ratios and capacity reference**
- The selected comparison uses **{denominator} observed stands** for {scope}, not verified usable, compatible capacity.
- Gate occupancy = distinct active gates / observed stands.
- Estimated stand demand = active flight records / observed stands. It counts overlapping aircraft requirements separately and can exceed 100%; gate occupancy cannot.
- A ratio below 100% does not rule out local conflicts or establish that suitable alternative stands are available.
- The optional planning target is an illustrative user assumption, not a Schiphol standard.

**Conflicts and limitations**
- Flags compare each flight's estimated start with its directly preceding same-gate flight's clear time. This is not an exhaustive count of overlapping pairs.
- Conflict counts cover the selected day; they are not simultaneous peak conflicts or confirmed operational clashes.
- Baseline overlaps also use estimated starts rather than actual gate arrivals.
- Peak occupancy and peak demand are separate maxima and need not occur at the same time.
- Sampling is every 15 minutes from 05:00 to 22:45. Brief peaks can be missed; adjacent-date records are excluded.
- Weather conditions are hypothetical, not verified for this historical date. No time-varying weather, affected-flight probability, gate reassignment, or knock-on delays are modelled.
- Scenario holds do not update the original EOBT–SOBT schedule-slippage chart.
- The dashboard is a historical-data stress-test prototype, not a validated operational forecast or automatic decision system.
""")
