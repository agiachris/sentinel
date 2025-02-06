import os
import re
import click
import plotly
import plotly.graph_objects as go
import datetime


def parse_logs(log_file_path):
    events = []
    start_time = None
    prev_details = None
    if log_file_path.endswith("dp.log"):
        pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]\[root\]\[INFO\] - \[(eval step)\](.*)"
    else:
        pattern = (
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):INFO:\[(start|done)\](.*)"
        )

    with open(log_file_path, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                timestamp_str, event_type, details = match.groups()
                timestamp = datetime.datetime.strptime(
                    timestamp_str + "000", "%Y-%m-%d %H:%M:%S,%f"
                )
                if event_type == "start":
                    start_time = timestamp
                elif event_type == "done" and start_time is not None:
                    events.append((start_time, timestamp, details.strip()))
                    start_time = None  # Reset start time after capturing the event
                elif event_type == "eval step":
                    if "poses" in details:
                        start_time = timestamp
                        prev_details = details
                    elif "done" in details:
                        events.append((start_time, timestamp, prev_details.strip()))
                        start_time = None
                        prev_details = None
                else:
                    print(event_type)
    return events


def create_timeline_plot(events_data):
    fig = go.Figure()

    start_time = min([data[0][1].timestamp() for _, data in events_data.items()])

    for process_name in [
        "obs_spinner.log",
        "policy_spinner.log",
        "real_eval_mp_dp.log",
    ]:
        events = events_data[process_name]
        for start, end, detail in events:
            duration = (end.timestamp() - start.timestamp()) * 1000
            fig.add_trace(
                go.Bar(
                    x=[duration],
                    y=[process_name],
                    base=[(start.timestamp() - start_time) * 1000],
                    orientation="h",
                    text=[detail],
                    textposition="none",
                    hoverinfo="text",
                    name="",
                )
            )
            print(
                f"{process_name} => start at {(start.timestamp() - start_time) * 1000}ms; end at {(start.timestamp() - start_time) * 1000 + duration}ms; duration {duration}ms"
            )

    fig.update_layout(
        title="Process Timeline from Logs",
        xaxis_title="Time (ms)",
        yaxis_title="Process",
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
            itemwidth=30,
        ),
    )
    return fig


@click.command()
@click.option("--log_dir", type=str, default=None)
def main(log_dir):
    events_data = {}
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            file_path = os.path.join(log_dir, filename)
            events = parse_logs(file_path)
            events_data[filename] = events

    fig = create_timeline_plot(events_data)
    plotly.io.write_html(fig, os.path.join(log_dir, "timeline.html"))


if __name__ == "__main__":
    main()
