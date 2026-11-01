"""An isolated, offline GUI fixture for Week 6 acceptance, never an external app.

The independent state journal is test evidence only. It is not read by the
planner, observer, adapter or executor. Closing this window affects no user file.
"""
from __future__ import annotations

import argparse
import json
import time
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "transient", "permanent"), default="normal")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    root = tk.Tk()
    root.title("Week6 Local Lab")
    root.attributes("-fullscreen", True)
    root.configure(bg="#f7f9fc")
    root.tk.call("tk", "scaling", 1.0)
    state = {"stage": "project", "project": "", "colour": "", "committed": False,
             "review_attempts": 0, "mode": args.mode, "events": 0}
    content = tk.Frame(root, bg="#f7f9fc")
    content.place(relx=.5, rely=.5, anchor="center")
    transition_until = 0.0

    def record(event):
        state["events"] += 1
        value = {"utc": datetime.now(UTC).isoformat(), "event": event, **state}
        with (args.directory / "independent_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")
        temp = args.directory / "independent_state.tmp"
        temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.directory / "independent_state.json")

    def label(text, size=26):
        tk.Label(content, text=text, font=("Arial", size), bg="#f7f9fc", fg="#152b48").pack(pady=14)

    def button(text, command):
        tk.Button(content, text=text, font=("Arial", 25), width=20, height=2,
                  bg="#e6eefb", fg="#152b48", command=command).pack(pady=16)

    def accept_transition():
        # A double click must not accidentally activate a different control
        # rendered under its second click after a page transition.
        nonlocal transition_until
        now = time.monotonic()
        if now < transition_until:
            record("transition_click_ignored")
            return False
        transition_until = now + .4
        return True

    def render(notice=""):
        for widget in content.winfo_children():
            widget.destroy()
        label("Week6 Local Lab", 34)
        label("Offline report preparation", 22)
        if state["project"]:
            label(f"Project {state['project']}")
        if state["colour"]:
            label(f"Theme {state['colour']}")
        if notice:
            label(notice, 22)
        if state["stage"] == "project":
            label("Choose a project")
            for project in ("ALPHA", "BETA"):
                button(project, lambda p=project: choose_project(p))
        elif state["stage"] == "theme":
            label("Choose report theme")
            for colour in ("BLUE", "GREEN"):
                button(colour, lambda c=colour: choose_colour(c))
        elif state["stage"] == "review":
            label("Draft ready")
            button("Review report", review)
        elif state["stage"] == "commit":
            label("Review complete")
            button("Commit locally", commit)
        else:
            label("LOCAL COMPLETE", 36)
            label("This report exists only in this test window", 20)
        root.update_idletasks()

    def choose_project(project):
        if not accept_transition():
            return
        state.update(project=project, stage="theme")
        record("project_selected")
        render()

    def choose_colour(colour):
        if not accept_transition():
            return
        state.update(colour=colour, stage="review")
        record("theme_selected")
        render()

    def review():
        if not accept_transition():
            return
        state["review_attempts"] += 1
        if args.mode == "permanent" or (args.mode == "transient" and state["review_attempts"] == 1):
            record("review_rejected")
            render("Review unavailable. Please retry.")
        else:
            state["stage"] = "commit"
            record("review_accepted")
            render()

    def commit():
        if not accept_transition():
            return
        state.update(stage="completed", committed=True)
        record("local_commit")
        render()

    record("started")
    render()
    root.mainloop()


if __name__ == "__main__":
    main()
