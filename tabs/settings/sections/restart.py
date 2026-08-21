import gradio as gr
import os
import sys
import json

from rvc.lib.i18n import _

now_dir = os.getcwd()

def stop_infer():
    pid_file_path = os.path.join(now_dir, "assets", "infer_pid.txt")
    try:
        with open(pid_file_path, "r") as pid_file:
            pids = [int(pid) for pid in pid_file.readlines()]
        for pid in pids:
            os.kill(pid, 9)
        os.remove(pid_file_path)
    except:
        pass


def restart_application():
    if os.name != "nt":
        os.system("clear")
    else:
        os.system("cls")
    python = sys.executable
    os.execl(python, python, *sys.argv)


def restart_tab():
    with gr.Row():
        with gr.Column():
            restart_button = gr.Button(_("Restart ShiroRVC"))
            restart_button.click(
                fn=restart_application,
                inputs=[],
                outputs=[],
            )
