"""Simple Tkinter GUI for meta-agent.

This module provides a basic graphical interface for users who prefer
not to use the terminal. It allows running the autonomous task loop
with a visual interface.

Usage:
    python -m metaagent.gui
    metaagent --gui
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from .license import FREE_TIER_LIMIT, PURCHASE_URL, display_license_status, verify_pro_key


class MetaAgentGUI:
    """Main GUI application for meta-agent."""

    def __init__(self, root: tk.Tk):
        """Initialize the GUI.

        Args:
            root: Tkinter root window.
        """
        self.root = root
        self.root.title("Meta-Agent - Autonomous Code Refinement")
        self.root.geometry("800x700")
        self.root.minsize(600, 500)

        # State
        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue = queue.Queue()
        self.is_running = False

        # Configure style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6)
        self.style.configure("TLabel", padding=2)

        self._create_widgets()
        self._setup_layout()
        self._poll_output()

    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")

        # Title
        self.title_label = ttk.Label(
            self.main_frame,
            text="Meta-Agent",
            font=("Helvetica", 18, "bold"),
        )
        self.subtitle_label = ttk.Label(
            self.main_frame,
            text="Autonomous Code Refinement Tool",
            font=("Helvetica", 10),
        )

        # License frame
        self.license_frame = ttk.LabelFrame(self.main_frame, text="License", padding="5")
        self.license_var = tk.StringVar(value=f"Free Tier ({FREE_TIER_LIMIT} iterations)")
        self.license_label = ttk.Label(self.license_frame, textvariable=self.license_var)
        self.pro_key_label = ttk.Label(self.license_frame, text="Pro Key:")
        self.pro_key_entry = ttk.Entry(self.license_frame, width=40, show="*")
        self.verify_btn = ttk.Button(
            self.license_frame,
            text="Verify Key",
            command=self._verify_license,
        )
        self.buy_btn = ttk.Button(
            self.license_frame,
            text="Get Pro Key",
            command=self._open_purchase_url,
        )

        # PRD frame
        self.prd_frame = ttk.LabelFrame(self.main_frame, text="PRD (Product Requirements)", padding="5")
        self.prd_path_label = ttk.Label(self.prd_frame, text="PRD File:")
        self.prd_path_var = tk.StringVar()
        self.prd_path_entry = ttk.Entry(self.prd_frame, textvariable=self.prd_path_var, width=50)
        self.browse_btn = ttk.Button(self.prd_frame, text="Browse...", command=self._browse_prd)

        self.prd_text_label = ttk.Label(self.prd_frame, text="Or paste PRD content:")
        self.prd_text = tk.Text(self.prd_frame, height=8, width=70, wrap=tk.WORD)
        self.prd_text.insert("1.0", "# My Project PRD\n\nDescribe what you want to build here...")

        # Options frame
        self.options_frame = ttk.LabelFrame(self.main_frame, text="Options", padding="5")
        self.human_approve_var = tk.BooleanVar(value=True)
        self.human_approve_cb = ttk.Checkbutton(
            self.options_frame,
            text="Require human approval for each task",
            variable=self.human_approve_var,
        )
        self.dry_run_var = tk.BooleanVar(value=False)
        self.dry_run_cb = ttk.Checkbutton(
            self.options_frame,
            text="Dry run (preview only, no changes)",
            variable=self.dry_run_var,
        )
        self.max_iter_label = ttk.Label(self.options_frame, text="Max iterations:")
        self.max_iter_var = tk.StringVar(value="")
        self.max_iter_entry = ttk.Entry(self.options_frame, textvariable=self.max_iter_var, width=10)

        # Control buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.start_btn = ttk.Button(
            self.control_frame,
            text="Start Loop",
            command=self._start_loop,
            style="TButton",
        )
        self.stop_btn = ttk.Button(
            self.control_frame,
            text="Stop",
            command=self._stop_loop,
            state=tk.DISABLED,
        )
        self.clear_btn = ttk.Button(
            self.control_frame,
            text="Clear Log",
            command=self._clear_log,
        )

        # Log output
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Output Log", padding="5")
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame,
            height=15,
            width=90,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 9),
        )

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )

    def _setup_layout(self) -> None:
        """Setup widget layout using grid."""
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        self.title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 0))
        self.subtitle_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # License frame
        self.license_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.license_label.grid(row=0, column=0, columnspan=4, sticky=tk.W)
        self.pro_key_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pro_key_entry.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.verify_btn.grid(row=1, column=2, padx=5)
        self.buy_btn.grid(row=1, column=3, padx=5)

        # PRD frame
        self.prd_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.prd_path_label.grid(row=0, column=0, sticky=tk.W)
        self.prd_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.browse_btn.grid(row=0, column=2)
        self.prd_text_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        self.prd_text.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=5)
        self.prd_frame.columnconfigure(1, weight=1)

        # Options frame
        self.options_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.human_approve_cb.grid(row=0, column=0, sticky=tk.W)
        self.dry_run_cb.grid(row=0, column=1, sticky=tk.W, padx=20)
        self.max_iter_label.grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.max_iter_entry.grid(row=0, column=3, sticky=tk.W)

        # Control buttons
        self.control_frame.grid(row=5, column=0, columnspan=2, pady=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Log frame
        self.log_frame.grid(row=6, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.main_frame.rowconfigure(6, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # Status bar
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _verify_license(self) -> None:
        """Verify the entered pro key."""
        key = self.pro_key_entry.get().strip()
        if not key:
            messagebox.showwarning("No Key", "Please enter a pro key to verify.")
            return

        if verify_pro_key(key):
            self.license_var.set("Pro Tier (Unlimited iterations)")
            os.environ["METAAGENT_PRO_KEY"] = key
            messagebox.showinfo("Success", "Pro key verified! You have unlimited iterations.")
        else:
            messagebox.showerror("Invalid Key", "The pro key is not valid. Please check and try again.")

    def _open_purchase_url(self) -> None:
        """Open the purchase URL in browser."""
        import webbrowser
        webbrowser.open(PURCHASE_URL)

    def _browse_prd(self) -> None:
        """Open file dialog to select PRD file."""
        file_path = filedialog.askopenfilename(
            title="Select PRD File",
            filetypes=[
                ("Markdown files", "*.md"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.prd_path_var.set(file_path)
            # Load content into text area
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.prd_text.delete("1.0", tk.END)
                self.prd_text.insert("1.0", content)
            except Exception as e:
                self._log(f"Error loading file: {e}\n")

    def _start_loop(self) -> None:
        """Start the meta-agent loop."""
        # Get PRD content
        prd_path = self.prd_path_var.get().strip()
        prd_content = self.prd_text.get("1.0", tk.END).strip()

        if not prd_path and not prd_content:
            messagebox.showwarning("No PRD", "Please provide a PRD file or content.")
            return

        # Build command
        cmd = [sys.executable, "-m", "metaagent", "loop"]

        # If we have a file path, use it; otherwise create temp file
        if prd_path and os.path.exists(prd_path):
            cmd.extend(["--prd", prd_path])
        else:
            # Create temp PRD file
            temp_prd = Path.cwd() / ".meta-agent-temp-prd.md"
            temp_prd.write_text(prd_content, encoding="utf-8")
            cmd.extend(["--prd", str(temp_prd)])

        # Add options
        if not self.human_approve_var.get():
            cmd.append("--no-human-approve")

        if self.dry_run_var.get():
            cmd.append("--dry-run")

        max_iter = self.max_iter_var.get().strip()
        if max_iter and max_iter.isdigit():
            cmd.extend(["--max-iterations", max_iter])

        # Add pro key if entered
        pro_key = self.pro_key_entry.get().strip()
        if pro_key:
            cmd.extend(["--pro-key", pro_key])

        # Start process
        self._log(f"Starting: {' '.join(cmd)}\n")
        self._log("-" * 60 + "\n")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=Path.cwd(),
            )
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Running...")

            # Start output reader thread
            thread = threading.Thread(target=self._read_output, daemon=True)
            thread.start()

        except Exception as e:
            self._log(f"Error starting process: {e}\n")
            messagebox.showerror("Error", f"Failed to start: {e}")

    def _stop_loop(self) -> None:
        """Stop the running loop."""
        if self.process:
            self.process.terminate()
            self._log("\n[Stopped by user]\n")
            self._on_process_complete()

    def _read_output(self) -> None:
        """Read process output in background thread."""
        try:
            for line in self.process.stdout:
                self.output_queue.put(line)
        except Exception:
            pass
        finally:
            self.output_queue.put(None)  # Signal completion

    def _poll_output(self) -> None:
        """Poll output queue and update log."""
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line is None:
                    self._on_process_complete()
                    break
                self._log(line)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_output)

    def _on_process_complete(self) -> None:
        """Handle process completion."""
        self.is_running = False
        self.process = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Completed")
        self._log("\n" + "=" * 60 + "\n")
        self._log("Process completed.\n")

    def _log(self, message: str) -> None:
        """Add message to log output.

        Args:
            message: Message to log.
        """
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        """Clear the log output."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)


def launch_gui() -> None:
    """Launch the meta-agent GUI."""
    root = tk.Tk()

    # Set icon if available
    try:
        # Try to set a window icon (optional)
        pass
    except Exception:
        pass

    app = MetaAgentGUI(root)
    root.mainloop()


def main() -> None:
    """Main entry point for GUI."""
    launch_gui()


if __name__ == "__main__":
    main()
