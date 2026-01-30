"""
Mining Claim Redistribution Dashboard - Professional Edition

A modern, professional GUI application with PDF reporting, embedded visualizations,
and exploration credit simulation capabilities.

Requirements:
- Python 3.8+
- geopandas, pandas, numpy, matplotlib, folium, tqdm, tkinter, PIL, reportlab

Usage:
    python Claim_Redistribution_App.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from datetime import date, datetime
from pathlib import Path
import json
import webbrowser
import tempfile

# Matplotlib backend for embedding
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Try to import PIL for image display
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Image as RLImage, Table, TableStyle, BaseDocTemplate, Frame, PageTemplate, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors as rl_colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Import the main redistribution module
try:
    script_dir = Path(__file__).parent
    main_script = script_dir / "Gestim_Claim_Redistribution.py"

    if main_script.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("redistribution", str(main_script))
        redistribution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(redistribution)
        MAIN_MODULE_AVAILABLE = True
    else:
        MAIN_MODULE_AVAILABLE = False
        redistribution = None
except Exception as e:
    MAIN_MODULE_AVAILABLE = False
    redistribution = None
    print(f"Warning: Could not import main redistribution module: {e}")


class ModernStyle:
    """Modern color scheme and styling constants."""
    # Color palette - Modern professional theme
    PRIMARY = '#2C3E50'      # Dark blue-gray
    SECONDARY = '#3498DB'    # Bright blue
    SUCCESS = '#27AE60'      # Green
    WARNING = '#F39C12'      # Orange
    DANGER = '#E74C3C'       # Red
    INFO = '#16A085'         # Teal
    LIGHT = '#ECF0F1'        # Light gray
    DARK = '#34495E'         # Darker gray
    WHITE = '#FFFFFF'

    # Fonts
    FONT_FAMILY = 'Segoe UI'
    FONT_TITLE = (FONT_FAMILY, 16, 'bold')
    FONT_HEADING = (FONT_FAMILY, 12, 'bold')
    FONT_BODY = (FONT_FAMILY, 10)
    FONT_SMALL = (FONT_FAMILY, 9)


class ClaimRedistributionApp:
    """Main application class for the Claim Redistribution Dashboard."""

    def __init__(self, root):
        self.root = root
        self.root.title("Mining Claim Redistribution Dashboard")
        self.root.geometry("1400x900")

        # Set modern icon if available
        try:
            # You can add an icon file here
            # self.root.iconbitmap('icon.ico')
            pass
        except:
            pass

        # Configure modern styling
        self.setup_styles()

        # Queue for thread-safe logging and progress updates
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()

        # Configuration variables
        self.config = {
            'MIDLAND_XLSX': tk.StringVar(),
            'WALLBRIDGE_XLSX': tk.StringVar(),
            'PROPERTY_CSV': tk.StringVar(),
            'SHP_PATH': tk.StringVar(),
            'OUTLINES_SHP': tk.StringVar(),
            'OUTPUT_DIR': tk.StringVar(),
            'MAX_DISTANCE': tk.StringVar(value='3900'),
            'MAX_YEAR': tk.IntVar(value=2060),
            'MAX_RENEWALS': 6,
            'SCORING_MODE': tk.StringVar(value='earliest_expiry'),
            'CURRENT_DATE': date.today().isoformat(),
            'CREDITS_XLSX': tk.StringVar(),  # Optional exploration credits Excel file
        }

        # Exploration credits option
        self.credits_enabled = tk.BooleanVar(value=False)

        # Projects configuration
        self.projects = ['CASAULT', 'MARTINIERE', 'FENELON', 'GRASSET', 'HARRI', 'DOIGT']
        self.project_vars = {proj: tk.BooleanVar(value=True) for proj in self.projects}

        # Running state and results
        self.is_running = False
        self.current_results = None

        # Create UI
        self.create_ui()

        # Start queue checkers
        self.check_log_queue()
        self.check_progress_queue()

        # Check if main module is available
        if not MAIN_MODULE_AVAILABLE:
            self.log("WARNING: Main redistribution module not found.", 'warning')

    def setup_styles(self):
        """Configure modern ttk styles."""
        style = ttk.Style()

        # Try to use modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'vista' in available_themes:
            style.theme_use('vista')
        elif 'aqua' in available_themes:
            style.theme_use('aqua')

        # Configure custom styles
        style.configure('Title.TLabel', font=ModernStyle.FONT_TITLE, foreground=ModernStyle.PRIMARY)
        style.configure('Heading.TLabel', font=ModernStyle.FONT_HEADING, foreground=ModernStyle.DARK)
        style.configure('Body.TLabel', font=ModernStyle.FONT_BODY)

        # Button styles
        style.configure('Primary.TButton', font=ModernStyle.FONT_BODY, padding=10)
        style.configure('Success.TButton', font=ModernStyle.FONT_BODY, padding=10)

        # Frame styles
        style.configure('Card.TFrame', background=ModernStyle.LIGHT, relief='raised')

        # Configure colors
        self.root.configure(bg=ModernStyle.WHITE)

    def create_ui(self):
        """Create the modern user interface."""
        # Create main container
        main_container = ttk.Frame(self.root, padding=0)
        main_container.pack(fill='both', expand=True)

        # Create notebook for tabs with modern styling
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 0: Welcome/README
        welcome_frame = ttk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="📖 Welcome")
        self.create_welcome_tab(welcome_frame)

        # Tab 1: Input Files
        files_frame = ttk.Frame(self.notebook)
        self.notebook.add(files_frame, text="📁 Files")
        self.create_files_tab(files_frame)

        # Tab 2: Configuration (includes projects now)
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")
        self.create_config_tab(config_frame)

        # Tab 3: Run & Progress
        run_frame = ttk.Frame(self.notebook)
        self.notebook.add(run_frame, text="▶️ Run")
        self.create_run_tab(run_frame)

        # Tab 4: Results Dashboard
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Results")
        self.create_results_tab(results_frame)

        # Tab 5: Map Viewer
        map_frame = ttk.Frame(self.notebook)
        self.notebook.add(map_frame, text="🗺️ Maps")
        self.create_map_tab(map_frame)

        # Tab 6: PDF Report
        report_frame = ttk.Frame(self.notebook)
        self.notebook.add(report_frame, text="📄 Report")
        self.create_report_tab(report_frame)

        # Modern status bar
        self.create_status_bar(main_container)

    def create_status_bar(self, parent):
        """Create modern status bar."""
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Left side - status message
        self.status_label = ttk.Label(status_frame, text="Ready", font=ModernStyle.FONT_SMALL,
                                     foreground=ModernStyle.SUCCESS)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=3)

        # Separator
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Right side - progress percentage
        self.progress_label = ttk.Label(status_frame, text="", font=ModernStyle.FONT_SMALL)
        self.progress_label.pack(side=tk.RIGHT, padx=10, pady=3)

    def create_welcome_tab(self, parent):
        """Create welcome/README tab with program guide and diagram."""
        # Main container with two columns
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure grid weights for 60-40 split
        main_frame.columnconfigure(0, weight=6)
        main_frame.columnconfigure(1, weight=4)
        main_frame.rowconfigure(0, weight=1)

        # LEFT COLUMN: Text content with scrolling
        left_canvas = tk.Canvas(main_frame, bg=ModernStyle.WHITE, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=left_canvas.yview)
        left_scrollable = ttk.Frame(left_canvas)

        left_scrollable.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        left_scrollbar.grid(row=0, column=0, sticky='nse')

        container = ttk.Frame(left_scrollable, padding=20)
        container.pack(fill='both', expand=True)

        # Header
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(header_frame, text="Mining Claim Redistribution Dashboard",
                 font=(ModernStyle.FONT_FAMILY, 20, 'bold'),
                 foreground=ModernStyle.PRIMARY).pack(anchor=tk.W)
        ttk.Label(header_frame, text="Version 2.0 Enhanced - Quebec Mining Claim Analysis",
                 font=ModernStyle.FONT_BODY, foreground='gray').pack(anchor=tk.W, pady=(5, 0))

        # Overview Card
        overview_card = ttk.LabelFrame(container, text="Program Overview", padding=20)
        overview_card.pack(fill=tk.X, pady=10)

        overview_text = """This application simulates the redistribution of work credits among mining claims in Quebec.
It helps identify which claims will lapse and when, based on accumulated credits and renewal requirements.

KEY FEATURES:
• Automated credit redistribution simulation
• Visual maps showing projected lapse years
• Optional exploration program credit loading
• Professional PDF reports
• Real-time progress tracking"""

        ttk.Label(overview_card, text=overview_text, font=ModernStyle.FONT_BODY,
                 wraplength=450, justify=tk.LEFT).pack(anchor=tk.W)

        # How It Works Card
        works_card = ttk.LabelFrame(container, text="How the Simulation Works", padding=20)
        works_card.pack(fill=tk.X, pady=10)

        works_text = """CREDIT REDISTRIBUTION ALGORITHM:

1. INITIAL STATE
   Each claim has:
   • Required work credits needed to maintain the claim
   • Surplus credits that exceed requirements
   • Expiration date based on current term

2. REDISTRIBUTION PROCESS
   When a claim has surplus credits:
   • Find nearby claims (within max distance) that need credits
   • Prioritize recipients by earliest expiration date
   • Transfer credits from donor to recipient claims
   • Update expiration dates based on new credit balances

3. RENEWAL SIMULATION
   For each year until max year:
   • Check which claims expire this year
   • Apply accumulated credits to extend life
   • Attempt to redistribute any remaining surpluses
   • Track which claims lapse when credits run out

4. SCORING MODES
   • Earliest Expiry: Prioritizes claims expiring soonest
   • Distance-Surplus: Balances distance with surplus amount

5. OUTPUT
   • Final expiry dates for all claims
   • Years of life remaining for each claim
   • Complete log of all credit redistributions
   • Visual maps and statistical summaries"""

        ttk.Label(works_card, text=works_text, font=(ModernStyle.FONT_FAMILY, 9),
                 wraplength=450, justify=tk.LEFT).pack(anchor=tk.W)

        # Usage Guide Card
        usage_card = ttk.LabelFrame(container, text="How to Use This Program", padding=20)
        usage_card.pack(fill=tk.X, pady=10)

        usage_text = """STEP-BY-STEP GUIDE:

1. FILES TAB
   • Browse and select all required input files:
     - Midland Excel file (.xlsx)
     - Wallbridge Excel file (.xlsx)
     - Property CSV file (.csv)
     - Claims shapefile (.shp)
     - Property outlines shapefile (.shp)
   • Set output directory for results
   • Optional: Save configuration for future use
   • Optional: Add exploration program credits from Excel file

2. CONFIGURATION TAB
   • Set Max Distance: Maximum distance in meters for credit redistribution
     (typical: 2000-5000m)
   • Max Year: Fixed at 2060 for projections
   • Select Projects: Choose which projects to include in analysis
   • Max Renewals: Fixed at 6 per Quebec regulations

3. RUN TAB
   • Click "Run Simulation" to start
   • Monitor real-time progress (0-100%)
   • View detailed logs in the console
   • Wait for completion (typically 5-15 minutes)

4. RESULTS TAB
   • View summary statistics (total claims, average life, etc.)
   • Examine bar charts showing average life by project
   • Click "Refresh Results" to update after simulation

5. MAPS TAB
   • Toggle between static PNG map and interactive HTML map
   • Use mouse wheel to zoom in/out on static maps
   • Pan with scrollbars when zoomed
   • Static map shows color-coded lapse years

6. SENSITIVITY TAB
   • Test multiple max distance values (e.g., "2000, 3900, 5000")
   • Compare results across different scenarios
   • View impact on claim life and credits redistributed
   • Each run saves to separate subfolder

7. REPORT TAB
   • Preview PDF report contents
   • Generate professional PDF with results table and map
   • Includes 1-2 year outlook of at-risk claims
   • Optionally open PDF immediately after generation"""

        ttk.Label(usage_card, text=usage_text, font=(ModernStyle.FONT_FAMILY, 9),
                 wraplength=450, justify=tk.LEFT).pack(anchor=tk.W)

        # Parameters Card
        params_card = ttk.LabelFrame(container, text="Key Parameters Explained", padding=20)
        params_card.pack(fill=tk.X, pady=10)

        params_text = """MAX DISTANCE (meters)
The maximum spatial distance for credit redistribution. Claims can only transfer
credits to other claims within this distance. Lower values = more conservative,
higher values = more aggressive redistribution.

MAX YEAR
The final year for projections. Simulation runs year-by-year until this year, showing
when each claim will lapse based on credit availability.

MAX RENEWALS
Fixed at 6 renewals per Quebec mining regulations. After 6 renewals, claims cannot
be extended further.

SCORING MODE
• Earliest Expiry: Recipients prioritized by soonest expiration (default)
• Distance-Surplus: Balanced scoring considering both distance and surplus amount

INCLUDED PROJECTS
Select which mining projects to include in the analysis. Only claims from selected
projects will be processed and shown in results."""

        ttk.Label(params_card, text=params_text, font=(ModernStyle.FONT_FAMILY, 9),
                 wraplength=450, justify=tk.LEFT).pack(anchor=tk.W)

        # Tips Card
        tips_card = ttk.LabelFrame(container, text="Pro Tips", padding=20)
        tips_card.pack(fill=tk.X, pady=10)

        tips_text = """• Start with the default max distance (3900m) for initial runs
• Add exploration program credits to simulate future work programs
• Save configurations to JSON files for repeatable analysis
• Check the 1-2 year outlook in reports to identify urgent risks
• Maps use UTM Zone 17N coordinate system (EPSG:2958)
• Larger max distance values will generally extend claim life more
• Review the full redistribution log CSV for detailed transaction history"""

        ttk.Label(tips_card, text=tips_text, font=ModernStyle.FONT_BODY,
                 wraplength=450, justify=tk.LEFT).pack(anchor=tk.W)

        # Get Started Button
        start_frame = ttk.Frame(container)
        start_frame.pack(pady=20)

        ttk.Button(start_frame, text="▶ Get Started - Go to Files Tab",
                  command=lambda: self.notebook.select(1),
                  style='Success.TButton').pack()

        # RIGHT COLUMN: State Diagram
        right_frame = ttk.LabelFrame(main_frame, text="Simulation State Flow", padding=15)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))

        # Create scrollable text widget for diagram
        diagram_scroll = ttk.Scrollbar(right_frame, orient="vertical")

        diagram_text = tk.Text(right_frame, wrap=tk.NONE, font=('Courier New', 8),
                              bg=ModernStyle.LIGHT, fg=ModernStyle.DARK,
                              relief=tk.FLAT, padx=10, pady=10,
                              yscrollcommand=diagram_scroll.set)

        diagram_scroll.config(command=diagram_text.yview)

        # ASCII State Diagram
        diagram_content = """
┌──────────────────────────────────────────┐
│         START SIMULATION                 │
│    (Current Date, Max Year, etc.)        │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│   STEP 1: LOAD & MERGE DATA              │
│  • Midland Excel → CSV                   │
│  • Wallbridge Excel → CSV                │
│  • Merge with Property mapping           │
│  • Join with Claims shapefile            │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│   STEP 2: INITIALIZE CLAIMS              │
│  For each claim:                         │
│    ├─ Current Expiry Date                │
│    ├─ Required Work Credits              │
│    ├─ Surplus Credits                    │
│    └─ Renewal Count                      │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│   STEP 3: PRECOMPUTE SPATIAL DATA        │
│  • Distance matrix between all claims    │
│  • R-tree spatial index                  │
│  • Neighbor identification               │
└────────────────┬─────────────────────────┘
                 │
                 ▼
    ┌────────────┴────────────┐
    │   YEAR-BY-YEAR LOOP     │
    │ (Current → Max Year)    │
    └────────────┬────────────┘
                 │
    ╔════════════▼════════════════════════╗
    ║  FOR EACH YEAR:                     ║
    ║                                     ║
    ║  ┌──────────────────────────────┐   ║
    ║  │ A. Identify Expiring Claims  │   ║
    ║  │    (expiry = current year)   │   ║
    ║  └──────────┬───────────────────┘   ║
    ║             │                       ║
    ║             ▼                       ║
    ║  ┌──────────────────────────────┐   ║
    ║  │ B. Check Surplus Credits     │   ║
    ║  │    Can extend life?          │   ║
    ║  └──────────┬───────────────────┘   ║ 
    ║             │                       ║
    ║      ┌──────┴──────┐                ║
    ║      │             │                ║
    ║  YES │             │ NO             ║
    ║      ▼             ▼                ║
    ║  ┌────────┐   ┌────────────┐        ║
    ║  │ EXTEND │   │ LAPSE      │        ║
    ║  │ EXPIRY │   │ CLAIM      │        ║
    ║  └────────┘   └────────────┘        ║
    ║                                     ║
    ║  ┌──────────────────────────────┐   ║
    ║  │ C. Redistribution Process    │   ║
    ║  │                              │   ║
    ║  │ For claims with surplus:     │   ║
    ║  │  1. Find neighbors           │   ║
    ║  │     (within MAX_DISTANCE)    │   ║
    ║  │  2. Score recipients:        │   ║
    ║  │     - Earliest expiry        │   ║
    ║  │     - Distance-surplus       │   ║
    ║  │  3. Transfer credits         │   ║
    ║  │  4. Update expiry dates      │   ║
    ║  │  5. Log transaction          │   ║
    ║  └──────────────────────────────┘   ║
    ║                                     ║
    ╚═════════════════════════════════════╝
                 │
                 │ (next year)
                 │
                 ▼
           Year < Max Year?
                 │
          ┌──────┴──────┐
          │             │
         YES           NO
          │             │
          └─────(loop)  │
                        ▼
           ┌────────────────────────────┐
           │  STEP 4: CALCULATE RESULTS │
           │  • Years of life remaining │
           │  • Final expiry dates      │
           │  • Total credits moved     │
           └────────────┬───────────────┘
                        │
                        ▼
           ┌────────────────────────────┐
           │  STEP 5: EXPORT OUTPUTS    │
           │  • CSV: Years of life      │
           │  • CSV: Redistribution log │
           │  • PNG: Static map         │
           │  • HTML: Interactive map   │
           │  • PNG: Summary charts     │
           └────────────┬───────────────┘
                        │
                        ▼
           ┌────────────────────────────┐
           │      END SIMULATION        │
           │   (Results Available)      │
           └────────────────────────────┘


LEGEND:
═══  Loop/Iteration
───  Sequential Flow
◄──  Decision Point
┌─┐  Process/Step
"""

        diagram_text.insert('1.0', diagram_content)
        diagram_text.configure(state='disabled')  # Make read-only

        diagram_scroll.pack(side=tk.RIGHT, fill='y')
        diagram_text.pack(side=tk.LEFT, fill='both', expand=True)

    def create_files_tab(self, parent):
        """Create modern file selection tab."""
        # Scrollable frame
        canvas = tk.Canvas(parent, bg=ModernStyle.WHITE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Title
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, padx=20, pady=20)

        ttk.Label(title_frame, text="Input Files", style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Select all required input files for the simulation",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(anchor=tk.W, pady=(5, 0))

        # Files section
        files_card = ttk.LabelFrame(scrollable_frame, text="Required Files", padding=20)
        files_card.pack(fill=tk.X, padx=20, pady=10)

        file_configs = [
            ("Midland Excel File:", 'MIDLAND_XLSX', "*.xlsx"),
            ("Wallbridge Excel File:", 'WALLBRIDGE_XLSX', "*.xlsx"),
            ("Property CSV File:", 'PROPERTY_CSV', "*.csv"),
            ("Claims Shapefile:", 'SHP_PATH', "*.shp"),
            ("Property Outlines Shapefile:", 'OUTLINES_SHP', "*.shp"),
        ]

        for i, (label, key, filetype) in enumerate(file_configs):
            file_frame = ttk.Frame(files_card)
            file_frame.pack(fill=tk.X, pady=8)

            ttk.Label(file_frame, text=label, font=ModernStyle.FONT_BODY, width=30).pack(side=tk.LEFT)
            ttk.Entry(file_frame, textvariable=self.config[key], width=60,
                     font=ModernStyle.FONT_SMALL).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            ttk.Button(file_frame, text="Browse",
                      command=lambda k=key, ft=filetype: self.browse_file(k, ft),
                      width=12).pack(side=tk.LEFT)

        # Output directory section
        output_card = ttk.LabelFrame(scrollable_frame, text="Output Location", padding=20)
        output_card.pack(fill=tk.X, padx=20, pady=10)

        output_frame = ttk.Frame(output_card)
        output_frame.pack(fill=tk.X)

        ttk.Label(output_frame, text="Output Directory:", font=ModernStyle.FONT_BODY, width=30).pack(side=tk.LEFT)
        ttk.Entry(output_frame, textvariable=self.config['OUTPUT_DIR'], width=60,
                 font=ModernStyle.FONT_SMALL).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_directory, width=12).pack(side=tk.LEFT)

        # Optional: Exploration Credits section
        credits_card = ttk.LabelFrame(scrollable_frame, text="Optional: Exploration Program Credits", padding=20)
        credits_card.pack(fill=tk.X, padx=20, pady=10)

        # Enable checkbox
        credits_enable_frame = ttk.Frame(credits_card)
        credits_enable_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(credits_enable_frame, text="Add credits from completed exploration program",
                       variable=self.credits_enabled).pack(side=tk.LEFT)
        ttk.Label(credits_enable_frame, text="(simulates future work program credits)",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(side=tk.LEFT, padx=10)

        # Credits file selector
        credits_file_frame = ttk.Frame(credits_card)
        credits_file_frame.pack(fill=tk.X, pady=5)

        ttk.Label(credits_file_frame, text="Credits Excel File:", font=ModernStyle.FONT_BODY, width=30).pack(side=tk.LEFT)
        ttk.Entry(credits_file_frame, textvariable=self.config['CREDITS_XLSX'], width=60,
                 font=ModernStyle.FONT_SMALL).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        ttk.Button(credits_file_frame, text="Browse",
                  command=lambda: self.browse_file('CREDITS_XLSX', '*.xlsx'),
                  width=12).pack(side=tk.LEFT)

        # Info about expected format
        credits_info_frame = ttk.Frame(credits_card)
        credits_info_frame.pack(fill=tk.X, pady=(10, 0))

        credits_info = """Expected Excel columns:
  • Title Number (String): The claim title number
  • Amount (Float): Credit amount in dollars
  • Start Date (YYYY-MM-DD): Allocation date
Credits expire after 12 years per Quebec Mining Act Section 75."""

        ttk.Label(credits_info_frame, text=credits_info, font=ModernStyle.FONT_SMALL,
                 foreground='gray', justify=tk.LEFT).pack(anchor=tk.W)

        # Actions section
        actions_frame = ttk.Frame(scrollable_frame)
        actions_frame.pack(fill=tk.X, padx=20, pady=20)

        ttk.Button(actions_frame, text="📂 Set Default Paths",
                  command=self.set_default_paths,
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="💾 Save Configuration",
                  command=self.save_config_to_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="📥 Load Configuration",
                  command=self.load_config_from_file).pack(side=tk.LEFT, padx=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_config_tab(self, parent):
        """Create modern configuration tab with projects."""
        # Scrollable frame
        canvas = tk.Canvas(parent, bg=ModernStyle.WHITE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Title
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, padx=20, pady=20)

        ttk.Label(title_frame, text="Configuration", style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Configure simulation parameters and select projects",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(anchor=tk.W, pady=(5, 0))

        # Parameters card
        params_card = ttk.LabelFrame(scrollable_frame, text="Simulation Parameters", padding=20)
        params_card.pack(fill=tk.X, padx=20, pady=10)

        # Max Distance
        dist_frame = ttk.Frame(params_card)
        dist_frame.pack(fill=tk.X, pady=8)
        ttk.Label(dist_frame, text="Max Distance (meters):", font=ModernStyle.FONT_BODY, width=25).pack(side=tk.LEFT)
        dist_entry = ttk.Entry(dist_frame, textvariable=self.config['MAX_DISTANCE'], width=15, font=ModernStyle.FONT_BODY)
        dist_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(dist_frame, text="Distance for credit redistribution",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(side=tk.LEFT, padx=10)

        # Max Year (fixed at 2060)
        year_frame = ttk.Frame(params_card)
        year_frame.pack(fill=tk.X, pady=8)
        ttk.Label(year_frame, text="Max Year:", font=ModernStyle.FONT_BODY, width=25).pack(side=tk.LEFT)
        ttk.Label(year_frame, text="2060 (fixed for simulation)",
                 font=ModernStyle.FONT_BODY, foreground=ModernStyle.INFO).pack(side=tk.LEFT, padx=10)

        # Max Renewals (fixed)
        renew_frame = ttk.Frame(params_card)
        renew_frame.pack(fill=tk.X, pady=8)
        ttk.Label(renew_frame, text="Max Renewals:", font=ModernStyle.FONT_BODY, width=25).pack(side=tk.LEFT)
        ttk.Label(renew_frame, text=f"{self.config['MAX_RENEWALS']} (fixed per regulations)",
                 font=ModernStyle.FONT_BODY, foreground=ModernStyle.INFO).pack(side=tk.LEFT, padx=10)

        # Scoring mode
        score_frame = ttk.Frame(params_card)
        score_frame.pack(fill=tk.X, pady=8)
        ttk.Label(score_frame, text="Scoring Mode:", font=ModernStyle.FONT_BODY, width=25).pack(side=tk.LEFT)

        mode_buttons = ttk.Frame(score_frame)
        mode_buttons.pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_buttons, text="Earliest Expiry", variable=self.config['SCORING_MODE'],
                       value='earliest_expiry').pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_buttons, text="Distance Surplus", variable=self.config['SCORING_MODE'],
                       value='distance_surplus').pack(side=tk.LEFT, padx=10)

        # Current date
        date_frame = ttk.Frame(params_card)
        date_frame.pack(fill=tk.X, pady=8)
        ttk.Label(date_frame, text="Current Date:", font=ModernStyle.FONT_BODY, width=25).pack(side=tk.LEFT)
        ttk.Label(date_frame, text=self.config['CURRENT_DATE'],
                 font=ModernStyle.FONT_HEADING, foreground=ModernStyle.SUCCESS).pack(side=tk.LEFT, padx=10)
        ttk.Label(date_frame, text="(automatically set to today)",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(side=tk.LEFT, padx=10)

        # Projects card
        projects_card = ttk.LabelFrame(scrollable_frame, text="Project Selection", padding=20)
        projects_card.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(projects_card, text="Select projects to include in analysis:",
                 font=ModernStyle.FONT_BODY).pack(anchor=tk.W, pady=(0, 10))

        # Project checkboxes in grid
        proj_grid = ttk.Frame(projects_card)
        proj_grid.pack(fill=tk.X)

        for i, project in enumerate(self.projects):
            row = i // 3
            col = i % 3
            cb_frame = ttk.Frame(proj_grid)
            cb_frame.grid(row=row, column=col, sticky=tk.W, padx=20, pady=5)

            cb = ttk.Checkbutton(cb_frame, text=project, variable=self.project_vars[project])
            cb.pack(side=tk.LEFT)

        # Select all/none buttons
        proj_buttons = ttk.Frame(projects_card)
        proj_buttons.pack(fill=tk.X, pady=(15, 0))

        ttk.Button(proj_buttons, text="✓ Select All",
                  command=self.select_all_projects, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(proj_buttons, text="✗ Deselect All",
                  command=self.deselect_all_projects, width=15).pack(side=tk.LEFT, padx=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_run_tab(self, parent):
        """Create modern run simulation tab."""
        container = ttk.Frame(parent, padding=20)
        container.pack(fill='both', expand=True)

        # Title
        title_frame = ttk.Frame(container)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(title_frame, text="Run Simulation", style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Execute the claim redistribution analysis",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(anchor=tk.W, pady=(5, 0))

        # Progress card
        progress_card = ttk.LabelFrame(container, text="Progress", padding=15)
        progress_card.pack(fill=tk.X, pady=10)

        # Progress bar
        ttk.Label(progress_card, text="Overall Progress:", font=ModernStyle.FONT_BODY).pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(progress_card, mode='determinate', maximum=100, length=500)
        self.overall_progress.pack(fill=tk.X, pady=5)

        # Step and detail labels
        self.step_label = ttk.Label(progress_card, text="Step: Waiting to start...",
                                    font=ModernStyle.FONT_HEADING, foreground=ModernStyle.SECONDARY)
        self.step_label.pack(anchor=tk.W, pady=(10, 5))

        self.detail_label = ttk.Label(progress_card, text="",
                                      font=ModernStyle.FONT_SMALL, foreground='gray')
        self.detail_label.pack(anchor=tk.W)

        # Control buttons
        button_frame = ttk.Frame(container)
        button_frame.pack(pady=15)

        self.run_button = ttk.Button(button_frame, text="▶ Run Simulation",
                                     command=self.run_simulation,
                                     style='Success.TButton', width=20)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="⬛ Stop",
                                      command=self.stop_simulation,
                                      state=tk.DISABLED, width=15)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="📁 Open Output Folder",
                  command=self.open_output_folder, width=20).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="🗑 Clear Log",
                  command=self.clear_log, width=15).pack(side=tk.LEFT, padx=5)

        # Log area
        log_card = ttk.LabelFrame(container, text="Simulation Log", padding=10)
        log_card.pack(fill='both', expand=True, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_card, height=20, width=100,
                                                  wrap=tk.WORD, font=('Consolas', 9),
                                                  bg=ModernStyle.DARK, fg=ModernStyle.WHITE)
        self.log_text.pack(fill='both', expand=True)

        # Configure text tags
        self.log_text.tag_config('error', foreground=ModernStyle.DANGER)
        self.log_text.tag_config('success', foreground=ModernStyle.SUCCESS)
        self.log_text.tag_config('warning', foreground=ModernStyle.WARNING)
        self.log_text.tag_config('info', foreground=ModernStyle.INFO)

    def create_results_tab(self, parent):
        """Create modern results dashboard tab with vertical scrolling layout."""
        # Main container with scrollbar
        main_container = ttk.Frame(parent)
        main_container.pack(fill='both', expand=True)

        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=20)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill='y')

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        container = scrollable_frame

        # Title
        title_frame = ttk.Frame(container)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(title_frame, text="Results Dashboard", style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Summary statistics and visualizations",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(anchor=tk.W, pady=(5, 0))

        # Refresh button at top
        ttk.Button(container, text="🔄 Refresh Results",
                  command=self.refresh_results,
                  style='Primary.TButton').pack(pady=10, anchor=tk.W)

        # Statistics grid
        stats_card = ttk.LabelFrame(container, text="Summary Statistics", padding=15)
        stats_card.pack(fill=tk.X, pady=10)

        stats_grid = ttk.Frame(stats_card)
        stats_grid.pack(fill=tk.X)

        self.stats_labels = {}
        stats = [
            ('total_claims', 'Total Claims', '0'),
            ('avg_life', 'Avg Years of Life', '0.00'),
            ('min_life', 'Min Years', '0.00'),
            ('max_life', 'Max Years', '0.00'),
            ('total_credits_available', 'Total Credits Available', '$0'),
            ('credits_unused', 'Credits Unused', '$0 (0%)'),
            ('credits_used_renewals', 'Credits Used (Renewals)', '$0 (0%)'),
            ('credits_redistributed', 'Credits Redistributed', '$0')
        ]

        for i, (key, label, default) in enumerate(stats):
            row = i // 4
            col = i % 4

            stat_frame = ttk.Frame(stats_grid)
            stat_frame.grid(row=row, column=col, padx=15, pady=10, sticky=tk.W)

            ttk.Label(stat_frame, text=label, font=ModernStyle.FONT_SMALL,
                     foreground='gray').pack(anchor=tk.W)
            value_label = ttk.Label(stat_frame, text=default, font=ModernStyle.FONT_HEADING,
                                   foreground=ModernStyle.SECONDARY)
            value_label.pack(anchor=tk.W)
            self.stats_labels[key] = value_label

        # Credits by Project subsection
        ttk.Separator(stats_card, orient='horizontal').pack(fill=tk.X, pady=10)

        credits_title = ttk.Label(stats_card, text="Credits by Project",
                                 font=ModernStyle.FONT_HEADING,
                                 foreground=ModernStyle.PRIMARY)
        credits_title.pack(anchor=tk.W, pady=(5, 10))

        # Create frame for credits by project grid
        self.credits_grid = ttk.Frame(stats_card)
        self.credits_grid.pack(fill=tk.X)

        # Chart 1: Number of Lapsed Claims by Project and Year
        chart1_card = ttk.LabelFrame(container, text="Number of Lapsed Claims by Project and Year", padding=10)
        chart1_card.pack(fill=tk.X, pady=10)

        self.chart1_figure = Figure(figsize=(12, 5), dpi=100, facecolor=ModernStyle.WHITE)
        self.chart1_canvas = FigureCanvasTkAgg(self.chart1_figure, chart1_card)
        self.chart1_canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar1 = NavigationToolbar2Tk(self.chart1_canvas, chart1_card)
        toolbar1.update()

        # Chart 2: Average Claim Life by Project
        chart2_card = ttk.LabelFrame(container, text="Average Claim Life by Project", padding=10)
        chart2_card.pack(fill=tk.X, pady=10)

        self.chart2_figure = Figure(figsize=(12, 5), dpi=100, facecolor=ModernStyle.WHITE)
        self.chart2_canvas = FigureCanvasTkAgg(self.chart2_figure, chart2_card)
        self.chart2_canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar2 = NavigationToolbar2Tk(self.chart2_canvas, chart2_card)
        toolbar2.update()

        # Pivot Table
        table_card = ttk.LabelFrame(container, text="Required Spending by Year and Project", padding=10)
        table_card.pack(fill=tk.X, pady=10)

        # Create scrollable text widget for pivot table
        table_scroll = ttk.Scrollbar(table_card, orient="vertical")
        self.pivot_table_text = tk.Text(table_card, height=12, wrap=tk.NONE,
                                        font=('Courier New', 9),
                                        yscrollcommand=table_scroll.set)
        table_scroll.config(command=self.pivot_table_text.yview)

        self.pivot_table_text.pack(side=tk.LEFT, fill='both', expand=True)
        table_scroll.pack(side=tk.RIGHT, fill='y')

    def create_map_tab(self, parent):
        """Create modern map viewer tab."""
        container = ttk.Frame(parent, padding=20)
        container.pack(fill='both', expand=True)

        # Title and controls
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="Map Viewer", style='Title.TLabel').pack(side=tk.LEFT)

        control_frame = ttk.Frame(header_frame)
        control_frame.pack(side=tk.RIGHT)

        self.map_type = tk.StringVar(value='static')
        ttk.Radiobutton(control_frame, text="📊 Static Map", variable=self.map_type,
                       value='static', command=self.load_map).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="🌐 Interactive Map", variable=self.map_type,
                       value='interactive', command=self.load_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔄 Reload", command=self.load_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Export PNG", command=self.export_map_png,
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=5)

        # Map display
        if PIL_AVAILABLE:
            map_frame = ttk.Frame(container, relief=tk.SUNKEN, borderwidth=2)
            map_frame.pack(fill='both', expand=True)

            self.map_canvas = tk.Canvas(map_frame, bg=ModernStyle.LIGHT)
            self.map_canvas.pack(side="left", fill="both", expand=True)

            h_scroll = ttk.Scrollbar(map_frame, orient=tk.HORIZONTAL, command=self.map_canvas.xview)
            h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            v_scroll = ttk.Scrollbar(map_frame, orient=tk.VERTICAL, command=self.map_canvas.yview)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            self.map_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

            # Add zoom capability with mouse wheel
            self.map_zoom_level = 1.0
            self.original_map_image = None
            self.map_canvas.bind("<MouseWheel>", self.zoom_map)
            self.map_canvas.bind("<Button-4>", self.zoom_map)  # Linux scroll up
            self.map_canvas.bind("<Button-5>", self.zoom_map)  # Linux scroll down
        else:
            ttk.Label(container, text="PIL library not available\nInstall Pillow to view maps:\npip install Pillow",
                     font=ModernStyle.FONT_BODY, foreground=ModernStyle.WARNING).pack(pady=20)

    def create_report_tab(self, parent):
        """Create PDF report generation tab."""
        container = ttk.Frame(parent, padding=20)
        container.pack(fill='both', expand=True)

        # Title
        title_frame = ttk.Frame(container)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(title_frame, text="Professional Report", style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Generate comprehensive PDF reports",
                 font=ModernStyle.FONT_SMALL, foreground='gray').pack(anchor=tk.W, pady=(5, 0))

        if not REPORTLAB_AVAILABLE:
            warning_frame = ttk.Frame(container, relief=tk.RAISED, borderwidth=2)
            warning_frame.pack(fill=tk.X, pady=20, padx=20)

            ttk.Label(warning_frame, text="⚠ ReportLab library not available",
                     font=ModernStyle.FONT_HEADING, foreground=ModernStyle.WARNING).pack(pady=10)
            ttk.Label(warning_frame, text="Install reportlab to enable PDF export:\npip install reportlab",
                     font=ModernStyle.FONT_BODY).pack(pady=10)
            return

        # Report preview/display
        preview_card = ttk.LabelFrame(container, text="Report Preview", padding=20)
        preview_card.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create canvas for PDF preview (if available)
        if PIL_AVAILABLE:
            self.report_canvas = tk.Canvas(preview_card, bg=ModernStyle.LIGHT)
            self.report_canvas.pack(side="left", fill="both", expand=True)

            report_scroll = ttk.Scrollbar(preview_card, orient=tk.VERTICAL, command=self.report_canvas.yview)
            report_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.report_canvas.configure(yscrollcommand=report_scroll.set)

            self.report_preview_image = None

            # Instructions
            ttk.Label(preview_card, text="Generate a report to see preview here",
                     font=ModernStyle.FONT_BODY, foreground='gray').place(relx=0.5, rely=0.5, anchor='center')
        else:
            # Fallback text preview
            preview_text = scrolledtext.ScrolledText(preview_card, height=20, width=80,
                                                     wrap=tk.WORD, font=ModernStyle.FONT_BODY,
                                                     bg=ModernStyle.LIGHT)
            preview_text.pack(fill='both', expand=True)

            preview_content = """REPORT CONTENTS:

• Results by Project Table
• Projected Lapse Years Map
• Upcoming Concerns (1-2 Years)

Generate report below to create PDF."""

            preview_text.insert('1.0', preview_content)
            preview_text.configure(state='disabled')

        # Buttons
        export_frame = ttk.Frame(container)
        export_frame.pack(pady=20)

        ttk.Button(export_frame, text="🔄 Refresh Preview",
                  command=lambda: self.show_pdf_preview(None),
                  style='Secondary.TButton',
                  width=20).pack(side=tk.LEFT, padx=5)

        ttk.Button(export_frame, text="📄 Generate PDF Report",
                  command=self.generate_pdf_report,
                  style='Primary.TButton',
                  width=30).pack(side=tk.LEFT, padx=5)

    # Helper methods (file browsing, project selection, etc.)
    def browse_file(self, config_key, filetype):
        filename = filedialog.askopenfilename(
            title=f"Select {config_key}",
            filetypes=[(f"{filetype} files", filetype), ("All files", "*.*")]
        )
        if filename:
            self.config[config_key].set(filename)

    def browse_directory(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.config['OUTPUT_DIR'].set(dirname)

    def set_default_paths(self):
        base_dir = Path.cwd()
        files_dir = base_dir / "Files"

        defaults = {
            'MIDLAND_XLSX': files_dir / "Gestim_Midland_09012026.xlsx",
            'WALLBRIDGE_XLSX': files_dir / "Translated" / "Gestim_Wallbridge_090126.xlsx",
            'PROPERTY_CSV': files_dir / "Property_to_Claim.csv",
            'SHP_PATH': files_dir / "Shapefile" / "gsm_claims_20250703.shp",
            'OUTLINES_SHP': files_dir / "Shapefile" / "wmc_property_outlines.shp",
            'OUTPUT_DIR': files_dir / "Redistribution"
        }

        for key, path in defaults.items():
            if path.exists():
                self.config[key].set(str(path))
            else:
                self.log(f"Default path not found: {path}", 'warning')

        self.log("Default paths set", 'success')

    def select_all_projects(self):
        for var in self.project_vars.values():
            var.set(True)

    def deselect_all_projects(self):
        for var in self.project_vars.values():
            var.set(False)

    def load_config_from_file(self):
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    saved_config = json.load(f)

                for key, value in saved_config.items():
                    if key in self.config and key not in ['CURRENT_DATE', 'MAX_RENEWALS']:
                        if isinstance(self.config[key], tk.StringVar):
                            self.config[key].set(value)
                        elif isinstance(self.config[key], tk.IntVar):
                            self.config[key].set(int(value))

                if 'INCLUDED_PROJECTS' in saved_config:
                    for proj in self.projects:
                        self.project_vars[proj].set(proj in saved_config['INCLUDED_PROJECTS'])

                self.log(f"Configuration loaded from {filename}", 'success')
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                self.log(f"Error loading configuration: {e}", 'error')
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    def save_config_to_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config_dict = {}
                for key, var in self.config.items():
                    if key not in ['CURRENT_DATE', 'MAX_RENEWALS']:
                        config_dict[key] = var.get()

                config_dict['MAX_RENEWALS'] = self.config['MAX_RENEWALS']
                config_dict['INCLUDED_PROJECTS'] = [proj for proj, var in self.project_vars.items() if var.get()]

                with open(filename, 'w') as f:
                    json.dump(config_dict, f, indent=4)

                self.log(f"Configuration saved to {filename}", 'success')
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                self.log(f"Error saving configuration: {e}", 'error')
                messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def validate_inputs(self):
        errors = []

        file_labels = {
            'MIDLAND_XLSX': 'Midland Excel',
            'WALLBRIDGE_XLSX': 'Wallbridge Excel',
            'PROPERTY_CSV': 'Property CSV',
            'SHP_PATH': 'Claims Shapefile',
            'OUTLINES_SHP': 'Property Outlines Shapefile'
        }

        for key in file_labels:
            path = self.config[key].get()
            if not path:
                errors.append(f"❌ {file_labels[key]}: Not selected (use Browse button)")
            elif not Path(path).exists():
                errors.append(f"❌ {file_labels[key]}: File not found\n   Path: {path}\n   → Please update this path in the Files tab")

        if not self.config['OUTPUT_DIR'].get():
            errors.append("❌ Output directory: Not specified")

        try:
            max_dist = float(self.config['MAX_DISTANCE'].get())
            if max_dist <= 0:
                errors.append("❌ Max distance must be positive")
        except ValueError:
            errors.append("❌ Max distance must be a valid number")

        if not any(var.get() for var in self.project_vars.values()):
            errors.append("❌ At least one project must be selected")

        return errors

    def update_progress(self, step, detail="", percent=0):
        self.progress_queue.put(('progress', step, detail, percent))

    def run_simulation(self):
        if not MAIN_MODULE_AVAILABLE:
            messagebox.showerror("Error", "Main redistribution module not found.")
            return

        errors = self.validate_inputs()
        if errors:
            error_msg = "Configuration Issues Found:\n\n" + "\n\n".join(errors)
            error_msg += "\n\n💡 TIP: Go to the 'Files' tab and click 'Browse' to select your data files."
            messagebox.showerror("Configuration Required", error_msg)
            # Switch to Files tab to help user fix the issue
            self.notebook.select(0)  # Switch to first tab (Files)
            return

        selected_projects = [proj for proj, var in self.project_vars.items() if var.get()]
        confirm = messagebox.askyesno(
            "Confirm Simulation",
            f"Run simulation?\n\nProjects: {', '.join(selected_projects)}\n"
            f"Max Distance: {self.config['MAX_DISTANCE'].get()} m\n"
            f"This may take several minutes."
        )

        if not confirm:
            return

        self.clear_log()
        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Running...", foreground=ModernStyle.SECONDARY)
        self.overall_progress['value'] = 0

        thread = threading.Thread(target=self.run_simulation_thread, daemon=True)
        thread.start()

    def run_simulation_thread(self):
        """Thread function to run simulation with progress updates."""
        try:
            config = self.build_config()

            Path(config['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
            Path(config['TEMP_DIR']).mkdir(parents=True, exist_ok=True)

            self.update_progress("Starting", "Initializing simulation...", 0)
            self.log(f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'info')

            # Steps 1-9 as before
            self.update_progress("Step 1/9", "Processing Midland file...", 5)
            midland_df = redistribution.process_midland_file(config['MIDLAND_XLSX'], config['PROPERTY_CSV'], config)
            self.log(f"✓ Processed {len(midland_df)} Midland claims", 'success')
            self.update_progress("Step 1/9", f"Processed {len(midland_df)} Midland claims", 15)

            self.update_progress("Step 2/9", "Processing Wallbridge file...", 20)
            wallbridge_df = redistribution.process_wallbridge_file(config['WALLBRIDGE_XLSX'], config['PROPERTY_CSV'], config)
            self.log(f"✓ Processed {len(wallbridge_df)} Wallbridge claims", 'success')
            self.update_progress("Step 2/9", f"Processed {len(wallbridge_df)} Wallbridge claims", 30)

            self.update_progress("Step 3/9", "Merging datasets...", 35)
            merged_csv = redistribution.merge_datasets(midland_df, wallbridge_df, config)
            self.log(f"✓ Merged CSV saved: {merged_csv}", 'success')
            self.update_progress("Step 3/9", "Datasets merged", 40)

            self.update_progress("Step 4/9", "Loading spatial data...", 45)
            gdf, gdf_outlines = redistribution.load_and_prepare_data(merged_csv, config['SHP_PATH'], config['OUTLINES_SHP'], config)
            self.log(f"✓ Loaded {len(gdf)} claims and {len(gdf_outlines)} property outlines", 'success')
            self.update_progress("Step 4/9", f"Loaded {len(gdf)} claims", 50)

            self.update_progress("Step 5/9", "Initializing simulation...", 52)
            gdf = redistribution.initialize_simulation(gdf)
            self.update_progress("Step 5/9", "Simulation initialized", 55)

            # Optional: Load exploration credits if enabled
            credited_titles = []
            if self.credits_enabled.get() and self.config['CREDITS_XLSX'].get():
                credits_path = self.config['CREDITS_XLSX'].get()
                if credits_path and Path(credits_path).exists():
                    self.update_progress("Step 5b/9", "Loading exploration credits...", 56)
                    self.log(f"Loading exploration credits from: {credits_path}", 'info')
                    try:
                        gdf, num_credits, total_credits, credited_titles = redistribution.load_and_apply_credits(
                            credits_path, gdf, config['OUTPUT_DIR']
                        )
                        self.log(f"✓ Added {num_credits} credit entries totaling ${total_credits:,.2f}", 'success')
                        self.log(f"  Credits expire after 12 years per Quebec Mining Act Section 75", 'info')
                    except Exception as e:
                        self.log(f"⚠ Warning: Could not load credits: {str(e)}", 'warning')
                        self.log("  Continuing simulation without additional credits...", 'warning')
                else:
                    self.log(f"⚠ Credits file not found: {credits_path}", 'warning')

            self.update_progress("Step 6/9", "Computing spatial relationships...", 57)
            dist_matrix, sindex = redistribution.precompute_spatial_data(gdf)
            self.log(f"✓ Computed distance matrix", 'success')
            self.update_progress("Step 6/9", "Spatial data computed", 60)

            self.update_progress("Step 7/9", "Running redistribution simulation...", 65)
            current_date_obj = date.fromisoformat(config['CURRENT_DATE'])
            log_table, unresolved = redistribution.run_simulation(gdf, dist_matrix, sindex, current_date_obj, config)
            self.log(f"✓ Simulation complete", 'success')
            self.update_progress("Step 7/9", "Simulation complete", 75)

            self.update_progress("Step 8/9", "Exporting results...", 78)
            redistribution.export_results(log_table, unresolved, gdf, config['OUTPUT_DIR'], current_date_obj)
            self.log(f"✓ Results exported", 'success')
            self.update_progress("Step 8/9", "Results exported", 85)

            self.update_progress("Step 9/9", "Creating visualizations...", 88)
            redistribution.plot_results(gdf, gdf_outlines, config['OUTPUT_DIR'], credited_titles)
            redistribution.plot_interactive_map(gdf, log_table, unresolved, gdf_outlines, config['OUTPUT_DIR'])
            redistribution.plot_summary_by_project(gdf, unresolved, config['OUTPUT_DIR'])
            self.log(f"✓ All visualizations created", 'success')
            self.update_progress("Complete", "Simulation finished successfully", 100)

            # Store results
            self.current_results = {
                'gdf': gdf,
                'unresolved': unresolved,
                'config': config,
                'timestamp': datetime.now(),
                'credited_titles': credited_titles
            }

            # Update UI
            self.root.after(0, self.refresh_results)
            self.root.after(0, self.load_map)
            self.root.after(0, lambda: self.show_pdf_preview(None))  # Auto-show report preview

            self.log("\n" + "="*80, 'success')
            self.log("SIMULATION COMPLETE", 'success')
            self.log("="*80, 'success')

            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Simulation completed!\n\nClaims analyzed: {len(gdf)}\nResults: {config['OUTPUT_DIR']}"
            ))

        except Exception as e:
            error_msg = str(e)
            self.log(f"\nERROR: {error_msg}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Simulation failed:\n\n{msg}"))

        finally:
            self.root.after(0, self.simulation_finished)

    def build_config(self):
        return {
            'MIDLAND_XLSX': self.config['MIDLAND_XLSX'].get(),
            'WALLBRIDGE_XLSX': self.config['WALLBRIDGE_XLSX'].get(),
            'PROPERTY_CSV': self.config['PROPERTY_CSV'].get(),
            'SHP_PATH': self.config['SHP_PATH'].get(),
            'OUTLINES_SHP': self.config['OUTLINES_SHP'].get(),
            'OUTPUT_DIR': self.config['OUTPUT_DIR'].get(),
            'MAX_DISTANCE': float(self.config['MAX_DISTANCE'].get()),
            'MAX_YEAR': self.config['MAX_YEAR'].get(),
            'MAX_RENEWALS': self.config['MAX_RENEWALS'],
            'SCORING_MODE': self.config['SCORING_MODE'].get(),
            'CURRENT_DATE': date.today().isoformat(),
            'INCLUDED_PROJECTS': [proj for proj, var in self.project_vars.items() if var.get()],
            'TEMP_DIR': str(Path(self.config['OUTPUT_DIR'].get()).parent / "Temp"),
            'LOG_DIR': str(Path(self.config['OUTPUT_DIR'].get()).parent / "Logs"),
            'SCORING_WEIGHTS': {'surplus': 0.3, 'distance': 0.7}
        }

    def refresh_results(self):
        """Refresh results dashboard."""
        if not self.current_results:
            self.log("No results available yet. Run a simulation first.", 'warning')
            return

        gdf = self.current_results['gdf']
        unresolved = self.current_results['unresolved']
        config = self.current_results['config']

        # Calculate credit statistics from redistribution log
        import pandas as pd
        output_dir = Path(config['OUTPUT_DIR'])

        # Get total credits available and used from full_redistribution_log
        full_log_path = output_dir / 'full_redistribution_log.csv'
        total_credits_available = 0
        credits_used_renewals = 0
        credits_redistributed = 0

        if full_log_path.exists():
            try:
                log_df = pd.read_csv(full_log_path)

                # Helper function to parse credit values from expiration strings
                def parse_credits(exp_str):
                    """Parse credits from format like '2029/04/09 (12946.51 $)'"""
                    if pd.isna(exp_str) or exp_str == '':
                        return 0
                    total = 0
                    for part in str(exp_str).split(';'):
                        if '(' in part and '$' in part:
                            try:
                                credit_str = part.split('(')[1].split('$')[0].strip()
                                total += float(credit_str)
                            except:
                                pass
                    return total

                # Calculate total credits available from original_expirations
                log_df['original_credits'] = log_df['original_expirations'].apply(parse_credits)
                total_credits_available = log_df['original_credits'].sum()

                # Calculate credits used for renewals
                renewal_rows = log_df[log_df['action_type'] == 'renewal']
                if not renewal_rows.empty and 'renewal_amount' in renewal_rows.columns:
                    credits_used_renewals = renewal_rows['renewal_amount'].sum()

                # Calculate credits redistributed
                redistribution_rows = log_df[log_df['action_type'] == 'redistribution']
                if not redistribution_rows.empty and 'credits_pulled' in redistribution_rows.columns:
                    credits_redistributed = redistribution_rows['credits_pulled'].sum()

            except Exception as e:
                self.log(f"Warning: Could not calculate credit statistics: {e}", 'warning')

        credits_unused = total_credits_available - credits_used_renewals - credits_redistributed

        # Update statistics
        self.stats_labels['total_claims'].config(text=str(len(gdf)))
        self.stats_labels['avg_life'].config(text=f"{gdf['years_of_life'].mean():.2f}")
        self.stats_labels['min_life'].config(text=f"{gdf['years_of_life'].min():.2f}")
        self.stats_labels['max_life'].config(text=f"{gdf['years_of_life'].max():.2f}")

        self.stats_labels['total_credits_available'].config(text=f"${total_credits_available:,.2f}")

        pct_unused = (credits_unused / total_credits_available * 100) if total_credits_available > 0 else 0
        self.stats_labels['credits_unused'].config(text=f"${credits_unused:,.2f} ({pct_unused:.1f}%)")

        pct_renewals = (credits_used_renewals / total_credits_available * 100) if total_credits_available > 0 else 0
        self.stats_labels['credits_used_renewals'].config(text=f"${credits_used_renewals:,.2f} ({pct_renewals:.1f}%)")

        self.stats_labels['credits_redistributed'].config(text=f"${credits_redistributed:,.2f}")

        # Update charts and pivot table
        self.plot_summary_chart(gdf, unresolved)
        self.update_pivot_table(gdf)
        self.update_credits_by_project(gdf, config)

        self.notebook.select(3)

    def plot_summary_chart(self, gdf, unresolved):
        """Plot charts separately in vertical layout."""
        import seaborn as sns

        # Clear both figures
        self.chart1_figure.clear()
        self.chart2_figure.clear()

        # Create axes for each chart
        ax1 = self.chart1_figure.add_subplot(111)
        ax2 = self.chart2_figure.add_subplot(111)

        # Left chart: Claims by Project and Year
        if 'life_category' not in gdf.columns:
            gdf['life_category'] = gdf['final_expiry_date'].apply(lambda x: str(x.year))

        order = sorted(gdf['life_category'].unique())
        sns.countplot(data=gdf, x='life_category', hue='project', order=order, ax=ax1, palette='tab10')

        ax1.set_xlabel('Lapse Year', fontweight='bold', fontsize=10)
        ax1.set_ylabel('Number of Lapsed Claims', fontweight='bold', fontsize=10)
        ax1.set_title('Number of Lapsed Claims by Project and Year', fontweight='bold', fontsize=11)
        ax1.legend(title='Project', fontsize=8, loc='upper left')
        ax1.grid(axis='y', alpha=0.2, linestyle='--')
        ax1.set_facecolor(ModernStyle.LIGHT)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)

        # Right chart: Average Claim Life by Project
        project_stats = gdf.groupby('project')['years_of_life'].mean().sort_values()

        colors = [ModernStyle.SECONDARY, ModernStyle.INFO, ModernStyle.SUCCESS,
                 ModernStyle.WARNING, ModernStyle.PRIMARY, ModernStyle.DANGER]

        bars = ax2.barh(project_stats.index, project_stats.values,
                      color=colors[:len(project_stats)], edgecolor=ModernStyle.DARK, linewidth=1.5)

        ax2.set_xlabel('Average Years of Life', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Project', fontweight='bold', fontsize=10)
        ax2.set_title('Average Claim Life by Project', fontweight='bold', fontsize=11)
        ax2.grid(axis='x', alpha=0.2, linestyle='--')
        ax2.set_facecolor(ModernStyle.LIGHT)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, f' {width:.1f}',
                   ha='left', va='center', fontweight='bold', fontsize=9)

        # Refresh both canvases
        self.chart1_figure.tight_layout()
        self.chart1_canvas.draw()

        self.chart2_figure.tight_layout()
        self.chart2_canvas.draw()

    def update_pivot_table(self, gdf):
        """Update the pivot table showing required spending by year and project."""
        import pandas as pd

        # Clear existing content
        self.pivot_table_text.config(state='normal')
        self.pivot_table_text.delete('1.0', tk.END)

        # Create pivot table: spending required by year and project
        # Years as rows, projects as columns
        gdf['lapse_year'] = gdf['final_expiry_date'].apply(lambda x: x.year)

        # Group by lapse year and project to count claims
        pivot_data = gdf.groupby(['lapse_year', 'project']).size().reset_index(name='claims_count')

        # Renewal cost per claim (actual cost from redistribution system)
        RENEWAL_COST_PER_CLAIM = 2500

        pivot_data['required_spending'] = pivot_data['claims_count'] * RENEWAL_COST_PER_CLAIM

        # Create pivot table with years as rows (index) and projects as columns
        pivot = pivot_data.pivot(index='lapse_year', columns='project', values='required_spending')
        pivot = pivot.fillna(0)

        # Add totals column (total per year) and row (total per project)
        pivot['TOTAL'] = pivot.sum(axis=1)
        pivot.loc['TOTAL'] = pivot.sum(axis=0)

        # Format as text table
        table_str = pivot.to_string(float_format=lambda x: f'${x:,.0f}')

        self.pivot_table_text.insert('1.0', table_str)
        self.pivot_table_text.config(state='disabled')

    def update_credits_by_project(self, gdf, config):
        """Update the credits by project grid in summary statistics."""
        import pandas as pd

        # Clear existing widgets in credits grid
        for widget in self.credits_grid.winfo_children():
            widget.destroy()

        output_dir = Path(config['OUTPUT_DIR'])
        full_log_path = output_dir / 'full_redistribution_log.csv'

        if not full_log_path.exists():
            ttk.Label(self.credits_grid, text="No redistribution log found.",
                     foreground='gray').pack()
            return

        try:
            # Read the redistribution log
            log_df = pd.read_csv(full_log_path)

            # Convert title_no to same type for merging
            # Convert float to int first to avoid .0 in string (e.g., 2284009.0 -> 2284009 -> "2284009")
            log_df['title_no'] = log_df['title_no'].fillna(0).astype(int).astype(str)
            gdf_merge = gdf[['title_no', 'project']].copy()
            gdf_merge['title_no'] = gdf_merge['title_no'].astype(str)

            # Merge with gdf to get project information
            log_with_project = log_df.merge(gdf_merge,
                                            left_on='title_no', right_on='title_no', how='left')

            # Calculate credits used for renewals by project
            renewal_rows = log_with_project[log_with_project['action_type'] == 'renewal']
            if not renewal_rows.empty and 'renewal_amount' in renewal_rows.columns:
                # Only group by rows that have a project
                renewal_with_project = renewal_rows[renewal_rows['project'].notna()]
                if not renewal_with_project.empty:
                    renewals_by_project = renewal_with_project.groupby('project')['renewal_amount'].sum()
                else:
                    renewals_by_project = pd.Series(dtype=float)
                    self.log("Warning: No renewals matched with project data", 'warning')
            else:
                renewals_by_project = pd.Series(dtype=float)

            # Calculate credits redistributed BY donor project
            redistribution_rows = log_df[log_df['action_type'] == 'redistribution'].copy()
            if not redistribution_rows.empty and 'credits_pulled' in redistribution_rows.columns and 'donor_title_no' in redistribution_rows.columns:
                # Convert donor_title_no to string for merging (float -> int -> string to remove .0)
                redistribution_rows['donor_title_no'] = redistribution_rows['donor_title_no'].fillna(0).astype(int).astype(str)

                # Get donor project - merge on donor_title_no
                redistr_with_donor = redistribution_rows.merge(
                    gdf_merge,
                    left_on='donor_title_no', right_on='title_no',
                    how='left')
                # Group by the donor's project (only non-null projects)
                redistr_with_project = redistr_with_donor[redistr_with_donor['project'].notna()]
                if not redistr_with_project.empty:
                    redistributed_by_project = redistr_with_project.groupby('project')['credits_pulled'].sum()
                else:
                    redistributed_by_project = pd.Series(dtype=float)
                    self.log("Warning: No redistributions matched with project data", 'warning')
            else:
                redistributed_by_project = pd.Series(dtype=float)

            # Helper function to parse credit values
            def parse_credits(exp_str):
                """Parse credits from format like '2029/04/09 (12946.51 $)'"""
                if pd.isna(exp_str) or exp_str == '':
                    return 0
                total = 0
                for part in str(exp_str).split(';'):
                    if '(' in part and '$' in part:
                        try:
                            credit_str = part.split('(')[1].split('$')[0].strip()
                            total += float(credit_str)
                        except:
                            pass
                return total

            # Calculate total initial credits by project
            log_with_project['original_credits'] = log_with_project['original_expirations'].apply(parse_credits)
            initial_credits_by_project = log_with_project.groupby('project')['original_credits'].sum()

            # Get all projects
            all_projects = sorted(gdf['project'].unique())

            # Create table with projects as column headers
            # Row 0: Project names (headers)
            col = 0
            for project in all_projects:
                ttk.Label(self.credits_grid, text=project,
                         font=ModernStyle.FONT_HEADING,
                         foreground=ModernStyle.PRIMARY).grid(row=0, column=col, padx=10, pady=5)
                col += 1

            # Row 1: Utilized
            col = 0
            for project in all_projects:
                utilized = renewals_by_project.get(project, 0)
                ttk.Label(self.credits_grid, text=f"Utilized: ${utilized:,.0f}",
                         font=('Segoe UI', 9),
                         foreground=ModernStyle.SUCCESS).grid(row=1, column=col, padx=10, pady=2)
                col += 1

            # Row 2: Redistributed
            col = 0
            for project in all_projects:
                redistributed = redistributed_by_project.get(project, 0)
                ttk.Label(self.credits_grid, text=f"Redistributed: ${redistributed:,.0f}",
                         font=('Segoe UI', 9),
                         foreground=ModernStyle.WARNING).grid(row=2, column=col, padx=10, pady=2)
                col += 1

            # Row 3: Unused
            col = 0
            for project in all_projects:
                initial = initial_credits_by_project.get(project, 0)
                utilized = renewals_by_project.get(project, 0)
                redistributed = redistributed_by_project.get(project, 0)
                unused = initial - utilized - redistributed
                ttk.Label(self.credits_grid, text=f"Unused: ${unused:,.0f}",
                         font=('Segoe UI', 9),
                         foreground=ModernStyle.DANGER).grid(row=3, column=col, padx=10, pady=2)
                col += 1

        except Exception as e:
            ttk.Label(self.credits_grid, text=f"Error: {str(e)}",
                     foreground='red').pack()

    def load_map(self):
        """Load and display map."""
        if not self.current_results:
            return

        output_dir = Path(self.current_results['config']['OUTPUT_DIR'])

        if self.map_type.get() == 'static':
            map_file = output_dir / 'gsm_claims_lapse_years_map.png'
            if map_file.exists() and PIL_AVAILABLE:
                try:
                    # Load original image and store it for zooming
                    self.original_map_image = Image.open(map_file)

                    # Calculate zoom to fit canvas perfectly
                    canvas_width = self.map_canvas.winfo_width()
                    canvas_height = self.map_canvas.winfo_height()
                    img_width, img_height = self.original_map_image.size

                    # Only auto-fit if canvas has been rendered (width > 100 to ensure it's fully initialized)
                    if canvas_width > 100 and canvas_height > 100:
                        zoom_x = canvas_width / img_width
                        zoom_y = canvas_height / img_height
                        self.map_zoom_level = min(zoom_x, zoom_y) * 0.90  # 90% to add margin
                    else:
                        # Set a reasonable default zoom (fit to typical canvas size)
                        # Assuming typical canvas is around 800x600
                        default_zoom_x = 800 / img_width
                        default_zoom_y = 600 / img_height
                        self.map_zoom_level = min(default_zoom_x, default_zoom_y) * 0.90

                    self.display_map_at_zoom()
                    self.log("Static map loaded (use mouse wheel to zoom)", 'success')
                except Exception as e:
                    self.log(f"Error loading map: {e}", 'error')
        else:
            # Open interactive map in browser
            map_file = output_dir / 'interactive_gsm_claims_map.html'
            if map_file.exists():
                webbrowser.open(str(map_file))
                self.log(f"Opened interactive map", 'success')

    def display_map_at_zoom(self):
        """Display map at current zoom level."""
        if not self.original_map_image:
            return

        # Calculate new size based on zoom level
        orig_width, orig_height = self.original_map_image.size
        new_width = int(orig_width * self.map_zoom_level)
        new_height = int(orig_height * self.map_zoom_level)

        # Resize image
        resized = self.original_map_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Display on canvas
        photo = ImageTk.PhotoImage(resized)
        self.map_canvas.delete("all")
        self.map_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.map_canvas.image = photo

        # Update scroll region
        self.map_canvas.config(scrollregion=(0, 0, new_width, new_height))

    def zoom_map(self, event):
        """Zoom map with mouse wheel."""
        if not self.original_map_image:
            return

        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Scroll up / zoom in
            self.map_zoom_level *= 1.1
        elif event.num == 5 or event.delta < 0:  # Scroll down / zoom out
            self.map_zoom_level /= 1.1

        # Limit zoom range
        self.map_zoom_level = max(0.1, min(self.map_zoom_level, 5.0))

        # Redisplay at new zoom
        self.display_map_at_zoom()

    def export_map_png(self):
        """Export the static map as PNG."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No map available. Run simulation first.")
            return

        output_dir = Path(self.current_results['config']['OUTPUT_DIR'])
        source_map = output_dir / 'gsm_claims_lapse_years_map.png'

        if not source_map.exists():
            messagebox.showerror("Error", "Static map file not found.")
            return

        filename = filedialog.asksaveasfilename(
            title="Export Map as PNG",
            defaultextension=".png",
            initialfile="mining_claims_map.png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if filename:
            try:
                import shutil
                shutil.copy(source_map, filename)
                self.log(f"Map exported to: {filename}", 'success')
                messagebox.showinfo("Success", f"Map exported successfully!\n\n{filename}")
            except Exception as e:
                self.log(f"Error exporting map: {e}", 'error')
                messagebox.showerror("Error", f"Failed to export map:\n{e}")

    def generate_pdf_report(self):
        """Generate professional PDF report with black border."""
        if not REPORTLAB_AVAILABLE:
            messagebox.showerror("Error", "ReportLab not installed.\nRun: pip install reportlab")
            return

        if not self.current_results:
            messagebox.showwarning("Warning", "No results available. Run simulation first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            self.log("Generating PDF report...", 'info')

            gdf = self.current_results['gdf']
            config = self.current_results['config']
            output_dir = Path(config['OUTPUT_DIR'])

            # Custom page template with black border
            class BorderedDocTemplate(BaseDocTemplate):
                def __init__(self, filename, **kwargs):
                    BaseDocTemplate.__init__(self, filename, **kwargs)

                def handle_pageBegin(self):
                    BaseDocTemplate.handle_pageBegin(self)
                    # Draw black border around page
                    self.canv.setStrokeColor(rl_colors.black)
                    self.canv.setLineWidth(2)
                    margin = 0.25 * inch
                    self.canv.rect(margin, margin,
                                   letter[0] - 2*margin,
                                   letter[1] - 2*margin)

            # Create PDF with bordered template
            doc = BorderedDocTemplate(filename, pagesize=letter,
                                     leftMargin=0.75*inch, rightMargin=0.75*inch,
                                     topMargin=0.75*inch, bottomMargin=0.75*inch)

            # Define frame and page template
            frame = Frame(doc.leftMargin, doc.bottomMargin,
                         doc.width, doc.height, id='normal')
            template = PageTemplate(id='bordered', frames=frame,
                                   onPage=lambda canvas, doc: None)
            doc.addPageTemplates([template])

            story = []
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=rl_colors.HexColor(ModernStyle.PRIMARY),
                spaceAfter=12,
                alignment=TA_CENTER
            )

            heading_style = ParagraphStyle(
                'Heading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=rl_colors.HexColor(ModernStyle.SECONDARY),
                spaceAfter=8,
                spaceBefore=8
            )

            # ========== PAGE 1 ==========
            # Title
            story.append(Paragraph("MINING CLAIM REDISTRIBUTION ANALYSIS", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}",
                                 ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER)))
            story.append(Spacer(1, 0.2*inch))

            # Summary Statistics Table
            story.append(Paragraph("SUMMARY STATISTICS", heading_style))

            # Calculate credit statistics
            # Calculate credit statistics from redistribution log
            import pandas as pd

            # Get total credits available and used from full_redistribution_log
            full_log_path = output_dir / 'full_redistribution_log.csv'
            total_credits_available = 0
            credits_used_renewals = 0
            credits_redistributed = 0

            if full_log_path.exists():
                try:
                    log_df = pd.read_csv(full_log_path)

                    # Helper function to parse credit values
                    def parse_credits(exp_str):
                        """Parse credits from format like '2029/04/09 (12946.51 $)'"""
                        if pd.isna(exp_str) or exp_str == '':
                            return 0
                        total = 0
                        for part in str(exp_str).split(';'):
                            if '(' in part and '$' in part:
                                try:
                                    credit_str = part.split('(')[1].split('$')[0].strip()
                                    total += float(credit_str)
                                except:
                                    pass
                        return total

                    # Calculate total credits available from original_expirations
                    log_df['original_credits'] = log_df['original_expirations'].apply(parse_credits)
                    total_credits_available = log_df['original_credits'].sum()

                    # Calculate credits used for renewals
                    renewal_rows = log_df[log_df['action_type'] == 'renewal']
                    if not renewal_rows.empty and 'renewal_amount' in renewal_rows.columns:
                        credits_used_renewals = renewal_rows['renewal_amount'].sum()

                    # Calculate credits redistributed
                    redistribution_rows = log_df[log_df['action_type'] == 'redistribution']
                    if not redistribution_rows.empty and 'credits_pulled' in redistribution_rows.columns:
                        credits_redistributed = redistribution_rows['credits_pulled'].sum()

                except Exception as e:
                    self.log(f"Warning: Could not calculate credit statistics: {e}", 'warning')

            credits_unused = total_credits_available - credits_used_renewals - credits_redistributed

            summary_data = [
                ['Total Claims', 'Avg Years Life', 'Total Credits Available', 'Credits Unused'],
                [str(len(gdf)), f"{gdf['years_of_life'].mean():.1f}",
                 f"${total_credits_available:,.0f}", f"${credits_unused:,.0f}"],
                ['Credits for Renewals', 'Credits Redistributed', '% Unused', '% Renewals'],
                [f"${credits_used_renewals:,.0f}", f"${credits_redistributed:,.0f}",
                 f"{(credits_unused/total_credits_available*100):.1f}%" if total_credits_available > 0 else "0%",
                 f"{(credits_used_renewals/total_credits_available*100):.1f}%" if total_credits_available > 0 else "0%"]
            ]

            summary_table = Table(summary_data, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(ModernStyle.SECONDARY)),
                ('BACKGROUND', (0, 2), (-1, 2), rl_colors.HexColor(ModernStyle.SECONDARY)),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                ('TEXTCOLOR', (0, 2), (-1, 2), rl_colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, 1), rl_colors.HexColor(ModernStyle.LIGHT)),
                ('BACKGROUND', (0, 3), (-1, 3), rl_colors.HexColor(ModernStyle.LIGHT)),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor(ModernStyle.DARK))
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.15*inch))

            # Projected Lapse Years Map (larger)
            story.append(Paragraph("PROJECTED LAPSE YEARS MAP", heading_style))
            map_file = output_dir / 'gsm_claims_lapse_years_map.png'
            if map_file.exists():
                from PIL import Image as PILImage
                pil_img = PILImage.open(str(map_file))
                img_width, img_height = pil_img.size
                aspect = img_height / img_width

                # Larger map for page 1
                max_width = 7.0*inch
                map_height = max_width * aspect

                if map_height > 5.0*inch:
                    map_height = 5.0*inch
                    max_width = map_height / aspect

                img = RLImage(str(map_file), width=max_width, height=map_height)
                story.append(img)
            story.append(Spacer(1, 0.15*inch))

            # Generate Charts Programmatically
            import seaborn as sns
            import matplotlib.pyplot as plt
            from io import BytesIO

            # Chart 1: Claims by Project and Year (on Page 1)
            story.append(Paragraph("NUMBER OF LAPSED CLAIMS BY PROJECT AND YEAR", heading_style))

            fig1, ax1 = plt.subplots(figsize=(10, 4))
            if 'life_category' not in gdf.columns:
                gdf['life_category'] = gdf['final_expiry_date'].apply(lambda x: str(x.year))

            order = sorted(gdf['life_category'].unique())
            sns.countplot(data=gdf, x='life_category', hue='project', order=order, ax=ax1, palette='tab10')

            ax1.set_xlabel('Lapse Year', fontweight='bold', fontsize=11)
            ax1.set_ylabel('Number of Lapsed Claims', fontweight='bold', fontsize=11)
            ax1.legend(title='Project', fontsize=9, loc='upper left')
            ax1.grid(axis='y', alpha=0.2, linestyle='--')
            ax1.tick_params(axis='x', rotation=45, labelsize=9)
            fig1.tight_layout()

            # Save to bytes and add to PDF
            buf1 = BytesIO()
            fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
            buf1.seek(0)
            plt.close(fig1)

            img1 = RLImage(buf1, width=7.0*inch, height=2.8*inch)
            story.append(img1)

            # Page break after first chart to start page 2
            story.append(PageBreak())

            # ========== PAGE 2 ==========
            # Chart 2: Average Claim Life by Project
            story.append(Paragraph("AVERAGE CLAIM LIFE BY PROJECT", heading_style))

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            project_stats = gdf.groupby('project')['years_of_life'].mean().sort_values()

            colors_list = ['#3498DB', '#16A085', '#27AE60', '#F39C12', '#2C3E50', '#E74C3C']
            bars = ax2.barh(project_stats.index, project_stats.values,
                          color=colors_list[:len(project_stats)], edgecolor='#34495E', linewidth=1.5)

            ax2.set_xlabel('Average Years of Life', fontweight='bold', fontsize=11)
            ax2.set_ylabel('Project', fontweight='bold', fontsize=11)
            ax2.grid(axis='x', alpha=0.2, linestyle='--')

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2, f' {width:.1f}',
                       ha='left', va='center', fontweight='bold', fontsize=9)

            fig2.tight_layout()

            # Save to bytes and add to PDF
            buf2 = BytesIO()
            fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
            buf2.seek(0)
            plt.close(fig2)

            img2 = RLImage(buf2, width=7.0*inch, height=2.8*inch)
            story.append(img2)
            story.append(Spacer(1, 0.2*inch))

            # Pivot Table (Years as rows, Projects as columns)
            story.append(Paragraph("REQUIRED SPENDING BY YEAR AND PROJECT", heading_style))

            gdf['lapse_year'] = gdf['final_expiry_date'].apply(lambda x: x.year)
            pivot_data = gdf.groupby(['lapse_year', 'project']).size().reset_index(name='claims_count')
            RENEWAL_COST_PER_CLAIM = 2500  # Actual cost per claim renewal
            pivot_data['required_spending'] = pivot_data['claims_count'] * RENEWAL_COST_PER_CLAIM

            # Years as rows, projects as columns
            pivot = pivot_data.pivot(index='lapse_year', columns='project', values='required_spending')
            pivot = pivot.fillna(0)

            # Get project list and limit to first 10 years
            projects = list(pivot.columns)
            years = sorted([int(y) for y in pivot.index])[:10]

            # Create table with years as rows
            pivot_table_data = [['Year'] + projects + ['TOTAL']]

            for year in years:
                row = [str(year)]
                year_total = 0
                for project in projects:
                    val = pivot.loc[year, project] if year in pivot.index and project in pivot.columns else 0
                    row.append(f"${val:,.0f}")
                    year_total += val
                row.append(f"${year_total:,.0f}")
                pivot_table_data.append(row)

            # Add totals row (total per project)
            totals_row = ['TOTAL']
            grand_total = 0
            for project in projects:
                if project in pivot.columns:
                    project_total = pivot[project].sum()
                    totals_row.append(f"${project_total:,.0f}")
                    grand_total += project_total
                else:
                    totals_row.append("$0")
            totals_row.append(f"${grand_total:,.0f}")
            pivot_table_data.append(totals_row)

            col_widths = [0.7*inch] + [1.0*inch] * len(projects) + [1.0*inch]
            pivot_table = Table(pivot_table_data, colWidths=col_widths)
            pivot_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(ModernStyle.SECONDARY)),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BACKGROUND', (0, 1), (-1, -2), rl_colors.HexColor(ModernStyle.LIGHT)),
                ('BACKGROUND', (0, -1), (-1, -1), rl_colors.HexColor('#E0E0E0')),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor(ModernStyle.DARK))
            ]))
            story.append(pivot_table)
            story.append(Spacer(1, 0.15*inch))

            # Upcoming Concerns
            story.append(Paragraph("UPCOMING CONCERNS (1-2 YEARS)", heading_style))
            current_year = date.today().year
            near_term = gdf[gdf['final_expiry_date'].dt.year <= current_year + 2]

            if len(near_term) > 0:
                concern_text = f"""<b>{len(near_term)} claims</b> expire within {current_year}-{current_year+2}. Projects at risk: """
                proj_list = []
                for proj in near_term['project'].unique():
                    proj_claims = near_term[near_term['project'] == proj]
                    proj_list.append(f"{proj} ({len(proj_claims)})")
                concern_text += ", ".join(proj_list) + "."
                story.append(Paragraph(concern_text, styles['Normal']))
            else:
                story.append(Paragraph("No critical concerns for next 1-2 years.", styles['Normal']))

            # Build PDF
            doc.build(story)

            self.log(f"PDF report generated: {filename}", 'success')

            # Store the PDF path for preview
            self.last_generated_pdf = filename

            # Show preview in Report tab
            self.show_pdf_preview(filename)

            messagebox.showinfo("Success", f"PDF report generated successfully!\n\n{filename}")

            # Ask if user wants to open it
            if messagebox.askyesno("Open Report", "Would you like to open the PDF report now?"):
                webbrowser.open(filename)

        except Exception as e:
            self.log(f"Error generating PDF: {e}", 'error')
            messagebox.showerror("Error", f"Failed to generate PDF:\n{e}")

    def show_pdf_preview(self, pdf_path=None):
        """Show PDF report preview - tries to render actual PDF pages or shows summary."""
        if not PIL_AVAILABLE or not hasattr(self, 'report_canvas'):
            return

        if not self.current_results:
            return

        try:
            # Clear existing preview
            self.report_canvas.delete("all")

            # If no path provided, use last generated PDF if available
            if not pdf_path and hasattr(self, 'last_generated_pdf'):
                pdf_path = self.last_generated_pdf

            # Try to use pdf2image if available and pdf_path provided
            # Note: Requires poppler to be installed on system
            if pdf_path and Path(pdf_path).exists():
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(pdf_path, dpi=100)

                    # Create a combined image showing both pages side by side
                    if len(images) >= 2:
                        page1, page2 = images[0], images[1]

                        # Resize pages to fit
                        max_height = 1000
                        page1.thumbnail((int(page1.width * (max_height / page1.height)), max_height), Image.Resampling.LANCZOS)
                        page2.thumbnail((int(page2.width * (max_height / page2.height)), max_height), Image.Resampling.LANCZOS)

                        # Combine side by side
                        total_width = page1.width + page2.width + 20
                        combined = Image.new('RGB', (total_width, max_height), color='white')
                        combined.paste(page1, (0, 0))
                        combined.paste(page2, (page1.width + 20, 0))

                        photo = ImageTk.PhotoImage(combined)
                        self.report_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                        self.report_preview_image = photo
                        self.report_canvas.config(scrollregion=self.report_canvas.bbox(tk.ALL))
                        self.log("PDF preview loaded (2 pages)", 'success')
                        return

                except ImportError:
                    self.log("pdf2image not available - using text preview (install poppler for PDF rendering)", 'info')
                except Exception as e:
                    # Likely poppler not installed or PDF conversion failed
                    if "poppler" in str(e).lower() or "unable to get page count" in str(e).lower():
                        self.log("Poppler not installed - using text preview (PDF file still generated successfully)", 'info')
                    else:
                        self.log(f"Could not render PDF preview: {e} - using text preview", 'info')

            # Fallback: Show simple message
            from PIL import Image, ImageDraw, ImageFont

            # Create simple preview message
            preview_img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(preview_img)

            try:
                title_font = ImageFont.truetype("arial.ttf", 20)
                body_font = ImageFont.truetype("arial.ttf", 14)
            except:
                title_font = ImageFont.load_default()
                body_font = ImageFont.load_default()

            # Draw message
            pdf_exists = pdf_path and Path(pdf_path).exists()
            message_lines = [
                "PDF Report Summary",
                "",
                "2-Page Report Includes:",
                "• Summary Statistics (8 metrics)",
                "• Projected Lapse Years Map",
                "• Claims by Project and Year Chart",
                "• Average Claim Life by Project Chart",
                "• Required Spending Pivot Table",
                "• Upcoming Concerns",
                "",
                f"Status: {'PDF Generated ✓' if pdf_exists else 'Not Yet Generated'}",
                "",
                "Note: Install poppler-utils for full PDF preview",
                "Text preview shown (PDF export works fine)"
            ]

            y = 150
            for line in message_lines:
                if line == "PDF Report Ready":
                    draw.text((400, y), line, fill='#2C3E50', font=title_font, anchor='mt')
                    y += 40
                else:
                    draw.text((400, y), line, fill='black', font=body_font, anchor='mt')
                    y += 30

            photo = ImageTk.PhotoImage(preview_img)
            self.report_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.report_preview_image = photo
            self.report_canvas.config(scrollregion=self.report_canvas.bbox(tk.ALL))

        except Exception as e:
            self.log(f"Could not display PDF preview: {e}", 'warning')
            import traceback
            self.log(traceback.format_exc(), 'error')

    def stop_simulation(self):
        self.is_running = False
        self.log("\nStop requested...", 'warning')

    def simulation_finished(self):
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready", foreground=ModernStyle.SUCCESS)

    def log(self, message, tag=''):
        self.log_queue.put((message, tag))

    def check_log_queue(self):
        try:
            while True:
                message, tag = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + '\n', tag)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.check_log_queue)

    def check_progress_queue(self):
        """Check for progress updates from background thread."""
        try:
            while True:
                msg_type, step, detail, percent = self.progress_queue.get_nowait()
                if msg_type == 'progress':
                    # Update progress bar if it exists
                    pass  # Progress updates handled via log messages
        except queue.Empty:
            pass
        self.root.after(100, self.check_progress_queue)

    def open_output_folder(self):
        """Open output folder in file explorer."""
        output_dir = self.config['OUTPUT_DIR'].get()
        if output_dir and Path(output_dir).exists():
            webbrowser.open(output_dir)
            self.log(f"Opened output folder: {output_dir}", 'info')
        else:
            messagebox.showwarning("Warning", "Output folder not set or does not exist.")

    def clear_log(self):
        """Clear the log text widget."""
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = ClaimRedistributionApp(root)
    root.mainloop()
