# Handles user input, identifying the ticker column, and calls the forecasting engine.




############################################################################### Imports ####################################################################

import os
import sys
import json
import pandas as pd

from FORECASTING_ENGINE import portfolio_analyser

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog, QStackedWidget, QComboBox,
    QMessageBox, QHBoxLayout, QDateEdit
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QDate

from datetime import datetime, time

import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

################################################# Functions for storing, accessing and modifying user settings #######################################################
def get_config_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.join(os.path.expanduser("~"), ".portfolio_tracker")
    else:
        base_dir = os.path.join(SCRIPT_DIR, "settings")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "config.json")


CONFIG_PATH = get_config_path()


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"button_colour": "blue", "light_or_dark_mode": "Dark", "font_size": 14}



def save_config(config, parent=None):
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
    except Exception:
        # Use parent so the dialog is correctly attached to the app window (or None if called outside GUI context)
        QMessageBox.information(parent, "ERROR", "Applied user settings cannot be saved.")

############################################ Stylesheet creation and application ##################################################################################
def apply_stylesheet(app, light_or_dark_mode, button_colour="blue", button_colour_only=False):
    button_colours_list = {"blue": "#0B78D0", "dark-blue": "#1E3C72", "green": "#2D8656"}
    button_colour = button_colours_list.get(button_colour, "#0B78D0")
    if light_or_dark_mode == "Dark":

        hover = "#555"
        base_bg = "#2e2e2e"
        base_fg = "white"

    else:
        hover = "#ccc"
        base_bg = "white"
        base_fg = "black"

    style = ""
    if not button_colour_only:
        style += f"""QWidget {{ background-color: {base_bg}; color: {base_fg}; }}QComboBox {{ background-color: {base_bg}; color: {base_fg}; padding: 4px;border: 1px solid #888; border-radius: 4px; }}"""
    if light_or_dark_mode == "Dark":
        style += f"""QPushButton {{ background-color: {button_colour}; color: white; padding: 6px;border: none; border-radius: 4px; }}QPushButton:hover {{ background-color: {hover}; color: {'white'}; }}"""
    else:
        style += f"""QPushButton {{ background-color: {button_colour}; color: white; padding: 6px;border: none; border-radius: 4px; }}QPushButton:hover {{ background-color: {hover}; color: {'black'}; }}"""
    app.setStyleSheet(style)


class MainWindow(QMainWindow):

    #################################################################################################################################################################
    def __init__(self):  # Creating Main Window
        super().__init__()

        self.default_font = QApplication.instance().font()

        self.config = load_config()  # load user settings for button colours and text size, if this fails a default stylesheet will be applied instead.
        self.config.setdefault("font_size", 14)

        self.labels_and_widgets = []

        self.setWindowTitle("Portfolio Analyser")
        self.setGeometry(100, 100, 500, 320)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.main_page = QWidget()
        self.settings_page = QWidget()
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.settings_page)

        apply_stylesheet(app, self.config.get("light_or_dark_mode", "Dark"), self.config.get("button_colour", "blue"))

        self.rebuild_main_page()  # Creates main page UI
        self.rebuild_settings_page()  # Creates settings page UI
        self.apply_font_size(self.config.get("font_size", 14))  # Applies size to text on the pages, including in buttons.

    ################################################################################################################################################################

    def apply_font_size(self, text_size):
        font = QFont(self.default_font)
        font.setPointSize(text_size)
        self.setFont(font)

        for widget in self.labels_and_widgets:
            widget.setFont(font)
            if isinstance(widget, QDateEdit):
                widget.lineEdit().setFont(font)
                widget.calendarWidget().setFont(font)

        if hasattr(self, "main_buttons") and self.main_buttons:
            self.update_main_buttons_size()


    def load_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilters(["CSV files (*.csv)"])
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                filepath = selected_files[0]
                try:
                    dataframe = pd.read_csv(filepath)
                    tickers_list = []

                    # Finds the column that contains ticker symbols
                    ticker_col = None
                    for col in dataframe.columns:
                        sample_values = dataframe[col].dropna().astype(str).head(10).tolist()
                        if all(val.isupper() and len(val) <= 7 for val in sample_values):
                            ticker_col = col
                            break

                    if ticker_col:
                        tickers = dataframe[ticker_col]  # Identifies ticker
                    else:
                        QMessageBox.critical(self, "ERROR", "No valid ticker column found. Please check the file you uploaded and modify it or select new file.")
                        return

                    tickers_list = tickers.unique().tolist()  # Makes list of tickers in the csv portfolio, once the ticker column has been identified.

                    returns_file_path, invalidtickerslist = self.call_portfolio_analyser(tickers_list)  # Calls function for forecasting future returns
                    if returns_file_path == "Error 1":
                        QMessageBox.critical(self, "ERROR", "Could not fetch historical adjusted close price data. Please check your internet connection and try again.")
                    elif returns_file_path == "Error 2":
                        QMessageBox.critical(self, "ERROR", "No valid tickers detected, please check your internet connection, that all tickers are valid, and try again later.")
                    elif returns_file_path == "Error 3":
                        QMessageBox.critical(self, "ERROR", "Please load the file again.")
                    elif returns_file_path == "Error 4":
                        QMessageBox.critical(self, "Missing Date", "Please click the 'Apply Time Period' button before loading a portfolio.")
                    else:
                        if len(invalidtickerslist) == 0:
                            QMessageBox.information(self, "File Loaded", f"Forecast produced successfully. Input Portfolio CSV:\n{filepath}\n Results saved in: {returns_file_path}")
                        else:
                            QMessageBox.information(self, "File Loaded", f"Forecast produced successfully. Input Portfolio CSV:\n{filepath}\n Results saved in: {returns_file_path}\n INVALID TICKERS: \n {invalidtickerslist}")

                except AttributeError:
                    QMessageBox.critical(self, "Date not saved", "Please click the 'Apply Time Period' button then try again.")
                except Exception as e:
                    QMessageBox.critical(self, "Error Loading File", f"Failed to load CSV file:\n{e}")

    def show_help(self):
        QMessageBox.information(self, "Ticker Format Help and Quick Tips",
        "Ensure correct ticker formatting before uploading your portfolio file.\n"
        "IMPORTANT: Tickers with exchange suffixes (e.g. .L, .T) may work but are not guaranteed to be supported.\n\n"
        "Region      Format     Example\n"
        "----------- ---------  ---------\n"
        "USA         No suffix  AAPL\n"
        "Japan       .T         7203.T\n"
        "UK          .L         VOD.L\n"
        "Canada      .TO        RY.TO\n"
        "Germany     .DE        BMW.DE\n"
        "France      .PA        AIR.PA\n"
        "Hong Kong   .HK        0700.HK\n\n Note that each ticker should only have one instance in the portfolio. So If AAPL is in your portfolio it should not be there more than once.\n\n All tickers must be written in UPPERCASE (e.g. AAPL, NOT aapl)." \
        "\n\n Quick Pointers\n\n Lookback period start date should never be beyond the present day or older than 1st January 2000.\n It is recommended to set a lookback period start date of 2 years ago or earlier for forecasting.\n If backtesting is also desired, lookback period start date should be 3 years and 1 week before the present day. ")

    def rebuild_main_page(self):
        main_page_layout = QVBoxLayout()
        main_page_layout.addStretch(1)

        # Set main label
        self.main_label = QLabel("Portfolio Analyser")
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_label.setStyleSheet("font-weight: bold; font-size: 20pt;")
        main_page_layout.addWidget(self.main_label)
        self.labels_and_widgets.append(self.main_label)

        main_page_layout.addSpacing(20)

        # Main buttons
        self.main_buttons = []
        button_data = [
            ("Upload Portfolio File (must be CSV)", self.load_file),
            ("Settings", self.go_to_settings),
            ("Ticker Format Help and Quick Tips", self.show_help),
            ("Exit", self.close),
        ]

        main_page_buttons = []
        for button_text, button_function in button_data:
            main_page_button = QPushButton(button_text)
            main_page_button.clicked.connect(button_function)
            main_page_button.setSizePolicy(main_page_button.sizePolicy().horizontalPolicy(), main_page_button.sizePolicy().verticalPolicy())
            self.labels_and_widgets.append(main_page_button)
            main_page_buttons.append(main_page_button)

        # Ensure all buttons are the same size
        max_width = 0
        max_height = 0
        for main_page_button in main_page_buttons:
            fm = main_page_button.fontMetrics()
            text_rect = fm.boundingRect(main_page_button.text())
            width = text_rect.width() + 40
            height = text_rect.height() + 20
            max_width = max(max_width, width)
            max_height = max(max_height, height)

        for main_page_button in main_page_buttons:
            main_page_button.setFixedSize(max_width, max_height)

        # Add buttons to main_page_layout
        for main_page_button in main_page_buttons:
            height = QHBoxLayout()
            height.addStretch()
            height.addWidget(main_page_button)
            height.addStretch()
            main_page_layout.addLayout(height)
            self.main_buttons.append(main_page_button)

            # Insert date widget, information on recommended models, save button for date immediately after the Upload Portfolio File (must be CSV) button
            if main_page_button.text() == "Upload Portfolio File (must be CSV)":

                # Initialise variable for storing lookback period start date input
                self.lookback_period_date = None

                # Label above DateEdit widget
                self.upload_helper_text_label = QLabel("Apply lookback period start date,\n then upload file to run forecast")
                self.upload_helper_text_label.setAlignment(Qt.AlignCenter)
                self.upload_helper_text_label.setStyleSheet("font-weight: bold;")
                self.upload_helper_text_label.setFixedWidth(max_width)

                self.upload_helper_text_label.setWordWrap(True)
                self.labels_and_widgets.append(self.upload_helper_text_label)
                helper_layout = QHBoxLayout()
                helper_layout.addStretch()
                helper_layout.addWidget(self.upload_helper_text_label)
                helper_layout.addStretch()
                main_page_layout.addLayout(helper_layout)

                self.qdate_lookback_period_label = QLabel("Lookback period start date\n (Historical data window start date)")
                self.qdate_lookback_period_label.setAlignment(Qt.AlignCenter)
                self.qdate_lookback_period_label.setStyleSheet("font-weight: bold;")
                self.qdate_lookback_period_label.setFixedWidth(max_width)
                self.qdate_lookback_period_label.setWordWrap(True)
                self.labels_and_widgets.append(self.qdate_lookback_period_label)

                date_label_layout = QHBoxLayout()
                date_label_layout.addStretch()
                date_label_layout.addWidget(self.qdate_lookback_period_label)
                date_label_layout.addStretch()
                main_page_layout.addLayout(date_label_layout)

                self.qdate_lookback_period = QDateEdit()
                self.qdate_lookback_period.setCalendarPopup(True)
                self.qdate_lookback_period.setDate(QDate.currentDate())
                self.qdate_lookback_period.setToolTip("Select date where lookback period start date begins")
                self.labels_and_widgets.append(self.qdate_lookback_period)
                date_widget_layout = QHBoxLayout()
                date_widget_layout.addStretch()
                date_widget_layout.addWidget(self.qdate_lookback_period)
                date_widget_layout.addStretch()
                main_page_layout.addLayout(date_widget_layout)

                # Save Button
                self.save_button = QPushButton("Apply Time Period")
                self.save_button.clicked.connect(self.save_date)
                self.labels_and_widgets.append(self.save_button)
                self.main_buttons.append(self.save_button)
                save_button_layout = QHBoxLayout()
                save_button_layout.addStretch()
                save_button_layout.addWidget(self.save_button)
                save_button_layout.addStretch()
                main_page_layout.addLayout(save_button_layout)

        main_page_layout.addSpacing(10)
        main_page_layout.addStretch(3)
        self.main_page.setLayout(main_page_layout)


    def update_main_buttons_size(self):
        if not hasattr(self, "main_buttons") or not self.main_buttons:  # Redundant????????
            return

        max_width = 0
        max_height = 0
        for main_page_button in self.main_buttons:
            fm = main_page_button.fontMetrics()
            text_rect = fm.boundingRect(main_page_button.text())
            width = text_rect.width() + 40
            height = text_rect.height() + 20
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height

        for main_page_button in self.main_buttons:
            main_page_button.setFixedSize(max_width, max_height)
        
        if hasattr(self, 'qdate_lookback_period'):
            self.qdate_lookback_period.setFixedSize(max_width, max_height)

        if hasattr(self, 'upload_helper_text_label'):
            self.upload_helper_text_label.setFixedWidth(max_width)
        if hasattr(self, 'qdate_lookback_period_label'):
            self.qdate_lookback_period_label.setFixedWidth(max_width)
        

    def rebuild_settings_page(self):
        settings_page_layout = QVBoxLayout()
        settings_page_layout.addSpacing(10)

        # Set settings label
        self.settings_label = QLabel("Settings")
        self.settings_label.setAlignment(Qt.AlignCenter)
        self.settings_label.setStyleSheet("font-weight: bold; font-size: 18pt;")
        settings_page_layout.addWidget(self.settings_label)
        self.labels_and_widgets.append(self.settings_label)

        settings_page_layout.addSpacing(20)

        # Toggle for setting light mode or dark mode
        self.toggle_button_widget = QPushButton(
            "Switch to Light" if self.config.get("light_or_dark_mode") == "Dark" else "Switch to Dark"
        )
        self.toggle_button_widget.clicked.connect(self.handle_toggle_button)
        settings_page_layout.addWidget(self.toggle_button_widget)
        self.labels_and_widgets.append(self.toggle_button_widget)

        # Label above QComboBox widget that sets button colour
        colour_label = QLabel("Select Button Colour:")
        colour_label.setAlignment(Qt.AlignCenter)
        colour_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        settings_page_layout.addWidget(colour_label)
        self.labels_and_widgets.append(colour_label)

        # QComboBox widget that sets button colour
        self.button_colour_combo = QComboBox()  # Try change to button colour and do the same in json file and stylesheet
        self.button_colour_combo.addItems(["blue", "dark-blue", "green"])
        self.button_colour_combo.setCurrentText(self.config.get("button_colour", "blue"))
        self.button_colour_combo.currentTextChanged.connect(self.change_button_colour)
        font = QFont()
        font.setPointSize(self.config.get("font_size", 14))
        self.button_colour_combo.setFont(font)

        settings_page_layout.addWidget(self.button_colour_combo)
        self.labels_and_widgets.append(self.button_colour_combo)

        # Label above QComboBox widget that sets text size
        size_label = QLabel("Select Text Size:")
        size_label.setAlignment(Qt.AlignCenter)
        size_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        settings_page_layout.addWidget(size_label)
        self.labels_and_widgets.append(size_label)

        # QComboBox widget that sets text size
        self.text_size_combobox = QComboBox()
        self.text_size_combobox.addItems(["Small", "Medium", "Large", "Extra Large"])
        mapping = {10: "Small", 14: "Medium", 18: "Large", 22: "Extra Large"}
        current_size = self.config.get("font_size", 14)
        self.text_size_combobox.setCurrentText(mapping.get(current_size, "Medium"))
        self.text_size_combobox.currentTextChanged.connect(self.change_font_size)
        settings_page_layout.addWidget(self.text_size_combobox)
        self.labels_and_widgets.append(self.text_size_combobox)

        # Button which applies the default stylesheet (essentially making the GUI have a default appearance)
        self.restore_defaults_button_widget = QPushButton("Restore Defaults")
        self.restore_defaults_button_widget.setToolTip("Reset to dark mode, default button colour, font, and size")
        self.restore_defaults_button_widget.clicked.connect(self.restore_defaults)
        settings_page_layout.addWidget(self.restore_defaults_button_widget)
        self.labels_and_widgets.append(self.restore_defaults_button_widget)

        # Button for returning to the main page, where user can upload their portfolio CSV file
        return_button_widget = QPushButton("Return to Main Menu")
        return_button_widget.clicked.connect(self.go_to_main)
        settings_page_layout.addWidget(return_button_widget)
        self.labels_and_widgets.append(return_button_widget)

        settings_page_layout.addStretch(2)
        self.settings_page.setLayout(settings_page_layout)

    def restore_defaults(self):
        self.config.update({"button_colour": "blue", "light_or_dark_mode": "Dark", "font_size": 14})
        save_config(self.config, self)

        self.button_colour_combo.setCurrentText("blue")
        self.text_size_combobox.setCurrentText("Medium")
        self.toggle_button_widget.setText("Switch to Light")

        apply_stylesheet(app, "Dark", "blue")
        self.apply_font_size(14)

    def toggle_light_or_dark_mode(self, button_widget):
        current = self.config.get("light_or_dark_mode", "Dark")
        if current == "Dark":
            new = "Light"
        else:
            new = "Dark"
        self.config["light_or_dark_mode"] = new
        save_config(self.config, self)

        if new == "Dark":
            button_widget.setText("Switch to Light")
        else:
            button_widget.setText("Switch to Dark")

        apply_stylesheet(app, new, self.config.get("button_colour", "blue"))
        self.apply_font_size(self.config.get("font_size", 14))

    def change_button_colour(self, button_colour):  # Also change this to set to light or dark mode if appearance can be consistently changed to light or dark in json and stylesheet
        self.config["button_colour"] = button_colour
        save_config(self.config, self)
        apply_stylesheet(app, self.config.get("light_or_dark_mode", "Dark"), button_colour)
        self.apply_font_size(self.config.get("font_size", 14))

    def change_font_size(self, size_text):
        size_map = {"Small": 10, "Medium": 14, "Large": 18, "Extra Large": 22}
        size = size_map.get(size_text, 14)
        self.config["font_size"] = size
        save_config(self.config, self)
        self.apply_font_size(size)

    def go_to_settings(self):
        self.stack.setCurrentWidget(self.settings_page)

    def go_to_main(self):
        self.stack.setCurrentWidget(self.main_page)

    def handle_toggle_button(self):
        self.toggle_light_or_dark_mode(self.toggle_button_widget)

    ######################################################## Functions for saving date inputs in DateEdit widgets ############################################

    def save_date(self):
        self.lookback_period_date = self.qdate_lookback_period.date()

        year = self.lookback_period_date.year()

        if year < 2000:
            QMessageBox.information(self, "Invalid Date", "Lookback period start date must not be older than 1st of January 2000.\nPlease select a new date.")
            return None

        candidate_date = (f"{year}-"f"{self.lookback_period_date.month():02d}-"f"{self.lookback_period_date.day():02d}")
        if candidate_date > str(datetime.fromtimestamp(time.time())):
            present_date = datetime.fromtimestamp(time.time())
            present_date_text = str(f"{present_date.year}-"f"{present_date.month:02d}-"f"{present_date.day:02d}")
            QMessageBox.information(self, "Invalid Date", f"Lookback period start date must not be later than the present date ({present_date_text}).\nPlease select a new date.")
            return None
        self.py_date = candidate_date
        return self.py_date

    def call_portfolio_analyser(self, tickers_list):
        if not hasattr(self, "py_date"):
            return "Error 4", []

        file_path, invalid_tickers = portfolio_analyser(
            self.py_date,
            tickers_list
        ) 
        return file_path, invalid_tickers



# Loading GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.update_main_buttons_size()
    sys.exit(app.exec())
