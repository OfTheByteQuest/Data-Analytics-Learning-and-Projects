"""Utility functions for stock analysis."""

from functools import wraps
import re

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException  # Import TimeoutException
from selenium.webdriver.common.keys import Keys  # For Keys.ESCAPE
import time


def scrape_bitcoin_data(start, end):
    # Configure Selenium to use headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(
        "--window-size=1920,1080"
    )  # Specifying window size can sometimes help in headless mode
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    )

    # Initialize the WebDriver
    # Ensure chromedriver is in your PATH or provide the executable_path
    # For example: driver = webdriver.Chrome(executable_path='/path/to/chromedriver', options=options)
    driver = webdriver.Chrome(options=options)

    try:
        # Navigate to the historical data page
        url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start={}&end={}".format(
            start, end
        )
        print(f"Navigating to URL: {url}")
        driver.get(url)

        # Increased timeout as CoinMarketCap can be slow to load dynamic content.
        timeout = 30  # You can increase this if needed, e.g., to 45 or 60

        # Attempt to send ESCAPE key to close any initial overlays
        try:
            print("Attempting to send ESCAPE key to close overlays...")
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            ).send_keys(Keys.ESCAPE)
            time.sleep(1)  # Give a moment for the action to take effect
            print("ESCAPE key sent.")
        except Exception as e:
            print(f"Could not send ESCAPE key or body not found quickly: {e}")

        # Attempt to accept cookies/close pop-ups that might obscure the table
        try:
            print("Attempting to find and click cookie consent button...")
            cookie_button_selectors = [
                (By.ID, "onetrust-accept-btn-handler"),
                (By.XPATH, "//button[.//p[contains(text(), 'Accept All Cookies')]]"),
                (By.XPATH, "//button[contains(text(), 'Accept All')]"),
                (By.XPATH, "//button[contains(text(), 'Accept')]"),
                (By.XPATH, "//button[contains(text(), 'Agree')]"),
                (
                    By.XPATH,
                    "//button[contains(@class, 'accept') or contains(@id, 'accept') or contains(@data-test, 'accept')]",
                ),
                (
                    By.XPATH,
                    "//div[div[contains(., 'cookies')]]//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
                ),
            ]

            cookie_button_found_and_clicked = False
            for by_type, selector_value in cookie_button_selectors:
                try:
                    cookie_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((by_type, selector_value))
                    )
                    cookie_button.click()
                    print(
                        f"Clicked cookie consent button using: {by_type} '{selector_value}'"
                    )
                    time.sleep(3)
                    cookie_button_found_and_clicked = True
                    break
                except TimeoutException:
                    print(
                        f"Cookie button not found or not clickable with: {by_type} '{selector_value}'"
                    )

            if not cookie_button_found_and_clicked:
                print("No common cookie consent button was found or clicked.")

        except Exception as e:
            print(f"An error occurred while trying to handle cookie consent: {e}")

        # Attempt to find and click a generic close button (e.g., 'X') for other popups
        try:
            print("Attempting to find and click a generic close button (e.g., 'X')...")
            generic_close_selectors = [
                (By.XPATH, "//button[@aria-label='Close']"),
                (By.XPATH, "//button[@aria-label='close']"),
                (
                    By.XPATH,
                    "//button[contains(@class, 'close') or contains(@id, 'close') or contains(@title, 'Close') or contains(@data-dismiss, 'modal')]",
                ),
                (
                    By.XPATH,
                    "//button/span[normalize-space(text())='×']",
                ),  # '×' (multiplication sign) common for X
                (By.XPATH, "//button[normalize-space(text())='×']"),
                (By.XPATH, "//i[contains(@class, 'close')]"),
            ]
            close_button_clicked_generic = False
            for by_type, selector_value in generic_close_selectors:
                try:
                    # Look for multiple instances in case there are several such buttons
                    buttons = driver.find_elements(by_type, selector_value)
                    for close_button in buttons:
                        if close_button.is_displayed() and close_button.is_enabled():
                            WebDriverWait(driver, 2).until(
                                EC.element_to_be_clickable(close_button)
                            )
                            close_button.click()
                            print(
                                f"Clicked generic close button using: {by_type} '{selector_value}'"
                            )
                            time.sleep(2)
                            close_button_clicked_generic = True
                            # break # or continue if multiple popups might exist
                except (
                    Exception
                ):  # Catch broader exceptions as elements might become stale
                    print(
                        f"Generic close button not found, not clickable, or error with: {by_type} '{selector_value}'"
                    )
            if not close_button_clicked_generic:
                print("No generic close button found or clicked.")
        except Exception as e:
            print(f"Error trying to click generic close button: {e}")

        # Scroll down to ensure the table is in view and potentially trigger lazy loading
        print("Scrolling down the page to trigger potential lazy loading...")
        try:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight / 4);"
            )  # Scroll 1/4
            time.sleep(1)
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight / 2);"
            )  # Scroll 1/2
            time.sleep(1)
            driver.execute_script(
                "window.scrollTo(0, 3 * document.body.scrollHeight / 4);"
            )  # Scroll 3/4
            time.sleep(1)
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )  # Scroll to bottom
            time.sleep(2)  # Give time for any lazy-loaded content to appear
            print("Scrolled page.")
        except Exception as e:
            print(f"Error during scrolling: {e}")

        # Refined waiting strategy for the table based on user-provided HTML
        # The div with class "sc-fabeca80-2 cntXkr" seems to be the direct parent of the table section
        # Or, "HistoricalData_container_qmdTH" is a higher-level container for the historical data section.
        # Let's try targeting the "cntXkr" class first as it's more specific to the table area.
        table_area_xpath = "//div[contains(@class, 'cntXkr')]"
        print(
            f"Waiting up to {timeout} seconds for the historical data table area: {table_area_xpath}"
        )

        try:
            # 1. Wait for the table's direct parent area to be visible
            table_area_container = WebDriverWait(driver, timeout).until(
                EC.visibility_of_element_located((By.XPATH, table_area_xpath))
            )
            print(
                "Table area container (e.g., div with class 'cntXkr') found and is visible."
            )

            # 2. Find the table element within this container
            #    The table itself has class 'cmc-table' based on the screenshot.
            table_element_relative_xpath = ".//table[contains(@class, 'cmc-table')]"
            table = WebDriverWait(
                table_area_container, timeout
            ).until(  # Search within table_area_container
                EC.visibility_of_element_located(
                    (By.XPATH, table_element_relative_xpath)
                )
            )
            print(
                "Table element (class 'cmc-table') found within its area and is visible."
            )

            # 3. Wait for the tbody of this specific table to be visible
            table_body_relative_xpath = ".//tbody"
            WebDriverWait(driver, timeout).until(
                lambda d: table.find_element(
                    By.XPATH, table_body_relative_xpath
                ).is_displayed()
            )
            print("Historical data table body (tbody) is visible.")

            # 'table' variable now holds the main table element

            # Extract headers
            headers = []
            header_elements = table.find_elements(By.XPATH, ".//thead//tr//th")
            for header_element in header_elements:
                headers.append(header_element.text.strip())
            print(f"Extracted headers: {headers}")

            # Extract rows data
            historical_data_list = []
            # It's good practice to re-locate rows within the tbody to avoid stale elements
            tbody_element = table.find_element(By.XPATH, table_body_relative_xpath)
            rows = tbody_element.find_elements(
                By.XPATH, ".//tr"
            )  # Get rows from the located tbody
            print(f"Found {len(rows)} data rows. Extracting data...")

            for row_idx, row in enumerate(rows):
                cols = row.find_elements(By.XPATH, ".//td")
                if not cols:
                    print(f"Skipping empty row {row_idx + 1}")
                    continue

                row_data = {}
                current_row_values = [col.text.strip() for col in cols]
                # Filter out '#' or '*' from headers if they are used for display only
                data_headers = [h for h in headers if h and h not in ("#", "*")]

                # Robust data mapping logic
                if data_headers and len(current_row_values) == len(data_headers):
                    # Ideal case: number of cells matches number of data headers
                    for i, header_name in enumerate(data_headers):
                        row_data[header_name] = current_row_values[i]
                elif (
                    data_headers
                    and headers
                    and headers[0] in ("#", "*")
                    and len(current_row_values) == len(data_headers) + 1
                ):
                    # Case: First HTML column is '#'/ '*', skip it for data mapping
                    for i, header_name in enumerate(data_headers):
                        row_data[header_name] = current_row_values[i + 1]
                elif (
                    len(current_row_values) >= 7
                ):  # Fallback if header mapping is complex
                    print(
                        f"Row {row_idx + 1}: Attempting fallback mapping for {len(current_row_values)} columns: {current_row_values}"
                    )
                    # Standard expected columns if direct mapping fails
                    expected_columns_data_map = [
                        "Date",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Market Cap",
                    ]
                    # Determine if there's an offset (e.g. if first actual column is a number/symbol not in expected_columns_data_map)
                    offset = 0
                    if len(current_row_values) > len(
                        expected_columns_data_map
                    ):  # If more cells than expected data points
                        # This simple offset might not be perfect if table structure is very dynamic
                        if headers and headers[0] in ("#", "*"):
                            offset = len(current_row_values) - len(
                                expected_columns_data_map
                            )

                    if (
                        len(current_row_values)
                        >= len(expected_columns_data_map) + offset
                    ):
                        for i, col_name in enumerate(expected_columns_data_map):
                            try:
                                row_data[col_name] = current_row_values[offset + i]
                            except IndexError:
                                row_data[col_name] = None  # Or some other placeholder
                                print(
                                    f"Row {row_idx + 1}: Data missing for {col_name} in fallback mapping."
                                )
                    else:
                        print(
                            f"Skipping row {row_idx + 1} due to column count mismatch in fallback. Headers: {headers}, Cols: {current_row_values}"
                        )
                        continue  # Skip this row
                else:
                    print(
                        f"Skipping row {row_idx + 1} due to insufficient columns for any mapping. Headers: {headers}, Cols: {current_row_values}"
                    )
                    continue  # Skip this row

                if (
                    row_data
                ):  # Only append if we successfully extracted some data for the row
                    historical_data_list.append(row_data)

            # Convert the data into a DataFrame
            if historical_data_list:
                # Determine columns for DataFrame: use data_headers if consistent, else infer
                df_columns = None
                if data_headers and all(
                    isinstance(item, dict) for item in historical_data_list
                ):
                    # Check if all dicts in historical_data_list contain all keys from data_headers
                    if all(
                        all(h in item for h in data_headers)
                        for item in historical_data_list
                    ):
                        df_columns = data_headers

                df = pd.DataFrame(
                    historical_data_list, columns=df_columns
                )  # df_columns can be None
                print("\nSuccessfully extracted data. DataFrame head:")
                print(df.head())
            else:
                print(
                    "No data was extracted. The table might be empty or the structure might have changed significantly."
                )

        except TimeoutException as e_table_wait:
            print(f"TimeoutException during table loading sequence: {e_table_wait}")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_file = f"coinmarketcap_timeout_screenshot_{timestamp}.png"
            try:
                driver.save_screenshot(screenshot_file)
                print(f"Screenshot saved to {screenshot_file} for debugging.")
            except Exception as e_screenshot:
                print(f"Failed to save screenshot: {e_screenshot}")
            print("Continuing to finally block after timeout.")

    finally:
        # Clean up and close the browser
        print("Closing the browser.")
        driver.quit()
        return df


def _sanitize_label(label):
    """
    Clean up a label by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters:
        - label: The test you want to fix.

    Returns: The sanitized label.
    """
    return re.sub(r"[^\w\s]", "", label).lower().replace(" ", "_")


def label_sanitizer(method):
    """
    Decorator around a method that returns a dataframe to
    clean up all the labels in said dataframe (column names and index
    name) by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters:
        - method: The method to wrap.

    Returns: A decorated method or function.
    """

    @wraps(method)  # keeps the docstring of the data method for help.
    def method_wrapper(self, *args, **kwargs):
        df = method(self, *args, **kwargs)

        # fix the column names
        df.columns = [_sanitize_label(col) for col in df.columns]

        # fix the index name
        if df.index.name:
            df.index.rename(_sanitize_label(df.index.name), inplace=True)
        else:
            df.index.name = "index"

        return df

    return method_wrapper


def group_stocks(mapping):
    """
    Create a new dataframe with many assets and a new column
    indicating the asset that row's data belongs to.

    Parameters:
        - mapping A key-value mapping of the form:
            {asset_name:asset_df}

    Returns: A new Pandas dataframe.
    """
    grouped_df = pd.DataFrame()

    for stock, stock_data in mapping.items():
        df = stock_data.copy(deep=True)
        df["name"] = stock  # Add a new column with the values being stock name
        grouped_df = pd.concat([grouped_df, df])

    grouped_df.index = pd.to_datetime(grouped_df.index)

    return grouped_df


def validate_df(columns, instance_method=True):
    """
    Decorator that raises a ValueError if the input isn't a pandas
    DataFrame or doesn't contain the proper columns. Note the
    DataFrame must be the first positional argument passed to this
    method.
    """

    def method_wrapper(method):
        @wraps(method)
        def validate_wrapper(self, *args, **kwargs):
            # functions and static methods don't pass self
            # so, self is the first positional argument in that case.

            df = (self, *args)[0 if not instance_method else 1]

            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")
            if columns.difference(df.columns):
                raise ValueError(
                    f"Input DataFrame must contain the following columns: {columns}"
                )
            return method(self, *args, **kwargs)

        return validate_wrapper

    return method_wrapper


@validate_df(columns={"name"}, instance_method=False)
def describe_group(data):
    """
    Run `describe` on the asset group created with `group_stocks()`.

    Parameters:
        - data: The group data resulting from `group_stocks()`.

    Returns: The transpose of the grouped description statistics.
    """
    return data.groupby("name").describe().transpose()


@validate_df(columns=set(), instance_method=False)
def make_portfolio(data, datae_column="date"):
    """
    Make a portfolio of assets from by grouping by date and summing all
    columns.

    Parameters:
        - data: The data to make a portfolio from.
        - date_column: The column to group by. Defaults to 'date'.

    Returns: A new DataFrame with the portfolio data.

    Note: The caller is responsible for making sure the dates line up
    accross assets and handling when they don't.
    """
    return data.reset_index().groupby("date").sum()
