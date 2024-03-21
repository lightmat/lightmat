from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.common.exceptions import NoSuchElementException
from fake_useragent import UserAgent
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime as dt
import pandas as pd
import os
import random

# Product list page on amazon.de: Krimis & Thriller, Sprache: Deutsch, Letzte 30 Tage, Sortiert nach: Veröffentlichungsdatum absteigend
URL = 'https://www.amazon.de/s?i=stripbooks&bbn=287480&rh=n%3A287480%2Cp_n_feature_three_browse-bin%3A15425222031%2Cp_n_publication_date%3A1778535031&s=date-desc-rank&dc&qid=1702832247&rnid=1778534031&ref=sr_st_date-desc-rank&ds=v1%3A%2B7zptr3Mr7mrj7aNisnrLWA3U%2FRZ75hPRTOthrtXtMA.'

# Max number of books to scrape
MAX_BOOKS = 1200

# Max number of known books before stopping
MAX_KNOWN_BOOKS = 5

# Dateiname nach dem timestamp
FILE_NAME = "Krimis_und_Thriller"

# Max number of navigation fails before stopping
MAX_NAVIGATION_FAILS = 5

# Define all the needed columns
BOOK_COLS = {
    "general": { 
        "erfasst am": "erfasst am",
        "Titel": "Titel",
        "Reihe?": "Reihe?",
        "Autor": "Autor",
        "verfügbare Formate": "verfügbare Formate",
        "Übersetzer": "Übersetzer",
        "Anmerkung": "Anmerkung",
    },
    "Kindle": {
        "is_available": "Kindle eBoook",
        "link": "Link Kindle E-Book",
        "publisher": "Herausgeber E-Books",
        "asin": "ASIN",
        "isbn_10": "ISBN-10 eBook",
        "isbn_13": "ISBN-13 eBook",
        "price": "Preis Kindle eBook",
        "in_kindle_unlimited": "in Kindle Unlimited enthalten",
        "page_count": "Seitenzahl der Print-Ausgabe",
        "page_count_source": "ISBN/ASIN-Quelle für Seitenzahl",
        "edition": "Edition eBook (sofern gesonderte Angabe)",
        "release_date": "Erscheinungsdatum Kindle eBook",
    },
    "Taschenbuch": {
        "is_available": "Taschenbuch",
        "link": "Link Taschenbuch",
        "publisher": "Herausgeber Taschenbuch",
        "isbn_10": "ISBN-10 Taschenbuch",
        "isbn_13": "ISBN-13 Taschenbuch",
        "price": "Preis Taschenbuch",
        "page_count": "Seitenzahl Taschenbuch",
        "edition": "Edition Taschenbuch (sofern gesonderte Angabe)",
        "release_date": "Erscheinungsdatum Taschenbuch",
    },
    "Paperback": {
        "is_available": "Paperback",
        "link": "Link Paperback",
        "publisher": "Herausgeber Paperback",
        "isbn_10": "ISBN-10 Paperback",
        "isbn_13": "ISBN-13 Paperback",
        "price": "Preis Paperback",
        "page_count": "Seitenzahl Paperback",
        "edition": "Edition Paperback",
        "release_date": "Erscheinungsdatum Paperback",
    },
    "Gebundenes Buch": {
        "is_available": "Gebundene Ausgabe",
        "link": "Link Gebundene Ausgabe",
        "publisher": "Herausgeber Gebundene Ausgabe",
        "isbn_10": "ISBN-10 Gebundene Ausgabe",
        "isbn_13": "ISBN-13 Gebundene Ausgabe",
        "price": "Preis Gebundene Ausgabe",
        "page_count": "Seitenzahl Gebundene Ausgabe",
        "edition": "Edition Gebundene Ausgabe",
        "release_date": "Erscheinungsdatum Gebundene Ausgabe",
    },
    "Broschiert": {
        "is_available": "Broschiert",
        "link": "Link Broschiert",
        "publisher": "Herausgeber Broschiert",
        "isbn_10": "Broschiert ISBN-10",
        "isbn_13": "Broschiert ISBN-13",
        "price": "Preis Broschiert",
        "page_count": "Seitenzahl Broschiert",
        "edition": "Edition Broschiert",
        "release_date": "Erscheinungsdatum Broschiert",
    },
    "Hörbuch": {
        "is_available": "Hörbuch",
        "link": "Link Hörbuch",
        "publisher": "Herausgeber Hörbuch",
        "asin": "ASIN Hörbuch",
        "price": "Preis Hörbuch",
        "duration": "Dauer Hörbuch",
        "edition": "Edition Hörbuch (sofern gesonderte Angabe)",
        "release_date": "Erscheinungsdatum Hörbuch",
    },
    "Audio-CD": {
        "is_available": "Audio-CD",
        "link": "Link Audio-CD",
        "publisher": "Herausgeber Audio-CD",
        "isbn_10": "Audio-CD ISBN-10",
        "isbn_13": "Audio-CD ISBN-13",
        "price": "Preis Audio-CD",
        "duration": "Dauer Audio-CD",
        "edition": "Edition Audio-CD (sofern gesonderte Angabe)",
        "release_date": "Erscheinungsdatum Audio-CD",
    },
}

format_page_count_name = {
    "Taschenbuch": "Taschenbuch",
    "Paperback": "Broschiert",
    "Gebundenes Buch": "Gebundene Ausgabe",
    "Broschiert": "Broschiert",
    "Kindle": "Not relevant",
    "Hörbuch": "Not relevant",
    "Audio-CD": "Not relevant",
}

def human_like_delay():
    time_to_sleep = random.uniform(1, 2)
    time.sleep(time_to_sleep)
    
def init_driver_with_random_useragent():
    ua = UserAgent(browsers=['chrome'])
    user_agent = ua.random
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument(f"user-agent={user_agent}")
    
    # Initialize the WebDriver with the options
    driver = webdriver.Chrome(options=options)
    return driver

def get_page_with_new_agent(driver, url):
    driver.quit()
    driver = init_driver_with_random_useragent()
    driver.get(url)
    return driver

def navigate_safely(driver: webdriver.Chrome, url: str, max_retries: int = 3) -> [webdriver.Chrome, bool]:

    attempt = 0
    while attempt < max_retries:
        try: 
            driver = get_page_with_new_agent(driver, url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            # Scroll down fifty percent of the time
            if random.random() < 0.5:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            break   
        except:
            attempt += 1
            print(f"Attempt {attempt}/{max_retries} to navigate to {url} failed. Trying again in 3 seconds.")
            time.sleep(3)
        finally:
            human_like_delay()
    
    if attempt == max_retries:
        success = False
    else:
        success = True

    return driver, success
    
def get_general_info(driver: webdriver.Chrome) -> dict:
    """
    Get the general information from the book detail page. It assumes that the driver is already on the book detail page.
    """
    general = {}
    general["erfasst am"] = dt.date.today().strftime("%d.%m.%Y")
    try:
        general["Titel"] = driver.find_element(By.ID, "productTitle").text
        # Print progress
        print(f"Name of the book: '{general['Titel']}'")
    except:
        print("No 'Titel' found.")
    try:
        part_of_series = driver.find_elements(By.CSS_SELECTOR, "#rpi-attribute-book_details-series")
        general["Reihe?"] = "Ja" if len(part_of_series) > 0 else "Nein"
    except:
        print("No 'Reihe?' information found.")
    try:
        general['Autor'] = ""
        general['Übersetzer'] = ""
        authors_translators = driver.find_elements(By.CSS_SELECTOR, "#bylineInfo > span.author")
        for elem in authors_translators:
            name = elem.find_element(By.CSS_SELECTOR, "a.a-link-normal").get_attribute("innerHTML")
            contribution = elem.find_element(By.CSS_SELECTOR, "span.a-color-secondary").get_attribute("innerHTML")
            if "Autor" in contribution:
                general["Autor"] = name if general["Autor"] == "" else general["Autor"] + ", " + name
            elif "Übersetzer" in contribution or "Übersetzung" in contribution:
                general["Übersetzer"] = name if general["Übersetzer"] == "" else general["Übersetzer"] + ", " + name
    except:
        print("No 'Autor' and 'Übersetzer' information found.")
    
    return general

def get_format_info(format_num: int, format_name: str, format_link: str, driver: webdriver.Chrome, book_cols: dict) -> dict:
    """
    Get the information for a specific format.
    """
    # Print progress
    print(f"Format: {format_name}")

    # Define the format dicts and get the needed columns
    format = {}
    try:
        format_cols = book_cols[format_name]
    except:
        print(f"Format '{format_name}' is not supported.")
        return format

    # Navigate to the format
    if "void" not in format_link:
        driver, success = navigate_safely(driver, format_link)
        if not success:
            print(f"Could not navigate to format '{format_name}'.")
            return format
    else:
        if format_num > 0:
            print(f"The link to format '{format_name}' is not available.")
            return format

    # Store if format is available and link
    format[format_cols["is_available"]] = "Ja"
    format[format_cols["link"]] = driver.current_url

    # Store publisher info
    try:
        format[format_cols["publisher"]] = driver.find_element(By.CSS_SELECTOR, "div#rpi-attribute-book_details-publisher > div.rpi-attribute-value > span").get_attribute("innerHTML")
    except:
        pass

    # Store release date
    try:
        format[format_cols["release_date"]] = driver.find_element(By.CSS_SELECTOR, "div#rpi-attribute-book_details-publication_date > div.rpi-attribute-value > span").get_attribute("innerHTML")
    except:
        pass

    # Store price and if in kindle unlimited
    try:
        for elem in driver.find_elements(By.CSS_SELECTOR, "div#tmmSwatches span.a-color-price"):
            if "€" in elem.text:
                format[format_cols["price"]] = elem.text
                break
        if "Kindle" in format_name:
            if "0,00" in format[format_cols["price"]]:
                format[format_cols["in_kindle_unlimited"]] = "Ja"
                format[format_cols["price"]] = driver.find_element(By.CSS_SELECTOR, "span#kindle-price").text
            else:
                format[format_cols["in_kindle_unlimited"]] = "Nein"
    except:
        print(f"No '{format_cols['price']}' information found.")
        if "Kindle" in format_name:
            print(f"No '{format_cols['in_kindle_unlimited']}' information found.")
    
    # Extract inforamtion from the product information section on the bottom of the page (for Hörbuch this section is structured differently)
    if "Hörbuch" not in format_name:
        try:
            product_info = driver.find_elements(By.CSS_SELECTOR, "div#detailBullets_feature_div > ul > li > span")
            for elem in product_info:
                if "Herausgeber" in elem.text and format.get(format_cols["publisher"]) is None:
                    format[format_cols["publisher"]] = elem.text.split(";")[0].split("(")[0].replace("Herausgeber", "").replace(":", "").strip()
                elif "ISBN-10" in elem.text:
                    format[format_cols["isbn_10"]]= elem.text.replace("ISBN-10", "").replace(":", "").strip()
                elif "ISBN-13" in elem.text:
                    format[format_cols["isbn_13"]] = elem.text.replace("ISBN-13", "").replace(":", "").strip()

                if "Herausgeber" in elem.text and ";" in elem.text:
                    try:
                        format[format_cols["edition"]] = elem.text.split(";")[1].split("(")[0].strip()
                    except:
                        pass

                # Extract page count for print books
                # if format_name != "Kindle":
                if format_page_count_name[format_name] in elem.text:
                    format[format_cols["page_count"]] = elem.text.split(":")[1].replace("Seiten", "").strip()

                # Kindle specific information
                if format_name == "Kindle":
                    if "ASIN" in elem.text:
                        format[format_cols["asin"]] = elem.text.replace("ASIN", "").replace(":", "").strip()
                    elif "Seitenzahl der Print-Ausgabe" in elem.text:
                        format[format_cols["page_count"]] = elem.text.replace("Seitenzahl der Print-Ausgabe", "").replace(":", "").replace("Seiten", "").strip()
                    elif "ISBN-Quelle für Seitenzahl" in elem.text:
                        format[format_cols["page_count_source"]] = elem.text.replace("ISBN-Quelle für Seitenzahl", "").replace(":", "").strip()
        except:
            print("The product information section on the bottom of the page couldn't be extracted.")
        
    # Hörbuch: Extract information from the product information section on the bottom of the page
    if "Hörbuch" in format_name:
        try:
            format[format_cols["duration"]] = driver.find_element(By.CSS_SELECTOR, "tr#detailsListeningLength td span").text.replace(" und", ",")
        except:
            print(f"No '{format_cols['duration']}' information found.")
        try:
            format[format_cols["asin"]] = driver.find_element(By.CSS_SELECTOR, "tr#detailsAsin td").text
        except:
            print(f"No '{format_cols['asin']}' information found.")
        try:
            format[format_cols["publisher"]] = driver.find_element(By.CSS_SELECTOR, "tr#detailspublisher td").text
        except:
            print(f"No '{format_cols['publisher']}' information found.")
        try:
            format[format_cols["edition"]] = driver.find_element(By.CSS_SELECTOR, "tr#detailsVersion td").text
        except:
            pass
        if format.get(format_cols["release_date"]) is None:
            try:
                format[format_cols["release_date"]] = driver.find_element(By.CSS_SELECTOR, "tr#detailsReleaseDate td").text
            except:
                print(f"No '{format_cols['release_date']}' information found.")
    
    return format

def german_datastring_to_date(date_string):
    # Map of German month names to English month names
    month_map = {
        'Januar': 'January',
        'Februar': 'February',
        'März': 'March',
        'April': 'April',
        'Mai': 'May',
        'Juni': 'June',
        'Juli': 'July',
        'August': 'August',
        'September': 'September',
        'Oktober': 'October',
        'November': 'November',
        'Dezember': 'December'
    }
    
    # Replace German month with English month
    for german_month, english_month in month_map.items():
        date_string = date_string.replace(german_month, english_month)
    
    # Convert the modified string to datetime
    return dt.datetime.strptime(date_string, '%d. %B %Y').date()

if __name__ == "__main__":
    # Setup the webdriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()), options=options)
    # driver_path = "/Users/jgross/.wdm/drivers/chromedriver/mac64/117.0.5938.149/chromedriver-mac-arm64/chromedriver"
    # driver_path = "/Users/mirjamweng/.wdm/drivers/chromedriver/mac64/118.0.5993.70/chromedriver-mac-x64/chromedriver"
    # driver = webdriver.Chrome(service = Service(executable_path=driver_path), options=options)
    

    # Read all of already scraped titles and authors
    if os.path.isfile(f"{FILE_NAME}_Titel_und_Autoren.csv"):
        title_and_authors_df = pd.read_csv(f"{FILE_NAME}_Titel_und_Autoren.csv", encoding="utf-16")
    else:
        title_and_authors_df = pd.DataFrame(columns=["Titel", "Autor", "Datum auf Suchseite", "erfasst am"])

    # Iterate over all books in search result until MAX_BOOKS is reached or known book is encountered
    # Only scrape books that came out before today
    books = []
    page_counter = 0
    num_known_books = 0
    navigation_fails = 0
    stop = False
    while not stop:
        # Increment page counter
        page_counter += 1
        
        # Set url to the initial url or the url of the next page
        if page_counter == 1:
            url = URL
        else:     
            # List of locator strategies to try
            locator_strategies = [
                (By.CSS_SELECTOR, "a.s-pagination-next"),
                (By.XPATH, "//a[@class='s-pagination-next']"),
                (By.CLASS_NAME, "s-pagination-next"),
                # (By.ID, "uniqueButtonID"),
                # (By.NAME, "buttonName"),
                # (By.LINK_TEXT, "Next Page"),
                # (By.PARTIAL_LINK_TEXT, "Next"),
            ]

            for locator_strategy, locator_string in locator_strategies:
                try:
                    #time.sleep(1)  # Consider whether the sleep is necessary in your specific case
                    next_page_button = driver.find_element(locator_strategy, locator_string)
                    url = next_page_button.get_attribute('href')
                    break  # Break out of the loop if element is found
                except NoSuchElementException:
                    # Continue to the next locator strategy if the element is not found
                    continue
            else:
                # This block executes if none of the locator strategies found the element
                print("No next page found.")
                stop = True   
                break

        # Navigate to the product list page and print progress
        driver, success = navigate_safely(driver, url)
        if not success:
            print("Could not navigate to the next page.")
            stop = True
            break
        print(f"Now on search page {page_counter}.")

        # Find all books in search result and extract the links
        book_containers = driver.find_elements(By.XPATH, "//div[@data-component-type='s-search-result']")
        all_book_links = [book.find_element(By.CSS_SELECTOR, "a.a-link-normal").get_attribute('href') for book in book_containers]
        num_books_on_page = len(all_book_links)
        
        # Create a list of all dates of the books on the current page, put None if no date is available
        all_book_dates = []
        for book in book_containers:
            try:
                all_book_dates.append(book.find_element(By.CSS_SELECTOR, "span.a-size-base.a-color-secondary.a-text-normal").get_attribute('innerHTML'))
            except:
                all_book_dates.append(None)
        
        # Only keep the books that came out before today or have no date
        book_links = []
        book_idxs = []
        book_dates = []
        for idx, (book_link, book_date) in enumerate(zip(all_book_links, all_book_dates)):
            if book_date is None:
                book_links.append(book_link)
                book_idxs.append(idx)
                book_dates.append("kein Datum")
            else:
                book_date_str = book_date
                book_date = german_datastring_to_date(book_date)
                if book_date < dt.date.today():
                    book_links.append(book_link)
                    book_idxs.append(idx)
                    book_dates.append(book_date_str)

        # Extract desired information from each book container
        for idx, book_link, book_date in zip(book_idxs, book_links, book_dates):
            # Stopping criteria
            if len(books) >= MAX_BOOKS or navigation_fails >= MAX_NAVIGATION_FAILS:
                stop = True
                break

            # Define book dict and add book date
            book = {}
            book['Datum auf Suchseite'] = book_date
            # Go to book detail page
            driver, success = navigate_safely(driver, book_link)
            if not success:
                print(f"Could not navigate to book '{book_link}'.")
                navigation_fails += 1
                continue

            # Print progress
            print(f"Now on book {idx+1} of {num_books_on_page} on page {page_counter}.")

            # Get general information
            general = get_general_info(driver)

            check_book = True
            if "Autor" not in general or "Titel" not in general:
                print("No 'Autor' or 'Titel' found. Hence, we can't check if the book is already known, but simply store it.")
                check_book = False

            # Check if book is already known
            if check_book:
                book_matches = (title_and_authors_df['Autor'] == general["Autor"]) & (title_and_authors_df['Titel'] == general["Titel"]) & (title_and_authors_df['Datum auf Suchseite'] == book['Datum auf Suchseite'])
                if book_matches.any():
                    if book_matches.iloc[-1]:
                        num_known_books += 1
                        print(f"Book '{general['Titel']}' by '{general['Autor']}' is the last book from the previous scraping session, so we don't store it again and stop.")
                        stop = True
                        break
                    else:
                        num_known_books += 1
                        print(f"Book '{general['Titel']}' by '{general['Autor']}' is already known.")
                        if num_known_books > MAX_KNOWN_BOOKS:
                            stop = True
                            break

            # Extract all avalailable format names and links
            # formats = driver.find_elements(By.CSS_SELECTOR, "div#tmmSwatches ul span.a-button-inner")
            formats = []
            for row in driver.find_elements(By.CSS_SELECTOR, "#tmmSwatches .a-row"):
                for format in row.find_elements(By.CSS_SELECTOR, "span.a-button-inner"):
                    formats.append(format)
            format_links = [format.find_element(By.CSS_SELECTOR, "a.a-button-text").get_attribute('href') for format in formats]          
            format_names = []
            for idx, format in enumerate(formats):
                unknown_format = True
                for elem in format.find_elements(By.CSS_SELECTOR, "a.a-button-text span"):
                    if elem.get_attribute("innerHTML").strip() in BOOK_COLS.keys():
                        format_names.append(elem.get_attribute("innerHTML").strip())
                        unknown_format = False
                        break
                
                if unknown_format:
                    format_links.pop(idx)
        

            # make sure that the format with a void link is the first in the list
            if any("javascript:void(0)" in link for link in format_links):
                void_idx = format_links.index("javascript:void(0)")
                format_links.insert(0, format_links.pop(void_idx))
                format_names.insert(0, format_names.pop(void_idx))

            # Get the information for each format
            for format_num, (format_link, format_name) in enumerate(zip(format_links, format_names)):
                format = get_format_info(format_num, format_name, format_link, driver, BOOK_COLS)
                book.update(format)
            
            # Get available format names 
            general["verfügbare Formate"] = ", ".join(sorted(format_names, key=str.lower))

            # Add general information to the book
            book.update(general)

            # Add book to the list of books
            books.append(book)

        # Check if stopping criteria is met
        if (len(books) >= MAX_BOOKS) or stop:
            stop = True
            print("The maximum number of total or known books is reached.")
            break
        
        # Navigate back to the search result page
        driver, success = navigate_safely(driver, url)
        if not success:
            print("Could not navigate back to the search result page.")
            stop = True
            break
    
    # Close the driver
    driver.quit()
    print("Finished scraping. Now saving the results.")

    # Extract all columns from book_cols
    columns = []
    for value in BOOK_COLS.values():
        columns.extend(value.values())
    columns = columns + ["Datum auf Suchseite"]

    # Create a dataframe and save it to a csv file
    books_df = pd.DataFrame.from_records(books, columns=columns)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    books_df.drop('Datum auf Suchseite', axis=1).to_csv(f"{timestamp}_{FILE_NAME}.csv", index=False, encoding="utf-16")

    # Add new books to the title_and_authors_df and save it to a csv file
    titel_col = BOOK_COLS["general"]["Titel"]
    autor_col = BOOK_COLS["general"]["Autor"]
    day_col = BOOK_COLS["general"]["erfasst am"]
    title_and_authors_df = pd.concat([title_and_authors_df, books_df[[day_col, titel_col, autor_col, 'Datum auf Suchseite']].iloc[::-1].reset_index(drop=True)])
    title_and_authors_df.to_csv(f"{FILE_NAME}_Titel_und_Autoren.csv", index=False, encoding="utf-16") 
