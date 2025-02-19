from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

class SimpleScraper:
    def __init__(self, 
                 output_dir: str = "./scraped_content",
                 headless: bool = True,
                 wait_time: int = 5):
        """Initialize the scraper"""
        self.output_dir = output_dir
        self.wait_time = wait_time
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        
        # Initialize webdriver
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(wait_time)

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        # Remove extra whitespace
        return " ".join(text.split())

    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape content from the specified div class"""
        try:
            self.driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.driver, self.wait_time).until(
                EC.presence_of_element_located((By.CLASS_NAME, "wpb-content-wrapper"))
            )
            
            # Get the content
            content_div = self.driver.find_element(By.CLASS_NAME, "wpb-content-wrapper")
            html_content = content_div.get_attribute('outerHTML')
            
            # Use BeautifulSoup to clean up the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Get clean text
            text_content = self.clean_text(soup.get_text())
            
            return {
                "url": url,
                "content": text_content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "url": url,
                "content": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def scrape_multiple_pages(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape content from multiple URLs"""
        results = []
        
        try:
            for url in urls:
                result = self.scrape_page(url)
                results.append(result)
                
            # Save results
            self.save_results(results)
            
        finally:
            self.cleanup()
            
        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"scraped_content_{timestamp}.json")
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()

# Example usage
if __name__ == "__main__":
    urls_to_scrape = [
        "https://www.expressanalytics.com/solutions/rfm-model/",
        "https://www.expressanalytics.com/product/voice-of-customer-analysis/"
    ]
    
    scraper = SimpleScraper(
        output_dir="./scraped_data",
        headless=True,
        wait_time=5
    )
    
    results = scraper.scrape_multiple_pages(urls_to_scrape)
    
    # Print results
    for result in results:
        print(f"URL: {result['url']}")
        if result.get('error'):
            print(f"Error: {result['error']}")
        else:
            print(f"Content length: {len(result['content'])} characters")
        print("-" * 50)