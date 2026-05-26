import os
import re
import sys
import json
import logging
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# 3. Call load_dotenv() at the very top to import environment variables securely
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
from google import genai
from google.genai import errors

# Configure production-ready logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("NewsBriefApp")

app = Flask(__name__)

# Constants
FEEDS = {
    "HackerNews": "https://news.ycombinator.com/rss",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Reuters World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
}

# Check if environment is configured
def get_env_variable(name, placeholder):
    val = os.environ.get(name)
    if not val or val == placeholder or val.strip() == "":
        return None
    return val

def check_config():
    tg_token = get_env_variable("TELEGRAM_TOKEN", "your_actual_bot_token_here")
    tg_chat_id = get_env_variable("TELEGRAM_CHAT_ID", "your_actual_chat_id_here")
    gemini_key = get_env_variable("GEMINI_API_KEY", "your_actual_gemini_key_here")
    return tg_token, tg_chat_id, gemini_key

# 4. Fetch RSS feeds using standard urllib libraries
def fetch_rss_feed(feed_name, url):
    logger.info(f"Fetching RSS feed for {feed_name} from {url}")
    # Custom headers to avoid scraping protection blocks (e.g. from CoinTelegraph)
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
        
        root = ET.fromstring(data)
        items = []
        
        # Parse standard RSS items
        for item in root.findall('.//item'):
            title_elem = item.find('title')
            link_elem = item.find('link')
            desc_elem = item.find('description')
            
            title = title_elem.text if title_elem is not None else ""
            link = link_elem.text if link_elem is not None else ""
            desc = desc_elem.text if desc_elem is not None else ""
            
            # Clean up whitespace and basic HTML if present
            title = title.strip() if title else ""
            link = link.strip() if link else ""
            if desc:
                # Remove common HTML tags for standard formatting
                desc = re.sub(r'<[^>]+>', '', desc)
                desc = re.sub(r'\s+', ' ', desc).strip()
            else:
                desc = ""
                
            items.append({
                "title": title,
                "link": link,
                "description": desc
            })
            
            # Limit items fetched per feed to prevent overloading LLM context
            if len(items) >= 15:
                break
                
        logger.info(f"Successfully fetched and parsed {len(items)} items from {feed_name}")
        return items
    except Exception as e:
        logger.error(f"Error fetching/parsing {feed_name}: {e}", exc_info=True)
        return []

# 5. Compile & send RSS data to official Google GenAI SDK (gemini-2.5-flash)
def generate_briefing(raw_news, gemini_api_key):
    logger.info("Initializing Google GenAI SDK and generating briefing...")
    
    # Format the news items into a structured readable string for Gemini
    formatted_input = ""
    for category, items in raw_news.items():
        formatted_input += f"=== Category: {category} ===\n"
        if not items:
            formatted_input += "No updates for this category.\n\n"
            continue
        for idx, item in enumerate(items, 1):
            formatted_input += f"{idx}. Title: {item['title']}\n"
            if item['link']:
                formatted_input += f"   Link: {item['link']}\n"
            if item['description']:
                formatted_input += f"   Summary: {item['description']}\n"
            formatted_input += "\n"

    # Initialize Gemini client
    # The SDK automatically uses GEMINI_API_KEY environment variable.
    # To be extremely robust and clean, we rely on genai.Client() as requested.
    client = genai.Client()
    
    prompt = (
        "You are an elite, concise editor. Review the following raw news items across categories:\n"
        f"{formatted_input}\n"
        "Your task is to compile a crisp, premium, and highly professional news briefing. Follow these rules:\n"
        "1. Filter out duplicate stories, noise, and low-quality summaries.\n"
        "2. Keep the briefing short, punchy, and direct.\n"
        "3. Output MUST be formatted strictly in Telegram's custom Markdown format.\n"
        "   - Use single asterisks for bold (*bold*). Do NOT use double asterisks (**bold**).\n"
        "   - Use single underscores for italics (_italics_).\n"
        "   - Inline links must use the pattern: [Anchor Text](URL).\n"
        "   - Do not use complex nested markdown styling.\n"
        "4. Structure the briefing with category headers using clear emojis and emojis for bullet points.\n"
        "5. Include a final sentence with a dynamic editorial sign-off."
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        briefing = response.text
        logger.info("Gemini briefing generated successfully.")
        return briefing
    except errors.APIError as api_err:
        logger.error(f"Gemini API Error occurred: {api_err}", exc_info=True)
        raise api_err
    except Exception as e:
        logger.error(f"Unexpected error in briefing generation: {e}", exc_info=True)
        raise e

# 6. Send payload directly to Telegram Bot API using urllib.request
def send_to_telegram(text, token, chat_id):
    logger.info(f"Sending brief to Telegram chat {chat_id}")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'NewsBriefApp/1.0'
        },
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            if res_json.get('ok'):
                logger.info("Successfully delivered brief to Telegram.")
                return True, res_json
            else:
                logger.error(f"Telegram returned error response: {res_json}")
                return False, res_json
    except urllib.error.HTTPError as he:
        err_content = he.read().decode('utf-8') if he else ""
        logger.error(f"Telegram HTTP Error {he.code}: {he.reason}. Response: {err_content}")
        return False, {"error": f"HTTP {he.code}", "details": err_content}
    except Exception as e:
        logger.error(f"Failed to make outbound connection to Telegram Bot API: {e}", exc_info=True)
        return False, {"error": str(e)}

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    tg_token, tg_chat_id, gemini_key = check_config()
    config_status = {
        "TELEGRAM_TOKEN": "Configured" if tg_token else "Missing/Placeholder",
        "TELEGRAM_CHAT_ID": "Configured" if tg_chat_id else "Missing/Placeholder",
        "GEMINI_API_KEY": "Configured" if gemini_key else "Missing/Placeholder"
    }
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config_status": config_status
    }), 200

# 4. Trigger route to run the entire flow
@app.route('/trigger', methods=['GET', 'POST'])
def trigger_brief():
    logger.info("Received request to /trigger")
    
    # 1. Validate environment configuration
    tg_token, tg_chat_id, gemini_key = check_config()
    if not (tg_token and tg_chat_id and gemini_key):
        missing_vars = []
        if not tg_token: missing_vars.append("TELEGRAM_TOKEN")
        if not tg_chat_id: missing_vars.append("TELEGRAM_CHAT_ID")
        if not gemini_key: missing_vars.append("GEMINI_API_KEY")
        
        logger.warning(f"Trigger request skipped due to missing or placeholder environment variables: {missing_vars}")
        return jsonify({
            "status": "error",
            "message": "Configuration required. Please configure valid credentials in your .env file.",
            "missing_variables": missing_vars,
            "instruction": "Ensure your .env file does not contain literal placeholder strings."
        }), 400

    # 2. Fetch and parse all feeds
    raw_news = {}
    total_articles = 0
    for feed_name, feed_url in FEEDS.items():
        feed_items = fetch_rss_feed(feed_name, feed_url)
        raw_news[feed_name] = feed_items
        total_articles += len(feed_items)
        
    if total_articles == 0:
        logger.warning("No articles fetched from any of the sources.")
        return jsonify({
            "status": "warning",
            "message": "No news articles could be fetched from the feeds."
        }), 502

    # 3. Generate summary via Gemini 2.5 Flash
    try:
        briefing = generate_briefing(raw_news, gemini_key)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to generate briefing using Gemini model: {e}"
        }), 500

    # 4. Dispatch brief to Telegram Bot
    success, tg_result = send_to_telegram(briefing, tg_token, tg_chat_id)
    if success:
        # CLEAN FIX: Pass back a lightweight confirmation. 
        return jsonify({
            "status": "success",
            "message": "Morning news briefing successfully compiled and sent to Telegram."
        }), 200
    else:
        # For debugging errors, it's fine to pass a small message description
        return jsonify({
            "status": "error",
            "message": "Failed to send briefing to Telegram.",
            "details": str(tg_result)[:200]  # Truncate to keep the payload tiny
        }), 502

if __name__ == '__main__':
    # Production-ready configuration defaults
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting Flask server on {host}:{port} in debug mode...")
    app.run(host=host, port=port, debug=True)
