"""
Inline CSS and JavaScript into templates/index.html
Ensures 100% self-contained delivery on Vercel Serverless Functions.
"""

import re

def inline_assets():
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    with open('static/css/style.css', 'r', encoding='utf-8') as f:
        css = f.read().strip()

    with open('static/js/script.js', 'r', encoding='utf-8') as f:
        js = f.read().strip()

    # Inlining CSS: replace existing <link> or <style>...</style> block
    css_block = f"<style>\n{css}\n    </style>"
    link_str = '<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/style.css\') }}">'
    
    if link_str in html:
        html = html.replace(link_str, css_block)
        print("Replaced CSS <link> successfully!")
    elif '<style>' in html and '</style>' in html:
        html = re.sub(r'<style>[\s\S]*?</style>', css_block, html, count=1)
        print("Replaced existing <style> tag successfully!")
    else:
        print("Warning: Neither CSS link nor <style> found!")

    # Inlining JS: replace existing <script src="...script.js"></script> or existing client <script> block
    js_block = f"<script>\n{js}\n    </script>"
    script_str = '<script src="{{ url_for(\'static\', filename=\'js/script.js\') }}"></script>'

    if script_str in html:
        html = html.replace(script_str, js_block)
        print("Replaced JS <script src> successfully!")
    elif re.search(r'<script>\s*/\*\*\s*\*\s*Spotify Song Recommendation Studio[\s\S]*?</script>', html):
        html = re.sub(
            r'<script>\s*/\*\*\s*\*\s*Spotify Song Recommendation Studio[\s\S]*?</script>',
            js_block,
            html,
            count=1
        )
        print("Replaced existing client <script> tag successfully!")
    else:
        print("Warning: Neither JS script tag nor client <script> found!")

    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("templates/index.html updated and inlined successfully!")

if __name__ == '__main__':
    inline_assets()
