"""
Inline CSS and JavaScript into templates/index.html
Ensures 100% self-contained delivery on Vercel Serverless Functions.
"""

def inline_assets():
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    with open('static/css/style.css', 'r', encoding='utf-8') as f:
        css = f.read()

    with open('static/js/script.js', 'r', encoding='utf-8') as f:
        js = f.read()

    # Inlining CSS
    css_block = f"<style>\n{css}\n</style>"
    link_str = '<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/style.css\') }}">'
    if link_str in html:
        html = html.replace(link_str, css_block)
        print("CSS inlined successfully!")
    else:
        print("CSS link string not found, skipping CSS inline.")

    # Inlining JS
    js_block = f"<script>\n{js}\n</script>"
    script_str = '<script src="{{ url_for(\'static\', filename=\'js/script.js\') }}"></script>'
    if script_str in html:
        html = html.replace(script_str, js_block)
        print("JS inlined successfully!")
    else:
        print("JS script string not found, skipping JS inline.")

    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("templates/index.html updated successfully!")

if __name__ == '__main__':
    inline_assets()
