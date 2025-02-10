import html
import logging
from jinja2 import Template
from pathlib import Path


def create_match_card(match):
    """Create HTML for a single match card with proper escaping"""
    return f"""
    <div class="match-card" data-similarity="{match['similarity']}">
        <div class="similarity-bar">
            <div class="similarity-fill" style="width: {match['similarity'] * 100}%"></div>
        </div>
        <p><strong>Similarity:</strong> {match['similarity']:.2f}</p>
        <div class="document-info">
            <div class="doc1-info">
                <strong>Document 1:</strong> {html.escape(match['doc1_name'])} (Page {match['doc1_page']})<br>
                <div class="sentence">{html.escape(match['doc1_sentence'])}</div>
            </div>
            <div class="doc2-info">
                <strong>Document 2:</strong> {html.escape(match['doc2_name'])} (Page {match['doc2_page']})<br>
                <div class="sentence">{html.escape(match['doc2_sentence'])}</div>
            </div>
        </div>
    </div>
    """


def generate_html_report(data, output_path):
    """Generate interactive HTML report with comparison results"""
    # Create templates directory if it doesn't exist
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)

    # Create template file if it doesn't exist
    template_path = template_dir / "report_template.html"
    if not template_path.exists():
        template_content = """
        <html>
        <head>
            <title>Document Comparison Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .slider-container { 
                    margin: 20px 0;
                    padding: 15px;
                    background: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .match-card { 
                    background: white;
                    border: 1px solid #ddd;
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .similarity-bar {
                    height: 6px;
                    background: #eee;
                    margin: 10px 0;
                    border-radius: 3px;
                }
                .similarity-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #4CAF50, #45a049);
                    border-radius: 3px;
                    transition: width 0.3s ease;
                }
                .document-info {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 15px;
                }
                .doc1-info, .doc2-info {
                    padding: 15px;
                    background: #f9f9f9;
                    border-radius: 5px;
                }
                .sentence {
                    margin-top: 10px;
                    line-height: 1.5;
                }
                h1, h2 {
                    color: #333;
                }
                #threshold {
                    width: 200px;
                    margin: 0 10px;
                }
                .stats {
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .footer {
                    margin-top: 40px;
                    padding: 20px;
                    background: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                    font-size: 0.9em;
                    color: #666;
                }
                .footer a {
                    color: #4CAF50;
                    text-decoration: none;
                    font-weight: bold;
                }
                .footer a:hover {
                    text-decoration: underline;
                }
                .footer-logo {
                    margin-bottom: 10px;
                    font-weight: bold;
                    font-size: 1.1em;
                }
            </style>
        </head>
        <body>
            <h1>Document Comparison Report</h1>
            
            <div class="stats">
                <h3>Documents Compared:</h3>
                <p>Document 1: {{ data.doc1.path.split('/')[-1] }}</p>
                <p>Document 2: {{ data.doc2.path.split('/')[-1] }}</p>
                <p>Total Similar Sentences Found: {{ data.similar_sentences|length }}</p>
            </div>

            <div class="slider-container">
                <label>Similarity Threshold: 
                    <input type="range" id="threshold" min="0" max="1" step="0.05" value="0.85"
                           oninput="updateThreshold(this.value)">
                    <span id="thresholdValue">0.85</span>
                </label>
            </div>

            <h2>Similar Sentences ({{ data.similar_sentences|length }})</h2>
            <div id="matchesContainer">
                {% for match in data.similar_sentences %}
                    {{ create_match_card(match) }}
                {% endfor %}
            </div>

            <div class="footer">
                <div class="footer-logo">🚀 Powered by FragDasPDF</div>
                <p>
                    Try our AI-powered PDF analysis tool at <a href="https://fragdaspdf.de" target="_blank">FragDasPDF.de</a><br>
                    View source code on <a href="https://github.com/FragDasPDF/regierungsprogramme" target="_blank">GitHub</a>
                </p>
            </div>

            <script>
                function updateThreshold(value) {
                    document.getElementById('thresholdValue').textContent = value;
                    const matches = document.querySelectorAll('.match-card');
                    matches.forEach(match => {
                        const matchScore = parseFloat(match.dataset.similarity);
                        match.style.display = matchScore >= parseFloat(value) ? 'block' : 'none';
                    });
                }

                // Sort matches by similarity on load
                document.addEventListener('DOMContentLoaded', function() {
                    const container = document.getElementById('matchesContainer');
                    const matches = Array.from(container.children);
                    matches.sort((a, b) => {
                        return parseFloat(b.dataset.similarity) - parseFloat(a.dataset.similarity);
                    });
                    matches.forEach(match => container.appendChild(match));
                });
            </script>
        </body>
        </html>
        """
        template_path.write_text(template_content)

    # Load and render template
    template = Template(template_path.read_text())
    rendered = template.render(data=data, create_match_card=create_match_card)

    # Write output file
    output_path = Path(output_path)
    output_path.write_text(rendered, encoding="utf-8")
