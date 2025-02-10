import html
import logging

def generate_html_report(results_by_chapter, output_file):
    """
    Generates an HTML report of similar sentences found in the documents.
    
    Args:
        results_by_chapter: Dictionary mapping chapter names to lists of match results
        output_file: Path where the HTML report should be saved
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .chapter { margin-bottom: 30px; }
            .match { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }
            .similarity { color: #666; }
        </style>
    </head>
    <body>
        <h1>Document Comparison Report</h1>
    """
    
    for chapter, matches in results_by_chapter.items():
        html_content += f"<div class='chapter'><h2>{html.escape(chapter)}</h2>"
        for match in matches:
            html_content += f"""
                <div class='match'>
                    <p>Page {match['page1']}: {html.escape(match['sentence1'])}</p>
                    <p>Page {match['page2']}: {html.escape(match['sentence2'])}</p>
                    <p class='similarity'>Similarity: {match['similarity']:.2%}</p>
                </div>
            """
        html_content += "</div>"
    
    html_content += "</body></html>"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    logging.info(f"HTML report generated: {output_file}") 