import html
import logging
import csv
import pandas as pd
from jinja2 import Template
from pathlib import Path
import argparse


def generate_csv_report(data, output_path):
    """Generate CSV file with comparison results"""
    csv_path = Path(output_path).with_suffix(".csv")

    # Filter out short sentences
    filtered_matches = [
        match
        for match in data.get("similar_sentences", [])
        if len(match.get("doc1_sentence", "").strip()) >= 15
        and len(match.get("doc2_sentence", "").strip()) >= 15
    ]

    # Prepare data for CSV
    csv_data = []
    for match in filtered_matches:
        csv_data.append(
            {
                "similarity": match.get("similarity", 0),
                "doc1_name": match.get("doc1_name", ""),
                "doc1_page": match.get("doc1_page", ""),
                "doc1_sentence": match.get("doc1_sentence", ""),
                "doc2_name": match.get("doc2_name", ""),
                "doc2_page": match.get("doc2_page", ""),
                "doc2_sentence": match.get("doc2_sentence", ""),
            }
        )

    # Write to CSV
    if csv_data:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    return csv_path


def generate_html_from_csv(csv_path, output_path):
    """Generate HTML report directly from a CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure 'similarity' is a float
        if "similarity" not in df.columns:
            raise ValueError("CSV file does not contain 'similarity' column")

        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0)

        # Calculate statistics
        stats = {
            "total_matches": len(df),
            "above_95": len(df[df["similarity"] >= 0.95]),
            "above_90": len(df[df["similarity"] >= 0.90]),
            "above_85": len(df[df["similarity"] >= 0.85]),
        }

        # Create the data structure
        data = {
            "doc1": {"path": "From CSV"},
            "doc2": {"path": "From CSV"},
            "similar_sentences": df.to_dict("records"),
            "stats": stats,
        }

        # Generate the HTML report
        generate_html_report(data, output_path)
        logging.info(f"Successfully generated HTML report at {output_path}")

    except Exception as e:
        logging.error(f"Error generating HTML from CSV: {str(e)}")
        raise


def generate_html_report(data, output_path):
    """Generate interactive HTML report with comparison results"""
    # Filter out short sentences (same as in generate_csv_report)
    filtered_matches = [
        match
        for match in data.get("similar_sentences", [])
        if len(match.get("doc1_sentence", "").strip()) >= 15
        and len(match.get("doc2_sentence", "").strip()) >= 15
    ]

    df = pd.DataFrame(filtered_matches) if filtered_matches else pd.DataFrame(columns=["similarity"])
    df["similarity"] = pd.to_numeric(df.get("similarity", 0), errors="coerce").fillna(0)

    stats = data.get(
        "stats",
        {
            "total_matches": len(df),
            "above_95": len(df[df["similarity"] >= 0.95]),
            "above_90": len(df[df["similarity"] >= 0.90]),
            "above_85": len(df[df["similarity"] >= 0.85]),
        },
    )

    template_content = """
    <html>
    <head>
        <title>Document Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: auto; }
            .match-card { padding: 15px; background: #f9f9f9; margin: 10px 0; border-radius: 5px; }
            .stats { margin: 20px 0; }
            .stats table { width: 100%; border-collapse: collapse; }
            .stats th, .stats td { padding: 10px; border-bottom: 1px solid #ddd; text-align: left; }
            .similarity-bar { background: #ddd; height: 6px; border-radius: 3px; margin-top: 5px; }
            .similarity-fill { height: 100%; background: #4CAF50; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Document Comparison Report</h1>
        
        <div class="stats">
            <h3>Comparison Statistics</h3>
            <table>
                <tr><th>Threshold</th><th>Matches</th><th>Percentage</th></tr>
                {% for key, val in stats.items() if key != "total_matches" %}
                <tr>
                    <td>≥ {{ key.split("_")[1] }}% similarity</td>
                    <td>{{ val }}</td>
                    <td>{{ "%.2f"|format(val / stats.total_matches * 100 if stats.total_matches else 0) }}%</td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <h2>Similar Sentences ({{ matches|length }})</h2>
        <div>
            {% for match in matches %}
            <div class="match-card">
                <p><strong>Similarity:</strong> {{ "%.2f"|format(match.similarity) }}</p>
                <div class="similarity-bar">
                    <div class="similarity-fill" style="width: {{ match.similarity * 100 }}%"></div>
                </div>
                <p><strong>Document 1:</strong> {{ match.doc1_name }} (Page {{ match.doc1_page }})</p>
                <p>{{ match.doc1_sentence }}</p>
                <p><strong>Document 2:</strong> {{ match.doc2_name }} (Page {{ match.doc2_page }})</p>
                <p>{{ match.doc2_sentence }}</p>
            </div>
            {% endfor %}
        </div>

        <p>Generated by <a href="https://fragdaspdf.de" target="_blank">FragDasPDF</a></p>
    </body>
    </html>
    """

    template = Template(template_content)
    rendered_html = template.render(stats=stats, matches=filtered_matches)

    Path(output_path).write_text(rendered_html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report from CSV")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument(
        "-o", "--output", help="Output HTML file", default="report.html"
    )
    args = parser.parse_args()

    try:
        generate_html_from_csv(args.csv_path, args.output)
        print(f"Successfully generated HTML report: {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)