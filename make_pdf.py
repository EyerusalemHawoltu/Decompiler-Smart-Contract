import csv
from fpdf import FPDF
csv.field_size_limit(10**9)
rows = list(csv.DictReader(open('results/full_scores_clean_table.csv')))

def clean(s):
    return (s or '').encode('latin-1', 'replace').decode('latin-1')

pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(True, margin=12)
pdf.set_margins(10, 10, 10)
pdf.add_page()

def cell(h, txt, size, color, bold=False):
    pdf.set_font('Helvetica' if bold else 'Courier', 'B' if bold else '', size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, h, clean(txt), wrapmode="CHAR", new_x="LMARGIN", new_y="NEXT")

ok = 0
for i, r in enumerate(rows):
    try:
        cell(4, f"{i+1}. {r['function']}  [v{r['version']}]  bleu4={r['bleu4']} exact={r['exact_match']}", 8, (0,0,160), bold=True)
        cell(3.4, "EXPECTED:\n" + r['expected_solidity'], 7, (0,110,0))
        pdf.ln(1)
        cell(3.4, "MODEL OUTPUT:\n" + r['model_output'], 7, (170,0,0))
        pdf.set_draw_color(200,200,200); y=pdf.get_y()+1; pdf.line(10,y,200,y); pdf.ln(3)
        ok += 1
    except Exception as e:
        pdf.ln(2)
        continue

pdf.output('results/test_expected_vs_output.pdf')
print("wrote results/test_expected_vs_output.pdf  (%d/%d functions rendered)" % (ok, len(rows)))
