import pdfplumber
import pandas as pd

tables = []


with pdfplumber.open('IB_BOUNDARIES/M22.pdf') as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        tables_on_page = page.extract_tables({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_x_tolerance": 10,
            "intersection_y_tolerance": 10
        })
        
        if tables_on_page:
            for table in tables_on_page:
                if table:
                    tables.append({
                        'page': page_num,
                        'data': table
                    })

for table in tables:
    with open('M22.txt', 'a') as f:
        df = pd.DataFrame(table['data'])
        f.write(df.to_string(index=False))
        f.write("\n" + "-" * 50 + "\n")
