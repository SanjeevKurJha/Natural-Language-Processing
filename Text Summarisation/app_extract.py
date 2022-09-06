import fitz
import glob
import pandas as pd
import re
import os
from pandas import ExcelWriter
from flask import Flask,render_template,request


app = Flask(__name__)

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9.]+", ' ', text)
    #text = re.sub(r"[^a-zA-Z0-9.!?']", ' ', text)
    text = re.sub("\n", '', text)
    text = re.sub(r"[..]", '.', text)
    return text


@app.route('/')




def text_extract():
    dic = {}
    print("+++++++++++++++++++++++++++++++++++++++++++",'HI')
    pdf_files = glob.glob("data/" + "/*.pdf")
    org_text = ""
    all_abstracts = ""
    if os.path.exists('Data.xlsx'):
        os.remove('Data.xlsx')
    for filename in pdf_files:
        print("+++++++++++++++++++++++++++++++++++++++++++",filename)
        doc = fitz.open(filename)
        #print ("number of pages: %i" % doc.pageCount)
        #print (doc.is_repaired)
        #print (doc.page_count)
        ##for page in doc:
        ##    page1text = page.getText()
        ##    page2text = page.getText()

        page1 = doc.load_page(0)
        page1text = page1.get_text("text")
        
        page2 = doc.load_page(1)
        page2text = page2.get_text("text")

        try:
            try:
                Abstract = page1text.split("Introduction",1)[1]
                partitioned_string = page2text.split('Data')[0]
            except:
                Abstract = page1text.split("Introduction",1)[0]
                partitioned_string = Abstract.split('Data')[0]
        except Exception as IndexError:
            try:
                Abstract = page2text.split("Introduction",1)[1]
                partitioned_string = Abstract.split('Data')[0]
            except:
                Abstract = page2text.split("Introduction",1)[0]
                partitioned_string = Abstract.split('Data')[0]
        partitioned_string = re.sub('\S+@\S+',"",partitioned_string)
        partitioned_string = re.sub(r'[0-9]',"",partitioned_string)
        #partitioned_string = " ".join(re.findall(r"[a-zA-Z0-9]+",partitioned_string ))
        all_abstracts = partitioned_string
        org_text=all_abstracts
        org_text = clean_text(org_text)
        dic[filename] = org_text
    
   
    df = pd.DataFrame(dic.items(), columns=['Filename', 'Abstract'])
    writer = ExcelWriter('Data.xlsx')
    df.to_excel(writer,sheet_name = 'Filename')
    writer.save()
    
      
    return (dic)

if __name__ == '__main__':
    app.run(debug = True,host="0.0.0.0", port=8006)

#,host="0.0.0.0", port=8006