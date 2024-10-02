from bs4 import BeautifulSoup

def crawlerXML(filePath, encoding="latin_1"):
    with open(filePath, "r", encoding=encoding) as file:
        content = file.read()
    doc = BeautifulSoup(content, "xml")
    return doc